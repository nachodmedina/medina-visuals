"""El motor en la GPU (OpenGL vía moderngl): la misma partitura, dibujada con shaders.

La GPU calcula la escena píxel por píxel (polvo o disco, lente, onda, horizonte, silencio) y todo el
acabado (estela, fantasma, bloom, resplandor del horizonte, viñeta, punto de negro, grano). La CPU
sigue dibujando lo disperso (estrellas y fenómenos, en capas que la GPU combina), la firma y el
glitch. Las cuentas son las de styles.py, space.py y post.py; el resultado es equivalente al motor
de la CPU (no idéntico bit a bit: la GPU redondea distinto y baja el supersampling promediando).

Misma interfaz que render.Renderer: draw(f, i), .grid, .post (con .trail y .reset()).
"""
import os

import numpy as np

from .analysis import bursts
from .logo import Logo
from .noise import hash32
from .post import Post
from .presets import LOOKS, PRESETS, RED
from .render import Grid, camera, glitch, palette
from .space import Phenomena, Starfield
from .styles import FrameCtx

SHADERS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shaders")


def available():
    """¿Hay moderngl y un contexto de OpenGL 3.3+?"""
    try:
        import moderngl
        moderngl.create_standalone_context(require=330).release()
        return True
    except Exception:  # noqa: BLE001
        return False


def _src(name):
    with open(os.path.join(SHADERS, name)) as fh:
        return fh.read()


def _gauss(sigma):
    """Pesos de scipy.ndimage.gaussian_filter (truncate=4): mitad derecha del núcleo."""
    radius = int(4.0 * sigma + 0.5)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    phi = np.exp(-0.5 / sigma ** 2 * x ** 2)
    phi /= phi.sum()
    w = np.zeros(128, np.float32)
    w[:radius + 1] = phi[radius:]
    return radius, w


def _set(prog, **kw):
    for k, v in kw.items():
        if k in prog:
            prog[k].value = v


class _GPUPost:
    """El estado del acabado que vive en la GPU. Las curvas por cuadro y el banco de grano son
    los de post.Post (mismo azar)."""

    def __init__(self, gpu, cpu_post):
        self.gpu, self.p = gpu, cpu_post
        self.trail = cpu_post.trail
        self.fresh = True

    def reset(self):
        self.fresh = True


class GPURenderer:
    def __init__(self, score, W, H, title="MED1NA", font=None, ss=2, look="luz"):
        import moderngl
        self.score = score
        sysm = score.system
        self.grid = g = Grid(W, H, ss, score.fps, sysm.distance, arrays=False)
        self.S = max(1, ss)
        self.set_accent(RED)
        self.stars = Starfield(score.seed, sysm.stars)
        self.phenomena = Phenomena(score.seed, sysm.phenomena)
        self.logo = Logo(title, font, W, H)
        self.ctx = ctx = moderngl.create_standalone_context(require=330)
        vert = _src("quad.vert")
        prog = lambda frag: ctx.program(vertex_shader=vert, fragment_shader=_src(frag))
        self.p_scene, self.p_down = prog("scene.frag"), prog("lanczos.frag")
        self.p_hole, self.p_sing = prog("hole.frag"), prog("singularity.frag")
        tri = ctx.buffer(np.array([-1, -1, 3, -1, -1, 3], np.float32).tobytes())
        self.vao = {k: ctx.vertex_array(p, [(tri, "2f", "p")]) for k, p in
                    (("scene", self.p_scene), ("down", self.p_down), ("hole", self.p_hole),
                     ("sing", self.p_sing))}
        # escena a S x la salida; Lanczos horizontal (intermedio en 8 bits) y vertical
        self.t_scene = ctx.texture((self.S * g.tw, self.S * g.th), 4)
        self.f_scene = ctx.framebuffer([self.t_scene])
        self.t_half = ctx.texture((g.tw, self.S * g.th), 4)
        self.f_half = ctx.framebuffer([self.t_half])
        self.t_frame = ctx.texture((W, H), 4)
        self.f_frame = ctx.framebuffer([self.t_frame])
        # capas de la CPU (a la resolución de la grilla) y tablas del disco
        self.lay_stars = np.zeros((g.hh, g.ww), np.uint8)
        self.lay_phen = np.zeros((g.hh, g.ww), np.uint8)
        self.t_stars = ctx.texture((g.ww, g.hh), 1, dtype="u1")
        self.t_phen = ctx.texture((g.ww, g.hh), 1, dtype="u1")
        self.t_lanes = ctx.texture((192, 5), 1, dtype="f4")
        for t in (self.t_stars, self.t_phen, self.t_lanes):
            t.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.ids = np.arange(-96, 96, dtype=np.int32)
        self.post = None
        if LOOKS[look]:
            self._init_post(look)

    # ----------------------------------------------------------------- acabado ----
    def _init_post(self, look):
        import moderngl
        ctx, g, sc = self.ctx, self.grid, self.score
        cpu = Post(g.W, g.H, sc.fps, sc.analysis, look, sc.seed, sc.system)
        self.post = _GPUPost(self, cpu)
        vert = _src("quad.vert")
        mk = lambda frag: ctx.program(vertex_shader=vert, fragment_shader=_src(frag))
        self.p_trail, self.p_quarter, self.p_blur, self.p_final = (mk(f) for f in
                                                                  ("trail.frag", "quarter.frag", "blur.frag", "final.frag"))
        tri = ctx.buffer(np.array([-1, -1, 3, -1, -1, 3], np.float32).tobytes())
        for k, p in (("trail", self.p_trail), ("quarter", self.p_quarter), ("blur", self.p_blur),
                     ("final", self.p_final)):
            self.vao[k] = ctx.vertex_array(p, [(tri, "2f", "p")])
        W, H = g.W, g.H
        self.t_acc = [ctx.texture((W, H), 4, dtype="f4") for _ in range(2)]
        self.t_y = ctx.texture((W, H), 4, dtype="f4")
        self.f_trail = [ctx.framebuffer([self.t_y, self.t_acc[k]]) for k in range(2)]
        self.cur = 0
        w4, h4 = W // 4, H // 4
        self.t_q = [ctx.texture((w4, h4), 4, dtype="f4") for _ in range(4)]   # 1/4, temp, escala 0, escala 1
        self.f_q = [ctx.framebuffer([t]) for t in self.t_q]
        for t in self.t_q:
            t.filter = (moderngl.LINEAR, moderngl.LINEAR)
            t.repeat_x = t.repeat_y = False
        self.t_out = ctx.texture((W, H), 4)
        self.f_out = ctx.framebuffer([self.t_out])
        L = cpu.L
        self.blur = [_gauss(v * cpu.s / 4) for v in L["bloom_sig"]]
        self.t_grain = []
        for tile in cpu.tiles:
            t = ctx.texture((tile.shape[1], tile.shape[0]), 1, np.ascontiguousarray(tile).tobytes(), dtype="i1")
            t.filter = (moderngl.NEAREST, moderngl.NEAREST)
            self.t_grain.append(t)

    def _post(self, i, hole, bloom_k=1.0, ghost=True):
        ctx, P, cpu = self.ctx, self.post, self.post.p
        L, W, H = cpu.L, self.grid.W, self.grid.H
        accent = tuple(np.asarray(self.accent, np.float32) / 255)
        prev, nxt = self.cur, 1 - self.cur
        hole_u = tuple(float(v) for v in hole) if hole is not None else (0.0, 0.0, 0.0)
        # estela + fantasma
        self.t_frame.use(0)
        self.t_acc[prev].use(1)
        gpx = int(round(float(cpu.ghost[i]) * L["ghost_px"] * cpu.s)) if ghost else 0
        _set(self.p_trail, frame=0, acc_prev=1, fresh=int(P.fresh), decay=float(cpu.decay[i]),
             trail_g=float(L["trail_g"]), ghost_px=gpx,
             accent=accent, size=(W, H), has_hole=int(hole is not None), hole=hole_u)
        self.f_trail[nxt].use()
        self.vao["trail"].render()
        self.cur, P.fresh = nxt, False
        # bloom a 1/4, en dos escalas
        b = float(cpu.bloom[i]) * bloom_k
        if b > 0.01:
            self.t_y.use(0)
            _set(self.p_quarter, img=0)
            self.f_q[0].use()
            self.vao["quarter"].render()
            for k, (radius, w) in enumerate(self.blur):
                for src, dst, d in ((0, 1, (1, 0)), (1, 2 + k, (0, 1))):
                    self.t_q[src].use(0)
                    _set(self.p_blur, img=0, dir=d, radius=radius)
                    self.p_blur["w"].write(w.tobytes())
                    self.f_q[dst].use()
                    self.vao["blur"].render()
        # final
        T = float(cpu.T[i])
        nerv = min(1.0, max(0.0, (T - 0.35) / 0.65))
        amp = (0.30 + 0.35 * T + 0.35 * float(cpu.flare[i]) + 0.25 * float(cpu.kick[i])
               + 0.35 * nerv * float(cpu.hat[i])) * cpu.light
        gr = np.random.default_rng([cpu.seed, 708, i])
        tile = int(gr.integers(0, len(cpu.tiles)))
        oy, ox = (int(v) for v in gr.integers(0, cpu.pad, 2))
        self.t_y.use(0)
        self.t_q[2].use(1)
        self.t_q[3].use(2)
        self.t_grain[tile].use(3)
        _set(self.p_final, ytex=0, bl0=1, bl1=2, grain_tex=3, bloom=b, bloom_mix=float(L["bloom_mix"]),
             size=(W, H), has_hole=int(hole is not None), hole=hole_u, amp=float(amp), s_=float(cpu.s),
             vignette=float(L["vignette"]), black=float(L.get("black", 0.0)), accent=accent,
             grain_off=(ox, oy), grain=float(cpu.grain[i]))
        self.f_out.use()
        self.vao["final"].render()
        return self.f_out

    # ----------------------------------------------------------------- escena ----
    def set_accent(self, c):
        self.accent = c
        self.pal = palette(c)

    def camera(self, i):
        return camera(self.score, self.grid, i)

    def _lanes(self, fr, K):
        """Tablas por carril del disco (las de styles._orbits)."""
        st, c = fr.st, fr.c
        ids = self.ids
        z = np.zeros_like(ids)
        scale = 1 - 0.35 * (c["tens"] if c["closed"] else 0.0)
        g = np.random.default_rng([fr.seed, 2024])
        f1, f2, f3 = g.uniform(0, 2 * np.pi, 3)
        r_l = np.exp((ids + 0.5) / K) * scale
        om = (0.25 / np.maximum(r_l, 0.04)) ** 1.2
        spd = 0.6 + 0.8 * hash32(ids + 40, z + 4, 0)
        lane_ang = st["rot"] * 2.0 * om * spd
        nseg = np.floor(1 + 6 * hash32(ids + 70, z + 6, st["mut"]))
        dens = np.clip(0.5 + 0.5 * np.sin(ids * 0.37 + f3) * np.sin(ids * 0.11 + f1), 0, 1)
        bri = (0.18 + 0.95 * hash32(ids + 2000, z + 7, 0) ** 2.4) * (0.55 + 0.6 * dens) \
            * np.clip(0.3 / np.maximum(r_l, 0.02), 0.35, 2.2)
        thick = 0.35 + 0.8 * hash32(ids + 2200, z + 8, 0) ** 1.5
        self.t_lanes.write(np.stack([lane_ang, nseg, dens, bri, thick]).astype(np.float32).tobytes())
        return f1, f2, f3

    def draw(self, f, i):
        """Dibuja el cuadro i en `f` (H, W, 3) uint8 y lo devuelve."""
        sc, g = self.score, self.grid
        A = sc.analysis
        fs = sc.frames[i]
        P = PRESETS[fs.preset]
        st, c = fs.st, fs.c
        self.set_accent(sc.accent[i])
        roll, box = self.camera(i)
        fr = FrameCtx(P, st, c, fs.kc, i, roll, box, sc.seed, sc.fps, sc.system)
        sy = sc.system
        silent = bool(c["silent"])
        if P["style"] == "singularity":                 # antes de que exista nada: ni estrellas ni fenómenos
            self._singularity(i)
            return self._finish(f, i, A, P, None)
        has_phen = 0
        if not silent:                                  # lo disperso: en la CPU
            self.lay_stars.fill(0)
            self.stars.draw(self.lay_stars, g, fr)
            self.t_stars.write(self.lay_stars)
            Ly = A.get("layers")
            if Ly and Ly["order"] and any(Ly["on"][i, k] for k in Ly["order"]):
                self.lay_phen.fill(0)
                self.phenomena.draw(self.lay_phen, g, fr, Ly)
                self.t_phen.write(self.lay_phen)
                has_phen = 1
        if P["style"] == "hole":
            self._hole(fr)
            return self._finish(f, i, A, P, None)          # (el resplandor 2D no va: el borde es real)
        style = 1 if P["style"] == "disk" else 0
        K = P.get("K", 20.0)
        f1, f2, f3 = self._lanes(fr, K) if style == 1 else (0.0, 0.0, 0.0)
        N = int(round(P.get("N", 48) * (1 + 0.5 * c["chaos"])))
        x0, y0, x1, y1 = box or (0, 0, g.ww, g.hh)
        p = self.p_scene
        self.t_lanes.use(0)
        self.t_stars.use(1)
        self.t_phen.use(2)
        _set(p, lanes=0, stars_tex=1, phen_tex=2, has_phen=has_phen,
             out_size=(self.S * g.tw, self.S * g.th), box=(x0, y0, x1, y1), grid=(g.ww, g.hh),
             dist=float(sy.distance), unit=float(g.unit), gs=float(g.gs), tg=float(g.tg), roll=float(roll),
             style=style, silent=int(silent), closed=int(c["closed"]),
             rh=float(st["rh"]), tw_=float(st["tw"]), glow=float(st.get("glow", 0.0)), rot=float(st["rot"]),
             ph=float(st["ph"]), kick=c["kick"], chaos=c["chaos"], tens=c["tens"], hat=c["hat"],
             low=c["low"], mid=c["mid"], warp=float(c.get("warp", 0.0)),
             mut=int(st["mut"]), kc=int(fs.kc), hat_n=int(c["hat_n"]), frame_i=int(i),
             lf=float(sy.lens), turb=float(sy.turbulence), matter=float(sy.matter), accent_k=float(sy.accent),
             incl=float(sy.incl * P.get("open", 1.0)), tilt=float(P.get("tilt", sy.tilt)), K=float(K),
             f=(float(f1), float(f2), float(f3)),
             N=float(N), Kd=float(P.get("Kd", 7.5)), p_=float(P.get("p", 0.2)), dens_=float(P.get("dens", 0.7)),
             dot_=float(P.get("dot", 0.30)), pxcap=float(0.5 * P.get("px", 3.0) * g.gs),
             flick=int(P.get("flicker", 0)), plus_=int(bool(P.get("plus"))))
        p["pal"].write((self.pal.astype(np.float32) / 255).tobytes())
        self.f_scene.use()
        self.vao["scene"].render()
        hole = None
        if not silent:                                  # el agujero en la salida (como space.horizon)
            cx, cy = g.ww // 2, g.hh // 2
            sxx, syy = g.tw / (x1 - x0), g.th / (y1 - y0)
            rr = float(st["rh"]) * 1.12
            hole = ((cx - x0) * sxx, (cy - y0) * syy, rr * g.unit * syy)
        return self._finish(f, i, A, P, hole)

    # el agujero en 3D: la cámara cae hacia él con la tensión (el radio de la sombra en pantalla
    # sigue al horizonte de siempre, agrandado)
    HOLE_SCALE, FOCAL = 3.2, 2.0

    def _band(self):
        """La banda galáctica de este track: orientación (de la semilla) y carácter (del ADN)."""
        if getattr(self, "_bd", None) is None:
            sc, sy = self.score, self.score.system
            ax = sc.dna or {}
            g = np.random.default_rng([sc.seed, 5151])
            n = g.normal(size=3)
            n /= np.linalg.norm(n)
            vac, den, luz = ax.get("vacio", 0.4), ax.get("densidad", 0.5), ax.get("luz", 0.5)
            self._bd = dict(band_n=tuple(float(v) for v in n), band_w=float(0.10 + 0.22 * den),
                            band_k=float((0.08 + 0.15 * (1 - vac)) * (0.7 + 0.6 * luz)),
                            band_dust=float(0.4 + 0.6 * den))
        return self._bd

    @staticmethod
    def _spring(x, fps, f_up, z_up, f_dn=None, z_dn=None, rate=None):
        """Masa y resorte: sigue a `x` con inercia (frecuencia en Hz, amortiguación; 1 = sin rebote).
        Con parámetros distintos según el objetivo suba o baje. `rate`: el paso del tiempo en cada
        cuadro (0 = quieto: la dilatación del tiempo también detiene la inercia)."""
        f_dn, z_dn = f_dn or f_up, z_dn or z_up
        y = np.empty(len(x))
        p, v = float(x[0]), 0.0
        for k, tgt in enumerate(x):
            dt = (1.0 if rate is None else float(rate[k])) / fps
            f, z = (f_up, z_up) if tgt > p else (f_dn, z_dn)
            w = 2 * np.pi * f
            v += (w * w * (tgt - p) - 2 * z * w * v) * dt
            p += v * dt
            y[k] = p
        return y

    def _hole_motion(self):
        """El movimiento con masa (una vez por partitura, así cada cuadro se dibuja solo):
        - la sombra sigue a la tensión con inercia: la atracción tira despacio y sin rebote, el
          escape es rápido y rebota apenas; el kick es un latido con ataque suave;
        - la cámara orbita el agujero: acelera en los drops y casi se detiene en las tensiones;
        - el giro de cámara es una deriva lenta (sin los saltos de la cámara 2D);
        - el tiempo musical (en beats) para que las ondas del borde vayan con el track."""
        sc = self.score
        if getattr(self, "_hm", None) is not None:
            return self._hm
        A, fps, sy = sc.analysis, sc.fps, sc.system
        n = len(sc.frames)
        t = np.arange(n) / fps
        sp = self._spring
        ons = A["onsets"]
        bpm = 60 * fps / float(np.median(np.diff(ons))) if len(ons) > 1 else 128.0
        rel = self._release_gesture(A, fps, bpm, n)
        speed = rel["speed"]                                    # el tiempo de la imagen (dilatación)
        rh = sp(np.array([f.st["rh"] for f in sc.frames]), fps, 0.35, 1.0, 1.0, 0.6, rate=speed)
        pulse = sp(A["kick"].astype(np.float64), fps, 2.5, 0.45, rate=speed)
        closed = ~A["sub_on"].astype(bool) | A["break_zone"].astype(bool)
        energy = np.where(closed, 0.05, 0.6 * A["low"] + 0.4 * A["mid"]).astype(np.float64)
        e_s = np.clip(sp(energy, fps, 0.12, 1.0), 0, 1.5)
        ph = np.random.default_rng([sc.seed, 4242]).uniform(0, 2 * np.pi, 4)
        # la velocidad de la luz arranca cuando el tiempo vuelve a correr (no durante la quietud)
        wv = sc.warp.astype(np.float64) * (1 - rel["freeze"]) if getattr(sc, "warp", None) is not None \
            else np.zeros(n)
        warp = np.clip(sp(wv, fps, 1.2, 0.9, 0.7, 1.0, rate=speed), 0, 1)
        tt = np.cumsum(speed) / fps
        pulse = pulse * (1 - rel["freeze"])                     # con el tiempo quieto, el latido se detiene
        w0 = 2 * np.pi / (240 / sy.orbit)                       # una vuelta cada ~4 min
        az = np.cumsum(w0 * (0.25 + 1.2 * e_s) * speed) / fps + 0.12 * np.sin(2 * np.pi * tt / 41 + ph[1]) + ph[0]
        el = 1.0 + 0.25 * np.sin(2 * np.pi * tt / 63 + ph[2])
        chaos = sp(A["chaos"].astype(np.float64), fps, 0.1, 1.0)
        roll = np.deg2rad(2.8) * np.sin(2 * np.pi * tt / 23 + ph[3]) * (0.5 + 0.5 * chaos)
        t0 = ons[0] / fps if len(ons) else 0.0
        beats = (tt - t0) * bpm / 60
        self._hm = dict(rh=rh, pulse=pulse, warp=warp, az=az, el=el, roll=roll, beats=beats, **rel)
        return self._hm

    @staticmethod
    def _release_gesture(A, fps, bpm, n):
        """El gesto de cada liberación, en secuencia: el impacto; el tiempo se detiene un beat (solo
        el eco de luz da la vuelta al borde); el tiempo vuelve, acelera para recuperar lo perdido y
        sale la onda gravitacional. Más intenso cuanto más larga fue la tensión."""
        beat = 60 / bpm
        hold, ease, catch = 1.0 * beat, 1.5 * beat, 3.0 * beat
        freeze, bump = np.zeros(n), np.zeros(n)
        echo_k, echo_q = np.zeros(n), np.zeros(n)
        wave_amp, wave_dt = np.zeros(n), np.full(n, -1.0)
        idx = np.arange(n)
        sm = lambda x: np.clip(x, 0, 1) ** 2 * (3 - 2 * np.clip(x, 0, 1))
        for r in np.where(A["rise"])[0]:
            j = r - 1
            while j > 0 and A["tension"][j] > 0:
                j -= 1
            imp = float(np.clip(0.55 + 0.45 * (r - j) / (16 * fps), 0.55, 1.0))
            sl = slice(r, min(n, r + int((hold + ease + catch + 6) * fps)))
            dt = (idx[sl] - r) / fps
            f = imp * sm(dt / 0.06) * (1 - sm((dt - hold) / ease))
            freeze[sl] = np.maximum(freeze[sl], 0.95 * f)
            b = np.where((dt > hold) & (dt < hold + catch), np.sin(np.pi * (dt - hold) / catch) ** 2, 0.0)
            area_f = 0.95 * f.sum()
            if b.sum() > 0:
                bump[sl] += b * area_f / b.sum()
            T_echo = 1.25 * beat
            q = dt / T_echo
            ek = np.where(q < 1, imp * np.sin(np.pi * np.clip(q, 0, 1)) ** 0.7, 0.0)
            new = ek > echo_k[sl]
            echo_k[sl] = np.where(new, ek, echo_k[sl])
            echo_q[sl] = np.where(new, np.clip(q, 0, 1), echo_q[sl])
            dw = dt - hold
            wa = np.where(dw >= 0, 0.07 * imp * np.exp(-dw / 1.5) * sm(dw / 0.1), 0.0)
            neww = wa > wave_amp[sl]
            wave_amp[sl] = np.where(neww, wa, wave_amp[sl])
            wave_dt[sl] = np.where(neww, dw, wave_dt[sl])
        speed = 1 - freeze + bump
        return dict(speed=speed, freeze=freeze, echo_k=echo_k, echo_q=echo_q, wave_amp=wave_amp, wave_dt=wave_dt)

    def _traits(self):
        """El carácter del agujero de este track (de su sistema estelar / ADN)."""
        sy = self.score.system
        cl = lambda x: float(np.clip(x, 0, 1))
        return dict(ring_w=0.03 + 0.04 * cl((sy.matter - 0.6) / 0.8),           # densidad: el borde más grueso
                    ring_amp=0.006 + 0.024 * cl((sy.turbulence - 0.4) / 1.6),   # caos: ondula más
                    ring_ang=float(np.radians(180 + sy.tilt * 2.5)),             # dónde cae el lado encendido
                    ring_tint=1 - cl((sy.light - 0.7) / 0.7),                    # luz: blanco; oscuridad: violeta
                    star_dens=float(sy.stars) / 1000,                           # vacío: más o menos cielo
                    **self._band())

    def _hole(self, fr):
        """El agujero negro en 3D (shaders/hole.frag). Solo se mueve la cámara 3D (la 2D no va)."""
        g, sc = self.grid, self.score
        c, sy = fr.c, sc.system
        hm = self._hole_motion()
        i = fr.i
        # el peso: con cada kick (y con los graves) la sombra late, como un corazón pesado
        rs = float(hm["rh"][i]) * 1.12 * self.HOLE_SCALE * (1.0 + 0.05 * sy.shake * float(hm["pulse"][i]))
        tr = self._traits()
        _set(self.p_hole, out_size=(self.S * g.tw, self.S * g.th), box=(0, 0, g.ww, g.hh), grid=(g.ww, g.hh),
             dist=float(sy.distance), tg=float(g.tg), roll=float(hm["roll"][i]), tilt=float(sy.tilt),
             silent=int(bool(c["silent"])), closed=int(c["closed"]),
             camD=float(2.598 * self.FOCAL / max(rs, 0.05)), focal=self.FOCAL,
             elev=float(0.7 * np.arcsin(min(0.9, sy.incl)) * hm["el"][i]), azim=float(hm["az"][i]),
             glow=float(fr.st.get("glow", 0.0)), kick=float(np.clip(hm["pulse"][i], 0, 1.5)), tens=c["tens"],
             hat=c["hat"], hat_n=int(c["hat_n"]), ring_k=1.5, light=float(sy.light), warp=float(hm["warp"][i]),
             t_s=float(hm["beats"][i]) * 0.5, seed=int(sc.seed % 1000003),
             star_px=float(g.gs * 2 * sy.distance / (g.hh * self.FOCAL)),   # ángulo de un píxel de salida
             accent=tuple(np.asarray(self.accent, np.float32) / 255),
             echo_k=float(hm["echo_k"][i]), echo_ang=float(tr["ring_ang"] + 2 * np.pi * hm["echo_q"][i]),
             wave_amp=float(hm["wave_amp"][i]) * float(min(sy.turbulence, 1.6) + 0.4),
             wave_r=float(rs + 1.1 * max(hm["wave_dt"][i], 0.0)), **tr)
        self.f_scene.use()
        self.vao["hole"].render()

    def _sing_motion(self):
        """El movimiento de la Singularidad (una vez por partitura, así cada cuadro se dibuja solo).
        El track no tiene kick: todo sale de su atmósfera, y todo se mueve con inercia.
        - la respiración: las olas del grave (el punto crece y se enciende con ellas);
        - la presencia: el nivel del track, lento (el punto emerge y se apaga con el track);
        - la apertura: el caos (los agudos se abren) despierta el medio;
        - las capas: cada sonido que aparece enciende y agita su región del medio;
        - la cámara: deriva alrededor del punto, más cuanto más suena; se acerca a medida que se abre;
        - los pares: cada ataque agudo hace nacer un par que se separa, cae apenas y se aniquila;
        - los relámpagos: cada estallido del rango medio enciende la nube desde adentro (2 a 4
          descargas en medio segundo, más un resplandor que se apaga);
        - el crecimiento: el medio se extiende y se densifica a medida que el track avanza (más rápido
          cuanto más suena); en el final, cuando el track se apaga, todo se contrae y cae al punto."""
        if getattr(self, "_sm", None) is not None:
            return self._sm
        sc = self.score
        A, fps = sc.analysis, sc.fps
        n, sp = len(sc.frames), self._spring
        t = np.arange(n) / fps

        def norm(x, lo=5, hi=97):
            a, b = np.percentile(x, lo), np.percentile(x, hi)
            return np.clip((x - a) / (b - a + 1e-9), 0, 1)

        breath = np.clip(sp(norm(A["low"].astype(np.float64)), fps, 0.9, 0.8), 0, 1.3)
        lvl = A["level_db"].astype(np.float64)
        pres = np.clip(sp(np.clip(1 - (np.percentile(lvl, 95) - lvl) / 24, 0, 1), fps, 0.15, 1.0), 0, 1)
        opn = np.clip(sp(A["chaos"].astype(np.float64), fps, 0.08, 1.0), 0, 1)
        # las capas: actividad con ataque rápido y caída lenta (como un fluido que se calma)
        Ly = A.get("layers") or dict(act=np.zeros((n, 0)), on=np.zeros((n, 0), bool), center=np.zeros(0))
        G = min(8, Ly["act"].shape[1])
        g = np.random.default_rng([sc.seed, 7171])
        acts, lays = [], []
        for k in range(G):
            gate = np.clip(sp(Ly["on"][:, k].astype(np.float64), fps, 0.25, 1.0), 0, 1)
            a = np.clip(sp(np.clip(Ly["act"][:, k], 0, 1.5) * gate, fps, 2.0, 0.9, 0.3, 1.0), 0, 1.4)
            acts.append(a)
            d = g.normal(size=3)
            oc = float(np.clip(np.log2(max(Ly["center"][k], 300) / 300) / 4.5, 0, 1))   # 0 grave .. 1 agudo
            lays.append(dict(d=d / np.linalg.norm(d), r=1.1 - 0.75 * oc, w=0.16 - 0.07 * oc, white=0.1 + 0.6 * oc))
        acts = np.array(acts).reshape(G, n)
        e_tot = np.clip(sp(np.clip(acts.sum(0) / 1.5, 0, 1.5), fps, 0.3, 1.0), 0, 1.5)
        speed = 0.3 + 0.9 * e_tot + 0.4 * opn
        swirl = np.cumsum(0.035 * speed) / fps
        grow = np.cumsum(pres * (0.4 + e_tot + 0.6 * opn))
        grow = grow / max(grow[-1], 1e-9)
        tail = np.clip((t - 0.8 * t[-1]) / (0.05 * t[-1] + 1e-9), 0, 1)
        fall = np.clip(sp((1 - pres) * tail, fps, 0.2, 1.0), 0, 1)       # el final: todo cae al punto
        env_r = (0.35 + 0.8 * grow ** 0.8) * (1 - 0.75 * fall)
        dens_k = 0.45 + 0.55 * grow ** 0.7
        flow = np.cumsum(0.05 + 0.07 * opn + 0.09 * e_tot + 1.2 * fall) / fps
        turb = np.cumsum(0.15 + 0.65 * e_tot + 0.3 * opn) / fps
        warp_k = 0.5 + 0.5 * e_tot
        # la cámara con masa
        ph = g.uniform(0, 2 * np.pi, 8)
        az = np.cumsum(2 * np.pi / 150 * speed) / fps + ph[0]
        el = 0.38 * np.sin(2 * np.pi * t / 83 + ph[1]) + 0.12 * np.sin(2 * np.pi * t / 37 + ph[2])
        D = sp(2.7 - 0.6 * opn - 0.15 * e_tot, fps, 0.05, 1.0) - 0.04 * breath
        C = np.stack([np.sin(az) * np.cos(el), np.sin(el), -np.cos(az) * np.cos(el)], 1) * D[:, None]
        tgt = 0.07 * np.stack([np.sin(2 * np.pi * t / 29 + ph[3]), np.sin(2 * np.pi * t / 41 + ph[4]),
                               np.sin(2 * np.pi * t / 53 + ph[5])], 1)
        fw = tgt - C
        fw /= np.linalg.norm(fw, axis=1, keepdims=True)
        rt = np.cross(np.array([0.0, 1.0, 0.0]), fw)
        rt /= np.linalg.norm(rt, axis=1, keepdims=True)
        up = np.cross(fw, rt)
        roll = np.deg2rad(4) * np.sin(2 * np.pi * t / 47 + ph[6])
        cr, sr = np.cos(roll)[:, None], np.sin(roll)[:, None]
        rt, up = rt * cr + up * sr, up * cr - rt * sr
        ax = g.normal(size=3)
        ax[1] = abs(ax[1]) + 1.5                                # el giro, más o menos horizontal
        ev = []
        born = np.flatnonzero(np.diff(A["hat_n"], prepend=A["hat_n"][:1]) > 0)
        for b in born:
            rg = np.random.default_rng([sc.seed, 6061, int(b)])
            u = rg.uniform(size=5)
            d = rg.normal(size=3)
            ev.append((int(b), int((0.35 + 0.45 * u[0]) * fps), (0.08 + 0.55 * u[1] ** 1.5) * d / np.linalg.norm(d),
                       2 * np.pi * u[2], 0.35 + 0.65 * u[3] ** 2, 0.012 + 0.02 * u[4]))
        fl = []
        for b, k in bursts(A, fps):
            rg = np.random.default_rng([sc.seed, 9090, b])
            d1, d2 = rg.normal(size=3), rg.normal(size=3)
            a = d1 / np.linalg.norm(d1) * env_r[b] * rg.uniform(0.35, 0.9)
            c = a + d2 / np.linalg.norm(d2) * rg.uniform(0.25, 0.6)
            m = int(rg.integers(2, 5))
            st = np.concatenate([[0.0], np.cumsum(rg.uniform(0.06, 0.18, m - 1))])
            amp = np.concatenate([[1.0], rg.uniform(0.4, 1.0, m - 1)])
            tt = np.arange(int(1.4 * fps)) / fps
            env = sum(a_ * np.exp(-(tt - s_) / 0.06) * (tt >= s_) for a_, s_ in zip(amp, st))
            env = env + 0.15 * np.exp(-tt / 0.35)
            fl.append((b, a, c, k * env))
        self._sm = dict(breath=breath, pres=pres, open=opn, acts=acts, lays=lays, swirl=swirl, flow=flow,
                        turb=turb, warp_k=warp_k, C=C, fw=fw, rt=rt, up=up, ax=ax / np.linalg.norm(ax),
                        events=ev, env_r=env_r, dens_k=dens_k, flashes=fl)
        return self._sm

    def _noise3d(self):
        """Ruido 3D periódico de 4 canales (de la semilla del track): la materia del medio."""
        if getattr(self, "t_noise", None) is None:
            import moderngl
            from scipy.ndimage import gaussian_filter
            N = 128
            g = np.random.default_rng([self.score.seed, 3131])
            vol = np.empty((N, N, N, 4), np.uint8)
            for c in range(4):
                x = gaussian_filter(g.standard_normal((N, N, N)).astype(np.float32), 2.2, mode="wrap")
                rk = np.empty(x.size, np.float32)                   # ecualizado: valores parejos 0..1
                rk[np.argsort(x, axis=None)] = np.linspace(0, 255, x.size)
                vol[..., c] = rk.reshape(x.shape).astype(np.uint8)
            self.t_noise = self.ctx.texture3d((N, N, N), 4, vol.tobytes())
            self.t_noise.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self.t_noise.repeat_x = self.t_noise.repeat_y = self.t_noise.repeat_z = True
        return self.t_noise

    def _singularity(self, i):
        """La Singularidad (shaders/singularity.frag)."""
        g, sc = self.grid, self.score
        sm = self._sing_motion()
        br, pr = float(sm["breath"][i]), float(sm["pres"][i])
        C, fw, rt, up = sm["C"][i], sm["fw"][i], sm["rt"][i], sm["up"][i]
        fps, focal = sc.fps, 1.6
        sw = float(sm["swirl"][i])
        # las capas: su región gira con el medio (al ritmo del giro a su radio)
        ld, lp = np.zeros((8, 4), np.float32), np.zeros((8, 4), np.float32)
        kx = sm["ax"]
        for k, L in enumerate(sm["lays"]):
            a = -sw * (1.0 + 0.9 * np.exp(-L["r"] / 0.7))
            d = L["d"]
            d = d * np.cos(a) + np.cross(kx, d) * np.sin(a) + kx * np.dot(kx, d) * (1 - np.cos(a))
            ld[k] = (*d, float(sm["acts"][k, i]) * pr)
            lp[k] = (L["r"], L["w"], L["white"], 0.0)
        # los pares, proyectados con la cámara de este cuadro
        ea, eb = np.zeros((48, 4), np.float32), np.zeros((48, 2), np.float32)
        k = 0
        for f0, life, pos, axr, bri, sep in reversed(sm["events"]):   # los más nuevos primero
            if f0 > i:
                continue
            q = (i - f0) / life
            if q >= 1:
                if i - f0 > 2 * fps:
                    break
                continue
            o = pos * np.exp(-0.6 * (i - f0) / fps) - C              # caen apenas hacia el punto
            z = float(np.dot(o, fw))
            if z <= 0.05:
                continue
            ea[k] = (np.dot(o, rt) * focal / z, -np.dot(o, up) * focal / z, axr, bri * pr)
            eb[k] = (q, sep * focal / z)
            k += 1
            if k == 48:
                break
        # los relámpagos activos (los dos más intensos)
        fa, fb = np.zeros((2, 4), np.float32), np.zeros((2, 4), np.float32)
        on = sorted(((env[i - b], a, c) for b, a, c, env in sm["flashes"] if 0 <= i - b < len(env)),
                    key=lambda x: -x[0])[:2]
        for j, (I, a, c) in enumerate(on):
            fa[j] = (*a, I * pr)
            fb[j] = (*c, 0.0)
        sy = sc.system
        p = self.p_sing
        self._noise3d().use(0)
        _set(p, out_size=(self.S * g.tw, self.S * g.th), px=(2.0 / g.th, 2.0 / g.th), aspect=float(g.tw / g.th),
             focal=focal, cam=tuple(C), fw=tuple(fw), rt=tuple(rt), up=tuple(up), noise=0,
             swirl_ax=tuple(kx), swirl=sw, flow=float(sm["flow"][i]), turb=float(sm["turb"][i]),
             warp_k=float(sm["warp_k"][i]), lens_k=0.012,
             L0=pr * (0.8 + 0.5 * br), med_k=0.4 * (0.35 + 0.65 * float(sm["open"][i])), sigma=1.2, ly_k=25.0,
             core_k=pr * (0.9 + 0.6 * br), halo_k=pr * (0.16 + 0.22 * br), core_px=1.5, halo_px=float(7 + 6 * br),
             tint=float(1 - np.clip((sy.light - 0.7) / 0.7, 0, 1) * 0.5),
             accent=tuple(np.asarray(self.accent, np.float32) / 255),
             env_r=float(sm["env_r"][i]), dens_k=float(sm["dens_k"][i]), n_fl=len(on), fl_col=(0.66, 0.6, 0.8),
             n_ly=len(sm["lays"]), n_ev=k)
        p["fl_a"].write(fa.tobytes())
        p["fl_b"].write(fb.tobytes())
        p["ly_d"].write(ld.tobytes())
        p["ly_p"].write(lp.tobytes())
        p["ev_a"].write(ea.tobytes())
        p["ev_b"].write(eb.tobytes())
        self.f_scene.use()
        self.vao["sing"].render()

    def _finish(self, f, i, A, P, hole):
        sc, g = self.score, self.grid
        # supersampling -> el cuadro con Lanczos (con el margen de la grilla en negro)
        self.f_half.use()
        self.t_scene.use(0)
        _set(self.p_down, img=0, dir=(1, 0), scale=float(self.S), offset=(0, 0))
        self.vao["down"].render()
        self.f_frame.use()
        self.f_frame.clear(0.0, 0.0, 0.0, 1.0)
        self.f_frame.viewport = (g.x_off, g.ty0, g.tw, g.th)
        self.t_half.use(0)
        _set(self.p_down, img=0, dir=(0, 1), scale=float(self.S), offset=(g.x_off, g.ty0))
        self.vao["down"].render()
        self.f_frame.viewport = (0, 0, g.W, g.H)
        fbo = (self._post(i, hole, P.get("bloom", 1.0), P.get("ghost", True)) if self.post is not None
               else self.f_frame)
        f[:] = np.frombuffer(fbo.read(components=3, alignment=1), np.uint8).reshape(g.H, g.W, 3)
        glitch(f, A, i, sc.seed, g.s, P)
        if P.get("title", True):
            self.logo.draw(f, float(A["kick"][i]) if P.get("pulse", True) else 0.0)
        return f
