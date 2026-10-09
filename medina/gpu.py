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

from .analysis import bursts, layer_attacks
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


def _h01(c, r, k, seed):
    """El hash de las celdas de capsules.frag (la misma cuenta en enteros de 32 bits)."""
    M = 0xFFFFFFFF

    def hu(x):
        x ^= x >> 16
        x = (x * 0x7FEB352D) & M
        x ^= x >> 15
        x = (x * 0x846CA68B) & M
        return x ^ (x >> 16)
    inner = (((r & M) * 0x85EBCA77) & M) ^ (((seed & M) * 0xC2B2AE3D) & M) ^ (((k & M) * 0x27D4EB2F) & M)
    return (hu((((c & M) * 0x9E3779B1) & M) ^ hu(inner)) & 0xFFFFFF) / 16777215.0


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
        self.p_infl, self.p_caps = prog("inflation.frag"), prog("capsules.frag")
        tri = ctx.buffer(np.array([-1, -1, 3, -1, -1, 3], np.float32).tobytes())
        self.vao = {k: ctx.vertex_array(p, [(tri, "2f", "p")]) for k, p in
                    (("scene", self.p_scene), ("down", self.p_down), ("hole", self.p_hole),
                     ("sing", self.p_sing), ("infl", self.p_infl), ("caps", self.p_caps))}
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

    def _post(self, i, hole, bloom_k=1.0, ghost=True, trail=True, grain_k=1.0):
        ctx, P, cpu = self.ctx, self.post, self.post.p
        L, W, H = cpu.L, self.grid.W, self.grid.H
        accent = tuple(np.asarray(self.accent, np.float32) / 255)
        prev, nxt = self.cur, 1 - self.cur
        hole_u = tuple(float(v) for v in hole) if hole is not None else (0.0, 0.0, 0.0)
        # estela + fantasma
        self.t_frame.use(0)
        self.t_acc[prev].use(1)
        gpx = int(round(float(cpu.ghost[i]) * L["ghost_px"] * cpu.s)) if ghost else 0
        _set(self.p_trail, frame=0, acc_prev=1, fresh=int(P.fresh), decay=float(cpu.decay[i]) if trail else 0.0,
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
             grain_off=(ox, oy), grain=float(cpu.grain[i]) * grain_k)
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
        if P["style"] == "inflation":
            self._inflation(i)
            return self._finish(f, i, A, P, None)
        if P["style"] == "capsules":
            self._capsules(i)
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
            m = 1 if w * dt <= 1.0 else int(np.ceil(w * dt / 0.5))   # pasos grandes: en partes (estable)
            h = dt / m
            for _ in range(m):
                v += (w * w * (tgt - p) - 2 * z * w * v) * h
                p += v * h
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
        wave_amp, wave_dt, wave_imp = np.zeros(n), np.full(n, -1.0), np.zeros(n)
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
            wave_imp[sl] = np.where(neww, imp, wave_imp[sl])
        speed = 1 - freeze + bump
        return dict(speed=speed, freeze=freeze, echo_k=echo_k, echo_q=echo_q, wave_amp=wave_amp, wave_dt=wave_dt,
                    wave_imp=wave_imp)

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
          cuanto más suena); en el final, cuando el track se apaga, todo se contrae y cae al punto;
        - los truenos: cada tom es una onda de presión que sale del punto, en un medio denso: avanza
          y se frena, y lo que empuja queda desplazado y se asienta despacio; el punto late (sin
          rebote) y la cámara recibe el empujón como una masa grande."""
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
            a = np.clip(sp(np.clip(Ly["act"][:, k], 0, 1.5) * gate, fps, 0.9, 1.0, 0.2, 1.0), 0, 1.4)
            acts.append(a)
            d = g.normal(size=3)
            oc = float(np.clip(np.log2(max(Ly["center"][k], 300) / 300) / 4.5, 0, 1))   # 0 grave .. 1 agudo
            lays.append(dict(d=d / np.linalg.norm(d), r=1.1 - 0.75 * oc, w=0.16 - 0.07 * oc, white=0.1 + 0.6 * oc))
        acts = np.array(acts).reshape(G, n)
        e_tot = np.clip(sp(np.clip(acts.sum(0) / 1.5, 0, 1.5), fps, 0.15, 1.0), 0, 1.5)
        speed = 0.3 + 0.9 * e_tot + 0.4 * opn
        swirl = np.cumsum(0.035 * speed) / fps
        grow = np.cumsum(pres * (0.4 + e_tot + 0.6 * opn))
        grow = grow / max(grow[-1], 1e-9)
        tail = np.clip((t - 0.8 * t[-1]) / (0.05 * t[-1] + 1e-9), 0, 1)
        fall = np.clip(sp((1 - pres) * tail, fps, 0.2, 1.0), 0, 1)       # el final: todo cae al punto
        env_r = (0.35 + 0.8 * grow ** 0.8) * (1 - 0.75 * fall)
        dens_k = 0.45 + 0.55 * grow ** 0.7
        flow = np.cumsum(0.05 + 0.07 * opn + 0.09 * e_tot + 1.2 * fall) / fps
        turb = np.cumsum(0.08 + 0.36 * e_tot + 0.16 * opn) / fps
        warp_k = 0.7 + 0.6 * e_tot
        # la cámara con masa
        ph = g.uniform(0, 2 * np.pi, 8)
        az = np.cumsum(2 * np.pi / 150 * speed) / fps + ph[0]
        el = 0.38 * np.sin(2 * np.pi * t / 83 + ph[1]) + 0.12 * np.sin(2 * np.pi * t / 37 + ph[2])
        toms = A.get("toms", [])
        imp = np.zeros(n)
        for b, k in toms:
            imp[b:b + int(0.12 * fps)] = np.maximum(imp[b:b + int(0.12 * fps)], k)
        beat = np.clip(sp(imp, fps, 2.2, 0.9, 0.6, 1.0), 0, 1.5)        # el latido del punto: sin rebote
        press = np.zeros(n)                                             # la presión de la onda dura un rato
        for b, k in toms:
            m = n - b
            press[b:] = np.maximum(press[b:], k * np.exp(-np.arange(m) / (0.9 * fps)))
        push = sp(press, fps, 0.35, 0.9, 0.25, 1.0)                      # la cámara, empujada como una masa
        D = sp(2.7 - 0.6 * opn - 0.15 * e_tot, fps, 0.05, 1.0) - 0.04 * breath + 0.22 * push
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
                        toms=toms, beat=beat,
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
        # los truenos activos (los más recientes): el frente se frena y se ensancha, la onda se apaga
        wv, wk = np.zeros((4, 4), np.float32), np.zeros((4, 2), np.float32)
        nw = 0
        for b, kt in reversed(sm["toms"]):
            t = (i - b) / fps
            if t < 0:
                continue
            if t > 4.5 or nw == 4:
                break
            R = 0.03 + 0.8 * t ** 0.65
            wv[nw] = (R, 0.9 * kt * pr * np.exp(-t / 1.3) / (1 + 0.5 * R), 0.12 + 0.10 * t, (b * 0.618) % 1.0)
            wk[nw] = (0.10 * kt * pr * (1 - np.exp(-t / 0.25)) * np.exp(-t / 1.8), 0.25 + 0.2 * t)
            nw += 1
        bt = float(sm["beat"][i])
        sy = sc.system
        p = self.p_sing
        self._noise3d().use(0)
        _set(p, out_size=(self.S * g.tw, self.S * g.th), px=(2.0 / g.th, 2.0 / g.th), aspect=float(g.tw / g.th),
             focal=focal, cam=tuple(C), fw=tuple(fw), rt=tuple(rt), up=tuple(up), noise=0,
             swirl_ax=tuple(kx), swirl=sw, flow=float(sm["flow"][i]), turb=float(sm["turb"][i]),
             warp_k=float(sm["warp_k"][i]), lens_k=0.012,
             L0=pr * (0.8 + 0.5 * br) * (1 + 1.4 * bt), med_k=0.4 * (0.35 + 0.65 * float(sm["open"][i])),
             sigma=2.0, ly_k=25.0,
             core_k=pr * (0.9 + 0.6 * br) * (1 + 1.8 * bt), halo_k=pr * (0.16 + 0.22 * br) * (1 + 3.0 * bt),
             core_px=1.5, halo_px=float(7 + 6 * br + 9 * bt), n_wv=nw,
             tint=float(1 - np.clip((sy.light - 0.7) / 0.7, 0, 1) * 0.5),
             accent=tuple(np.asarray(self.accent, np.float32) / 255),
             env_r=float(sm["env_r"][i]), dens_k=float(sm["dens_k"][i]), n_fl=len(on), fl_col=(0.66, 0.6, 0.8),
             n_ly=len(sm["lays"]), n_ev=k)
        p["wv"].write(wv.tobytes())
        p["wk"].write(wk.tobytes())
        p["fl_a"].write(fa.tobytes())
        p["fl_b"].write(fb.tobytes())
        p["ly_d"].write(ld.tobytes())
        p["ly_p"].write(lp.tobytes())
        p["ev_a"].write(ea.tobytes())
        p["ev_b"].write(eb.tobytes())
        self.f_scene.use()
        self.vao["sing"].render()

    def _infl_motion(self):
        """El movimiento de la Inflación (una vez por partitura, así cada cuadro se dibuja solo): el
        viaje a través del campo. Nada elástico: avanza siempre, en una dirección, cada vez más rápido.
        - la velocidad: un avance lento siempre (el rodar lento); crece con la energía; cada kick es un
          empuje hacia adelante que se apaga sin volver; en la liberación el tiempo se detiene un beat y
          después la velocidad crece exponencialmente medio compás (un "e-fold") y se asienta;
        - el estiramiento: las fluctuaciones crecen y se congelan, sobre todo en los e-folds;
        - la cámara: avanza y gira muy despacio, siempre para el mismo lado (sin vaivén ni rebotes);
        - la tensión: el campo se erosiona y se oscurece (solo quedan las fluctuaciones finas);
        - las capas: cada sonido enciende su escala (grave: lo grueso; agudo: lo fino);
        - las luces: cada capa es una luz dentro de la red que viaja con la cámara en su lugar del espacio
          de adelante (no se ve la fuente: solo los filamentos que le pasan cerca), con la presencia de la
          capa; lo grave, más grande. Cada ataque de una capa enciende un destello breve en su zona, fijo
          en el espacio (la cámara pasa de largo);
        - el final: el recalentamiento (el campo se vuelve luz) y el apagón."""
        if getattr(self, "_im", None) is not None:
            return self._im
        sc = self.score
        A, fps, sy = sc.analysis, sc.fps, sc.system
        n, sp = len(sc.frames), self._spring
        ons = A["onsets"]
        bpm = 60 * fps / float(np.median(np.diff(ons))) if len(ons) > 1 else 128.0
        rel = self._release_gesture(A, fps, bpm, n)
        speed = rel["speed"]                                    # el paso del tiempo (0 en la quietud)
        kick = np.clip(A["kick"].astype(np.float64), 0, 1.5)    # el empuje de cada kick (sin resorte)
        closed = ~A["sub_on"].astype(bool) | A["break_zone"].astype(bool)
        energy = np.where(closed, 0.05, 0.6 * A["low"] + 0.4 * A["mid"]).astype(np.float64)
        e_s = np.clip(sp(energy, fps, 0.12, 1.0, rate=speed), 0, 1.5)
        # se arma despacio; en la liberación se suelta de golpe (antes de que el tiempo se detenga)
        tens = np.clip(sp(np.where(closed, A["tension"], 0.0).astype(np.float64), fps, 0.5, 1.0, 4.0, 1.0), 0, 1)
        hat = np.clip(sp(A["hat"].astype(np.float64), fps, 6.0, 0.9, rate=speed), 0, 1.2)
        # el e-fold: desde que el tiempo vuelve, la velocidad crece exponencialmente medio compás
        T = 2 * 60 / bpm
        dw, imp = rel["wave_dt"], rel["wave_imp"]
        efold = np.where(dw < 0, 0.0, np.where(dw < T, (np.exp(4 * np.maximum(dw, 0) / T) - 1) / (np.exp(4) - 1),
                                               np.exp(-(dw - T) / 1.6))) * imp
        v = (0.12 + 0.55 * e_s + 0.45 * kick) * (0.6 + 0.4 * sy.orbit) + 3.2 * efold
        z = np.cumsum((0.04 + 0.08 * e_s + 0.7 * efold) * speed) / fps
        # la cámara: gira despacio alrededor de un eje fijo (siempre para el mismo lado) y avanza
        g = np.random.default_rng([sc.seed, 5252])
        f0 = g.normal(size=3)
        f0 /= np.linalg.norm(f0)
        ax = np.cross(f0, g.normal(size=3))
        ax /= np.linalg.norm(ax)
        th = np.cumsum(0.035 * (0.4 + e_s) * speed) / fps
        c, s_ = np.cos(th)[:, None], np.sin(th)[:, None]
        fwv = f0 * c + np.cross(ax, f0) * s_                    # (ax ⟂ f0: rotación simple)
        cam = np.cumsum(fwv * (v * speed)[:, None], axis=0) / fps
        upw = np.array([0.0, 1.0, 0.0])
        rt = np.cross(upw, fwv)
        rt /= np.linalg.norm(rt, axis=1, keepdims=True)
        up = np.cross(fwv, rt)
        # las capas
        Ly = A.get("layers") or dict(act=np.zeros((n, 0)), on=np.zeros((n, 0), bool), center=np.zeros(0))
        G = min(4, Ly["act"].shape[1])
        acts, pres, xs = np.zeros((G, n)), np.zeros((G, n)), np.zeros(G)
        for k in range(G):
            gate = np.clip(sp(Ly["on"][:, k].astype(np.float64), fps, 0.25, 1.0), 0, 1)
            a = np.clip(Ly["act"][:, k], 0, 1.5) * gate
            acts[k] = np.clip(sp(a, fps, 2.0, 0.9, 0.3, 1.0), 0, 1.4)
            pres[k] = np.clip(sp(a, fps, 0.5, 1.0, 0.12, 1.0), 0, 1.2)       # la presencia: lenta
            xs[k] = 0.15 + 0.75 * float(np.clip(np.log2(max(Ly["center"][k], 300) / 300) / 4.5, 0, 1))
        # las luces de las capas: repartidas alrededor del centro, a su profundidad, y giran muy despacio
        # (siempre para el mismo lado, con el tiempo de la imagen)
        oc = (xs - 0.15) / 0.75                                 # 0 grave .. 1 agudo
        lr = 0.1 + 0.6 * (1 - oc) ** 1.5                        # radio de cada luz
        lw = lr / max(lr.max(), 1e-9) if G else lr              # peso: la más grande domina
        gl = np.random.default_rng([sc.seed, 4343])
        phi0 = gl.uniform(0, 2 * np.pi) + 2 * np.pi * np.arange(G) / max(G, 1) + gl.normal(0, 0.3, G)
        rho = gl.uniform(0.25, 0.6, G)
        dep = 0.7 + 0.5 * gl.uniform(size=G)
        spin = np.cumsum(0.05 * (0.4 + e_s) * speed) / fps * (1.0 if gl.uniform() < 0.5 else -1.0)
        # los destellos: en el espacio de adelante, cerca de la luz de su capa. Para no saturar, las capas
        # chicas (agudas) solo encienden sus ataques más fuertes, y cada luz necesita un respiro entre
        # destellos (más largo cuanto más grande: la del synth, más o menos un beat)
        tau = np.cumsum(speed) / fps                            # el tiempo de la imagen
        fl, last = [], np.full(G, -(10 ** 9))
        for b, k, s in layer_attacks(Ly, fps, A.get("silent")):
            if k >= G or s * lw[k] < 0.3 or b - last[k] < (0.2 + 0.25 * lw[k]) * fps:
                continue
            last[k] = b
            rg = np.random.default_rng([sc.seed, 4444, k, b])
            ph = phi0[k] + spin[b] + rg.normal(0, 0.6)
            r_ = float(np.clip(rho[k] + rg.normal(0, 0.2), 0.05, 0.95))
            t_ = float(np.clip(dep[k] + rg.normal(0, 0.25), 0.45, 1.8))
            pc = self._ahead(ph, r_, t_)
            fl.append((b, cam[b] + rt[b] * pc[0] + up[b] * pc[1] + fwv[b] * pc[2], s * lw[k],
                       (0.4 + 0.3 * lw[k]) * lr[k], 0.08 + 0.08 * lw[k]))
        # el final: recalentamiento y apagón
        fd = np.clip(A["fade"].astype(np.float64), 0, 1)
        smt = lambda x: np.clip(x, 0, 1) ** 2 * (3 - 2 * np.clip(x, 0, 1))
        reheat = 0.4 * smt(fd / 0.5) * (1 - smt((fd - 0.5) / 0.5))
        fade_k = 1 - smt((fd - 0.45) / 0.55)
        self._im = dict(z=z, kick=kick, tens=tens, hat=hat, acts=acts, xs=xs, cam=cam, v=v,
                        fw=fwv, rt=rt, up=up, reheat=reheat, fade_k=fade_k, pres=pres, lr=lr, lw=lw, phi0=phi0,
                        rho=rho, dep=dep, spin=spin, tau=tau, flashes=fl,
                        fl_b=np.array([f[0] for f in fl], np.int64))
        return self._im

    LANT_K, FLASH_K = 7.0, 15.0
    INFL_FOCAL = 1.4

    def _ahead(self, ph, r, t):
        """Un punto del espacio de adelante (coordenadas de la cámara): a `r` del centro de la pantalla
        (en media altura, más abierto en horizontal), en la dirección `ph`, a profundidad `t`."""
        return np.array([1.4 * r * np.cos(ph), r * np.sin(ph), self.INFL_FOCAL]) * t / self.INFL_FOCAL

    def _infl_lights(self, i):
        """Las luces de este cuadro (las 6 más intensas): posición en coordenadas de la cámara,
        intensidad y radio."""
        im, fps = self._im, self.score.fps
        out = []
        for k in range(len(im["xs"])):
            I = self.LANT_K * float(im["pres"][k, i] * im["lw"][k])
            if I < 0.01:
                continue
            pc = self._ahead(im["phi0"][k] + im["spin"][i], im["rho"][k], im["dep"][k])
            out.append((*pc, I, im["lr"][k]))
        j0, j1 = np.searchsorted(im["fl_b"], [i - 4 * fps, i + 1])
        C, tau = im["cam"][i], im["tau"]
        for b, pw, s, R, dec in im["flashes"][j0:j1]:
            age = tau[i] - tau[b]
            I = self.FLASH_K * s * np.exp(-age / dec)
            if I < 0.01:
                continue
            d = pw - C
            pz = float(np.dot(d, im["fw"][i]))
            if pz < 0.05:                                       # ya quedó atrás
                continue
            out.append((float(np.dot(d, im["rt"][i])), float(np.dot(d, im["up"][i])), pz, I, R))
        return sorted(out, key=lambda x: -x[3])[:6]

    INFL_K = 30.0
    INFL_SIGMA = 0.3                             # absorción de los filamentos
    INFL_BODY, INFL_GLOW, INFL_BODY_W = 1.0, 0.15, 5.0   # el gas: absorción, luz de los nudos, ancho
    INFL_LIT = 0.35                              # cuánto iluminan las luces de los sonidos al gas
    INFL_THR, INFL_FINE = 0.032, (0.14, 0.55)    # ancho de los filamentos; peso de lo fino (con los hats)

    def _inflation(self, i):
        """La Inflación (shaders/inflation.frag)."""
        g, sc = self.grid, self.score
        im, sy = self._infl_motion(), sc.system
        z = float(im["z"][i])
        tens = float(im["tens"][i])
        ly = np.zeros((4, 4), np.float32)
        for k in range(len(im["xs"])):
            ly[k] = (im["xs"][k], im["acts"][k, i], 0.0, 0.0)
        la, lb = np.zeros((6, 4), np.float32), np.zeros((6, 4), np.float32)
        lights = self._infl_lights(i)
        for j, (x, y, t_, I, R) in enumerate(lights):
            la[j] = (x, y, t_, I)
            lb[j] = (R, 0.0, 0.0, 0.0)
        p = self.p_infl
        self._noise3d().use(0)
        _set(p, out_size=(self.S * g.tw, self.S * g.th), aspect=float(g.tw / g.th), focal=self.INFL_FOCAL,
             fw=tuple(im["fw"][i]), rt=tuple(im["rt"][i]), up=tuple(im["up"][i]), cam=tuple(im["cam"][i]), noise=0,
             zf=float(z - np.floor(z)), zi=int(np.floor(z)), zt=z, base_f=0.2, hz_off=1.0 - 0.6 * 4,
             dens_k=self.INFL_K * (1 - 0.5 * tens) * (0.7 + 0.3 * sy.matter) * (1 + 1.5 * float(sc.frames[i].st.get("glow", 0.0))),
             gas_r=1.0 / (1 + 1.5 * float(sc.frames[i].st.get("glow", 0.0))),
             thr=self.INFL_THR * (1 - 0.45 * tens), pxs=2.0 / g.th,
             sigma=self.INFL_SIGMA, body_k=self.INFL_BODY, glow_k=self.INFL_GLOW, body_w=self.INFL_BODY_W, lit_k=self.INFL_LIT, depth=2.6, expo=1.6,
             fine_k=self.INFL_FINE[0] + self.INFL_FINE[1] * float(im["hat"][i]), coarse_k=1.0, kick=float(max(im["kick"][i], 0)),
             tint=float(1 - np.clip((sy.light - 0.7) / 0.7, 0, 1) * 0.6),
             light=float(max(sy.light, 0.75)), reheat=float(im["reheat"][i]), fade_k=float(im["fade_k"][i]),
             accent=tuple(np.asarray(self.accent, np.float32) / 255), n_ly=len(im["xs"]), n_lt=len(lights))
        p["ly"].write(ly.tobytes())
        p["lt_a"].write(la.tobytes())
        p["lt_b"].write(lb.tobytes())
        self.f_scene.use()
        self.vao["infl"].render()

    CAPS_FOCAL, CAPS_D = 2.0, 4.0

    def _caps_traits(self):
        """El carácter de las cápsulas de este track (del ADN): separación y vacío, forma, orden, la
        familia de color de la película. La separación sigue al lado corto (en vertical, más columnas)."""
        ax = self.score.dna or {}
        g = lambda k: float(ax.get(k, 0.5))
        asp = self.grid.tw / self.grid.th
        return dict(sp=(0.95 + 0.25 * g("vacio") - 0.15 * g("densidad")) * float(np.sqrt(np.clip(asp, 0.5, 1.0))),
                    vac=0.15 + 0.35 * g("vacio"), rad=0.16 + 0.05 * g("densidad"),
                    hlen=0.2 + 0.12 * (1 - g("hipnosis")), disorder=0.35 + 0.65 * g("caos"),
                    d0=200.0 + 120.0 * g("luz"), d_rng=160.0)

    def _caps_cell(self, fall, x, y, tr, seed):
        """La celda (columna, fila) que ocupa el punto (x, y) del plano de las cápsulas con la caída
        `fall` (la misma cuenta del shader)."""
        sp = tr["sp"]
        c = int(np.floor(x / sp + 0.5))
        Y = fall * (0.8 + 0.4 * _h01(c, 0, 11, seed)) + sp * _h01(c, 0, 12, seed)
        return c, Y

    def _caps_motion(self):
        """El movimiento de las cápsulas (una vez por partitura, así cada cuadro se dibuja solo).
        - la caída: lenta siempre, crece con la energía; cada kick es un empuje hacia abajo que se apaga
          sin volver; en la tensión casi se suspende;
        - el giro: cada cápsula gira sobre su eje, más con la energía; el kick le suma un impulso;
        - la tensión: la película se adelgaza hasta volverse negra (como una pompa antes de romperse);
        - la respiración (el beat antes de la liberación): negro;
        - la liberación: el tiempo se detiene un beat y una onda de color sale del centro; después la
          caída acelera (crece exponencialmente medio compás) y se asienta, y cae la píldora roja;
        - la apertura: cuando el tiempo vuelve, algunas cápsulas se abren (más cuanto más larga fue la
          tensión) y sueltan polvo: sale con un estallido que el aire frena, y cae despacio con la escena;
        - los ataques de las capas: una cápsula brilla y las bandas le corren por la superficie;
        - el final: se apagan con el track."""
        if getattr(self, "_cm", None) is not None:
            return self._cm
        sc = self.score
        A, fps, sy = sc.analysis, sc.fps, sc.system
        n, sp = len(sc.frames), self._spring
        ons = A["onsets"]
        bpm = 60 * fps / float(np.median(np.diff(ons))) if len(ons) > 1 else 128.0
        rel = self._release_gesture(A, fps, bpm, n)
        speed = rel["speed"]
        kick = np.clip(A["kick"].astype(np.float64), 0, 1.5)
        closed = ~A["sub_on"].astype(bool) | A["break_zone"].astype(bool)
        energy = np.where(closed, 0.05, 0.6 * A["low"] + 0.4 * A["mid"]).astype(np.float64)
        e_s = np.clip(sp(energy, fps, 0.12, 1.0, rate=speed), 0, 1.5)
        tens = np.clip(sp(np.where(closed, A["tension"], 0.0).astype(np.float64), fps, 0.5, 1.0, 4.0, 1.0), 0, 1)
        T = 2 * 60 / bpm
        dw, imp = rel["wave_dt"], rel["wave_imp"]
        efold = np.where(dw < 0, 0.0, np.where(dw < T, (np.exp(4 * np.maximum(dw, 0) / T) - 1) / (np.exp(4) - 1),
                                               np.exp(-(dw - T) / 1.6))) * imp
        v = (0.15 + 0.45 * e_s + 0.2 * kick) * (0.6 + 0.4 * sy.orbit) * (1 - 0.8 * tens) + 1.4 * efold
        fall = np.cumsum(v * speed) / fps
        tumble = np.cumsum((0.12 + 0.3 * e_s + 0.5 * kick) * (1 - 0.7 * tens) * speed + 0.8 * efold) / fps
        # negro: la respiración, el silencio y el final
        smt = lambda x: np.clip(x, 0, 1) ** 2 * (3 - 2 * np.clip(x, 0, 1))
        dark = np.clip(sp((A["breath"] | A["silent"]).astype(np.float64), fps, 6.0, 1.0, 3.0, 1.0), 0, 1)
        fd = np.clip(A["fade"].astype(np.float64), 0, 1)
        fade_k = (1 - dark) * (1 - smt(fd))
        # la onda de color de la liberación: corre mientras el tiempo está quieto
        wave_r = 0.1 + 2.2 * rel["echo_q"]
        wave_amp = rel["echo_k"]
        tau = np.cumsum(speed) / fps                            # el tiempo de la imagen
        tr, seed = self._caps_traits(), int(sc.seed % 1000003)
        hh = self.CAPS_D / self.CAPS_FOCAL                      # media altura visible en el plano
        hw = hh * self.grid.tw / self.grid.th
        # la píldora roja: entra por arriba en cada liberación
        reds = []
        for r in np.where(A["rise"])[0]:
            rg = np.random.default_rng([sc.seed, 7777, int(r)])
            c, Y = self._caps_cell(fall[r], rg.uniform(-0.6, 0.6) * hw, 0.0, tr, seed)
            reds.append((int(r), c, int(np.ceil((hh + 0.65 * tr["sp"] + Y) / tr["sp"]))))
        red_cells = {(c, w) for _, c, w in reds}
        # las que se abren: cuando el tiempo vuelve a correr, celdas visibles y ocupadas
        hold = int(round(60 / bpm * fps))
        opens, dust = [], []
        for r in np.where(A["rise"])[0]:
            o = min(n - 1, int(r) + hold)
            imp = float(rel["wave_imp"][min(n - 1, o + 1)]) or 0.55
            rg = np.random.default_rng([sc.seed, 9191, int(r)])
            want, got = int(round(2 + 3 * imp)), set()
            for _ in range(40):
                if len(got) == want:
                    break
                x, y = rg.uniform(-0.75, 0.75) * hw, rg.uniform(-0.7, 0.7) * hh
                c, Y = self._caps_cell(fall[o], x, y, tr, seed)
                w = int(np.floor((y + Y) / tr["sp"] + 0.5))
                if _h01(c, w, 1, seed) >= tr["vac"] and (c, w) not in red_cells and (c, w) not in got:
                    got.add((c, w))
                    opens.append((o, c, w, imp))
                    dust.append(self._caps_dust(o, c, w, imp, fall, tr, seed, rg))
        # los ataques de las capas: cada uno enciende una cápsula visible (las graves, más)
        Ly = A.get("layers") or dict(act=np.zeros((n, 0)), on=np.zeros((n, 0), bool), center=np.zeros(0))
        G = Ly["act"].shape[1]
        oc = np.clip(np.log2(np.maximum(Ly["center"][:G], 300) / 300) / 4.5, 0, 1)
        lw = (0.1 + 0.6 * (1 - oc) ** 1.5) / max(float((0.1 + 0.6 * (1 - oc) ** 1.5).max()), 1e-9) if G else oc
        evs, last = [], np.full(G, -(10 ** 9))
        for b, k, s in layer_attacks(Ly, fps, A.get("silent")):
            if s * lw[k] < 0.3 or b - last[k] < (0.2 + 0.25 * lw[k]) * fps:
                continue
            rg = np.random.default_rng([sc.seed, 8888, k, b])
            for _ in range(8):                                  # una celda visible y ocupada
                x, y = rg.uniform(-0.85, 0.85) * hw, rg.uniform(-0.85, 0.85) * hh
                c, Y = self._caps_cell(fall[b], x, y, tr, seed)
                w = int(np.floor((y + Y) / tr["sp"] + 0.5))
                if _h01(c, w, 1, seed) >= tr["vac"] and (c, w) not in red_cells:
                    evs.append((int(b), c, w, float(s * lw[k])))
                    last[k] = b
                    break
        self._cm = dict(fall=fall, tumble=tumble, thin=tens, wave_r=wave_r, wave_amp=wave_amp, fade_k=fade_k,
                        tau=tau, reds=reds, evs=evs, ev_b=np.array([e[0] for e in evs], np.int64),
                        opens=opens, dust=dust)
        return self._cm

    DUST_N, ZFAR = 600, 20.0

    def _caps_dust(self, o, c, w, imp, fall, tr, seed, rg):
        """El polvo de una cápsula que se abre en el cuadro o: dónde nace cada mota (cerca del corte),
        su velocidad, cuánto la frena el aire y cuánto acompaña la caída; su tamaño, color y vida. Muchas
        motas finas y brillantes, y algunas grandes y tenues (el volumen de la nube)."""
        sp, N, Nh = tr["sp"], self.DUST_N, self.DUST_N // 8
        vc = 0.8 + 0.4 * _h01(c, 0, 11, seed)
        Y = fall[o] * vc + sp * _h01(c, 0, 12, seed)
        ctr = np.array([(c + 0.24 * (_h01(c, w, 3, seed) - 0.5)) * sp,
                        w * sp - Y + 0.2 * sp * (_h01(c, w, 4, seed) - 0.5),
                        self.CAPS_D + 0.5 * sp * (_h01(c, w, 5, seed) - 0.5)])
        M = N + Nh
        d = rg.normal(size=(M, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        d[:, 2] *= 0.5                                          # más hacia los costados que hacia la cámara
        v0 = d * sp * (0.25 + 0.9 * rg.uniform(size=(M, 1)) ** 1.5) * (0.6 + 0.6 * imp)
        v0[N:] *= 0.5                                           # la nube se abre más despacio
        p0 = ctr + rg.normal(0, 0.06 * sp, (M, 3))
        acc = np.asarray(self.accent, np.float64) / 255
        tone = np.stack([np.ones(3), 0.25 * acc + 0.75, 0.55 * acc + 0.45 * np.array([1.0, 0.8, 1.0]), acc * 1.3])
        col = tone[rg.integers(0, len(tone), M)]
        size = np.concatenate([1.0 + 2.6 * rg.uniform(size=N) ** 2, 8.0 + 16.0 * rg.uniform(size=Nh)])
        bri = np.concatenate([rg.uniform(0.6, 1.6, N), rg.uniform(0.08, 0.2, Nh)]) * (0.6 + 0.4 * imp)
        return dict(o=o, vc=vc, p0=p0, v0=v0, td=rg.uniform(0.35, 1.2, M), kf=rg.uniform(0.35, 0.85, M),
                    size=size, col=col, bri=bri, life=np.concatenate([rg.uniform(1.5, 4.0, N), rg.uniform(2.0, 4.5, Nh)]))

    def _caps_points(self, i):
        """Las motas de polvo vivas en el cuadro i: (N, 8) = posición, color, tamaño (px de la escena), brillo."""
        cm, S = self._cm, self.S
        tau, fall = cm["tau"], cm["fall"]
        out = []
        for P in cm["dust"]:
            o = P["o"]
            if i < o:
                continue
            t = tau[i] - tau[o]
            if t > 6.0:
                continue
            td = P["td"][:, None]
            pos = P["p0"] + P["v0"] * td * (1 - np.exp(-t / td))
            pos[:, 1] -= P["kf"] * (fall[i] - fall[o]) * P["vc"] + 0.02 * t * t
            bri = P["bri"] * np.exp(-t / P["life"]) * min(1.0, t / 0.08) * float(cm["fade_k"][i])
            size = P["size"] * S * self.CAPS_D / np.maximum(pos[:, 2], 0.5)
            out.append(np.column_stack([pos, P["col"], size, bri]))
        return np.concatenate(out).astype(np.float32) if out else np.zeros((0, 8), np.float32)

    def _capsules(self, i):
        """Las cápsulas (shaders/capsules.frag)."""
        g, sc = self.grid, self.score
        cm, sy = self._caps_motion(), sc.system
        fps, tau = sc.fps, cm["tau"]
        ev = np.zeros((8, 4), np.float32)
        j0, j1 = np.searchsorted(cm["ev_b"], [i - 3 * fps, i + 1])
        on = sorted(((s * np.exp(-(tau[i] - tau[b]) / 0.3), c, w, s, tau[i] - tau[b])
                     for b, c, w, s in cm["evs"][j0:j1]), reverse=True)[:8]
        for k, (_, c, w, s, age) in enumerate(on):
            ev[k] = (c, w, s, age)
        red = np.zeros((4, 2), np.float32)
        live = [(c, w) for b, c, w in cm["reds"] if b <= i < b + 60 * fps][-4:]
        for k, (c, w) in enumerate(live):
            red[k] = (c, w)
        opn = np.zeros((6, 4), np.float32)
        act = [(c, w, tau[i] - tau[o], imp) for o, c, w, imp in cm["opens"] if o <= i < o + 30 * fps][-6:]
        for k, row in enumerate(act):
            opn[k] = row
        p = self.p_caps
        _set(p, out_size=(self.S * g.tw, self.S * g.th), aspect=float(g.tw / g.th), focal=self.CAPS_FOCAL,
             D=self.CAPS_D, fall=float(cm["fall"][i]), tumble=float(cm["tumble"][i]), thin=float(cm["thin"][i]),
             wave_r=float(cm["wave_r"][i]), wave_amp=float(cm["wave_amp"][i]), bright=1.0,
             light=float(max(sy.light, 0.75)), fade_k=float(cm["fade_k"][i]),
             accent=tuple(np.asarray(self.accent, np.float32) / 255), seed=int(sc.seed % 1000003),
             n_ev=len(on), n_red=len(live), n_op=len(act), zfar=self.ZFAR, **self._caps_traits())
        p["ev"].write(ev.tobytes())
        p["red"].write(red.tobytes())
        p["op"].write(opn.tobytes())
        import moderngl
        ctx = self.ctx
        if getattr(self, "f_caps", None) is None:            # la escena con profundidad (para el polvo)
            self.rb_depth = ctx.depth_renderbuffer(self.t_scene.size)
            self.f_caps = ctx.framebuffer([self.t_scene], self.rb_depth)
            vert, frag = _src("dust.vert"), _src("dust.frag")
            self.p_dust = ctx.program(vertex_shader=vert, fragment_shader=frag)
            self.b_dust = ctx.buffer(reserve=8192 * 8 * 4, dynamic=True)
            self.vao["dust"] = ctx.vertex_array(self.p_dust, [(self.b_dust, "3f 3f 1f 1f", "pos", "col", "size", "bri")])
        self.f_caps.use()
        self.f_caps.clear(0.0, 0.0, 0.0, 1.0, depth=1.0)
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.depth_func = "<="
        self.vao["caps"].render()
        pts = self._caps_points(i)[:8192]
        if len(pts):
            _set(self.p_dust, aspect=float(g.tw / g.th), focal=self.CAPS_FOCAL, zfar=self.ZFAR)
            self.b_dust.write(pts.tobytes())
            self.f_caps.depth_mask = False
            ctx.enable(moderngl.BLEND | moderngl.PROGRAM_POINT_SIZE)
            ctx.blend_func = moderngl.ONE, moderngl.ONE
            ctx.depth_func = "<"
            self.vao["dust"].render(moderngl.POINTS, vertices=len(pts))
            ctx.disable(moderngl.BLEND | moderngl.PROGRAM_POINT_SIZE)
            ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            self.f_caps.depth_mask = True
        ctx.disable(moderngl.DEPTH_TEST)

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
        fbo = (self._post(i, hole, P.get("bloom", 1.0), P.get("ghost", True), P.get("trail", True),
                          P.get("grain", 1.0))
               if self.post is not None
               else self.f_frame)
        f[:] = np.frombuffer(fbo.read(components=3, alignment=1), np.uint8).reshape(g.H, g.W, 3)
        glitch(f, A, i, sc.seed, g.s, P)
        if P.get("title", True):
            self.logo.draw(f, float(A["kick"][i]) if P.get("pulse", True) else 0.0)
        return f
