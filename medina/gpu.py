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

from .logo import Logo
from .noise import hash32
from .post import Post
from .presets import LOOKS, PRESETS, RED
from .render import Grid, camera, glitch, palette
from .space import Phenomena, Starfield
from .styles import FrameCtx

SHADERS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shaders")


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
        tri = ctx.buffer(np.array([-1, -1, 3, -1, -1, 3], np.float32).tobytes())
        self.vao = {k: ctx.vertex_array(p, [(tri, "2f", "p")]) for k, p in
                    (("scene", self.p_scene), ("down", self.p_down))}
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

    def _post(self, i, hole):
        ctx, P, cpu = self.ctx, self.post, self.post.p
        L, W, H = cpu.L, self.grid.W, self.grid.H
        accent = tuple(np.asarray(self.accent, np.float32) / 255)
        prev, nxt = self.cur, 1 - self.cur
        hole_u = tuple(float(v) for v in hole) if hole is not None else (0.0, 0.0, 0.0)
        # estela + fantasma
        self.t_frame.use(0)
        self.t_acc[prev].use(1)
        _set(self.p_trail, frame=0, acc_prev=1, fresh=int(P.fresh), decay=float(cpu.decay[i]),
             trail_g=float(L["trail_g"]), ghost_px=int(round(float(cpu.ghost[i]) * L["ghost_px"] * cpu.s)),
             accent=accent, size=(W, H), has_hole=int(hole is not None), hole=hole_u)
        self.f_trail[nxt].use()
        self.vao["trail"].render()
        self.cur, P.fresh = nxt, False
        # bloom a 1/4, en dos escalas
        b = float(cpu.bloom[i])
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
        # el agujero en la salida (como space.horizon), para el resplandor
        hole = None
        if not silent:
            cx, cy = g.ww // 2, g.hh // 2
            sxx, syy = g.tw / (x1 - x0), g.th / (y1 - y0)
            rr = float(st["rh"]) * 1.12
            hole = ((cx - x0) * sxx, (cy - y0) * syy, rr * g.unit * syy)
        fbo = self._post(i, hole) if self.post is not None else self.f_frame
        f[:] = np.frombuffer(fbo.read(components=3, alignment=1), np.uint8).reshape(g.H, g.W, 3)
        glitch(f, A, i, sc.seed, g.s, P)
        if P.get("title", True):
            self.logo.draw(f, float(A["kick"][i]))
        return f
