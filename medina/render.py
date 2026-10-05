"""Armar un cuadro a partir de la partitura: estilo -> estrellas -> silencio -> fenómenos ->
agujero -> supersampling con la cámara -> post-proceso -> glitch -> firma."""
import numpy as np
from PIL import Image

from .logo import Logo
from .post import Post
from .presets import BLACK, LOOKS, PRESETS, RED, WHITE
from .space import Phenomena, Starfield, horizon, silence
from .styles import STYLES, FrameCtx


class Grid:
    """La grilla de cálculo: coordenadas de cada píxel normalizadas a media altura.
    ss >= 2 (acabado B): se calcula a ss× la salida y se baja con Lanczos (bordes limpios);
    ss = 1: a media resolución y se agranda sin interpolar (píxel duro)."""

    def __init__(self, W, H, ss=2, fps=30, distance=1.0, arrays=True):
        self.W, self.H, self.fps = W, H, fps
        s = self.s = H / 1080
        self.m = int(56 * s)
        self.t = max(2, int(4 * s))
        self.ds = 2
        self.sup = ss
        self.ty0 = 0
        self.th = ((H - self.ty0) // self.ds) * self.ds
        self.tw = (W // self.ds) * self.ds
        self.x_off = (W - self.tw) // 2
        if ss > 1:
            hh, ww = self.th * ss, self.tw * ss
        else:
            hh, ww = self.th // self.ds, self.tw // self.ds
        self.hh, self.ww = hh, ww
        self.gs = ww / self.tw                    # píxeles de grilla por píxel de salida
        self.tg = max(1, int(round(self.t * self.gs)) if ss > 1 else self.t // self.ds)
        self.unit = hh / 2 / distance              # píxeles de grilla por unidad del espacio
        self._disk = {}
        if not arrays:                             # el motor de GPU solo usa las medidas
            return
        yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
        r = hh / 2
        self.yy = (yy - hh / 2 + 0.5) / r
        self.xx = (xx - ww / 2 + 0.5) / r
        if distance != 1.0:                        # cámara más lejos: el mismo espacio se ve más chico
            self.yy, self.xx = self.yy * distance, self.xx * distance
        self.RR = np.hypot(self.xx, self.yy).astype(np.float32)
        self.TH = np.arctan2(self.yy, self.xx).astype(np.float32)
        self.logRR = np.log(self.RR + 1e-4).astype(np.float32)
        # remolino: torsión que crece hacia el centro (el agujero negro arrastra lo cercano)
        self.SWIRL = (0.3 / (self.RR + 0.05)).astype(np.float32)

    def disk_coords(self, tilt, incl):
        """Coordenadas del plano del disco de acreción (inclinado `tilt`°, visto con `incl`)."""
        key = (tilt, incl)
        if key not in self._disk:
            al = np.deg2rad(tilt)
            XR = (self.xx * np.cos(al) + self.yy * np.sin(al)).astype(np.float32)
            YR = (-self.xx * np.sin(al) + self.yy * np.cos(al)).astype(np.float32)
            yd = YR / incl
            self._disk[key] = (XR, YR, np.hypot(XR, yd).astype(np.float32),
                               np.arctan2(yd, XR).astype(np.float32))
        return self._disk[key]


def palette(accent):
    """0 negro, 1 blanco, 2 acento; 3-5 blancos tenues (18/40/70 %), 6-7 acento tenue (35/65 %)."""
    pal = np.array([BLACK, WHITE, accent] + [BLACK] * 5, np.uint8)
    for j, f in ((3, 0.18), (4, 0.40), (5, 0.70)):
        pal[j] = np.round(255 * f)
    c = np.asarray(accent, np.float32)
    for j, f in ((6, 0.35), (7, 0.65)):
        pal[j] = np.round(c * f).astype(np.uint8)
    return pal


def camera(score, g, i):
    """(giro, encuadre) del cuadro i. Sin cámara (o con ss=1): (0, None)."""
    C = score.camera
    if C is None or g.sup <= 1:
        return 0.0, None
    roll = float(C["roll"][i])
    z = float(C["zoom"][i])
    gw, gh = g.ww, g.hh
    bw, bh = gw / z, gh / z
    mx, my = (gw - bw) / 2, (gh - bh) / 2
    k = g.s * g.gs                                  # píxeles de 1080p -> píxeles de grilla
    x0 = mx + float(np.clip(C["dx"][i] * k, -mx, mx))
    y0 = my + float(np.clip(C["dy"][i] * k, -my, my))
    return roll, (x0, y0, x0 + bw, y0 + bh)


def glitch(f, A, i, seed, s, P):
    """Glitch de franjas: kicks fuertes con caos alto (no en los estilos del espacio)."""
    if A["strength"][i] >= 1.0 and A["chaos"][i] > 0.5 and P.get("glitch", True):
        gr = np.random.default_rng(seed + i)
        H = f.shape[0]
        for _ in range(int(gr.integers(1, 4))):
            y = int(gr.integers(0, H - 10))
            hh = int(gr.integers(int(6 * s), int(40 * s)))
            band = f[y:y + hh]
            band[:] = np.roll(band, int(gr.integers(-140, 140) * s), axis=1)


class Renderer:
    def __init__(self, score, W, H, title="MED1NA", font=None, ss=2, look="luz"):
        self.score = score
        sysm = score.system
        self.grid = Grid(W, H, ss, score.fps, sysm.distance)
        # 0 negro, 1 blanco, 2 acento; 3-5 blancos tenues (18/40/70 %), 6-7 acento tenue (35/65 %)
        self.pal = np.array([BLACK, WHITE, RED] + [BLACK] * 5, np.uint8)
        self.set_accent(RED)
        self.stars = Starfield(score.seed, sysm.stars)
        self.phenomena = Phenomena(score.seed, sysm.phenomena)
        self.logo = Logo(title, font, W, H)
        self.post = Post(W, H, score.fps, score.analysis, look, score.seed, sysm) if LOOKS[look] else None

    def set_accent(self, c):
        self.accent = c
        self.pal = palette(c)

    def camera(self, i):
        return camera(self.score, self.grid, i)

    def up(self, img, box):
        """Grilla -> píxeles de salida (con el encuadre de la cámara, si hay)."""
        g = self.grid
        if g.sup > 1:
            return np.asarray(Image.fromarray(img).resize((g.tw, g.th), Image.LANCZOS, box=box))
        return np.repeat(np.repeat(img, g.ds, axis=0), g.ds, axis=1)

    def draw(self, f, i):
        """Dibuja el cuadro i en `f` (H, W, 3) uint8 y lo devuelve."""
        sc, g = self.score, self.grid
        A = sc.analysis
        fs = sc.frames[i]
        P = PRESETS[fs.preset]
        c = fs.c
        self.set_accent(sc.accent[i])
        roll, box = self.camera(i)
        fr = FrameCtx(P, fs.st, c, fs.kc, i, roll, box, sc.seed, sc.fps, sc.system)
        idx = STYLES[P["style"]](g, fr)
        hole = None
        idx = self.stars.draw(idx, g, fr)
        idx = silence(idx, g, c)
        if not c["silent"]:
            idx = self.phenomena.draw(idx, g, fr, A.get("layers"))
            idx, hole = horizon(idx, g, fr)            # el agujero se traga también los fenómenos
        f[:] = 0
        f[g.ty0:g.ty0 + g.th, g.x_off:g.x_off + g.tw] = self.up(self.pal[idx], box)
        if self.post is not None:
            f[:] = self.post.apply(f, i, self.accent, hole)
        glitch(f, A, i, sc.seed, g.s, P)
        if P.get("title", True):
            self.logo.draw(f, float(A["kick"][i]))
        return f
