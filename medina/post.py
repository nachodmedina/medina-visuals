"""Acabado reactivo: estela, fantasma en el color de acento, bloom, resplandor del horizonte,
viñeta, punto de negro y grano. Cada efecto lee una señal del track (ver presets.LOOKS)."""
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from .audio import envelope
from .presets import LOOKS, RED


class Post:
    def __init__(self, W, H, fps, A, look, seed):
        L = self.L = LOOKS[look]
        self.W, self.H, self.fps, self.seed = W, H, fps, seed
        self.s = H / 1080
        n = A["n"]
        idx = np.arange(n)
        # bloom: base + destello en cada liberación que se apaga exponencialmente
        last = np.maximum.accumulate(np.where(A["rise"], idx, -1))
        env = np.where(last >= 0, np.exp(-(idx - last) / (L["bloom_tau"] * fps)), 0.0)
        self.bloom = (L["bloom"][0] + L["bloom"][1] * env * (0.7 + 0.3 * A["kick"])).astype(np.float32)
        # estela: vida media según la tensión (y el fade final)
        T = np.maximum(A["tension"], A["fade"])
        hl = L["trail_hl"][0] + (L["trail_hl"][1] - L["trail_hl"][0]) * T ** 1.5
        self.decay = (0.5 ** (1 / (hl * fps))).astype(np.float32)
        self.decay[A["breath"]] = 0.0              # la respiración corta la estela: negro limpio
        # grano según el caos
        self.grain = (L["grain"][0] + (L["grain"][1] - L["grain"][0]) * A["chaos"]).astype(np.float32)
        # fantasma en kicks fuertes; más leve en tensión
        strong = np.where(A["strength"] >= 1.0, 1.0, 0.0).astype(np.float32)
        self.ghost = envelope(strong, 0.55) * (1 - 0.6 * T)
        self.T = T.astype(np.float32)
        self.flare = env.astype(np.float32)          # destello de la liberación (para el horizonte)
        self.kick = A["kick"].astype(np.float32)
        self.hat = A["hat"].astype(np.float32)
        # viñeta fija
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.hypot((xx - W / 2) / (W / 2), (yy - H / 2) / (H / 2)) / np.sqrt(2)
        e = np.clip((r - 0.3) / 0.7, 0, 1)
        self.vig = (1 - L["vignette"] * e * e * (3 - 2 * e))[..., None].astype(np.float32)
        # banco de grano (reproducible), a escala 1 o 2 px
        g = np.random.default_rng([seed, 707])
        gp, pad = L["grain_px"], 64
        self.tiles = []
        for _ in range(8):
            t = np.clip(g.standard_normal(((H + pad) // gp + 1, (W + pad) // gp + 1)) * 40, -127, 127)
            t = np.repeat(np.repeat(t.astype(np.int8), gp, 0), gp, 1)
            self.tiles.append(t)
        self.pad = pad
        self.acc = None
        self._gmaps = {}

    def reset(self):
        """Olvidar la estela (el próximo cuadro arranca sin memoria de los anteriores)."""
        self.acc = None

    def _glow(self, y, i, hole, accent):
        """Horizonte difuso (tipo Gargantua): un filo de luz finísimo pegado al borde y un halo que
        se abre hacia afuera, blanco adentro y del color de acento afuera. Crece con la atracción
        (tensión) y se enciende en las liberaciones. Adentro, negro absoluto: se traga todo."""
        cx, cy, r = hole
        W, H = self.W, self.H
        ext = r * 3.2 + 8
        x0, x1 = max(0, int(cx - ext)), min(W, int(cx + ext) + 1)
        y0, y1 = max(0, int(cy - ext)), min(H, int(cy + ext) + 1)
        if x1 <= x0 or y1 <= y0:
            return y
        yy, xx = np.ogrid[y0:y1, x0:x1]
        d = np.sqrt((xx - cx + 0.5) ** 2 + (yy - cy + 0.5) ** 2).astype(np.float32)
        out = np.maximum(d - r, 0)
        T = float(self.T[i])
        nerv = min(1.0, max(0.0, (T - 0.35) / 0.65))
        amp = (0.30 + 0.35 * T + 0.35 * float(self.flare[i])       # energía que se concentra en el borde:
               + 0.25 * float(self.kick[i]) + 0.35 * nerv * float(self.hat[i]))   # late con kicks y hats
        core = 0.55 * np.exp(-(out / (1.2 * self.s + 0.012 * r)) ** 2)          # filo finísimo
        halo = amp * np.exp(-out / (0.28 * r + 2))                               # difuso, hacia afuera
        ac = np.asarray(accent, np.float32) / 255
        mix = np.clip(out / (0.5 * r + 1), 0, 1)[..., None]                      # blanco -> acento
        col = (1 - mix) + mix * ac
        g = (core + halo)[..., None] * col
        inside = np.clip((d - r * 0.985) / (0.02 * r + 0.8), 0, 1)[..., None]  # borde del negro
        sub = y[y0:y1, x0:x1]
        sub = (sub + np.minimum(g, 1) * (1 - sub)) * inside
        y[y0:y1, x0:x1] = sub
        self.acc[y0:y1, x0:x1] *= inside                                         # la estela también cae adentro
        return y

    def _gmap(self, px):
        """Mapas de muestreo para escalar la imagen hacia afuera `px` píxeles en el borde."""
        if px not in self._gmaps:
            W, H = self.W, self.H
            k = 1 + px / (W / 2)
            xs = np.clip(np.round(W / 2 + (np.arange(W) + 0.5 - W / 2) / k - 0.5), 0, W - 1).astype(np.intp)
            ys = np.clip(np.round(H / 2 + (np.arange(H) + 0.5 - H / 2) / k - 0.5), 0, H - 1).astype(np.intp)
            self._gmaps[px] = (ys, xs)
        return self._gmaps[px]

    def apply(self, f, i, accent=RED, hole=None):
        L, W, H = self.L, self.W, self.H
        x = f.astype(np.float32) * (1 / 255)
        # estela: lo vivo siempre al frente; lo anterior queda como un recuerdo tenue
        # que dura más cuanto mayor es la tensión
        if self.acc is not None:
            mem = self.acc * self.decay[i]
            self.acc = np.maximum(x, mem)
            x = np.maximum(x, mem * L["trail_g"])
        else:
            self.acc = x.copy()
        y = x
        # fantasma: lo encendido, desplazado hacia afuera, en el color de acento
        # (aberración dentro de la paleta)
        px = int(round(float(self.ghost[i]) * L["ghost_px"] * self.s))
        if px >= 1:
            ys, xs = self._gmap(px)
            m = x.max(-1)[ys][:, xs]
            y = np.maximum(x, (0.85 * m)[..., None] * (np.asarray(accent, np.float32) / 255))
        # bloom en dos escalas, calculado a 1/4 de resolución
        b = float(self.bloom[i])
        if b > 0.01:
            h4, w4 = H // 4, W // 4
            sm = y[:h4 * 4, :w4 * 4].reshape(h4, 4, w4, 4, 3).mean((1, 3))
            s0, s1 = (v * self.s / 4 for v in L["bloom_sig"])
            mx = L["bloom_mix"]
            gl = (1 - mx) * gaussian_filter(sm, (s0, s0, 0)) + mx * gaussian_filter(sm, (s1, s1, 0))
            up = np.stack([np.asarray(Image.fromarray(np.ascontiguousarray(gl[..., c]), "F")
                                      .resize((W, H), Image.BILINEAR)) for c in range(3)], -1)
            y = y + np.minimum(b * up, 1) * (1 - y)          # screen: brilla sin quemar el blanco
        if hole is not None:
            y = self._glow(y, i, hole, accent)
        y = y * self.vig
        # punto de negro: la bruma más tenue (bloom + estela) vuelve a negro; lo encendido no cambia
        k = L.get("black", 0.0)
        if k:
            y = np.maximum(y - k, 0) * (1 / (1 - k))
        # grano (luma), casi nada sobre el negro; también disimula el banding del bloom
        a = float(self.grain[i])
        g = np.random.default_rng([self.seed, 708, i])
        t = self.tiles[int(g.integers(0, len(self.tiles)))]
        oy, ox = (int(v) for v in g.integers(0, self.pad, 2))
        nz = t[oy:oy + H, ox:ox + W].astype(np.float32) * (a / 40)
        lum = y.mean(-1)
        y = y + (nz * (0.12 + 0.88 * np.minimum(lum, 1)))[..., None]
        return (np.clip(y, 0, 1) * 255 + 0.5).astype(np.uint8)
