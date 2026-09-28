"""La firma: MED1NA como un objeto masivo, igual que el agujero."""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_dilation, gaussian_filter

from .presets import WHITE

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",       # Linux
    "/System/Library/Fonts/Supplemental/Arial Black.ttf",         # macOS
    "/System/Library/Fonts/Supplemental/Impact.ttf",
    "/Library/Fonts/Arial Black.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "C:/Windows/Fonts/ariblk.ttf",                                # Windows
    "C:/Windows/Fonts/impact.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
]


def load_font(user_path, candidates, size):
    for p in ([user_path] if user_path else []) + candidates:
        if p and os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default(size=size)


def text_mask(text, font):
    """Texto sin antialias (píxeles puros) -> máscara booleana."""
    l, t, r, b = font.getbbox(text)
    img = Image.new("1", (max(1, r - l), max(1, b - t)), 0)
    ImageDraw.Draw(img).text((-l, -t), text, font=font, fill=1)
    return np.array(img, dtype=bool)


def blit(frame, mask, x, y, color):
    h, w = mask.shape
    H, W = frame.shape[:2]
    if x < 0 or y < 0 or x + w > W or y + h > H:
        return
    frame[y:y + h, x:x + w][mask] = color


class Logo:
    """Letras negras con un filo tenue y un halo apenas perceptible, abajo a la derecha;
    el espacio detrás se curva hacia ellas (lente), un poco más en cada kick."""

    def __init__(self, title, font_path, W, H):
        s = H / 1080
        self.W, self.H, self.m = W, H, int(56 * s)
        self.mask = text_mask(title.upper(), load_font(font_path, FONT_CANDIDATES, int(32 * s)))
        th, tw = self.mask.shape
        pad = int(max(th, 24 * s) * 1.1)
        big = np.zeros((th + 2 * pad, tw + 2 * pad), bool)
        big[pad:pad + th, pad:pad + tw] = self.mask
        self.edge = binary_dilation(big, iterations=max(1, int(round(1.1 * s)))) & ~big
        glow = gaussian_filter(self.edge.astype(np.float32), 3.0 * s)
        self.glow = glow / (glow.max() + 1e-6)
        hb, wb = big.shape
        self.yy, self.xx = np.mgrid[0:hb, 0:wb].astype(np.float32)
        self.c = ((wb - 1) / 2, (hb - 1) / 2)
        self.dn = np.hypot((self.xx - self.c[0]) / (wb / 2), (self.yy - self.c[1]) / (hb / 2))
        self.pad, self.big = pad, big

    def draw(self, f, kick):
        W, H, m, mask = self.W, self.H, self.m, self.mask
        th, tw = mask.shape
        pad, big = self.pad, self.big
        hb, wb = big.shape
        bx0, by0 = W - m - tw - pad, H - m - th - pad
        if bx0 < 0 or by0 < 0 or bx0 + wb > W or by0 + hb > H:
            blit(f, mask, W - m - tw, H - m - th, WHITE)
            return
        region = f[by0:by0 + hb, bx0:bx0 + wb].astype(np.float32)
        # lente: lo de atrás se ve ampliado hacia el centro de las letras
        cxb, cyb = self.c
        k = 0.28 + 0.08 * kick
        sf = 1 - k * np.exp(-2.2 * self.dn ** 2)
        sx = np.clip(cxb + (self.xx - cxb) * sf, 0, wb - 1.001)
        sy = np.clip(cyb + (self.yy - cyb) * sf, 0, hb - 1.001)
        x0i, y0i = sx.astype(np.int32), sy.astype(np.int32)
        fx, fy = (sx - x0i)[..., None], (sy - y0i)[..., None]
        out = ((region[y0i, x0i] * (1 - fx) + region[y0i, x0i + 1] * fx) * (1 - fy)
               + (region[y0i + 1, x0i] * (1 - fx) + region[y0i + 1, x0i + 1] * fx) * fy)
        out = out + (0.14 * self.glow)[..., None] * (255 - out)          # halo apenas perceptible
        out[self.edge] = np.maximum(out[self.edge], 105)                # filo tenue
        out[big] = 0                                                    # letras negras
        f[by0:by0 + hb, bx0:bx0 + wb] = np.clip(out, 0, 255).astype(np.uint8)
