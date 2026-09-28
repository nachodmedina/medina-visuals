#!/usr/bin/env python3
"""Master técnico: EQ + ensanchado M/S (grave en mono) + compresión de bus + limitador
con objetivo de loudness y true peak. Uso:
  python3 master.py entrada.wav salida.wav --lufs -8 --tp -1.0 [--hi 2.5 --air 2.0 --width 3]
"""
import argparse, numpy as np, subprocess, json
import pyloudnorm as pyln
from pedalboard import Pedalboard, HighpassFilter, LowShelfFilter, HighShelfFilter, PeakFilter, Compressor, Limiter, Gain
from pedalboard.io import AudioFile
from scipy.signal import butter, sosfiltfilt, resample_poly

ap = argparse.ArgumentParser()
ap.add_argument("inp"); ap.add_argument("out")
ap.add_argument("--lufs", type=float, default=-8.0)
ap.add_argument("--tp", type=float, default=-1.0)
ap.add_argument("--hi", type=float, default=2.5, help="shelf de presencia (dB, 3.5 kHz)")
ap.add_argument("--air", type=float, default=2.0, help="shelf de aire (dB, 10 kHz)")
ap.add_argument("--lowmid", type=float, default=-1.5, help="corte de medios-graves (dB, 280 Hz)")
ap.add_argument("--width", type=float, default=3.0, help="ganancia del lado en dB por encima de 500 Hz")
ap.add_argument("--clip", type=float, default=3.0, help="rodilla del clipper en dB por debajo del techo")
a = ap.parse_args()

with AudioFile(a.inp) as f:
    sr = f.samplerate; x = f.read(f.frames)          # (ch, n)
x = x.astype(np.float32)

# 1) EQ
eq = Pedalboard([
    HighpassFilter(cutoff_frequency_hz=24),
    PeakFilter(cutoff_frequency_hz=280, gain_db=a.lowmid, q=0.9),
    HighShelfFilter(cutoff_frequency_hz=3500, gain_db=a.hi, q=0.6),
    HighShelfFilter(cutoff_frequency_hz=10000, gain_db=a.air, q=0.6),
])
y = eq(x, sr)

# 2) M/S: grave < 120 Hz a mono, lado +width dB por encima de 500 Hz
L, R = y[0].astype(np.float64), y[1].astype(np.float64)
M, S = (L + R) / 2, (L - R) / 2
S = sosfiltfilt(butter(4, 120, "high", fs=sr, output="sos"), S)
S_hi = sosfiltfilt(butter(2, 500, "high", fs=sr, output="sos"), S)
S = S + S_hi * (10 ** (a.width / 20) - 1)
y = np.stack([M + S, M - S]).astype(np.float32)

# 3) compresión de bus suave (pegamento)
y = Pedalboard([Compressor(threshold_db=-14, ratio=2.0, attack_ms=30, release_ms=120)])(y, sr)

# 4) ganancia hacia el objetivo + limitador; iterar hasta clavar LUFS y true peak
meter = pyln.Meter(sr)
def true_peak(z):
    up = resample_poly(z, 4, 1, axis=1)
    return 20 * np.log10(np.abs(up).max() + 1e-12)
pre = y.copy()

def limiter(z, ceiling_db, sr, look_ms=5.0, release_ms=90.0, B=32):
    """Limitador brickwall con lookahead (procesado por bloques de B muestras)."""
    c = 10 ** (ceiling_db / 20)
    pk = np.abs(z).max(0)
    nb = int(np.ceil(len(pk) / B))
    pk = np.pad(pk, (0, nb * B - len(pk)))
    bp = pk.reshape(nb, B).max(1)
    La = max(1, int(look_ms / 1000 * sr / B))
    from scipy.ndimage import maximum_filter1d, uniform_filter1d
    ahead = maximum_filter1d(bp, size=2 * La + 1, mode="nearest")
    req = np.minimum(1.0, c / np.maximum(ahead, 1e-9))
    rel = 1 - np.exp(-B / (release_ms / 1000 * sr))
    g = np.empty_like(req); v = 1.0
    for k in range(nb):
        v = min(req[k], v + (1 - v) * rel)
        g[k] = v
    g = np.minimum(uniform_filter1d(g, La, mode="nearest"), g)
    gs = np.interp(np.arange(nb * B), np.arange(nb) * B + B / 2, g)[:z.shape[1]]
    return z * gs, 20 * np.log10(g.min())

def clipper(z, ceiling_db, knee_db):
    """Clipper suave con sobremuestreo x4: recorta los picos del kick antes del limitador."""
    c = 10 ** (ceiling_db / 20); k = 10 ** ((ceiling_db - knee_db) / 20)
    out = np.empty_like(z)
    for ch in range(z.shape[0]):
        u = resample_poly(z[ch].astype(np.float32), 4, 1)
        a_ = np.abs(u)
        over = a_ > k
        u[over] = np.sign(u[over]) * (k + (c - k) * np.tanh((a_[over] - k) / (c - k)))
        out[ch] = resample_poly(u, 1, 4)[:z.shape[1]]
    return out

ceil = a.tp - 0.4
gain = a.lufs - meter.integrated_loudness(pre.T.astype(np.float64))
for it in range(8):
    g_in = pre * 10 ** (gain / 20)
    clip_amt = 20 * np.log10(np.abs(g_in).max() + 1e-12) - ceil
    zc = clipper(g_in, ceil, a.clip)
    z, gr = limiter(zc, ceil, sr)
    tp = true_peak(z)
    if tp > a.tp:
        z = z * 10 ** ((a.tp - tp - 0.02) / 20)
    l = meter.integrated_loudness(z.T.astype(np.float64))
    if abs(l - a.lufs) < 0.1:
        break
    gain += (a.lufs - l)
print(f"picos del kick sobre el techo antes del clipper: {clip_amt:.1f} dB · "
      f"reducción máxima del limitador después: {-gr:.1f} dB")
tp = true_peak(z)
with AudioFile(a.out, "w", sr, 2, bit_depth=24) as f:
    f.write(z)
print(json.dumps(dict(lufs=round(float(l), 2), true_peak=round(float(tp), 2), gain_applied=round(float(gain), 2))))
