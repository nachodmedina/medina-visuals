"""Lectura del archivo de audio y herramientas de señal: tempo, beats, envolventes."""
import hashlib
import subprocess

import numpy as np
from scipy.signal import butter, sosfilt


def load_audio(path, sr, start=0, duration=0):
    """Audio mono a `sr` Hz como float32 (vía ffmpeg: wav, aiff, flac, mp3...)."""
    cmd = ["ffmpeg", "-v", "error"]
    if start:
        cmd += ["-ss", str(start)]
    if duration:
        cmd += ["-t", str(duration)]
    cmd += ["-i", path, "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


def mute_report(y, sr):
    """Control de la exportación: si el audio está casi mudo (típico de pistas en solo o mute
    olvidadas) devuelve un mensaje para frenar; si no, None."""
    nb = len(y) // sr
    lv = 20 * np.log10(np.sqrt((y[:nb * sr].reshape(nb, sr) ** 2).mean(1)) + 1e-9)
    if nb and (lv > -60).mean() < 0.2:
        return (f"El audio está casi en silencio ({(lv > -60).sum()} de {nb} s con sonido). "
                "Revisá la exportación (pistas en solo/mute, rango de exportación).")
    return None


def seed_from_audio(y):
    """Semilla del azar sacada del propio audio: la misma pieza da siempre el mismo video."""
    return int(hashlib.sha1(y[::997].tobytes()).hexdigest()[:6], 16) % 100000


def norm_db(x, lo_pct=5, hi_pct=99.5):
    """Potencia -> dB -> 0..1 usando percentiles de todo el track (por columna)."""
    db = 10 * np.log10(x + 1e-10)
    lo = np.percentile(db, lo_pct, axis=0)
    hi = np.percentile(db, hi_pct, axis=0)
    return np.clip((db - lo) / np.maximum(hi - lo, 1e-6), 0, 1)


def envelope(x, decay):
    """Ataque instantáneo, caída exponencial."""
    out = np.empty_like(x)
    v = 0.0
    for i, s in enumerate(x):
        v = s if s > v else v * decay
        out[i] = v
    return out


def detect_bpm(path, lo=110, hi=160):
    """Tempo exacto (0.01 BPM) por autocorrelación en peine de la envolvente de graves."""
    sr = 8000
    y = load_audio(path, sr)
    lo_band = sosfilt(butter(4, 150, "low", fs=sr, output="sos"), y)
    hop = 8
    n = len(lo_band) // hop
    e = np.sqrt((lo_band[:n * hop].reshape(n, hop) ** 2).mean(1))
    le = np.log(e + 1e-5)
    on = np.maximum(0, np.diff(le, prepend=le[0]))
    on -= on.mean()
    rate = sr / hop
    best = (-np.inf, None)
    for coarse in (True, False):
        grid = np.arange(lo, hi, 0.25) if coarse else np.arange(best[1] - 0.3, best[1] + 0.3, 0.01)
        for bpm in grid:
            p = rate * 60 / bpm
            sc = sum(np.dot(on[:-int(round(p * k))], on[int(round(p * k)):]) for k in (1, 2, 4, 8, 16))
            if sc > best[0]:
                best = (sc, bpm)
    return float(best[1])


def beat_track(onset, fps, bpm_min=100, bpm_max=165, tightness=100.0):
    """Beat tracker por programación dinámica (Ellis 2007). Devuelve los cuadros de cada beat."""
    n = len(onset)
    if n < fps * 4:
        return np.zeros(0, np.int64)
    o = onset / (onset.std() + 1e-9)
    x = o - o.mean()
    lags = np.arange(int(fps * 60 / bpm_max), int(fps * 60 / bpm_min) + 2)
    ac = np.array([np.dot(x[:-l], x[l:]) for l in lags])
    p0 = lags[int(np.argmax(ac))]
    # afinar el período mirando 8 beats (más resolución que 1 cuadro)
    L = np.arange(8 * p0 - 8, 8 * p0 + 9)
    ac8 = np.array([np.dot(x[:-l], x[l:]) for l in L])
    p = L[int(np.argmax(ac8))] / 8.0
    lo_off, hi_off = int(round(2 * p)), max(1, int(round(p / 2)))
    dts = np.arange(hi_off, lo_off + 1)                  # distancia hacia atrás
    wts = -tightness * np.log(dts / p) ** 2
    C = np.zeros(n)
    P = -np.ones(n, np.int64)
    for t in range(n):
        a, b = t - lo_off, t - hi_off
        if b < 0:
            C[t] = o[t]
            continue
        a0 = max(a, 0)
        seg = C[a0:b + 1][::-1] + wts[:b + 1 - a0]        # dt = hi_off..
        k = int(np.argmax(seg))
        C[t] = o[t] + seg[k]
        P[t] = b - k
    t = n - int(p) + int(np.argmax(C[n - int(p):]))
    out = []
    while t >= 0:
        out.append(t)
        t = P[t]
    return np.array(out[::-1], np.int64)
