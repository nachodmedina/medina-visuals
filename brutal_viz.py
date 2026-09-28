#!/usr/bin/env python3
"""
brutal_viz.py — visuales audio-reactivos brutalistas (negro / blanco / rojo)
para sets de techno. Analiza el audio y genera un video listo para YouTube.

Requisitos:  Python 3.9+, ffmpeg en el PATH,  pip install numpy scipy pillow

Uso básico:
    python brutal_viz.py mi_set.wav -o mi_set.mp4 --bpm 140-142

Probar un fragmento antes de renderizar todo (30 s desde el minuto 20):
    python brutal_viz.py mi_set.wav -o preview.mp4 --start 1200 --duration 30

Qué reacciona a qué:
    - Túnel de cuadrados    -> avanza más rápido con graves + medios
    - Cada kick             -> salto hacia adelante, anillos más gruesos,
                               y el anillo rojo cambia de lugar
    - Cada 32 kicks         -> el túnel gira 45° de golpe (cuadrado <-> rombo)
    - Vacío central         -> late con los graves; núcleo rojo con agudos
    - Rotación              -> lenta y escalonada, acelera con los medios
    - Barras inferiores     -> espectro (64 bandas); rojas en los picos
    - Glitch de franjas     -> kicks fuertes
    - Inversión blanco/negro-> el primer kick después de un break (el drop)
    - Break sin kick        -> el túnel se ralentiza y queda casi vacío
"""
import argparse
import os
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import maximum_filter1d, uniform_filter1d

BLACK = np.array([0, 0, 0], np.uint8)
WHITE = np.array([255, 255, 255], np.uint8)
RED = np.array([255, 0, 0], np.uint8)
GREEN = np.array([0, 255, 0], np.uint8)
BLUE = np.array([0, 0, 255], np.uint8)
ACCENTS = [RED, GREEN, BLUE]
VIOLET = np.array([125, 55, 200], np.uint8)
# paleta del modo limpio: violeta = negro / blanco / violeta (sin RGB);
# violeta_rojo = base violeta; en cada liberación alterna violeta <-> rojo por kick (32 kicks);
# rgb = negro / blanco / rojo, con rojo -> verde -> azul por kick en las liberaciones
PALETTES = {"violeta": VIOLET, "violeta_rojo": [VIOLET, RED], "rgb": None}
PITCHES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
# Camelot: número (1-12) por tónica, para menor (A) y mayor (B)
CAMELOT_MIN = {8: 1, 3: 2, 10: 3, 5: 4, 0: 5, 7: 6, 2: 7, 9: 8, 4: 9, 11: 10, 6: 11, 1: 12}
CAMELOT_MAJ = {11: 1, 6: 2, 1: 3, 8: 4, 3: 5, 10: 6, 5: 7, 0: 8, 7: 9, 2: 10, 9: 11, 4: 12}
KS_MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KS_MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# ------------------------------------------------------------- presets ----
# Cada preset = un estilo + parámetros. Todos leen las mismas señales del track
# (kick, tensión, liberación, caos, capítulos); el preset solo cambia el lenguaje.
PRESETS = {
    "barras":        dict(style="bars"),
    "anillos":       dict(style="rings", K=4.6),
    "anillos_finos": dict(style="rings", K=7.5, thin=0.55),
    "geometria":     dict(style="geo", sides=[4, 3, 6]),
    "triangulos":    dict(style="geo", sides=[3, 3, 6], K=3.4),
    "arcos":         dict(style="arcs", K=3.2),
    # espacio: líneas finas en órbita y el disco de acreción visto casi de canto (tipo Gargantua)
    "estelas":       dict(style="streaks", K=14.0, glitch=False),
    "disco":         dict(style="disk", K=20.0, incl=0.21, glitch=False),
    # partículas sobre el túnel: viajan hacia la cámara y crecen con la distancia
    "puntos":        dict(style="dots", polar=True, N=48, Kd=7.5, p=0.22),
    "puntos_densos": dict(style="dots", polar=True, N=80, Kd=12, p=0.32, dot=0.36),
    "brillos":       dict(style="dots", polar=True, N=88, Kd=13, p=0.03, flicker=4, plus=True, dot=0.2),
    "puntos_grilla": dict(style="dots", spacing=18, p=0.22),
    "rayos":         dict(style="rays"),
    # viaje: el track decide el estilo. Antes del primer silencio, mundo de
    # partículas; después, mundo geométrico. Solo cambia en las liberaciones.
    "viaje":         dict(style="journey",
                          worlds=[["brillos", "puntos", "puntos_densos"],
                                  ["arcos", "estelas", "disco"]]),
}

ANALYSIS_SR = 22050
N_FFT = 2048


# ----------------------------------------------------------------- audio ----
def load_audio(path, sr, start, duration):
    cmd = ["ffmpeg", "-v", "error"]
    if start:
        cmd += ["-ss", str(start)]
    if duration:
        cmd += ["-t", str(duration)]
    cmd += ["-i", path, "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


def norm_db(x, lo_pct=5, hi_pct=99.5):
    """Potencia -> dB -> 0..1 usando percentiles de todo el set (por columna)."""
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
    from scipy.signal import butter, sosfilt
    sr = 8000
    y = load_audio(path, sr, 0, 0)
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
    """Beat tracker por programación dinámica (Ellis 2007). Devuelve frames de beats."""
    n = len(onset)
    if n < fps * 4:
        return np.zeros(0, np.int64)
    o = onset / (onset.std() + 1e-9)
    x = o - o.mean()
    lags = np.arange(int(fps * 60 / bpm_max), int(fps * 60 / bpm_min) + 2)
    ac = np.array([np.dot(x[:-l], x[l:]) for l in lags])
    p0 = lags[int(np.argmax(ac))]
    # afinar el período mirando 8 beats (más resolución que 1 frame)
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


def analyze(y, fps, n_bars, bpm_range=(100, 165)):
    sr = ANALYSIS_SR
    n_frames = int(len(y) / sr * fps)
    hop = sr / fps
    y_p = np.pad(y, (N_FFT // 2, N_FFT // 2 + int(hop) + 1))
    win = np.hanning(N_FFT).astype(np.float32)
    freqs = np.fft.rfftfreq(N_FFT, 1 / sr)

    # matriz bins -> barras (espaciado logarítmico)
    edges = np.geomspace(30, 12000, n_bars + 1)
    M = np.zeros((len(freqs), n_bars), np.float32)
    for b in range(n_bars):
        idx = np.where((freqs >= edges[b]) & (freqs < edges[b + 1]))[0]
        if len(idx) == 0:
            idx = [np.argmin(np.abs(freqs - np.sqrt(edges[b] * edges[b + 1])))]
        M[idx, b] = 1.0 / len(idx)
    band = lambda lo, hi: (freqs >= lo) & (freqs < hi)
    b_low, b_mid, b_high = band(20, 140), band(140, 2500), band(2500, 11000)

    b_kick = np.where(band(35, 130))[0]
    bars = np.zeros((n_frames, n_bars), np.float32)
    lmh = np.zeros((n_frames, 3), np.float32)
    kick_spec = np.zeros((n_frames, len(b_kick)), np.float32)
    b_hat = np.where(band(5000, 11000))[0]
    hat_spec = np.zeros((n_frames, len(b_hat)), np.float32)
    b_sub = band(25, 60)
    subp = np.zeros(n_frames, np.float32)
    totp = np.zeros(n_frames, np.float32)
    # matriz bins -> clases de altura (chroma), 110 Hz - 3.5 kHz
    Cm = np.zeros((len(freqs), 12), np.float32)
    for k, fq in enumerate(freqs):
        if 110 <= fq <= 3500:
            Cm[k, int(round(12 * np.log2(fq / 440.0) + 9)) % 12] = 1.0
    chroma = np.zeros((n_frames, 12), np.float32)
    # bandas logarítmicas 300 Hz - 11 kHz para separar capas (NMF)
    le = np.geomspace(300, 11000, 73)
    Lm = np.zeros((len(freqs), 72), np.float32)
    for b in range(72):
        m_ = (freqs >= le[b]) & (freqs < le[b + 1])
        if not m_.any():
            m_[np.argmin(np.abs(freqs - np.sqrt(le[b] * le[b + 1])))] = True
        Lm[m_, b] = 1.0 / m_.sum()
    lspec = np.zeros((n_frames, 72), np.float32)
    offs = np.arange(N_FFT)
    for s in range(0, n_frames, 4096):
        e = min(s + 4096, n_frames)
        idx = (np.arange(s, e) * hop).astype(np.int64)
        P = np.abs(np.fft.rfft(y_p[idx[:, None] + offs] * win, axis=1)) ** 2
        bars[s:e] = P @ M
        lmh[s:e, 0] = P[:, b_low].sum(1)
        lmh[s:e, 1] = P[:, b_mid].sum(1)
        lmh[s:e, 2] = P[:, b_high].sum(1)
        kick_spec[s:e] = P[:, b_kick]
        hat_spec[s:e] = P[:, b_hat]
        subp[s:e] = P[:, b_sub].sum(1)
        totp[s:e] = P.sum(1)
        lspec[s:e] = P @ Lm
        chroma[s:e] = np.log1p(1e3 * P / (P.max(1, keepdims=True) + 1e-12)) @ Cm

    bars = norm_db(bars, 10, 99.5)
    lmh_n = norm_db(lmh, 20, 99.5)
    low, mid, high = lmh_n[:, 0], lmh_n[:, 1], lmh_n[:, 2]

    # --- flujo espectral en graves (35-130 Hz), bin por bin
    ks = 10 * np.log10(kick_spec + 1e-10)
    flux = np.maximum(0, np.diff(ks, axis=0, prepend=ks[:1])).mean(1)

    # --- seguimiento de tempo: grilla de beats + ¿hay kick en cada beat?
    beats = beat_track(flux, fps, bpm_range[0], bpm_range[1])
    kf = np.array([flux[max(0, b - 1):b + 2].max() for b in beats]) if len(beats) else np.zeros(0)
    onsets = np.zeros(0, np.int64)
    strength = np.zeros(n_frames, np.float32)
    if len(beats):
        ref = maximum_filter1d(uniform_filter1d(kf, 16), 64)
        g75 = np.percentile(kf, 75)
        present = (kf > 0.40 * ref) & (kf > 0.30 * g75) & (low[beats] > 0.2)
        onsets = beats[present]
        st = kf[present]
        if len(st):
            strength[onsets] = np.clip(st / np.percentile(st, 90), 0.05, 1.5)

    # --- break / drop: primer kick tras >= 4 s sin kicks
    since_kick = np.zeros(n_frames, np.float32)
    drops = np.zeros(n_frames, bool)
    last = -10 ** 9
    on_set = set(onsets.tolist())
    for i in range(n_frames):
        if i in on_set:
            if (i - last) / fps >= 4.0 and last > -10 ** 9:
                drops[i] = True
            last = i
        since_kick[i] = (i - last) / fps if last > -10 ** 9 else 99.0

    # --- zonas de bajada (mirando hacia adelante): del último kick antes de un
    # hueco de >= 4 s hasta el kick que vuelve. También antes del primer kick.
    break_zone = np.zeros(n_frames, bool)
    break_id = np.zeros(n_frames, np.int32)
    ons = onsets.tolist()
    edges_k = [-1] + ons + [n_frames]
    bid = 0
    for p, q in zip(edges_k[:-1], edges_k[1:]):
        if (q - p) / fps >= 4.0 or p < 0 or q >= n_frames:
            a0 = max(0, p + int(0.4 * fps)) if p >= 0 else 0
            if a0 < q:
                break_zone[a0:q] = True
                bid += 1
                break_id[a0:] = bid

    # --- key por ventana de 90 s (paso 15 s), Krumhansl-Schmuckler
    win, stp = int(90 * fps), int(15 * fps)
    profs = np.array([np.roll(KS_MIN, t) for t in range(12)] + [np.roll(KS_MAJ, t) for t in range(12)])
    profs = (profs - profs.mean(1, keepdims=True)) / profs.std(1, keepdims=True)
    keys = []
    for c0 in range(0, max(1, n_frames - win // 2), stp):
        v = chroma[c0:c0 + win].mean(0)
        v = (v - v.mean()) / (v.std() + 1e-9)
        keys.append(int(np.argmax(profs @ v)))
    keys = np.array(keys)
    # suavizado: moda en ventana de 5 (evita saltos de 15 s)
    sm = keys.copy()
    for j in range(len(keys)):
        w5 = keys[max(0, j - 2):j + 3]
        sm[j] = np.bincount(w5, minlength=24).argmax()
    key_frame = np.repeat(sm, stp)[:n_frames]
    if len(key_frame) < n_frames:
        key_frame = np.pad(key_frame, (0, n_frames - len(key_frame)), mode="edge")

    # --- momentos RGB: en cada drop fuerte (kick que vuelve tras >= 8 s),
    # el acento rota rojo -> verde -> azul con cada kick durante 32 kicks
    accent = np.zeros(n_frames, np.int8)          # 0 rojo, 1 verde, 2 azul
    n_big = 0
    prev_on = None
    for j, o in enumerate(ons):
        if prev_on is not None and (o - prev_on) / fps >= 8.0:
            n_big += 1
            for m in range(32):
                if j + m >= len(ons):
                    break
                a0 = ons[j + m]
                a1 = ons[j + m + 1] if j + m + 1 < len(ons) else n_frames
                accent[a0:a1] = m % 3
        prev_on = o

    # ================= capa de lectura: estados del track =================
    sec = max(1, int(fps))
    # 1) sub presente / filtrado (histéresis + suavizado de ~1 s)
    sub_db = uniform_filter1d(10 * np.log10(subp + 1e-10), sec)
    lo_s, hi_s = np.percentile(sub_db, 10), np.percentile(sub_db, 90)
    sub_on = np.zeros(n_frames, bool)
    if hi_s - lo_s > 8:                       # el track realmente abre/cierra el sub
        # relativo al sub pleno: un corte parcial (filtro a medias) también es tensión
        up, dn = hi_s - 3.5, hi_s - 6.5
        st_on = sub_db[0] > (up + dn) / 2
        for i in range(n_frames):
            if st_on and sub_db[i] < dn:
                st_on = False
            elif not st_on and sub_db[i] > up:
                st_on = True
            sub_on[i] = st_on
    else:
        sub_on[:] = True
    # 2) silencios (>= 0.5 s muy por debajo del nivel típico) -> capítulos
    tot_db = 10 * np.log10(uniform_filter1d(totp, max(1, sec // 2)) + 1e-10)
    from scipy.ndimage import median_filter as _mf
    ctx = _mf(tot_db, size=int(12 * fps) | 1, mode="nearest")
    silent = (tot_db < np.percentile(tot_db, 90) - 22) & (tot_db < ctx - 10)
    # silencio = corte breve (0.7-4 s) con música después; un tramo largo en bajo nivel
    # (fade out, outro) no es un silencio: es una desaparición gradual
    fade = np.zeros(n_frames, np.float32)
    ref_db = np.percentile(tot_db, 90)
    i = 0
    while i < n_frames:
        if silent[i]:
            j = i
            while j < n_frames and silent[j]:
                j += 1
            if (j - i) > 4 * fps or j >= n_frames - int(1 * fps):
                silent[i:j] = False
            i = j
        else:
            i += 1
    # fade: nivel largo (8 s) muy por debajo del típico, sin kicks cerca
    lvl = uniform_filter1d(tot_db, int(8 * fps))
    fade = np.clip((ref_db - 14 - lvl) / 20, 0, 1).astype(np.float32)
    full = np.where(lvl > ref_db - 8)[0]
    last_full = int(full[-1]) if len(full) else 0
    fade[:last_full] = 0                           # la intro suave no es un fade out
    chapter = np.zeros(n_frames, np.int8)
    ch, run = 0, 0
    for i in range(n_frames):
        run = run + 1 if silent[i] else 0
        if run == int(0.7 * fps) and fade[i] < 0.5:   # un hueco dentro del fade final no abre capítulo
            ch += 1
        chapter[i] = ch
    # 3) liberaciones: vuelve el sub (o vuelve el kick tras una bajada larga)
    rise = np.zeros(n_frames, bool)
    ch_start = np.zeros(n_frames, bool)
    ons_arr = np.array(ons, np.int64)
    # nivel real de sub (sin suavizar) en los 150 ms después de cada kick: el detector suavizado
    # se entera tarde, así que la liberación se adelanta al primer kick que ya trae el sub entero
    from scipy.signal import butter, sosfilt
    sub_raw = sosfilt(butter(4, [25, 60], "band", fs=sr, output="sos"), y)
    w15 = int(0.15 * sr)
    kick_sub = np.array([10 * np.log10((sub_raw[int(o / fps * sr):int(o / fps * sr) + w15] ** 2).mean() + 1e-12)
                         for o in ons]) if len(ons) else np.zeros(0)
    ks_full = np.percentile(kick_sub, 90) if len(kick_sub) else 0.0
    for i in np.where(np.diff(sub_on.astype(np.int8)) == 1)[0] + 1:
        j = int(np.searchsorted(ons_arr, i - int(0.3 * fps)))
        if j >= len(ons):
            rise[i] = True
            continue
        while j > 0 and (ons[j] - ons[j - 1]) / fps < 0.8 and kick_sub[j - 1] > ks_full - 6:
            j -= 1
        rise[ons[j]] = True                        # anclado al primer kick con el sub entero
    for i, o in enumerate(ons):
        if i and (o - ons[i - 1]) / fps >= 8.0:
            rise[o] = True
    rise[:int(2 * fps)] = False
    # vuelve el sub y vuelve el kick casi juntos = una sola liberación (queda la primera)
    last_r = -10 ** 9
    for r in np.where(rise)[0]:
        if r - last_r < 2 * fps:
            rise[r] = False
        else:
            last_r = r
    # la tensión termina en la liberación: el detector de sub (suavizado ~1 s) llega tarde
    for r in np.where(rise)[0]:
        if not sub_on[r]:
            nxt = np.where(sub_on[r:r + int(2 * fps)])[0]
            if len(nxt):
                sub_on[r:r + nxt[0]] = True
    # 4) tensión: progreso 0->1 dentro de cada tramo filtrado / bajada
    tension = np.zeros(n_frames, np.float32)
    closed = (~sub_on) | break_zone
    i = 0
    while i < n_frames:
        if closed[i]:
            j = i
            while j < n_frames and closed[j]:
                j += 1
            tension[i:j] = np.linspace(0, 1, j - i, dtype=np.float32)
            i = j
        else:
            i += 1
    # 5) caos: curva lenta de agudos (~8 s) normalizada al track
    hi_db = uniform_filter1d(10 * np.log10(lmh[:, 2] + 1e-10), int(8 * fps))
    a_, b_ = np.percentile(hi_db, 5), np.percentile(hi_db, 95)
    chaos = np.clip((hi_db - a_) / max(b_ - a_, 1e-6), 0, 1).astype(np.float32)
    # 6) RGB: en cada liberación, rota R->G->B por kick durante 32 kicks
    accent[:] = 0
    rel_kick = -np.ones(n_frames, np.int16)     # n° de kick dentro de cada liberación (-1 fuera)
    kidx = np.searchsorted(ons_arr, np.arange(n_frames), side="right") - 1
    n_rise = 0
    for r in np.where(rise)[0]:
        n_rise += 1
        j0 = int(np.searchsorted(ons_arr, r))
        for m in range(32):
            if j0 + m >= len(ons):
                break
            a0 = ons[j0 + m]
            a1 = ons[j0 + m + 1] if j0 + m + 1 < len(ons) else n_frames
            accent[a0:a1] = m % 3
            rel_kick[a0:a1] = m
    n_big = n_rise

    # 7) hats / percusión aguda: flujo espectral en 5-11 kHz, picos locales sobre el entorno
    hs = 10 * np.log10(hat_spec + 1e-10)
    del hat_spec
    hflux = np.maximum(0, np.diff(hs, axis=0, prepend=hs[:1])).mean(1)
    hloc = uniform_filter1d(hflux, max(3, int(1.5 * fps)))
    peak = (hflux >= maximum_filter1d(hflux, 3)) & (hflux > 1.6 * hloc) \
        & (hflux > 0.25 * np.percentile(hflux, 99))
    hat_on = np.zeros(n_frames, bool)
    last_h = -10 ** 9
    for i in np.where(peak)[0]:
        if i - last_h >= 2:                   # a 30 fps, más o menos una semicorchea
            hat_on[i] = True
            last_h = i
    hat = envelope(hat_on.astype(np.float32), 0.55)
    hat_n = np.cumsum(hat_on).astype(np.int32)

    layers = detect_layers(lspec, np.sqrt(le[:-1] * le[1:]), fps, silent)

    return dict(
        n=n_frames,
        fade=fade,
        layers=layers,
        sub_on=sub_on,
        silent=silent,
        chapter=chapter,
        rise=rise,
        tension=tension,
        chaos=chaos,
        accent=accent,
        rel_kick=rel_kick,
        hat=hat,
        hat_n=hat_n,
        n_big_drops=n_big,
        break_zone=break_zone,
        break_id=break_id,
        key=key_frame,
        bars=envelope_2d(bars, 0.80),
        low=envelope(low, 0.86),
        mid=envelope(mid, 0.90),
        high=envelope(high, 0.80),
        kick=envelope(np.where(strength > 0, 1.0, 0.0).astype(np.float32), 0.80),
        strength=strength,
        onsets=onsets,
        drops=drops,
        since_kick=since_kick,
    )


def detect_layers(lspec, cf, fps, silent, K=8):
    """Separa el track en K capas (NMF) por encima de 300 Hz y detecta cuándo entra
    cada una. Devuelve actividad rápida por capa y el orden de entrada."""
    try:
        from sklearn.decomposition import NMF
    except Exception:
        return None
    import warnings
    warnings.filterwarnings("ignore")
    n = len(lspec)
    V = np.log1p(lspec / (np.percentile(lspec, 97, axis=0) + 1e-12) * 30)
    step = max(1, int(fps // 10))                 # ajuste a 10 fps, proyección a fps completo
    model = NMF(n_components=K, init="nndsvda", max_iter=500, beta_loss="kullback-leibler",
                solver="mu", random_state=0)
    model.fit(V[::step])
    Hc = model.components_
    act = np.maximum(0, V @ np.linalg.pinv(Hc))   # activación a resolución completa
    act = act / (np.percentile(act, 97, axis=0) + 1e-9)
    slow = uniform_filter1d(act, int(2 * fps), axis=0)
    ref = np.percentile(slow, 95, axis=0) + 1e-9
    on = slow > 0.35 * ref
    # rellenar huecos cortos (< 3 s) y descartar tramos cortos (< 6 s)
    gap, minlen = int(3 * fps), int(6 * fps)
    for k in range(K):
        x = on[:, k].copy()
        i = 0
        while i < n:
            if not x[i]:
                j = i
                while j < n and not x[j]:
                    j += 1
                if 0 < i and j < n and j - i < gap and not silent[i:j].all():
                    x[i:j] = True
                i = j
            else:
                i += 1
        i = 0
        while i < n:
            if x[i]:
                j = i
                while j < n and x[j]:
                    j += 1
                if j - i < minlen:
                    x[i:j] = False
                i = j
            else:
                i += 1
        on[:, k] = x
    first = np.array([int(np.argmax(on[:, k])) if on[:, k].any() else n for k in range(K)])
    center = np.array([float(np.exp((np.log(cf) * h).sum() / h.sum())) for h in Hc])
    base = first < int(4 * fps)                    # sonaban desde el arranque
    order = [k for k in np.argsort(first) if not base[k] and first[k] < n]
    # capas que entran juntas (< 3 s) y en frecuencias vecinas (< 1 octava) = un solo elemento
    groups = []
    for k in order:
        if groups:
            g = groups[-1]
            if first[k] - first[g[0]] < 3 * fps and max(center[k], center[g[0]]) / min(center[k], center[g[0]]) < 2:
                g.append(k)
                continue
        groups.append([k])
    fast = np.clip(act, 0, 1.5).astype(np.float32)
    G = len(groups)
    gact = np.zeros((n, G), np.float32)
    gon = np.zeros((n, G), bool)
    gfirst = np.zeros(G, np.int64)
    gcen = np.zeros(G)
    for j, g in enumerate(groups):
        gact[:, j] = fast[:, g].max(1)
        gon[:, j] = on[:, g].any(1)
        gfirst[j] = first[g].min()
        gcen[j] = float(np.exp(np.mean(np.log(center[g]))))
    return dict(act=gact, on=gon, first=gfirst, center=gcen, order=list(range(G)))


def envelope_2d(x, decay):
    out = np.empty_like(x)
    v = np.zeros(x.shape[1], np.float32)
    for i in range(x.shape[0]):
        v = np.maximum(x[i], v * decay)
        out[i] = v
    return out


# ------------------------------------------------------ video del celular ----
def audio_features(path, start=0, duration=0):
    """(envolvente lenta a 10 Hz, ataques a 100 Hz), ambos normalizados."""
    sr = 8000
    y = load_audio(path, sr, start, duration)
    hop = sr // 100
    n = len(y) // hop
    le = np.log(np.sqrt((y[:n * hop].reshape(n, hop) ** 2).mean(1)) + 1e-5)
    on = np.maximum(0, np.diff(le, prepend=le[0]))
    on = (on - on.mean()) / (on.std() + 1e-9)
    m = n // 10
    slow = le[:m * 10].reshape(m, 10).mean(1)
    slow = slow - uniform_filter1d(slow, 300)      # saca tendencia de ~30 s
    slow = (slow - slow.mean()) / (slow.std() + 1e-9)
    return slow, on


def xcorr(ref, clip):
    n = len(ref) + len(clip)
    N = 1 << (n - 1).bit_length()
    cc = np.fft.irfft(np.fft.rfft(ref, N) * np.conj(np.fft.rfft(clip, N)), N)
    lags = np.arange(N)
    lags[lags >= N - len(clip)] -= N      # positivos hasta len(ref), negativos hasta len(clip)
    return cc, lags


def xcorr_offset(ref_feats, clip_feats):
    """Dos pasos: grueso con la envolvente lenta, fino (±0.3 s) con los ataques."""
    (rs, ro), (cs, co) = ref_feats, clip_feats
    cc, lags = xcorr(rs, cs)
    k = int(np.argmax(cc))
    coarse = lags[k] / 10
    conf = float(cc[k] / (cc.std() + 1e-9))
    cc2, lags2 = xcorr(ro, co)
    sel = np.abs(lags2 / 100 - coarse) <= 0.3
    k2 = int(np.argmax(np.where(sel, cc2, -np.inf)))
    return lags2[k2] / 100, conf


def video_duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", path], capture_output=True, text=True).stdout
    return float(out.strip() or 0)


def sync_video(audio_path, video_path, win=600):
    """Sincroniza el audio del celular con el set. Devuelve (offset_s, drift).
    offset = segundo del set en que arranca el video (negativo si arrancó antes)."""
    print("Sincronizando video con el audio del set...", flush=True)
    ref = audio_features(audio_path)
    set_dur = len(ref[1]) / 100
    vdur = video_duration(video_path)
    win = min(win, vdur / 2)
    t_a = min(60, vdur * 0.05)
    off_a, conf_a = xcorr_offset(ref, audio_features(video_path, t_a, win))
    off_a -= t_a
    print(f"  tramo {t_a/60:5.1f} min del video -> offset {off_a:8.2f}s  (confianza {conf_a:.1f})")
    # segundo control: lo más tarde posible dentro de la parte compartida
    t_b = min(vdur, set_dur - off_a) - win - 30
    drift = 0.0
    if t_b > t_a + win:
        off_b, conf_b = xcorr_offset(ref, audio_features(video_path, t_b, win))
        off_b -= t_b
        print(f"  tramo {t_b/60:5.1f} min del video -> offset {off_b:8.2f}s  (confianza {conf_b:.1f})")
        d = (off_b - off_a) / (t_b - t_a)
        if conf_b >= 6 and abs(d) < 0.001:
            drift = d
    if conf_a < 6:
        print("  ⚠ sincronización dudosa: revisá la prueba o pasá --video-offset a mano")
    return off_a, drift


class VideoSource:
    """Lee el video del celular en gris, ya escalado y recortado, a `fps` fijos."""

    def __init__(self, path, w, h, fps, set_start, offset, drift):
        self.w, self.h, self.fps = w, h, fps
        self.offset, self.drift = offset, drift
        self.dur = video_duration(path)
        pt0 = max(0.0, (set_start - offset) / (1 + drift))
        self.base = pt0
        vf = (f"fps={fps},scale={w}:{h}:force_original_aspect_ratio=increase,"
              f"crop={w}:{h},format=gray")
        self.proc = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-ss", f"{pt0:.3f}", "-i", path, "-an", "-vf", vf,
             "-f", "rawvideo", "-pix_fmt", "gray", "-"], stdout=subprocess.PIPE)
        self.idx = -1
        self.cur = None
        self.lo = self.hi = None

    def frame_at(self, set_time):
        pt = (set_time - self.offset) / (1 + self.drift)
        if pt < 0 or pt >= self.dur - 0.1:
            return None
        j = int(round((pt - self.base) * self.fps))
        size = self.w * self.h
        while self.idx < j:
            buf = self.proc.stdout.read(size)
            if len(buf) < size:
                return self.cur
            self.cur = np.frombuffer(buf, np.uint8).reshape(self.h, self.w)
            self.idx += 1
        return self.cur

    def posterize(self, g, kick):
        """Gris -> negro / blanco / rojo con umbrales adaptativos (sin parpadeo)."""
        lo, hi = np.percentile(g[::4, ::4], (5, 99.5))
        if self.lo is None:
            self.lo, self.hi = lo, hi
        self.lo += 0.05 * (lo - self.lo)
        self.hi += 0.05 * (hi - self.hi)
        n = (g.astype(np.float32) - self.lo) / max(self.hi - self.lo, 8)
        idx = (n > 0.55).astype(np.uint8)          # 1 = blanco
        if kick > 0.5:                              # en el kick, medios tonos en rojo
            idx[(n > 0.30) & (n <= 0.55)] = 2
        return idx

    def close(self):
        try:
            self.proc.kill()
        except Exception:
            pass


# ------------------------------------------------------------ tipografía ----
FONT_CANDIDATES = [
    # Linux
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    # macOS
    "/System/Library/Fonts/Supplemental/Arial Black.ttf",
    "/System/Library/Fonts/Supplemental/Impact.ttf",
    "/Library/Fonts/Arial Black.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    # Windows
    "C:/Windows/Fonts/ariblk.ttf",
    "C:/Windows/Fonts/impact.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
]
MONO_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/Monaco.ttf",
    "C:/Windows/Fonts/consolab.ttf",
    "C:/Windows/Fonts/courbd.ttf",
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


# -------------------------------------------------------------- dibujo ----
def rect(frame, x0, y0, x1, y1, color):
    frame[max(0, int(y0)):max(0, int(y1)), max(0, int(x0)):max(0, int(x1))] = color


def outline(frame, x0, y0, x1, y1, t, color):
    rect(frame, x0, y0, x1, y0 + t, color)
    rect(frame, x0, y1 - t, x1, y1, color)
    rect(frame, x0, y0, x0 + t, y1, color)
    rect(frame, x1 - t, y0, x1, y1, color)


def invert_keep_red(f):
    """Invierte solo blanco/negro; los colores (rojo, verde, azul) quedan igual."""
    out = f.copy()
    bw = (f[..., 0] == f[..., 1]) & (f[..., 1] == f[..., 2])
    out[bw] = 255 - f[bw]
    return out


# ------------------------------------------------------------- acabado ----
# Post-proceso reactivo (modo limpio). Cada efecto lee una señal del track:
#   bloom     <- liberaciones: se enciende en el rise y se apaga en unos segundos
#   estela    <- tensión: feedback que se alarga mientras el sub está filtrado
#   grano     <- caos: el track se abre -> más grano
#   fantasma  <- kicks fuertes: un eco en el color de acento se desplaza hacia afuera (sin salir de la paleta)
#   viñeta    <- fija; no se cierra con la tensión (nada de iris)
LOOKS = {
    # bloom=(base, pico en la liberación); trail_hl=(vida media abierta, en tensión máx) en s;
    # trail_g = brillo del recuerdo (lo vivo siempre va al frente); black = punto de negro
    "luz":       dict(bloom=(0.10, 0.75), bloom_tau=3.5, bloom_sig=(8, 32), bloom_mix=0.15,
                      trail_hl=(0.012, 0.40), trail_g=0.55, grain=(0.012, 0.045), grain_px=1,
                      ghost_px=5, vignette=0.30, black=0.05),
    "atmosfera": dict(bloom=(0.12, 0.8), bloom_tau=6.0, bloom_sig=(12, 56), bloom_mix=0.5,
                      trail_hl=(0.03, 0.90), trail_g=0.70, grain=(0.02, 0.07), grain_px=2,
                      ghost_px=8, vignette=0.50),
    "seco":      None,
}


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
        if "breath" in A:
            self.decay[A["breath"]] = 0.0          # la respiración corta la estela: negro limpio
        # grano según el caos
        self.grain = (L["grain"][0] + (L["grain"][1] - L["grain"][0]) * A["chaos"]).astype(np.float32)
        # fantasma rojo en kicks fuertes; más leve en tensión
        strong = np.where(A["strength"] >= 1.0, 1.0, 0.0).astype(np.float32)
        self.ghost = envelope(strong, 0.55) * (1 - 0.6 * T)
        self.T = T.astype(np.float32)
        self.flare = env.astype(np.float32)          # destello de la liberación (para el horizonte)
        self.kick = A["kick"].astype(np.float32)
        self.hat = A["hat"].astype(np.float32) if "hat" in A else np.zeros(n, np.float32)
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
        y = y.copy() if y is self.acc else y
        y[y0:y1, x0:x1] = sub
        self.acc[y0:y1, x0:x1] *= inside                                         # la estela también cae adentro
        return y

    def _gmap(self, px):
        """Mapas de muestreo para escalar el canal rojo hacia afuera `px` píxeles en el borde."""
        if px not in self._gmaps:
            W, H = self.W, self.H
            k = 1 + px / (W / 2)
            xs = np.clip(np.round(W / 2 + (np.arange(W) + 0.5 - W / 2) / k - 0.5), 0, W - 1).astype(np.intp)
            ys = np.clip(np.round(H / 2 + (np.arange(H) + 0.5 - H / 2) / k - 0.5), 0, H - 1).astype(np.intp)
            self._gmaps[px] = (ys, xs)
        return self._gmaps[px]

    def apply(self, f, i, accent=RED, hole=None):
        from scipy.ndimage import gaussian_filter
        L, W, H = self.L, self.W, self.H
        x = f.astype(np.float32) * (1 / 255)
        # estela: lo vivo siempre al frente; lo anterior queda como un recuerdo tenue
        # que dura más cuanto mayor es la tensión
        if self.acc is not None:
            mem = self.acc * self.decay[i]
            self.acc = np.maximum(x, mem)
            x = np.maximum(x, mem * L["trail_g"])
        else:
            self.acc = x
        y = x
        # fantasma rojo
        px = int(round(float(self.ghost[i]) * L["ghost_px"] * self.s))
        if px >= 1:
            ys, xs = self._gmap(px)
            # lo encendido, desplazado hacia afuera, en el color de acento: aberración dentro de la paleta
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


class Renderer:
    """Túnel hipnótico de cuadrados concéntricos. Bordes duros, solo negro/blanco/rojo."""

    def __init__(self, W, H, fps, title, subtitle, font, seed, start_offset, pattern="bars",
                 clean=False, preset=None, ss=1):
        self.W, self.H, self.fps = W, H, fps
        self.pattern = pattern
        self.clean = clean
        self.P = dict(preset or {})
        s = H / 1080
        self.s = s
        self.m = int(56 * s)
        self.t = max(2, int(4 * s))
        if clean:
            self.top, self.bot = -self.t, H          # imagen a pantalla completa
        else:
            self.top = int((110 if subtitle else 50) * s)
            self.bot = H - int(190 * s)
        self.start_offset = start_offset
        self.rng_seed = seed

        big = load_font(font, FONT_CANDIDATES, int(32 * s))
        med = load_font(font, FONT_CANDIDATES, int(34 * s))
        self.mono = load_font(None, MONO_CANDIDATES, int(24 * s))
        self.title_mask = text_mask(title.upper(), big)
        self.sub_mask = text_mask(subtitle.upper(), med) if subtitle else None
        self._tc_cache = (None, None)

        # zona del túnel. ss=1: se calcula a media resolución y se agranda sin
        # interpolar (píxeles duros). ss>=2 (acabado B): se calcula a ss× la salida
        # y se baja con lanczos -> bordes limpios, misma geometría.
        self.ds = 2
        self.sup = ss
        self.ty0 = self.top + self.t
        self.th = ((self.bot - self.ty0) // self.ds) * self.ds
        self.tw = (W // self.ds) * self.ds
        if ss > 1:
            hh, ww = self.th * ss, self.tw * ss
        else:
            hh, ww = self.th // self.ds, self.tw // self.ds
        self.hh, self.ww = hh, ww
        self.gs = ww / self.tw                    # píxeles de grilla por píxel de salida
        self.u = 2 * self.gs                      # píxeles de grilla por unidad del diseño (medio píxel de salida)
        self.tg = max(1, int(round(self.t * self.gs)) if ss > 1 else self.t // self.ds)
        self.post = None
        self.cam = None                           # cámara (se arma con el análisis)
        self.roll, self.box = 0.0, None
        self.cam_on = True
        self.hole = None                          # (x, y, radio) del horizonte en píxeles de salida
        self.accent_fixed = None                  # color de acento fijo (paleta), o RGB por liberación
        self.accent_cycle = None                  # colores que alternan por kick en las liberaciones
        yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
        r = hh / 2
        self.yy = (yy - hh / 2 + 0.5) / r
        self.xx = (xx - ww / 2 + 0.5) / r
        # 0 negro, 1 blanco, 2 acento; 3-5 blancos tenues (18/40/70 %), 6-7 acento tenue (35/65 %)
        self.pal = np.array([BLACK, WHITE, RED] + [BLACK] * 5, np.uint8)
        self.set_accent(RED)

        # ventana vertical 9:16 para el video del celular (centrada)
        pad = int(26 * s)
        self.vh = ((self.th - 2 * pad) // 4) * 4
        self.vw = ((self.vh * 9 // 16) // 4) * 4
        self.vx = (W - self.vw) // 2
        self.vy = self.ty0 + (self.th - self.vh) // 2

        # grosores de barras tipo código de barras (cambian cada 32 kicks)
        self.wrng = np.random.default_rng(seed + 99)
        self.wtab = self.new_widths()
        self.xrow = np.abs(self.xx[0])            # distancia horizontal al centro
        self.xs_half = np.arange(ww, dtype=np.float32) + 0.5
        self.r_half = r
        self.x_off = (W - self.tw) // 2
        self.jump_every = 0                       # 0 = video fijo en el centro
        self.red_share = 0.6
        self.accent = RED
        self.vx_center = self.vx
        self._pos_key = None

        # estado de la animación
        self.phase = 0.0
        self.angle = 0.0
        self.snap = 0.0
        self.last_kc = 0

        # coordenadas polares (media resolución) para los estilos
        self.RR = np.hypot(self.xx, self.yy).astype(np.float32)
        self.TH = np.arctan2(self.yy, self.xx).astype(np.float32)
        self.logRR = np.log(self.RR + 1e-4).astype(np.float32)
        # remolino: torsión que crece hacia el centro (el agujero negro arrastra lo cercano)
        self.SWIRL = (0.3 / (self.RR + 0.05)).astype(np.float32)
        # coordenadas cartesianas en unidades del diseño (enteras en ss=1)
        self.PX = ((xx - ww / 2) / self.u).astype(np.float32)
        self.PY = ((yy - hh / 2) / self.u).astype(np.float32)

        # capa estática (marco + tipografía)
        st = np.zeros((H, W, 3), np.uint8)
        if clean:
            self.static = st
            return
        rect(st, 0, self.top, W, self.top + self.t, WHITE)
        rect(st, 0, self.bot, W, self.bot + self.t, WHITE)
        if self.sub_mask is not None:
            blit(st, self.sub_mask, self.m, (self.top - self.sub_mask.shape[0]) // 2, WHITE)
        # título en rojo, abajo a la derecha, a la izquierda del REC + timecode
        tc_w = text_mask("00:00:00", self.mono).shape[1]
        d = int(16 * s)
        row_bottom = H - int(44 * s) + text_mask("0", self.mono).shape[0]
        tx = W - self.m - tc_w - 2 * d - int(22 * s) - self.title_mask.shape[1]
        blit(st, self.title_mask, tx, row_bottom - self.title_mask.shape[0], RED)
        self.static = st

    def _camera(self, A, i):
        """Cámara, función pura del cuadro (sin estado: sirve igual en tramos y en paralelo).
        zoom: empuja hacia adentro con la tensión (sobre todo al final) y suelta de golpe en la
        liberación, con un rebote; en lo abierto respira lento. kick: golpe de zoom + empujón.
        giro: deriva lenta (más amplia con el caos) y un giro seco en cada liberación."""
        if self.cam is None:
            fps, n = self.fps, A["n"]
            g = np.random.default_rng([self.rng_seed, 909])
            ph = g.uniform(0, 2 * np.pi, 2)
            idx = np.arange(n)
            T = np.maximum(A["tension"], A["fade"]).astype(np.float64)
            Ts = np.empty(n)                       # sube con la tensión; al liberar suelta en ~0.15 s
            v, k = 0.0, np.exp(-1 / (0.15 * fps))
            for j in range(n):
                v = T[j] if T[j] > v else v * k
                Ts[j] = v
            t = idx / fps
            last_r = np.maximum.accumulate(np.where(A["rise"], idx, -1))
            dt_r = np.where(last_r >= 0, (idx - last_r) / fps, 1e9)
            bounce = 0.03 * np.exp(-dt_r / 0.35) * (1 - np.exp(-dt_r / 0.06))   # se pasa hacia afuera y vuelve
            # kick: fuerza del golpe, rápida; menor en tensión
            kamp = envelope(np.clip(A["strength"], 0, 1.5).astype(np.float32), 0.55) * (1 - 0.6 * Ts)
            zoom = (1.05 + 0.20 * Ts ** 2 + 0.016 * (1 - Ts) * np.sin(2 * np.pi * t / 28 + ph[0])
                    - bounce + 0.028 * kamp)
            # giro: deriva ±2.8° y un paso seco de 2.2° por liberación (alterna el sentido: no se acumula)
            snaps = np.zeros(n)
            sign = 1.0 if g.random() < 0.5 else -1.0
            for r in np.where(A["rise"])[0]:
                snaps[r:] += sign * np.deg2rad(2.2)
                sign = -sign
            sn, v = np.empty(n), 0.0
            for j in range(n):                     # llega en ~3 cuadros
                v += 0.45 * (snaps[j] - v)
                sn[j] = v
            roll = np.deg2rad(2.8) * np.sin(2 * np.pi * t / 23 + ph[1]) * (0.5 + 0.5 * A["chaos"]) + sn
            # empujón: dirección propia de cada kick, amplitud según su fuerza
            on = np.where(A["strength"] > 0, idx, -1)
            last = np.maximum.accumulate(on)
            ang = self._hash(np.maximum(last, 0), np.zeros(n, np.int64) + 31, self.rng_seed) * 2 * np.pi
            sh = 15.0 * self.s * kamp
            dx, dy = sh * np.cos(ang), sh * np.sin(ang)
            if "hat" in A:                         # nervio: temblor chico en los hats al final de la tensión
                nerv = np.clip((T - 0.35) / 0.65, 0, 1) * (T > 0)
                hang = self._hash(A["hat_n"].astype(np.int64), np.zeros(n, np.int64) + 57, self.rng_seed) * 2 * np.pi
                hs_ = 5.0 * self.s * A["hat"] * nerv
                dx, dy = dx + hs_ * np.cos(hang), dy + hs_ * np.sin(hang)
            self.cam = dict(zoom=zoom, roll=roll, dx=dx, dy=dy)
        C = self.cam
        self.roll = float(C["roll"][i])
        z = float(C["zoom"][i])
        gw, gh = self.ww, self.hh
        bw, bh = gw / z, gh / z
        mx, my = (gw - bw) / 2, (gh - bh) / 2
        x0 = mx + float(np.clip(C["dx"][i] * self.gs, -mx, mx))
        y0 = my + float(np.clip(C["dy"][i] * self.gs, -my, my))
        self.box = (x0, y0, x0 + bw, y0 + bh)

    def _up(self, img):
        """Grilla de cálculo -> píxeles de salida (con el encuadre de la cámara, si hay)."""
        if self.sup > 1:
            return np.asarray(Image.fromarray(img).resize((self.tw, self.th), Image.LANCZOS, box=self.box))
        return np.repeat(np.repeat(img, self.ds, axis=0), self.ds, axis=1)

    def _finish_clean(self, f, A, i, kick):
        W, H, s, m = self.W, self.H, self.s, self.m
        if A["strength"][i] >= 1.0 and A["chaos"][i] > 0.5 and self.P.get("glitch", True):   # glitch solo con caos alto
            g = np.random.default_rng(self.rng_seed + i)
            for k in range(int(g.integers(1, 4))):
                y = int(g.integers(0, H - 10))
                hh = int(g.integers(int(6 * s), int(40 * s)))
                band = f[y:y + hh]
                band[:] = np.roll(band, int(g.integers(-140, 140) * s), axis=1)
        if self.P.get("title", True):
            th, tw = self.title_mask.shape
            x0, y0 = W - m - tw, H - m - th
            pad = int(10 * s)
            rect(f, x0 - pad, y0 - pad, x0 + tw + pad, y0 + th + pad, BLACK)
            blit(f, self.title_mask, x0, y0, RED)
        return f

    def timecode(self, i):
        sec = int(i / self.fps + self.start_offset)
        if self._tc_cache[0] != sec:
            txt = f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"
            self._tc_cache = (sec, text_mask(txt, self.mono))
        return self._tc_cache[1]

    def new_widths(self):
        # mezcla de barras finas, medias y gruesas
        return self.wrng.choice([0.25, 0.4, 0.6, 1.0, 1.5, 2.1], size=32,
                                p=[0.22, 0.2, 0.18, 0.18, 0.12, 0.10]).astype(np.float32)

    def set_accent(self, c):
        self.accent = c
        self.pal[2] = c
        c = np.asarray(c, np.float32)
        for j, f in ((3, 0.18), (4, 0.40), (5, 0.70)):
            self.pal[j] = np.round(255 * f)
        for j, f in ((6, 0.35), (7, 0.65)):
            self.pal[j] = np.round(c * f).astype(np.uint8)

    @staticmethod
    def _levels(I, acc):
        """Intensidad continua -> tonos de la paleta (blanco o acento, en 4 niveles)."""
        lev = np.zeros(I.shape, np.uint8)
        w = ~acc
        for thr, j in ((0.06, 3), (0.22, 4), (0.45, 5), (0.75, 1)):
            lev[(I > thr) & w] = j
        for thr, j in ((0.06, 6), (0.30, 7), (0.70, 2)):
            lev[(I > thr) & acc] = j
        return lev

    def set_pos(self, kick_count, in_break=False, break_id=0, music_key=0):
        """Cada `jump_every` kicks el video salta a una posición horizontal al azar
        y cambia el color de acento. En las bajadas vuelve al centro, en rojo.
        Todo es determinístico, así los tramos coinciden."""
        if not self.jump_every:
            return
        if in_break:
            key = ("break", int(break_id))
        else:
            key = (int(kick_count // self.jump_every), int(break_id))
        if key == self._pos_key:
            return
        self._pos_key = key
        if in_break:
            x = self.vx_center
            self.set_accent(RED)
        else:
            lo = self.m
            hi = self.W - self.vw - self.m
            prev = self.vx
            g = np.random.default_rng([self.rng_seed, key[0], key[1]])
            for _ in range(20):                   # que el salto se note
                x = int(g.integers(lo, hi + 1))
                if abs(x - prev) > self.vw * 0.8:
                    break
            # color: rojo la mayoría de las veces; si no, el color del track
            # (según su número Camelot: impar = verde, par = azul)
            mk = int(music_key)
            cam = CAMELOT_MIN[mk] if mk < 12 else CAMELOT_MAJ[mk - 12]
            track_col = GREEN if cam % 2 else BLUE
            self.set_accent(RED if g.random() < self.red_share else track_col)
        self.vx = (x // self.ds) * self.ds
        cxh = (self.vx + self.vw / 2 - self.x_off) * self.gs
        self.xrow = np.abs(self.xs_half - cxh) / self.r_half

    def bars(self, low, mid, high, kick, in_break, kick_count, break_id=0, music_key=0):
        self.set_pos(kick_count, in_break, break_id, music_key)
        """Barras verticales que nacen junto al video y se alejan hacia los bordes."""
        fps = self.fps
        energy = 0.6 * low + 0.4 * mid
        speed = 0.3 + 0.15 * mid if in_break else 0.7 + 2.0 * energy
        self.phase += speed / fps
        new_kicks = kick_count - self.last_kc
        if new_kicks > 0:
            self.phase += 0.22 * new_kicks
            if kick_count // 32 != self.last_kc // 32:
                self.wtab = self.new_widths()       # nuevo patrón cada 8 compases
        self.last_kc = kick_count

        d = self.xrow
        K = 4.5
        u = K * np.log(d + 1e-4) - self.phase
        n = np.floor(u)
        f = u - n
        ni = n.astype(np.int32)
        w = self.wtab[ni % len(self.wtab)]
        if in_break:
            duty = np.clip((0.05 + 0.08 * mid) * w, 0.02, 0.9)
            on = (f < duty) & (ni % 3 == 0)
        else:
            duty = np.clip((0.16 + 0.30 * kick) * w, 0.03, 0.92)
            on = f < duty
        red = on & ((ni + kick_count) % 7 == 0)
        idx = on.astype(np.uint8) + red.astype(np.uint8)
        row = np.repeat(self.pal[idx], self.ds, axis=0)          # (tw, 3)
        return np.broadcast_to(row[None], (self.th, self.tw, 3))

    def flow(self, A, i, kick_count):
        """Barras que leen el track: se cierran con la tensión, explotan en la
        liberación, se desordenan con el caos y cambian de geometría por capítulo."""
        fps = self.fps
        low, mid, kick = A["low"][i], A["mid"][i], A["kick"][i]
        chaos = float(A["chaos"][i])
        tens = float(A["tension"][i])
        closed = (not A["sub_on"][i]) or bool(A["break_zone"][i])
        chapter = int(A["chapter"][i])
        silent = bool(A["silent"][i]) or ("breath" in A and bool(A["breath"][i]))
        if not hasattr(self, "fl"):
            self.fl = dict(ph=0.0, phy=0.0, kc=0, ch=-1, wt=None, wty=None, rel=0)
        S = self.fl
        if chapter != S["ch"]:                     # capítulo nuevo: nuevo ADN
            S["ch"] = chapter
            g = np.random.default_rng([self.rng_seed, 101, chapter])
            S["g"] = g
            S["wt"] = g.choice([0.25, 0.4, 0.6, 1.0, 1.5, 2.1], size=32).astype(np.float32)
            S["wty"] = g.choice([0.3, 0.6, 1.0, 1.8], size=16).astype(np.float32)
            S["K"] = float(g.uniform(3.8, 5.2))
        if A["rise"][i]:
            S["rel"] = int(0.5 * fps)             # medio segundo de "explosión"
        # velocidad: cerrado = lento y creciendo con la tensión; pleno = empuja
        if closed:
            speed = 0.25 + 1.2 * tens * tens
        else:
            speed = 0.8 + 2.0 * (0.6 * low + 0.4 * mid) + 1.2 * chaos
        if S["rel"] > 0:
            speed += 6.0 * S["rel"] / (0.5 * fps)
            S["rel"] -= 1
        S["ph"] += speed / fps
        nk = kick_count - S["kc"]
        if nk > 0:
            S["ph"] += (0.10 if closed else 0.22) * nk
            # con caos alto, el patrón muta más seguido (cada 32 -> cada 8 kicks)
            every = int(round(32 - 24 * chaos))
            if every > 0 and kick_count // every != S["kc"] // every:
                g = S["g"]
                spread = 0.35 + 1.4 * chaos
                S["wt"] = np.clip(np.exp(g.normal(0, spread, 32)) * 0.7, 0.12, 2.6).astype(np.float32)
        S["kc"] = kick_count

        d = self.xrow
        K = S["K"]
        u = K * np.log(d + 1e-4) - S["ph"]
        n = np.floor(u)
        f = u - n
        ni = n.astype(np.int32)
        w = S["wt"][ni % 32] * self.P.get("dens", 1.0 if not self.clean else 0.7)
        if closed:
            duty = np.clip((0.06 + 0.10 * mid) * w, 0.02, 0.6)
            on = (f < duty) & (ni % 2 == 0)
        else:
            duty = np.clip((0.14 + 0.32 * kick) * w, 0.03, 0.92)
            on = f < duty
        red_mod = int(round(7 - 4 * chaos))       # más caos -> más barras de acento
        red = on & ((ni + kick_count) % max(2, red_mod) == 0)
        row_on, row_red = on, red

        hh = self.hh
        if chapter >= 1:                           # capítulo 2: entra la trama horizontal
            S["phy"] += (0.4 + 1.5 * mid) / fps + (0.08 * nk if nk > 0 else 0)
            yy = np.abs(self.yy[:, 0])
            uy = 3.2 * np.log(yy + 1e-4) - S["phy"]
            ny = np.floor(uy)
            fy = uy - ny
            dy = np.clip((0.12 + 0.25 * kick) * S["wty"][ny.astype(np.int32) % 16], 0.03, 0.8)
            col_on = fy < dy
            onm = row_on[None, :] ^ col_on[:, None]
            redm = row_red[None, :] & onm
            idx = onm.astype(np.uint8) + redm.astype(np.uint8)
        else:
            idx = np.broadcast_to((row_on.astype(np.uint8) + row_red.astype(np.uint8))[None, :],
                                  (hh, len(row_on))).copy()

        # tensión: las barras se van apagando y afinando (sin telones)
        if closed and not silent and self.clean:
            cc = dict(closed=True, silent=False, tens=tens)
            keep = self._erode(ni, cc, 2)
            thin_keep = f < np.clip(duty * (1 - 0.7 * tens), 0.01, 0.6)
            m = keep & thin_keep
            idx = idx * m[None, :].astype(np.uint8) if chapter < 1 else idx
        elif closed and not silent:
            ww = idx.shape[1]
            gap = int((ww / 2) * (1 - 0.72 * tens))
            cx = ww // 2
            idx[:, :max(0, cx - gap)] = 0
            idx[:, min(ww, cx + gap):] = 0
        if silent:
            idx[:] = 0
            yc = hh // 2
            idx[yc:yc + self.tg, :] = 1           # una sola línea en el silencio

        # glitch por caos: filas desplazadas en kicks
        if kick > 0.8 and chaos > 0.35 and not closed:
            g = np.random.default_rng([self.rng_seed, i])
            u = self.u
            for _ in range(int(1 + 5 * chaos)):
                y0 = int(g.integers(0, int(hh // u) - 4) * u)
                h_ = int(g.integers(2, max(3, int(18 * chaos))) * u)
                idx[y0:y0 + h_] = np.roll(idx[y0:y0 + h_], int(g.integers(-80, 80) * u), axis=1)

        return self._up(self.pal[idx])

    # ================================================================ estilos
    def _step(self, A, i, kc):
        """Estado compartido por todos los estilos (movimiento, capítulo, azar)."""
        fps = self.fps
        st = getattr(self, "ss", None)
        if st is None:
            st = self.ss = dict(ph=0.0, rot=0.0, snap=0.0, kc=0, ch=-1, rel=0, mut=0, tw=0.35, rh=0.08, glow=0.0)
        c = dict(low=float(A["low"][i]), mid=float(A["mid"][i]), high=float(A["high"][i]),
                 kick=float(A["kick"][i]), chaos=float(A["chaos"][i]), tens=float(A["tension"][i]),
                 closed=(not A["sub_on"][i]) or bool(A["break_zone"][i]),
                 silent=bool(A["silent"][i]) or bool(A["breath"][i]) if "breath" in A
                 else bool(A["silent"][i]), chapter=int(A["chapter"][i]))
        c["hat"] = float(A["hat"][i]) if "hat" in A else 0.0
        c["hat_n"] = int(A["hat_n"][i]) if "hat_n" in A else 0
        fd = float(A["fade"][i]) if "fade" in A else 0.0
        if fd > 0.05:
            c["closed"] = True
            c["tens"] = max(c["tens"], min(1.0, fd))
        if c["chapter"] != st["ch"]:
            st["ch"] = c["chapter"]
            st["g"] = np.random.default_rng([self.rng_seed, 202, c["chapter"]])
            st["wt"] = st["g"].choice([0.3, 0.5, 0.8, 1.0, 1.6, 2.2], size=32).astype(np.float32)
        st["glow"] *= np.exp(-1 / (2.5 * fps))      # destello de la materia: se calma en ~2.5 s
        if A["rise"][i]:
            st["rel"] = int(0.5 * fps)
            st["glow"] = 1.0
        if c["closed"]:
            speed = 0.25 + 1.2 * c["tens"] ** 2
        else:
            speed = 0.8 + 2.0 * (0.6 * c["low"] + 0.4 * c["mid"]) + 1.2 * c["chaos"]
        if st["rel"] > 0:
            speed += 6.0 * st["rel"] / (0.5 * fps)
            st["rel"] -= 1
        # agujero negro: en la tensión todo cae hacia el centro; al liberar, sale disparado
        flow = -1.0 if c["closed"] and st["rel"] <= 0 else 1.0
        st["ph"] += flow * speed / fps
        nk = kc - st["kc"]
        if nk > 0:
            st["ph"] += flow * (0.10 if c["closed"] else 0.22) * nk
            every = int(round(32 - 24 * c["chaos"]))
            if every > 0 and kc // every != st["kc"] // every:
                spread = 0.35 + 1.4 * c["chaos"]
                st["wt"] = np.clip(np.exp(st["g"].normal(0, spread, 32)) * 0.7, 0.12, 2.6).astype(np.float32)
                st["mut"] += 1
            if kc // 32 != st["kc"] // 32:
                st["snap"] += 1                    # giro/cambio duro cada 8 compases
        st["kc"] = kc
        st["rot"] += (0.05 + 0.35 * c["mid"] + 0.3 * c["chaos"]) / fps
        # torsión del remolino y tamaño del horizonte: siguen a la tensión (suavizados)
        tens_c = c["tens"] if c["closed"] else 0.0
        st["tw"] += (0.35 + 2.2 * tens_c - st["tw"]) * 0.15
        st["rh"] += (0.07 + 0.10 * tens_c + 0.03 * c["low"] - st["rh"]) * 0.25
        c["nk"] = nk
        return st, c

    def _iris(self, idx, c, rmax_full=2.1):
        """Tensión: todo lo que queda fuera de un radio que se cierra se apaga."""
        if c["closed"] and not c["silent"]:
            rmax = rmax_full * (1 - 0.78 * c["tens"]) + 0.12
            ring = (self.RR > rmax) & (self.RR < rmax + 0.012 + 0.01 * self.s)
            idx[self.RR > rmax] = 0
            idx[ring] = 2
        return idx

    def _horizon(self, idx, st):
        """El agujero negro: disco negro en la grilla (late con los graves y crece con la tensión).
        El resplandor difuso del borde lo agrega el post, en valores continuos."""
        rh = float(st["rh"]) * 1.12
        half = int(rh * self.hh / 2) + 2
        cy, cx = self.hh // 2, self.ww // 2
        sl = (slice(max(0, cy - half), cy + half), slice(max(0, cx - half), cx + half))
        idx[sl][self.RR[sl] < rh] = 0
        # dónde cae en la salida (con el encuadre de la cámara), para el resplandor del post
        x0, y0, x1, y1 = self.box or (0, 0, self.ww, self.hh)
        sx, sy = self.tw / (x1 - x0), self.th / (y1 - y0)
        self.hole = ((cx - x0) * sx, (cy - y0) * sy, rh * self.hh / 2 * sy)
        return idx

    def _stars(self, idx, st, c, i):
        """Espacio profundo: estrellas lejanas, casi quietas, que rodean el agujero muy despacio.
        Se ven más en el vacío (tensión, intro) y casi nada cuando el track está lleno.
        Solo encienden píxeles negros: siempre quedan detrás de todo."""
        if not hasattr(self, "_st_r"):
            g = np.random.default_rng([self.rng_seed, 1717])
            n = 900
            self._st_r = (2.1 * np.sqrt(g.random(n))).astype(np.float32)
            self._st_a = (g.random(n) * 2 * np.pi).astype(np.float32)
            self._st_big = g.random(n) < 0.12
            self._st_h = g.random(n).astype(np.float32)
        void = 1.0 if c["closed"] else 0.25
        vis = self._hash(np.arange(900), np.zeros(900, np.int64) + 91, i // 9) < (0.25 + 0.6 * void)
        vis &= self._st_h < 0.75 + 0.25 * void
        t = i / self.fps
        Re = float(st["rh"]) * 1.12
        r = self._st_r
        El = float(np.sqrt(Re * Re - 0.06 * Re * Re))
        rl = (r + np.sqrt(r * r + 4 * El * El)) / 2          # lente puntual: la imagen se aparta del horizonte
        a = self._st_a + 0.012 * t + 0.08 * st["tw"] * (0.3 / (r + 0.05)) + self.roll
        hh2 = self.hh / 2
        xs = (self.ww / 2 + rl * np.cos(a) * hh2).astype(np.int32)
        ys = (self.hh / 2 + rl * np.sin(a) * hh2).astype(np.int32)
        px = max(1, int(round(self.gs)))                    # 1 píxel de salida
        for x, y_, big in zip(xs[vis], ys[vis], self._st_big[vis]):
            sz = px * (2 if big else 1)
            if 0 <= x < self.ww - sz and 0 <= y_ < self.hh - sz:
                sub = idx[y_:y_ + sz, x:x + sz]
                sub[sub == 0] = 1
        # destellos: con cada hat algunas estrellas se encienden en cruz (más en el vacío)
        if c["hat"] > 0.3:
            gl = vis & (self._hash(np.arange(900), np.zeros(900, np.int64) + 93, c["hat_n"]) < 0.012 + 0.03 * void)
            arm = int((4 + 14 * c["hat"]) * self.gs)
            for x, y_ in zip(xs[gl], ys[gl]):
                if arm < x < self.ww - arm and arm < y_ < self.hh - arm:
                    for sub in (idx[y_:y_ + px, x - arm:x + arm + px], idx[y_ - arm:y_ + arm + px, x:x + px]):
                        sub[sub == 0] = 1
        return idx

    def _lens(self, st):
        """Lente gravitacional: cerca del horizonte se ve lo que está 'detrás' del agujero,
        apretado en un anillo muy denso alrededor del borde (lente puntual: fuente = r - El²/r,
        con El elegido para que justo en el borde la fuente sea casi el centro: todo lo de atrás
        cabe en ese anillo), y transición suave a espacio plano hacia 6 horizontes. También arma
        la torsión: la espiral general más un remolino que se aprieta pegado al borde.
        Devuelve log(r) curvado; se calcula solo en la zona central."""
        Re = float(st["rh"]) * 1.12
        s0 = 0.06 * Re                             # lo que se ve pegado al borde viene de casi el centro
        El = float(np.sqrt(Re * Re - s0 * Re))
        L = self.logRR.copy()
        Tw = st["tw"] * self.SWIRL
        half = int(6 * Re * self.hh / 2) + 2
        cy, cx = self.hh // 2, self.ww // 2
        sl = (slice(max(0, cy - half), cy + half), slice(max(0, cx - half), cx + half))
        R = self.RR[sl]
        src = R - El * El / np.maximum(R, 1e-4)
        w = np.clip((6 * Re - R) / (4 * Re), 0, 1)
        w = w * w * (3 - 2 * w)
        r_eff = np.maximum(R + w * (src - R), s0)
        L[sl] = np.log(r_eff + 1e-4)
        Tw[sl] += (1.0 + 1.2 * st["tw"]) * w * (Re / np.maximum(R, Re)) ** 2
        self.lensL, self.lensT = L, Tw
        self.lens_box = (sl, (r_eff / np.maximum(R, 1e-4)).astype(np.float32))
        return L

    def _silence(self, idx, c):
        if c["silent"]:
            idx[:] = 0
            h = idx.shape[0] // 2
            idx[h:h + self.tg, :] = 1
        return idx

    @staticmethod
    def _hash(a, b, t):
        h = (a.astype(np.uint32) * np.uint32(73856093)) ^ (b.astype(np.uint32) * np.uint32(19349663)) \
            ^ np.uint32((int(t) * 83492791) & 0xFFFFFFFF)
        h ^= h >> np.uint32(13)
        h *= np.uint32(0x5bd1e995)
        h ^= h >> np.uint32(15)
        return (h & np.uint32(0xFFFFFF)).astype(np.float32) / float(0xFFFFFF)

    @staticmethod
    def _box(idx, y0, y1, x0, x1, col):
        """Rectángulo recortado a la pantalla (sin índices negativos que 'dan la vuelta')."""
        h, w = idx.shape
        y0, y1 = max(0, int(y0)), min(h, int(y1))
        x0, x1 = max(0, int(x0)), min(w, int(x1))
        if y1 > y0 and x1 > x0:
            idx[y0:y1, x0:x1] = col

    def _line(self, idx, x0, y0, x1, y1, th, col):
        """Línea fina entre dos puntos de la grilla (recortada a pantalla)."""
        n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
        xs = np.linspace(x0, x1, n).astype(np.int32)
        ys = np.linspace(y0, y1, n).astype(np.int32)
        hh, ww = idx.shape
        for oy in range(th):
            for ox in range(th):
                xx, yy = xs + ox, ys + oy
                m = (xx >= 0) & (xx < ww) & (yy >= 0) & (yy < hh)
                idx[yy[m], xx[m]] = col

    def _details(self, idx, A, i):
        """Un fenómeno del espacio por cada capa que entró al track, reaccionando a SU actividad
        (no al kick). La capa más reciente va en color de acento; las viejas, solo en sus golpes
        fuertes. Posiciones y trayectorias salen de la semilla del audio (reproducibles)."""
        Ly = A.get("layers")
        if not Ly or not Ly["order"]:
            return idx
        hh, ww = idx.shape
        active = [k for k in Ly["order"] if Ly["on"][i, k]]
        if not active:
            return idx
        newest = max(active, key=lambda k: Ly["first"][k])
        g0 = np.random.default_rng([self.rng_seed, 404])
        kinds = list(g0.permutation(["cometa", "pulsar", "luna", "estrella", "meteoros", "jets"]))
        R0 = hh / 2                                  # unidad: media altura
        cx, cy = ww / 2, hh / 2
        fine = max(1, int(round(self.gs)))            # 1 píxel de salida
        bold = 2 * fine
        t = i / self.fps
        rh = float(self.ss["rh"]) * 1.12 if hasattr(self, "ss") else 0.1
        P = lambda r, ang: (cx + r * np.cos(ang) * R0, cy + r * np.sin(ang) * R0)
        # sin repetir tipo: el orden de entrada elige el fenómeno
        for rank, k in enumerate(Ly["order"]):
            if k not in active:
                continue
            a = float(Ly["act"][i, k])
            since = (i - Ly["first"][k]) / self.fps
            if since < 1.0:
                a = max(a, 1.0)                    # presentación: entra fuerte
            thr = 0.45 if k == newest else 0.9      # las capas viejas solo en sus golpes fuertes
            if a < thr:
                continue
            a = min(a, 1.3)
            col = 2 if k == newest else 1
            kind = kinds[rank % len(kinds)]
            gk = np.random.default_rng([self.rng_seed, 606, k])      # constantes de esta capa
            if kind == "cometa":                   # cruza la pantalla; cola que se alarga con la capa
                per = 7.0 + 4.0 * gk.random()
                ev, ph = int(t // per), (t % per) / per
                ge = np.random.default_rng([self.rng_seed, 607, k, ev])
                th0 = ge.uniform(0, 2 * np.pi)
                th1 = th0 + np.pi + ge.uniform(-0.7, 0.7)
                (x0, y0), (x1, y1) = P(2.0, th0), P(2.0, th1)
                hx, hy = x0 + (x1 - x0) * ph, y0 + (y1 - y0) * ph
                L = np.hypot(x1 - x0, y1 - y0) + 1e-6
                tl = (0.15 + 0.35 * a) * R0
                self._line(idx, hx, hy, hx - (x1 - x0) / L * tl, hy - (y1 - y0) / L * tl, fine, col)
                self._box(idx, hy - bold, hy + bold, hx - bold, hx + bold, col)
            elif kind == "pulsar":                 # punto lejano que titila con dos haces que giran
                px_, py_ = P(gk.uniform(0.7, 1.4), gk.uniform(0, 2 * np.pi))
                sz = int(fine * (1 + 2 * a))
                self._box(idx, py_ - sz, py_ + sz, px_ - sz, px_ + sz, col)
                ang = t * 2 * np.pi * (0.35 + 0.3 * gk.random())
                bl = (0.06 + 0.22 * a) * R0
                for sgn in (1, -1):
                    self._line(idx, px_, py_, px_ + sgn * np.cos(ang) * bl, py_ + sgn * np.sin(ang) * bl, fine, col)
            elif kind == "luna":                   # cuerpo chico en órbita; se esconde detrás del agujero
                Ro = gk.uniform(0.35, 0.65)
                ang = t * 0.9 * (0.25 / Ro) ** 1.5 + gk.uniform(0, 2 * np.pi)
                mx, my = cx + Ro * np.cos(ang) * R0, cy + Ro * np.sin(ang) * 0.35 * R0
                behind = np.sin(ang) < 0 and abs(mx - cx) < rh * R0 * 1.3
                if not behind:
                    sz = int(fine * (2 + 3 * a))
                    self._box(idx, my - sz, my + sz, mx - sz, mx + sz, col)
            elif kind == "estrella":               # estrella fija que se enciende en cruz
                sx_, sy_ = P(gk.uniform(0.5, 1.5), gk.uniform(0, 2 * np.pi))
                arm = (0.02 + 0.12 * a) * R0
                self._line(idx, sx_ - arm, sy_, sx_ + arm, sy_, fine, col)
                self._line(idx, sx_, sy_ - arm, sx_, sy_ + arm, fine, col)
                self._box(idx, sy_ - fine, sy_ + 2 * fine, sx_ - fine, sx_ + 2 * fine, col)
            elif kind == "meteoros":               # estelas cortas que caen hacia el agujero
                gm = np.random.default_rng([self.rng_seed, 608, k, int(t // 0.5)])
                for _ in range(1 + int(3 * a)):
                    r0, ang = gm.uniform(0.8, 1.9), gm.uniform(0, 2 * np.pi)
                    ln = gm.uniform(0.05, 0.16) * (0.6 + 0.6 * a)
                    (x0, y0), (x1, y1) = P(r0, ang), P(max(r0 - ln, 0.05), ang + 0.04)
                    self._line(idx, x0, y0, x1, y1, fine, col)
            elif kind == "jets":                   # chorros polares del agujero: se alargan con la capa
                jl = (0.12 + 0.55 * a) * R0
                e0 = rh * R0 * 1.05
                for sgn in (1, -1):
                    self._line(idx, cx, cy + sgn * e0, cx, cy + sgn * (e0 + jl), fine, col)
        return idx

    def _erode(self, ids, c, salt=0):
        """Tensión sin marcos: cada elemento (anillo, rayo, barra) tiene un umbral
        fijo; a medida que sube la tensión se van apagando hasta quedar pocos."""
        if not c["closed"] or c["silent"]:
            return True
        nerv = self._nerv(c)
        # en cada hat vuelve un instante parte de lo erosionado (más cuanto más cargada la tensión)
        e = 0.88 * c["tens"] ** 1.2 * (1 - 0.65 * nerv * c["hat"])
        keep = self._hash(ids + 7919, np.zeros_like(ids) + 3 + salt, 11) >= e
        if nerv > 0:
            keep &= self._hash(ids + 104729, np.zeros_like(ids) + 5 + salt, c["hat_n"]) \
                >= 0.5 * nerv * (1 - c["hat"])
        return keep

    @staticmethod
    def _nerv(c):
        """Nervio de la tensión: 0 hasta la mitad del tramo filtrado, 1 al final."""
        if not c["closed"] or c["silent"]:
            return 0.0
        return float(np.clip((c["tens"] - 0.35) / 0.65, 0, 1))

    def _thin(self, c):
        # y los que quedan se van afinando
        return 1.0 - 0.7 * c["tens"] if c["closed"] else 1.0

    def _radial(self, d, st, c, kc, K, thin=1.0, logd=None):
        u = K * (np.log(d + 1e-4) if logd is None else logd) - st["ph"]
        n = np.floor(u)
        f = u - n
        ni = n.astype(np.int32)
        w = st["wt"][ni % 32] * thin * self.P.get("dens", 0.7)
        if c["closed"]:
            on = (f < np.clip((0.06 + 0.10 * c["mid"]) * w * self._thin(c), 0.012, 0.6)) \
                 & (ni % 2 == 0) & self._erode(ni, c)
        else:
            on = f < np.clip((0.14 + 0.32 * c["kick"]) * w, 0.03, 0.92)
        red = on & ((ni + kc) % max(2, int(round(7 - 4 * c["chaos"]))) == 0)
        return on, red, ni

    def st_rings(self, st, c, kc):
        K = self.P.get("K", 4.6)
        on, red, ni = self._radial(self.RR, st, c, kc, K, self.P.get("thin", 1.0), self.logRR)
        if c["chapter"] >= 1 or c["chaos"] > 0.8:   # anillos partidos en arcos que giran
            nseg = 6 + 2 * (st["snap"] % 4)
            step = np.deg2rad(1.5)
            rot = round(st["rot"] / step) * step
            seg = np.floor((self.TH + (np.pi + self.roll) + rot * (1 - 2 * (ni % 2))) / (2 * np.pi / nseg)).astype(np.int32)
            keep = ((seg + ni) % 2 == 0)
            on &= keep
            red &= keep
        return on.astype(np.uint8) + red.astype(np.uint8)

    def st_arcs(self, st, c, kc):
        """Anillos siempre partidos en arcos irregulares; cada anillo gira en sentido
        contrario al vecino, a su propia velocidad, y engrana medio segmento en cada kick.
        Pocos, gruesos, mucho negro; hueco en el centro (nada de diana)."""
        K = self.P.get("K", 3.2)
        u = K * self._lens(st) - st["ph"]
        n = np.floor(u)
        f = u - n
        ni = n.astype(np.int32)
        # tabla por anillo (ids de anillo en un rango chico): sentido, velocidad, segmentos
        ids = np.arange(-64, 64, dtype=np.int32)
        hsp = self._hash(ids + 300, np.zeros_like(ids) + 1, 0)
        hsg = self._hash(ids + 600, np.zeros_like(ids) + 2, st["mut"])
        nseg = np.floor(5 + (5 + 8 * c["chaos"]) * hsg).astype(np.int32)       # 5..18 segmentos
        sgn = np.where(ids % 2 == 0, 1.0, -1.0)
        ang = sgn * (st["rot"] * (0.6 + 1.2 * hsp) * 2.2) + kc * np.pi / nseg * sgn
        ang = (ang + self.roll).astype(np.float32)
        k = (ni + 64) % 128
        a = np.mod((self.TH + ang[k] + self.lensT) * (1 / (2 * np.pi)), 1.0) * nseg[k]
        sg = np.floor(a).astype(np.int32)
        fa = a - sg
        hseg = self._hash(ni + 900, sg, st["mut"])
        keep = hseg < (0.55 + 0.15 * c["chaos"])
        keep &= (fa > 0.04) & (fa < 0.96)                                      # corte seco entre arcos
        w = st["wt"][ni % 32] * self.P.get("dens", 0.7)
        if c["closed"]:
            on = (f < np.clip((0.08 + 0.10 * c["mid"]) * w * self._thin(c), 0.015, 0.5)) & self._erode(ni, c)
        else:
            duty = np.clip((0.22 + 0.30 * c["kick"]) * w, 0.05, 0.85)
            duty = np.minimum(duty, 0.30 * K / (self.RR + 0.3))                # los de afuera no se engordan de más
            on = f < duty
        on &= keep
        acc = on & (self._hash(ni + 1200, sg, kc // 8) < (0.14 + 0.22 * c["chaos"]))
        return on.astype(np.uint8) + acc.astype(np.uint8)

    def _orbits(self, Lr, ang_src, st, c, kc, K, lw_px, wl_scale=None):
        """Materia en órbita: carriles en log(r) (en la tensión se contraen hacia el agujero) con
        estelas que giran a velocidad orbital (las de adentro, más rápido; el kick las estira).
        Nada homogéneo: anillos densos y huecos, la mayoría tenues y pocas brillantes, más brillo
        cerca del agujero, grosores distintos, ondas espirales que deforman las órbitas en conjunto,
        y cada estela con la cabeza brillante y la cola que se apaga.
        Devuelve (intensidad 0..1, acento, id de carril)."""
        tens_c = c["tens"] if c["closed"] else 0.0
        scale = 1 - 0.35 * tens_c
        u = K * (Lr - np.float32(np.log(scale)))
        # ondas de densidad espirales (colectivas: todas las órbitas se deforman juntas)
        g = np.random.default_rng([self.rng_seed, 2024])
        f1, f2, f3 = g.uniform(0, 2 * np.pi, 3)
        u = u + 0.35 * np.sin(2 * ang_src + f1 + 0.15 * st["rot"]) + 0.2 * np.sin(3 * ang_src - f2 - 0.1 * st["rot"])
        li = np.floor(u).astype(np.int32)
        fr = u - li
        ids = np.arange(-96, 96, dtype=np.int32)
        z = np.zeros_like(ids)
        r_l = np.exp((ids + 0.5) / K) * scale
        om = (0.25 / np.maximum(r_l, 0.04)) ** 1.2
        spd = 0.6 + 0.8 * self._hash(ids + 40, z + 4, 0)
        lane_ang = (st["rot"] * 2.0 * om * spd).astype(np.float32)
        nseg = np.floor(1 + 6 * self._hash(ids + 70, z + 6, st["mut"])).astype(np.int32)
        dens = np.clip(0.5 + 0.5 * np.sin(ids * 0.37 + f3) * np.sin(ids * 0.11 + f1), 0, 1)   # anillos y huecos
        bri = (0.25 + 0.9 * self._hash(ids + 2000, z + 7, 0) ** 1.6) * (0.55 + 0.6 * dens) \
            * np.clip(0.3 / np.maximum(r_l, 0.02), 0.35, 2.2)                                   # tenues; cerca del agujero, mucho brillo
        thick = 0.45 + 1.1 * self._hash(ids + 2200, z + 8, 0)
        k = np.clip(li + 96, 0, 191)
        a = np.mod((ang_src + lane_ang[k]) * (1 / (2 * np.pi)), 1.0) * nseg[k]
        si = np.floor(a).astype(np.int32)
        fa = a - si
        hl = self._hash(li + 300, si, st["mut"])
        ln = np.minimum((0.08 + 0.8 * hl ** 1.5) * (1 + 1.4 * c["kick"]), 0.97)
        seg_on = (fa < ln) & (self._hash(li + 500, si, st["mut"] // 2)
                              < (0.3 + 0.5 * dens[k]) * (1 + 0.3 * c["chaos"] + 0.8 * st["glow"]))
        wl = lw_px / (self.hh / 2) * K / (self.RR + 1e-3) if wl_scale is None else wl_scale
        wl = wl * thick[k]
        if c["closed"]:
            wl = wl * max(0.45, self._thin(c))
        on = (np.abs(fr - 0.5) < wl) & seg_on
        if c["closed"]:
            on &= self._erode(li, c, 4)
        along = (1 - fa / np.maximum(ln, 1e-3)) ** 1.6                                           # cabeza -> cola
        I = np.where(on, bri[k] * along * (0.65 + 0.6 * c["kick"]) * (1 + 1.6 * st["glow"]), 0).astype(np.float32)
        if c["closed"]:
            I *= 0.7 + 0.5 * self._nerv(c) * c["hat"]
        acc = on & (self._hash(li + 800, np.zeros_like(li) + 8, kc // 16) < (0.12 + 0.2 * c["chaos"]))
        return I, acc, li

    def st_streaks(self, st, c, kc):
        """Estelas: materia en órbita alrededor del agujero, curvada por la lente."""
        K = self.P.get("K", 14.0)
        L = self._lens(st)
        I, acc, _ = self._orbits(L, self.TH + self.lensT + self.roll, st, c, kc, K, 0.55 * self.tg)
        return self._levels(I, acc)

    def st_disk(self, st, c, kc):
        """Disco de acreción visto casi de canto: una banda de estelas que cruza el agujero; la
        lente dobla la parte de atrás por arriba y por abajo del horizonte (la imagen de
        Gargantua). El lado que se acerca, blanco y más brillante; el que se aleja, en acento."""
        K = self.P.get("K", 16.0)
        si_ = self.P.get("incl", 0.16)
        self._lens(st)
        sl, ratio = self.lens_box
        if not hasattr(self, "_RHO"):
            yd = self.yy / si_
            self._RHO = np.hypot(self.xx, yd).astype(np.float32)
            self._PHI = np.arctan2(yd, self.xx).astype(np.float32)
        rho, phi = self._RHO.copy(), self._PHI.copy()
        xs, ys = self.xx[sl] * ratio, self.yy[sl] * ratio / si_
        rho[sl] = np.hypot(xs, ys)
        phi[sl] = np.arctan2(ys, xs)
        Re = float(st["rh"]) * 1.12
        I, acc, li = self._orbits(np.log(rho + 1e-4), phi + self.roll, st, c, kc, K, 0.5 * self.tg,
                                  wl_scale=np.float32(0.10))
        I *= (rho > 2.0 * Re) & (rho < 2.2)
        I *= 1.9 * np.clip(1 - 0.45 * self.xx, 0.55, 1.5)             # doppler: brilla más el lado que se acerca
        # lado que se aleja (derecha) en acento; cada órbita cambia de color en un punto distinto
        doppler = self._hash(li + 1500, np.zeros_like(li) + 9, 0) < np.clip(0.5 + 0.9 * self.xx, 0, 1)
        return self._levels(I, (I > 0) & (doppler | acc))

    def st_pdots(self, st, c, kc, frame):
        """Partículas sobre el túnel: una celda por (anillo, sector); viajan hacia afuera con el
        avance del túnel y crecen con la distancia. Cada celda recuerda si está encendida
        (se renueva cada 8 kicks) para que el viaje se perciba; en la tensión titila con los hats."""
        N = int(round(self.P.get("N", 48) * (1 + 0.5 * c["chaos"])))
        Kd = self.P.get("Kd", 7.5)
        u = Kd * self._lens(st) - st["ph"] * (Kd / 5.0)
        ri = np.floor(u).astype(np.int32)
        fr = u - ri
        a = np.mod((self.TH + 0.35 * st["rot"] + self.roll + self.lensT)
                   * (1 / (2 * np.pi)), 1.0) * N
        ai = np.floor(a).astype(np.int32)
        fa = a - ai
        nerv = self._nerv(c)
        tkey = c["hat_n"] if nerv > 0 else kc // 8
        r1 = self._hash(ri + 5000, ai + 5000, tkey)
        r2 = self._hash(ri + 9000, ai + 1000, kc // 8 + 7)
        p = self.P.get("p", 0.2)
        energy = 0.5 * c["low"] + 0.5 * c["mid"]
        if c["closed"]:
            pr = p * self.P.get("dens", 0.7) * 0.35 * ((1 - 0.9 * c["tens"]) * (1 - 0.5 * nerv * (1 - c["hat"]))
                                                   + 1.6 * nerv * c["hat"])      # fogonazo en cada hat
        else:
            pr = p * self.P.get("dens", 0.7) * (0.4 + 0.9 * energy + 0.8 * c["kick"])
        # onda que sale del centro en cada kick
        wave = np.abs(self.RR - (0.15 + 1.6 * (1 - c["kick"]))) < 0.09
        lit = (r1 < pr) | (wave & (c["kick"] > 0.2) & (r1 < 0.6) & (not c["closed"]))
        flick = self.P.get("flicker", 0)
        if flick:
            lit &= self._hash(ri + 77, ai + 33, frame // flick) < 0.7
        th_ = self._thin(c)
        th_ += (1 - th_) * nerv * c["hat"]                # en el fogonazo vuelven a tamaño pleno
        sz = (self.P.get("dot", 0.30) + 0.25 * c["kick"]) * th_ + 0.08
        half = sz / 2
        inside = (np.abs(fr - 0.5) < half) & (np.abs(fa - 0.5) < half)
        if self.P.get("plus"):
            arm = min(0.5, half + 0.12 + 0.08 * c["kick"])
            thin = 0.05
            inside |= (((np.abs(fr - 0.5) < arm) & (np.abs(fa - 0.5) < thin)) |
                       ((np.abs(fa - 0.5) < arm) & (np.abs(fr - 0.5) < thin))) & (r1 < pr * 0.5)
        on = lit & inside
        acc = on & (r2 < (0.08 + 0.2 * c["chaos"]))
        return on.astype(np.uint8) + acc.astype(np.uint8)

    def st_geo(self, st, c, kc):
        sides = self.P.get("sides", [4, 3, 6])
        nsd = sides[c["chapter"] % len(sides)]
        sector = 2 * np.pi / nsd
        step = np.deg2rad(1.0)
        a0 = round(st["rot"] / step) * step + st["snap"] * sector / 2 + self.roll
        self._lens(st)                             # los polígonos también se curvan y se tuercen cerca del horizonte
        a = np.mod(self.TH + a0 + self.lensT, sector) - sector / 2
        d = self.RR * np.cos(a) / np.cos(sector / 2)
        sl, ratio = self.lens_box
        d[sl] *= ratio
        on, red, ni = self._radial(d, st, c, kc, self.P.get("K", 4.0))
        return on.astype(np.uint8) + red.astype(np.uint8)

    def st_rays(self, st, c, kc):
        nr = int(2 * round((10 + 30 * c["chaos"]) / 2))
        step = np.deg2rad(0.75)
        rot = round(st["rot"] * 0.6 / step) * step + st["snap"] * np.pi / nr + self.roll
        frac = np.mod((self.TH + rot) * nr / (2 * np.pi), 1.0)
        ray_i = np.floor((self.TH + rot) * nr / (2 * np.pi)).astype(np.int32)
        w = st["wt"][ray_i % 32]
        duty = np.clip((0.05 + (0.06 if c["closed"] else 0.30) * c["kick"] + 0.04) * w
                       * self.P.get("dens", 0.7), 0.015, 0.7)
        on = (frac < duty * self._thin(c)) & self._erode(ray_i % 997, c, 1)
        inner = 0.05 + 0.18 * c["low"]
        on &= self.RR > inner
        if c["chapter"] >= 1:                        # rayos cortados en tramos que viajan hacia afuera
            u = 3.0 * np.log(self.RR + 1e-4) - st["ph"]
            on &= np.mod(u, 1.0) < 0.55
        red = on & ((ray_i + kc) % max(2, int(round(6 - 3 * c["chaos"]))) == 0)
        idx = on.astype(np.uint8) + red.astype(np.uint8)
        core = self.RR < 0.02 + 0.03 * c["kick"]
        idx[core] = 2
        return idx

    def st_dots(self, st, c, kc, frame):
        base = self.P.get("spacing", 16)
        sp = max(4, int(round(base * (1.25 - 0.5 * c["chaos"]))))   # más caos -> más juntos
        PX, PY = self.PX, self.PY
        if abs(self.roll) > 1e-5:                  # giro de cámara: la retícula gira entera
            cr_, sr_ = np.float32(np.cos(self.roll)), np.float32(np.sin(self.roll))
            PX, PY = PX * cr_ - PY * sr_, PX * sr_ + PY * cr_
        cx = np.floor_divide(PX, sp)
        cy = np.floor_divide(PY, sp)
        lx = PX - cx * sp
        ly = PY - cy * sp
        flick = self.P.get("flicker", 0)
        tkey = (frame // flick) if flick else kc
        r1 = self._hash(cx + 5000, cy + 5000, tkey)
        r2 = self._hash(cx + 9000, cy + 1000, tkey + 7)
        p = self.P.get("p", 0.2)
        energy = 0.5 * c["low"] + 0.5 * c["mid"]
        pr = p * self.P.get("dens", 0.7) * ((0.35 * (1 - 0.9 * c["tens"])) if c["closed"]
                                              else (0.4 + 0.9 * energy + 0.8 * c["kick"]))
        # onda que sale del centro en cada kick: coherencia sobre el azar
        cr = np.hypot(cx * sp + sp / 2, cy * sp + sp / 2) / (self.hh / self.u / 2)
        wave = np.abs(cr - (0.15 + 1.6 * (1 - c["kick"]))) < 0.09
        lit = (r1 < pr) | (wave & (c["kick"] > 0.2) & (r1 < 0.6) & (not c["closed"]))
        size = max(1 if c["closed"] else 2,
                   int(round(sp * (self.P.get("dot", 0.30) + 0.25 * c["kick"]) * self._thin(c))))
        o = (sp - size) // 2
        inside = (lx >= o) & (lx < o + size) & (ly >= o) & (ly < o + size)
        if self.P.get("plus"):
            arm = size + 2 + int(3 * c["kick"])
            mcx, mcy = lx - sp // 2, ly - sp // 2
            inside = inside | (((np.abs(mcx) <= arm) & (np.abs(mcy) < 1)) |
                               ((np.abs(mcy) <= arm) & (np.abs(mcx) < 1))) & (r1 < pr * 0.5)
        on = lit & inside
        red = on & (r2 < (0.08 + 0.2 * c["chaos"]))
        return on.astype(np.uint8) + red.astype(np.uint8)

    def _journey(self, A, i):
        """Elige el estilo según la historia del track: capítulo -> mundo,
        caos -> escalón dentro del mundo. Solo cambia en liberaciones o capítulos."""
        if not hasattr(self, "_jr"):
            self._jr = dict(base=dict(self.P), cur=None, ch=-1, n_rise=0, k=0, ch_i=0)
        J = self._jr
        worlds = J["base"]["worlds"]
        n_ch = int(A["chapter"].max()) + 1
        ch = int(A["chapter"][i])
        if n_ch > 1:
            world = worlds[min(ch, len(worlds) - 1)] if ch < n_ch - 1 or n_ch <= len(worlds) \
                else worlds[-1]
        else:
            world = worlds[0] + worlds[1]
        chaos = float(A["chaos"][i])
        change = J["cur"] is None or ch != J["ch"] or bool(A["rise"][i])
        if change:
            if ch != J["ch"]:
                J["n_rise"] = 0
                J["ch_i"] = i
                J["k"] = 0
            elif A["rise"][i]:
                if i - J["ch_i"] < 5 * self.fps:      # el drop pegado al inicio del capítulo no cuenta
                    return
                J["n_rise"] += 1
            if world is worlds[0] or n_ch == 1:
                # sube de a un escalón por liberación, solo si el track se abrió lo suficiente;
                # sin capítulos, la escalera es completa (partículas -> geometría)
                if n_ch == 1:
                    target = min(len(world) - 1, int(chaos * len(world)))
                else:
                    target = 0 if chaos < 0.35 else (1 if chaos < 0.75 else 2)
                stale = (i - J.get("t_change", 0)) > 60 * self.fps
                if A["rise"][i] and (target > J["k"] or stale):
                    J["k"] += 1
                k = min(J["k"], len(world) - 1)
            else:
                k = J["n_rise"] % len(world)          # el mundo geométrico evoluciona por drop
            name = world[k]
            if name != J["cur"]:
                J["t_change"] = i
                p = dict(PRESETS[name])
                if "sides" in p:                   # en el viaje, cada estilo mantiene su forma
                    p["sides"] = [p["sides"][0]]
                self.P = p
                J["cur"] = name
            J["ch"] = ch

    def styled(self, A, i, kc, render=True):
        if not hasattr(self, "_is_journey"):
            self._is_journey = self.P.get("style") == "journey"
        style = "journey" if self._is_journey else self.P.get("style", "rings")
        if render and self.cam_on and self.sup > 1:
            self._camera(A, i)
        if style == "bars":
            out = self.flow(A, i, kc)
            return out if render else None
        if style == "journey":
            self._journey(A, i)
            style = self.P["style"]
        st, c = self._step(A, i, kc)
        if not render:
            return None
        if style == "rings":
            idx = self.st_rings(st, c, kc)
        elif style == "arcs":
            idx = self.st_arcs(st, c, kc)
        elif style == "streaks":
            idx = self.st_streaks(st, c, kc)
        elif style == "disk":
            idx = self.st_disk(st, c, kc)
        elif style == "dots" and self.P.get("polar"):
            idx = self.st_pdots(st, c, kc, i)
        elif style == "geo":
            idx = self.st_geo(st, c, kc)
        elif style == "rays":
            idx = self.st_rays(st, c, kc)
        else:
            idx = self.st_dots(st, c, kc, i)
        self.hole = None
        idx = self._stars(idx, st, c, i)
        idx = self._silence(idx, c)
        if not c["silent"]:
            idx = self._details(idx, A, i)
            idx = self._horizon(idx, st)           # el agujero se traga también los detalles
        return self._up(self.pal[idx])

    def tunnel(self, low, mid, high, kick, in_break, kick_count):
        fps = self.fps
        # velocidad de avance (anillos por segundo)
        energy = 0.6 * low + 0.4 * mid
        speed = 0.35 + 0.15 * mid if in_break else 0.8 + 2.2 * energy
        self.phase += speed / fps
        # cada kick: salto duro hacia adelante
        new_kicks = kick_count - self.last_kc
        if new_kicks > 0:
            self.phase += 0.22 * new_kicks
            # cada 32 kicks (8 compases) el túnel gira 45°: cuadrado <-> rombo
            if kick_count // 32 != self.last_kc // 32:
                self.snap += np.pi / 4
        self.last_kc = kick_count
        # rotación lenta y escalonada (pasos de 0.75°)
        self.angle += (0.10 + 0.45 * mid) / fps
        step = np.deg2rad(0.75)
        a = round(self.angle / step) * step + self.snap

        c, sn = np.float32(np.cos(a)), np.float32(np.sin(a))
        xr = self.xx * c + self.yy * sn
        yr = self.yy * c - self.xx * sn
        d = np.maximum(np.abs(xr), np.abs(yr))
        K = 4.0  # densidad de anillos
        u = K * np.log(d + 1e-4) - self.phase
        n = np.floor(u)
        f = u - n
        ni = n.astype(np.int32)

        if in_break:
            duty = 0.07 + 0.10 * mid
            on = (f < duty) & (ni % 3 == 0)
        else:
            duty = 0.14 + 0.46 * kick
            on = f < duty
        red = on & ((ni + kick_count) % 6 == 0)

        idx = on.astype(np.uint8) + red.astype(np.uint8)
        # vacío central que late con los graves + núcleo rojo
        void = 0.07 + 0.13 * low
        idx[d < void] = 0
        core = 0.02 + 0.035 * high + 0.03 * kick
        idx[d < core] = 2
        return self._up(self.pal[idx])

    def draw_video(self, f, idx_half, kick):
        """Pega el video (a media resolución, píxeles duros) con marco que late."""
        img = self.pal[idx_half]
        img = np.repeat(np.repeat(img, 2, axis=0), 2, axis=1)
        x0, y0, w, h = self.vx, self.vy, self.vw, self.vh
        b = self.t * 2 + (self.t * 2 if kick > 0.6 else 0)
        rect(f, x0 - b - self.t * 3, self.ty0, x0 + w + b + self.t * 3,
             self.ty0 + self.th, BLACK)
        rect(f, x0 - b, y0 - b, x0 + w + b, y0 + h + b, self.accent if kick > 0.6 else WHITE)
        f[y0:y0 + h, x0:x0 + w] = img[:h, :w]

    def draw(self, f, A, i, kick_count, video_idx=None):
        W, H, s, t, m = self.W, self.H, self.s, self.t, self.m
        f[:] = self.static

        low, mid, high = A["low"][i], A["mid"][i], A["high"][i]
        kick = A["kick"][i]
        in_break = bool(A["break_zone"][i])
        if self.accent_cycle is not None:
            m = int(A["rel_kick"][i]) if "rel_kick" in A else -1
            self.set_accent(self.accent_cycle[m % len(self.accent_cycle)] if m >= 0 else self.accent_cycle[0])
        elif self.accent_fixed is not None:
            self.set_accent(self.accent_fixed)
        elif not self.jump_every:
            self.set_accent(ACCENTS[int(A["accent"][i])])

        x_off = (W - self.tw) // 2
        gen = self.bars if self.pattern == "bars" else self.tunnel
        if self.pattern == "style":
            layer = self.styled(A, i, kick_count)
        elif self.pattern == "flow":
            layer = self.flow(A, i, kick_count)
        elif self.pattern == "bars":
            layer = self.bars(low, mid, high, kick, in_break, kick_count,
                              A["break_id"][i], A["key"][i])
        else:
            layer = self.tunnel(low, mid, high, kick, in_break, kick_count)
        f[self.ty0:self.ty0 + self.th, x_off:x_off + self.tw] = layer
        if video_idx is not None:
            self.draw_video(f, video_idx, kick)

        if self.clean:
            if self.post is not None:
                f[:] = self.post.apply(f, i, self.accent, self.hole)
            return self._finish_clean(f, A, i, kick)

        # --- espectro (banda inferior)
        bars = A["bars"][i]
        nb = len(bars)
        sx0, sx1 = m, W - m
        sy1 = H - int(62 * s)
        max_h = sy1 - (self.bot + t + int(22 * s))
        gap = max(2, int(5 * s))
        bw_ = (sx1 - sx0 - gap * (nb - 1)) / nb
        q = 8 * s
        for b in range(nb):
            v = bars[b]
            h_ = int(round(v * max_h / q) * q)
            if h_ <= 0:
                continue
            bx = int(sx0 + b * (bw_ + gap))
            rect(f, bx, sy1 - h_, int(bx + bw_), sy1, self.accent if v > 0.95 else WHITE)

        # --- lecturas
        if i % 2 == 0 or not hasattr(self, "_ro"):
            txt = (f"LOW {int(low*100):03d}  MID {int(mid*100):03d}  "
                   f"HIGH {int(high*100):03d}  KICK {kick_count:05d}")
            self._ro = text_mask(txt, self.mono)
        blit(f, self._ro, m, H - int(44 * s), WHITE)
        tc = self.timecode(i)
        blit(f, tc, W - m - tc.shape[1], H - int(44 * s), WHITE)
        if kick > 0.5:
            d = int(16 * s)
            x = W - m - tc.shape[1] - 2 * d
            rect(f, x, H - int(40 * s), x + d, H - int(40 * s) + d, self.accent)

        # --- glitch en kicks fuertes: franjas desplazadas, una en rojo
        if A["strength"][i] >= 1.0 or (kick > 0.8 and A["strength"][max(0, i - 1)] >= 1.0):
            g = np.random.default_rng(self.rng_seed + i)
            for k in range(int(g.integers(2, 5))):
                y = int(g.integers(self.ty0, self.ty0 + self.th - 10))
                hh = int(g.integers(int(6 * s), int(48 * s)))
                shift = int(g.integers(-140, 140) * s)
                band = f[y:y + hh]
                band[:] = np.roll(band, shift, axis=1)
                if k == 0:
                    band[band.all(axis=2)] = self.accent

        return f



def breath_mask(A, fps):
    """Respiración: el último beat antes de cada liberación queda a negro (solo la línea),
    así la liberación entra desde la nada. Anclado a la grilla de kicks del propio track."""
    n = A["n"]
    out = np.zeros(n, bool)
    ons = A["onsets"]
    if len(ons) < 2:
        return out
    beat = int(round(float(np.median(np.diff(ons)))))
    for r in np.where(A["rise"])[0]:
        out[max(0, r - beat):r] = True
    return out


def render_parallel(a, A, i0, i1, jobs, t_start, t_len):
    """Parte el render en `jobs` tramos que se dibujan en paralelo (cada uno repite el
    estado de la animación desde el inicio y precalienta la estela) y los une sin recodificar."""
    import pickle
    import tempfile
    tmp = tempfile.mkdtemp(prefix=".bviz_", dir=os.path.dirname(os.path.abspath(a.output)))
    try:
        cache = os.path.join(tmp, "analysis.pkl")
        with open(cache, "wb") as fh:
            pickle.dump(A, fh)
        cmd = [sys.executable, os.path.abspath(__file__), a.audio, "--analysis", cache,
               "--seed", str(a.seed), "--res", a.res, "--fps", str(a.fps), "--start", str(a.start),
               "--duration", str(a.duration), "--bars", str(a.bars), "--title", a.title,
               "--subtitle", a.subtitle, "--encoder-preset", a.x264, "--crf", str(a.crf),
               "--pattern", a.pattern, "--look", a.look, "--ss", str(a.ss),
               "--no-audio", "--jobs", "1", "--quiet"]
        if a.preset:
            cmd += ["--preset", a.preset]
        if a.clean:
            cmd += ["--clean"]
        if not a.cam:
            cmd += ["--no-cam"]
        cmd += ["--paleta", a.paleta]
        if a.font:
            cmd += ["--font", a.font]
        cuts = np.linspace(i0, i1, jobs + 1).round().astype(int)
        parts, procs = [], []
        for k in range(jobs):
            parts.append(os.path.join(tmp, f"part{k:02d}.mp4"))
            procs.append(subprocess.Popen(cmd + ["-o", parts[-1], "--chunk",
                                                 f"{cuts[k] / a.fps}-{cuts[k + 1] / a.fps}"]))
        print(f"  render en {jobs} procesos...", flush=True)
        t1 = time.time()
        while any(p.poll() is None for p in procs):
            time.sleep(5)
            done = sum(p.poll() is not None for p in procs)
            print(f"\r  tramos listos {done}/{jobs}  ({int(time.time() - t1)}s)", end="", flush=True)
        print()
        if any(p.returncode for p in procs):
            sys.exit("Falló algún tramo del render en paralelo.")
        lst = os.path.join(tmp, "parts.txt")
        with open(lst, "w") as fh:
            fh.writelines(f"file '{p}'\n" for p in parts)
        cat = ["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst]
        if a.no_audio:
            cat += ["-c", "copy", "-movflags", "+faststart", a.output]
        else:
            cat += ["-ss", f"{t_start:.3f}", "-t", f"{t_len:.3f}", "-i", a.audio, "-map", "0:v", "-map", "1:a",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "320k", "-shortest", "-movflags", "+faststart",
                    a.output]
        subprocess.run(cat, check=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser(description="Visuales brutalistas audio-reactivos (negro/blanco/rojo)")
    ap.add_argument("audio", help="archivo de audio (wav, flac, mp3, aiff...)")
    ap.add_argument("-o", "--output", default="set_visual.mp4")
    ap.add_argument("--title", default="MED1NA", help="texto grande arriba a la izquierda")
    ap.add_argument("--subtitle", default="", help="texto chico arriba a la derecha")
    ap.add_argument("--res", default="1920x1080", help="ej: 1920x1080, 1280x720, 2560x1440")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--start", type=float, default=0, help="segundo de inicio (para previews)")
    ap.add_argument("--duration", type=float, default=0, help="duración en segundos (0 = todo)")
    ap.add_argument("--bars", type=int, default=64, help="cantidad de barras del espectro")
    ap.add_argument("--font", default=None, help="ruta a una fuente .ttf/.otf para el título")
    ap.add_argument("--encoder-preset", dest="x264", default="veryfast", help="preset x264 (ultrafast..slow)")
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--seed", type=int, default=None,
                    help="variación visual; por defecto sale del propio audio (estable por track)")
    ap.add_argument("--video", default=None, help="video del celular (vertical) para la ventana central")
    ap.add_argument("--video-offset", type=float, default=None,
                    help="segundo del set en que arranca el video (si no, se sincroniza solo)")
    ap.add_argument("--video-drift", type=float, default=0.0)
    ap.add_argument("--preset", default=None, choices=sorted(PRESETS),
                    help="estilo visual listo para usar (activa el modo limpio)")
    ap.add_argument("--clean", action="store_true", help="sin espectro ni datos: imagen a pantalla completa")
    ap.add_argument("--list-presets", action="store_true")
    ap.add_argument("--pattern", default="bars", choices=["flow", "bars", "tunnel", "style"],
                    help="bars = barras verticales (ideal con video), tunnel = túnel de cuadrados")
    ap.add_argument("--video-jump", type=int, default=0,
                    help="el video salta a otra posición cada N kicks (0 = fijo en el centro)")
    ap.add_argument("--bpm", default="auto",
                    help="rango de tempo del set, ej: 140-142 (ayuda a seguir el pulso)")
    ap.add_argument("--chunk", default=None,
                    help="renderizar solo un tramo 'desde-hasta' en segundos (para partir renders largos); "
                         "el análisis y la animación siguen siendo los del set completo")
    ap.add_argument("--no-audio", action="store_true", help="exportar solo video")
    ap.add_argument("--look", default="luz", choices=sorted(LOOKS),
                    help="acabado reactivo (modo limpio): luz, atmosfera, seco = sin post-proceso")
    ap.add_argument("--ss", type=int, default=2,
                    help="supersampling de los estilos (2 = acabado B; 1 = píxel duro a media resolución)")
    ap.add_argument("--paleta", default="violeta", choices=sorted(PALETTES),
                    help="modo limpio: violeta = negro/blanco/violeta; violeta_rojo = violeta y en las liberaciones "
                         "alterna con rojo por kick; rgb = negro/blanco/rojo con RGB en las liberaciones")
    ap.add_argument("--no-cam", dest="cam", action="store_false",
                    help="sin movimiento de cámara (zoom lento, micro-giro, temblor en el kick)")
    ap.add_argument("--stills", default=None,
                    help="en vez de video, guardar cuadros PNG en estos segundos, ej: 150,167.2,180")
    ap.add_argument("--jobs", type=int, default=0,
                    help="procesos en paralelo para renders largos (0 = automático, 1 = uno solo)")
    ap.add_argument("--analysis", default=None, help=argparse.SUPPRESS)   # uso interno (--jobs)
    ap.add_argument("--quiet", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--show-keys", action="store_true", help="mostrar la key detectada a lo largo del set")
    a = ap.parse_args()
    if a.list_presets:
        for k, v in PRESETS.items():
            print(f"  {k:14s} {v}")
        return
    if a.preset:
        a.pattern = "style"
        a.clean = True

    if not shutil.which("ffmpeg"):
        sys.exit("No encuentro ffmpeg. Instalalo (brew install ffmpeg / winget install ffmpeg) y reintentá.")
    W, H = (int(v) for v in a.res.lower().split("x"))

    t0 = time.time()
    if a.analysis:                                  # tramo de un render en paralelo
        import pickle
        with open(a.analysis, "rb") as fh:
            A = pickle.load(fh)
        n = A["n"]
    else:
        print("Analizando audio...", flush=True)
        y = load_audio(a.audio, ANALYSIS_SR, a.start, a.duration)
        if len(y) == 0:
            sys.exit("No pude leer audio de ese archivo.")
        # control de la exportación: un bounce casi mudo suele ser un solo/mute olvidado
        blk = ANALYSIS_SR
        nb = len(y) // blk
        lv = 20 * np.log10(np.sqrt((y[:nb * blk].reshape(nb, blk) ** 2).mean(1)) + 1e-9)
        if nb and (lv > -60).mean() < 0.2:
            sys.exit(f"El audio está casi en silencio ({(lv > -60).sum()} de {nb} s con sonido). "
                     "Revisá la exportación (pistas en solo/mute, rango de exportación).")
        if a.bpm == "auto":
            bpm = detect_bpm(a.audio)
            print(f"  tempo detectado: {bpm:.2f} BPM", flush=True)
            lo_b, hi_b = bpm - 1, bpm + 1
        else:
            lo_b, hi_b = (float(v) for v in a.bpm.split("-"))
        if a.seed is None:
            import hashlib
            a.seed = int(hashlib.sha1(y[::997].tobytes()).hexdigest()[:6], 16) % 100000
        A = analyze(y, a.fps, a.bars, (lo_b - 2, hi_b + 2))
        n = A["n"]
        Ly = A.get("layers")
        if Ly and Ly["order"]:
            print("  capas que entran: " + " · ".join(
                f"{int(Ly['first'][k] / a.fps) // 60}:{int(Ly['first'][k] / a.fps) % 60:02d} "
                f"(~{Ly['center'][k]:.0f} Hz)" for k in Ly["order"]), flush=True)
        print(f"  capítulos: {int(A['chapter'].max()) + 1} · liberaciones: {int(A['rise'].sum())} · "
              f"sub filtrado: {100 * (1 - A['sub_on'].mean()):.0f}% del tiempo", flush=True)
        print(f"  {n/a.fps/60:.1f} min · {len(A['onsets'])} kicks · {A['n_big_drops']} drops fuertes (RGB) "
              f"· {time.time()-t0:.1f}s", flush=True)

    R = Renderer(W, H, a.fps, a.title, a.subtitle, a.font, a.seed, a.start, a.pattern,
                 clean=a.clean, preset=PRESETS.get(a.preset),
                 ss=a.ss if a.pattern == "style" else 1)
    R.cam_on = a.cam
    if a.clean and isinstance(PALETTES[a.paleta], list):
        R.accent_cycle = PALETTES[a.paleta]
    elif a.clean and PALETTES[a.paleta] is not None:
        R.accent_fixed = PALETTES[a.paleta]
    if a.pattern == "style" and "breath" not in A:
        A["breath"] = breath_mask(A, a.fps)
    if a.clean and LOOKS[a.look]:
        R.post = Post(W, H, a.fps, A, a.look, a.seed)
    if a.show_keys:
        prev = None
        for i in range(0, n, a.fps * 15):
            k = int(A["key"][i])
            if k != prev:
                cam = f"{CAMELOT_MIN[k]}A" if k < 12 else f"{CAMELOT_MAJ[k-12]}B"
                name = PITCHES[k % 12] + ("m" if k < 12 else "")
                t = int(a.start + i / a.fps)
                print(f"  {t//60:02d}:{t%60:02d}  {name:4s} ({cam})")
                prev = k

    if a.video:
        R.jump_every = a.video_jump
    i0, i1 = 0, n
    if a.chunk:
        c0, c1 = (float(v) for v in a.chunk.split("-"))
        i0, i1 = int(round(c0 * a.fps)), min(n, int(round(c1 * a.fps)))
    stills = None
    if a.stills:
        stills = sorted({min(n - 1, int(round(float(v) * a.fps))) for v in a.stills.split(",")})
        i0, i1 = stills[0], stills[-1] + 1
    t_start = a.start + i0 / a.fps
    t_len = (i1 - i0) / a.fps

    jobs = a.jobs or max(1, min(8, (os.cpu_count() or 2) - 2))
    if stills is None and not a.video and not a.analysis and jobs > 1 \
            and (i1 - i0) >= jobs * 10 * a.fps:
        render_parallel(a, A, i0, i1, jobs, t_start, t_len)
        print(f"Listo: {a.output}  ({(time.time()-t0)/60:.1f} min)")
        return

    enc = None
    if stills is None:
        cmd = ["ffmpeg", "-y", "-v", "error",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(a.fps), "-i", "-"]
        if a.no_audio:
            cmd += ["-c:v", "libx264", "-preset", a.x264, "-crf", str(a.crf), "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", a.output]
        else:
            cmd += ["-ss", f"{t_start:.3f}", "-t", f"{t_len:.3f}", "-i", a.audio, "-map", "0:v", "-map", "1:a",
                    "-c:v", "libx264", "-preset", a.x264, "-crf", str(a.crf), "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "320k", "-shortest", "-movflags", "+faststart", a.output]
        enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    V = None
    if a.video:
        if a.video_offset is None:
            off, drift = sync_video(a.audio, a.video)
        else:
            off, drift = a.video_offset, a.video_drift
        print(f"  video arranca en el segundo {off:.2f} del set (deriva {drift*1e6:.0f} ppm)", flush=True)
        V = VideoSource(a.video, R.vw // 2, R.vh // 2, a.fps, t_start, off, drift)

    frame = np.zeros((H, W, 3), np.uint8)
    onset_set = set(A["onsets"].tolist())
    drop_frames = int(0.12 * a.fps) + 1
    drop_left = 0
    kick_count = 0
    # la estela depende de los cuadros anteriores: se dibujan (sin exportar) ~3 s antes del tramo
    warm = int(3 * a.fps) if R.post is not None else 0
    if stills is not None:
        still_set = set(stills)
        draw_it = lambda i: any(s - warm <= i <= s for s in stills)
        keep_it = lambda i: i in still_set
    else:
        draw_it = lambda i: i >= i0 - warm
        keep_it = lambda i: i >= i0
    base = os.path.splitext(a.output)[0]
    n_out = i1 - i0
    k = 0
    t1 = last_print = time.time()
    try:
        for i in range(i1):
            if i in onset_set:
                kick_count += 1
            if A["drops"][i]:
                drop_left = drop_frames
            if not draw_it(i):
                # avanzar el estado de la animación sin dibujar
                if a.pattern == "style":
                    R.styled(A, i, kick_count, render=False)
                elif a.pattern == "flow":
                    R.flow(A, i, kick_count)
                elif a.pattern == "bars":
                    R.bars(A["low"][i], A["mid"][i], A["high"][i], A["kick"][i],
                           bool(A["break_zone"][i]), kick_count, A["break_id"][i], A["key"][i])
                if drop_left > 0:
                    drop_left -= 1
                continue
            vid = None
            if V is not None:
                g = V.frame_at(a.start + i / a.fps)
                if g is not None:
                    vid = V.posterize(g, A["kick"][i])
            R.draw(frame, A, i, kick_count, vid)
            out = frame
            if drop_left > 0 and not a.clean:
                out = invert_keep_red(frame)
                drop_left -= 1
            if not keep_it(i):
                continue
            if stills is not None:
                ts = a.start + i / a.fps
                Image.fromarray(out).save(f"{base}_{int(ts // 60)}m{ts % 60:05.2f}s.png")
                continue
            enc.stdin.write(out.tobytes())
            k += 1
            now = time.time()
            if not a.quiet and (now - last_print > 5 or i == i1 - 1):
                rate = k / (now - t1)
                eta = (n_out - k) / rate
                print(f"\r  render {100*k/n_out:5.1f}%  {rate:5.1f} fps  ETA {int(eta//60)}m{int(eta%60):02d}s ",
                      end="", flush=True)
                last_print = now
    except BrokenPipeError:
        pass
    if enc is not None:
        enc.stdin.close()
        enc.wait()
    if V is not None:
        V.close()
    if not a.quiet:
        print(f"\nListo: {a.output}  ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
