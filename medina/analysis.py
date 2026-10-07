"""Escuchar el track. Todo lo que el video hace sale de acá:

    kicks        beats con energía en 35-130 Hz (tempo + programación dinámica)
    sub_on       sub 25-60 Hz relativo al sub pleno (un corte parcial también es tensión)
    rise         liberaciones: vuelve el sub o vuelve el kick tras >= 8 s (y se queda: un golpe
                 suelto no cuenta), anclada al primer kick que ya trae el sub entero
    tension      0 -> 1 dentro de cada tramo filtrado; termina en la liberación
    breath       el último beat antes de cada liberación (respiración a negro)
    chaos        curva lenta de agudos (el track se abre)
    hat          ataques de 5-11 kHz
    silent/fade  cortes breves (capítulos) y desaparición gradual al final
    layers       elementos que entran (NMF sobre 300 Hz - 11 kHz)
    key          tonalidad por ventanas de 90 s
"""
import warnings

import numpy as np
from scipy.ndimage import maximum_filter1d, median_filter, uniform_filter1d
from scipy.signal import butter, sosfilt

from .audio import beat_track, envelope, norm_db

ANALYSIS_SR = 22050
N_FFT = 2048

PITCHES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
# Camelot: número (1-12) por tónica, para menor (A) y mayor (B)
CAMELOT_MIN = {8: 1, 3: 2, 10: 3, 5: 4, 0: 5, 7: 6, 2: 7, 9: 8, 4: 9, 11: 10, 6: 11, 1: 12}
CAMELOT_MAJ = {11: 1, 6: 2, 1: 3, 8: 4, 3: 5, 10: 6, 5: 7, 0: 8, 7: 9, 2: 10, 9: 11, 4: 12}
KS_MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KS_MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def key_name(k):
    """Índice de key (0-11 menor, 12-23 mayor) -> (nombre, código Camelot), ej. ('Am', '8A')."""
    cam = f"{CAMELOT_MIN[k]}A" if k < 12 else f"{CAMELOT_MAJ[k - 12]}B"
    return PITCHES[k % 12] + ("m" if k < 12 else ""), cam


def breath_mask(n, onsets, rise):
    """Respiración: el último beat antes de cada liberación queda a negro (solo la línea),
    así la liberación entra desde la nada. Anclado a la grilla de kicks del propio track."""
    out = np.zeros(n, bool)
    if len(onsets) < 2:
        return out
    beat = int(round(float(np.median(np.diff(onsets)))))
    for r in np.where(rise)[0]:
        out[max(0, r - beat):r] = True
    return out


def analyze(y, fps, bpm_range=(100, 165)):
    """Audio mono a ANALYSIS_SR -> señales del track, una muestra por cuadro de video."""
    sr = ANALYSIS_SR
    n_frames = int(len(y) / sr * fps)
    hop = sr / fps
    y_p = np.pad(y, (N_FFT // 2, N_FFT // 2 + int(hop) + 1))
    hann = np.hanning(N_FFT).astype(np.float32)
    freqs = np.fft.rfftfreq(N_FFT, 1 / sr)

    band = lambda lo, hi: (freqs >= lo) & (freqs < hi)
    b_low, b_mid, b_high = band(20, 140), band(140, 2500), band(2500, 11000)
    b_kick = np.where(band(35, 130))[0]
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
    # para la capa emocional: brillo (centroide 60 Hz - 11 kHz) y aspereza (planitud 300 Hz - 8 kHz)
    b_reg = band(60, 11000)
    b_flat = band(300, 8000)
    f_reg = freqs[b_reg]
    centroid = np.zeros(n_frames, np.float32)
    flatness = np.zeros(n_frames, np.float32)
    offs = np.arange(N_FFT)
    for s in range(0, n_frames, 4096):
        e = min(s + 4096, n_frames)
        idx = (np.arange(s, e) * hop).astype(np.int64)
        P = np.abs(np.fft.rfft(y_p[idx[:, None] + offs] * hann, axis=1)) ** 2
        lmh[s:e, 0] = P[:, b_low].sum(1)
        lmh[s:e, 1] = P[:, b_mid].sum(1)
        lmh[s:e, 2] = P[:, b_high].sum(1)
        kick_spec[s:e] = P[:, b_kick]
        hat_spec[s:e] = P[:, b_hat]
        subp[s:e] = P[:, b_sub].sum(1)
        totp[s:e] = P.sum(1)
        lspec[s:e] = P @ Lm
        chroma[s:e] = np.log1p(1e3 * P / (P.max(1, keepdims=True) + 1e-12)) @ Cm
        Pr = P[:, b_reg]
        centroid[s:e] = (Pr @ f_reg) / (Pr.sum(1) + 1e-12)
        Pf = P[:, b_flat] + 1e-12
        flatness[s:e] = np.exp(np.log(Pf).mean(1)) / Pf.mean(1)

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

    # --- zonas de bajada (mirando hacia adelante): del último kick antes de un
    # hueco de >= 4 s hasta el kick que vuelve. También antes del primer kick.
    break_zone = np.zeros(n_frames, bool)
    ons = onsets.tolist()
    edges_k = [-1] + ons + [n_frames]
    for p, q in zip(edges_k[:-1], edges_k[1:]):
        if (q - p) / fps >= 4.0 or p < 0 or q >= n_frames:
            a0 = max(0, p + int(0.4 * fps)) if p >= 0 else 0
            if a0 < q:
                break_zone[a0:q] = True

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

    # ================= estados del track =================
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
    ctx = median_filter(tot_db, size=int(12 * fps) | 1, mode="nearest")
    silent = (tot_db < np.percentile(tot_db, 90) - 22) & (tot_db < ctx - 10)
    # silencio = corte breve (0.7-4 s) con música después; un tramo largo en bajo nivel
    # (fade out, outro) no es un silencio: es una desaparición gradual
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
    ons_arr = np.array(ons, np.int64)
    # nivel real de sub (sin suavizar) en los 150 ms después de cada kick: el detector suavizado
    # se entera tarde, así que la liberación se adelanta al primer kick que ya trae el sub entero
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
        # vuelve el kick tras una bajada larga, y se queda: un golpe suelto en el filtrado no es un drop
        if i and (o - ons[i - 1]) / fps >= 8.0 and i + 2 < len(ons) and (ons[i + 2] - o) / fps < 2.0:
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
    # 6) acento por liberación (paletas con color por kick): índice R->G->B y n° de kick
    #    dentro de los 32 kicks que siguen a cada liberación
    accent = np.zeros(n_frames, np.int8)
    rel_kick = -np.ones(n_frames, np.int16)     # -1 fuera de las liberaciones
    for r in np.where(rise)[0]:
        j0 = int(np.searchsorted(ons_arr, r))
        for m in range(32):
            if j0 + m >= len(ons):
                break
            a0 = ons[j0 + m]
            a1 = ons[j0 + m + 1] if j0 + m + 1 < len(ons) else n_frames
            accent[a0:a1] = m % 3
            rel_kick[a0:a1] = m

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
        breath=breath_mask(n_frames, onsets, rise),
        tension=tension,
        chaos=chaos,
        accent=accent,
        rel_kick=rel_kick,
        hat=hat,
        hat_n=hat_n,
        break_zone=break_zone,
        key=key_frame,
        low=envelope(low, 0.86),
        mid=envelope(mid, 0.90),
        high=envelope(high, 0.80),
        kick=envelope(np.where(strength > 0, 1.0, 0.0).astype(np.float32), 0.80),
        strength=strength,
        onsets=onsets,
        # señales para la capa emocional (medina/emotion.py)
        level_db=tot_db.astype(np.float32),
        centroid=centroid,
        flatness=flatness,
        chroma=chroma,
        bands=np.log1p(lspec / (np.percentile(lspec, 97, axis=0) + 1e-12) * 30).astype(np.float32),
    )


def detect_layers(lspec, cf, fps, silent, K=8):
    """Separa el track en K capas (NMF) por encima de 300 Hz y detecta cuándo entra
    cada una. Devuelve actividad rápida por capa y el orden de entrada."""
    try:
        from sklearn.decomposition import NMF
    except Exception:
        return None
    n = len(lspec)
    V = np.log1p(lspec / (np.percentile(lspec, 97, axis=0) + 1e-12) * 30)
    step = max(1, int(fps // 10))                 # ajuste a 10 fps, proyección a fps completo
    model = NMF(n_components=K, init="nndsvda", max_iter=500, beta_loss="kullback-leibler",
                solver="mu", random_state=0)
    with warnings.catch_warnings():               # avisos de convergencia de sklearn: esperados
        warnings.simplefilter("ignore")
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


def bursts(A, fps, thr=8.0, gap=1.5):
    """Estallidos del rango medio (700 Hz - 3 kHz) por encima de lo que suena estable: barridos que
    caen, ráfagas. A cada banda se le resta su mediana de los últimos ~12 s (los tonos sostenidos no
    cuentan). Devuelve [(cuadro, fuerza 0..1)], separados al menos `gap` segundos."""
    le = np.geomspace(300, 11000, 73)
    cf = np.sqrt(le[:-1] * le[1:])
    X = A["bands"][:, (cf > 700) & (cf < 3000)]
    med = np.repeat(median_filter(X[::5], size=(max(1, int(12 * fps / 5)), 1), mode="nearest"), 5, axis=0)[:len(X)]
    E = uniform_filter1d(np.maximum(X - med, 0).mean(1), 4)
    m = np.median(E)
    z = (E - m) / (np.median(np.abs(E - m)) + 1e-9)
    peaks = []
    for i in np.argsort(-z):
        if z[i] < thr:
            break
        if all(abs(i - p) > gap * fps for p, _ in peaks):
            peaks.append((int(i), float(np.clip((z[i] - thr) / 10 + 0.3, 0.3, 1.0))))
    return sorted(peaks)
