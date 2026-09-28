"""Un track de techno sintético con una historia conocida, para probar el motor sin música real.

    0 s ........ kick a 128.57 BPM (14 cuadros por beat a 30 fps) con sub pleno;
                 hats en corcheas desde el segundo 8
    24 s ....... el sub se filtra (tensión): el kick sigue, pero sin graves debajo de 60 Hz
    ~40 s ...... vuelve el sub, justo en un beat (la liberación)
    44 s ....... entra un sinte en semicorcheas (una capa nueva)
    64 s ....... fin
"""
import numpy as np

SR = 22050
BPM = 60 * 30 / 14             # 128.57 BPM: cada beat dura exactamente 14 cuadros de video
BEAT = 60.0 / BPM
T0 = 8 / 30 + 0.008              # primer kick: justo después del centro de un cuadro
DUR = 64.0
FILTER_FROM = 24.0             # empieza la tensión (en el primer beat desde acá)
RELEASE_NEAR = 40.0            # la liberación cae en el primer beat desde acá
HATS_FROM = 8.0
LAYER_AT = 44.0


def beat_times(t0=T0):
    n = int((DUR - t0) / BEAT) + 1
    return t0 + BEAT * np.arange(n)


def _first_beat_after(t, t0=T0):
    b = beat_times(t0)
    return float(b[b >= t][0])


def expected(t0=T0):
    """Lo que el motor debería leer."""
    return dict(bpm=BPM, filter_start=_first_beat_after(FILTER_FROM, t0),
                release=_first_beat_after(RELEASE_NEAR, t0), layer_at=LAYER_AT, hats_from=HATS_FROM)


def _taper(sig, fade_in=0.001, fade_out=0.02):
    """Fundido de entrada y de salida: sin clics (un corte seco suena como un hat)."""
    sig = sig.copy()
    a, b = max(1, int(fade_in * SR)), max(1, int(fade_out * SR))
    sig[:a] *= np.linspace(0, 1, a)
    sig[-b:] *= np.linspace(1, 0, b)
    return sig


def _add(y, t, sig):
    a = int(round(t * SR))
    b = min(len(y), a + len(sig))
    if b > a:
        y[a:b] += sig[:b - a]


def make_track(seed=0, t0=T0):
    """Devuelve el audio mono float32 a 22050 Hz. `t0` = momento del primer kick."""
    rng = np.random.default_rng(seed)
    y = np.zeros(int(DUR * SR), np.float32)
    ex = expected(t0)
    t = np.arange(int(0.36 * SR)) / SR
    env = np.exp(-t / 0.16)

    def kick(f_end):                                           # el kick domina los graves
        f = f_end + (160 - f_end) * np.exp(-t / 0.03)          # barrido de tono
        return _taper((0.95 * np.sin(2 * np.pi * np.cumsum(f) / SR) * env).astype(np.float32))

    kick_full, kick_filtered = kick(45.0), kick(85.0)          # filtrado = sin graves bajo ~60 Hz
    ts = np.arange(int(0.2 * SR)) / SR
    sub = _taper((0.22 * np.sin(2 * np.pi * 45.0 * ts) * np.minimum(1, ts / 0.01) * np.exp(-ts / 0.12)).astype(np.float32))
    tn = np.arange(int(0.03 * SR)) / SR
    for bt in beat_times(t0):
        tense = ex["filter_start"] <= bt < ex["release"]
        _add(y, bt, kick_filtered if tense else kick_full)
        if not tense:
            _add(y, bt + BEAT / 2, sub)                           # bajo en el contratiempo
        if bt >= HATS_FROM:                                       # hats en corcheas
            for off in (0.0, BEAT / 2):
                noise = rng.standard_normal(len(tn)).astype(np.float32)
                hat = np.diff(noise, prepend=0) * 0.10 * np.exp(-tn / 0.008)
                _add(y, bt + off + 0.01, _taper(hat.astype(np.float32), 0.0005, 0.005))
    # sinte en semicorcheas desde LAYER_AT (una capa nueva, 330 Hz con armónicos)
    tsy = np.arange(int(BEAT / 4 * 0.8 * SR)) / SR
    note = _taper((0.12 * np.sign(np.sin(2 * np.pi * 330 * tsy)) * np.exp(-tsy / 0.05)).astype(np.float32), 0.002, 0.01)
    for bt in beat_times(t0):
        if bt >= LAYER_AT:
            for q in range(4):
                _add(y, bt + q * BEAT / 4, note)
    y += (3e-4 * rng.standard_normal(len(y))).astype(np.float32)     # piso de ruido (~ -70 dBFS), como una grabación real
    return np.clip(y, -1, 1).astype(np.float32)


def write_wav(path, y):
    from scipy.io import wavfile
    wavfile.write(path, SR, (y * 32767).astype(np.int16))
