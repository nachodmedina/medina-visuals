"""La capa emocional: qué te hace sentir cada momento del track (lo que genera el espacio) y el
ADN del track, que define su sistema estelar.

Tres niveles:
    ADN del track      dónde estás: su sistema estelar (constante para todo el track)
    emociones          qué te pasa en cada momento del recorrido
    pulso              kicks, hats, drops (ya en analysis)

Rasgos por momento (0..1): energía, densidad, brillo, aspereza, claridad y color de la armonía,
sorpresa (novedad), repetición, pulso, tensión, subidas de ruido y contraste con lo reciente.
Cada emoción es una combinación de rasgos; las escalas están calibradas con los tracks de
tracks/ para que los valores sean comparables entre tracks.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter1d, uniform_filter1d

from .analysis import KS_MAJ, KS_MIN

EMOTIONS = ("incertidumbre", "miedo", "esperanza", "enigma", "soledad", "fuerza", "vulnerabilidad")


def _sm(x, fps, sec):
    return uniform_filter1d(np.asarray(x, np.float64), max(1, int(sec * fps)), mode="nearest")


def _norm(x, lo=5, hi=95):
    a, b = np.percentile(x, [lo, hi])
    return np.clip((np.asarray(x, np.float64) - a) / max(b - a, 1e-9), 0, 1)


def _resample(x, n):
    """Curva a baja resolución -> una muestra por cuadro."""
    return np.interp(np.arange(n), np.linspace(0, n - 1, len(x)), x)


def _pool(X, step):
    m = len(X) // step
    return X[:m * step].reshape(m, step, -1).mean(1)


def _harmony(chroma, fps, n):
    """Claridad de la tonalidad (qué tan definida está) y color (mayor = luminoso, menor = oscuro),
    por ventanas de 8 s."""
    profs = np.array([np.roll(KS_MIN, t) for t in range(12)] + [np.roll(KS_MAJ, t) for t in range(12)])
    profs = (profs - profs.mean(1, keepdims=True)) / profs.std(1, keepdims=True)
    C = _pool(chroma, fps)                                   # 1 muestra por segundo
    W = uniform_filter1d(C, 8, axis=0, mode="nearest")
    W = (W - W.mean(1, keepdims=True)) / (W.std(1, keepdims=True) + 1e-9)
    r = W @ profs.T / 12                                     # correlación con cada tonalidad
    clarity = r.max(1)
    color = r[:, 12:].max(1) - r[:, :12].max(1)              # > 0: suena más a mayor
    return _resample(clarity, n), _resample(color, n)


def _novelty(bands, fps, n, half_s=8):
    """Sorpresa: cambio de timbre a escala de sección (novedad de Foote sobre las bandas)."""
    step = max(1, fps // 2)                                  # 2 muestras por segundo
    X = _pool(bands, step)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    S = X @ X.T
    L = int(half_s * fps / step)
    g = np.exp(-0.5 * (np.arange(-L, L) + 0.5) ** 2 / (0.5 * L) ** 2)
    K = np.outer(g, g) * np.sign(np.arange(-L, L) + 0.5)[:, None] * np.sign(np.arange(-L, L) + 0.5)[None, :]
    N = len(X)
    nov = np.zeros(N)
    for t in range(L, N - L):
        nov[t] = (K * S[t - L:t + L, t - L:t + L]).sum()
    nov = np.maximum(nov, 0)
    return _resample(nov, n)


def _repetition(bands, fps, n, bpm):
    """Repetición: qué tanto se parece cada momento a lo que sonó un compás y dos compases antes
    (el loop hipnótico). Coseno entre bandas de timbre, 0..1."""
    step = 3
    X = _pool(bands, step)                                   # 10 muestras por segundo a 30 fps
    X = X - X.mean(0)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    lag = max(1, int(round(4 * 60 / bpm * fps / step)))
    rep = np.zeros(len(X))
    for k in (1, 2):
        lk = lag * k
        rep[lk:] += (X[lk:] * X[:-lk]).sum(1) / 2
    rep = uniform_filter1d(np.clip(rep, 0, 1), max(1, int(4 * fps / step)), mode="nearest")
    return _resample(rep, n)


def features(A, fps, bpm):
    """Rasgos del track, una muestra por cuadro, en 0..1."""
    n = A["n"]
    closed = (~A["sub_on"]) | A["break_zone"] | (A["fade"] > 0.05)
    tension = np.where(closed, np.maximum(A["tension"], A["fade"]), 0.0)
    beat = 60 / bpm * fps
    ind = np.zeros(n)
    ind[A["onsets"]] = 1
    pulse = np.clip(uniform_filter1d(ind, int(4 * beat), mode="nearest") * beat, 0, 1)
    sub_s = _sm(A["sub_on"].astype(float), fps, 2)
    level = _norm(A["level_db"], 5, 98)
    energy = _sm(0.45 * level + 0.25 * pulse + 0.2 * sub_s + 0.1 * _norm(A["mid"]), fps, 2)
    Ly = A.get("layers")
    if Ly and Ly["order"]:
        cnt = Ly["on"].sum(1).astype(float)
        dens_l = cnt / max(1.0, cnt.max())
    else:
        cnt, dens_l = np.zeros(n), np.zeros(n)
    hat_on = np.diff(np.r_[0, A["hat_n"]]) > 0
    hat_rate = _sm(hat_on.astype(float), fps, 4) * fps              # hats por segundo
    density = _sm(0.5 * dens_l + 0.25 * np.clip(hat_rate / 6, 0, 1) + 0.25 * (A["mid"] + A["high"]) / 2, fps, 3)
    brightness = _sm(_norm(np.log2(A["centroid"] + 1)), fps, 3)
    rough = _sm(_norm(A["flatness"]), fps, 2)
    clarity, color = _harmony(A["chroma"], fps, n)
    novelty = _norm(_novelty(A["bands"], fps, n), 5, 98)
    repetition = _repetition(A["bands"], fps, n, bpm)
    hi = _sm(A["high"], fps, 1)
    slope = hi - np.r_[np.full(int(4 * fps), hi[0]), hi[:-int(4 * fps)]]
    riser = np.clip(slope * 3, 0, 1) * _sm(closed.astype(float), fps, 1)
    past = maximum_filter1d(energy, int(16 * fps), origin=int(8 * fps) - 1, mode="nearest")
    contrast = np.clip(past - energy, 0, 1)
    idx = np.arange(n)
    last = np.maximum.accumulate(np.where(A["rise"], idx, -1))
    post = np.where(last >= 0, np.exp(-(idx - last) / (8 * fps)), 0.0)
    # alivio: la liberación, pesada por cuánta tensión la precede (los 30 s anteriores)
    prior = maximum_filter1d(tension, int(30 * fps), origin=int(15 * fps) - 1, mode="nearest")
    relief = np.where(last >= 0, np.exp(-(idx - last) / (6 * fps)), 0.0) * prior[np.maximum(last, 0)]
    # luz que sube: el brillo asciende durante unos segundos
    b8 = np.r_[np.full(int(8 * fps), brightness[0]), brightness[:-int(8 * fps)]]
    rising = np.clip((brightness - b8) * 3, 0, 1)
    # entrada tonal: entra un elemento nuevo, agudo (> 700 Hz), que suena unos segundos
    entry = np.zeros(n)
    if Ly and Ly["order"]:
        for k in Ly["order"]:
            if Ly["center"][k] > 700 and Ly["first"][k] < n:
                f0 = int(Ly["first"][k])
                entry[f0:] = np.maximum(entry[f0:], np.exp(-(idx[f0:] - f0) / (8 * fps)))
    return dict(alivio=relief, sube=rising, entrada=entry, energia=energy, densidad=density, brillo=brightness, aspereza=rough,
                claridad=_norm(clarity), color=np.clip(0.5 + color * 2, 0, 1), sorpresa=novelty,
                repeticion=_norm(repetition), pulso=pulse, tension=tension, subida=riser,
                contraste=contrast, cerrado=closed.astype(float), silencio=_sm(A["silent"], fps, 1),
                post=post, capas=cnt, hats=hat_rate, repeticion_abs=repetition)


# escala de cada emoción (calibrada con los tracks de tracks/: su percentil 98 conjunto)
EMO_SCALE = dict(incertidumbre=0.713, miedo=0.359, esperanza=0.635, enigma=0.709, soledad=0.891,
                 fuerza=0.954, vulnerabilidad=0.347)


def emotions(F, fps, scale=EMO_SCALE):
    """Rasgos -> las siete emociones, una curva por emoción en 0..1."""
    dark = 1 - F["brillo"]
    tonal = 1 - F["aspereza"]
    raw = dict(
        # la atracción: tensión que crece, oscura, áspera o con ruido que sube
        miedo=_sm(F["tension"] ** 0.8 * (0.35 + 0.65 * dark) * (0.45 + 0.55 * np.maximum(F["aspereza"], F["subida"])), fps, 2),
        # masa y gravedad: energía plena, pulso firme, sub entero
        fuerza=_sm(F["energia"] ** 1.3 * (0.4 + 0.6 * F["pulso"]), fps, 1.5),
        # sin pulso claro, armonía indefinida, cambios inesperados, el comienzo de un tramo filtrado
        incertidumbre=_sm((0.35 * (1 - F["pulso"]) + 0.25 * (1 - F["claridad"]) + 0.25 * F["sorpresa"]
                           + 0.15 * F["cerrado"] * (1 - F["tension"])), fps, 3),
        # la luz: el alivio de una liberación tras una tensión larga, el brillo que sube,
        # un elemento tonal y agudo que entra; más si la armonía es luminosa
        esperanza=_sm(np.maximum.reduce([F["alivio"], F["sube"], F["entrada"] * tonal])
                      * (0.6 + 0.4 * F["color"]), fps, 2),
        # señales de lo desconocido: repetición hipnótica, armonía ambigua, tonal
        enigma=_sm(F["repeticion"] * (0.5 + 0.5 * (1 - F["claridad"])) * (0.4 + 0.6 * tonal), fps, 4),
        # un solo elemento en el vacío: poca densidad, poca energía (pero no silencio)
        soledad=_sm((1 - F["densidad"]) ** 1.2 * (1 - F["energia"]) ** 0.6 * (1 - F["silencio"]), fps, 3),
        # lo frágil después de la intensidad: cae la energía, queda poco
        vulnerabilidad=_sm(F["contraste"] * (0.4 + 0.6 * (1 - F["densidad"])) * (0.5 + 0.5 * F["brillo"]), fps, 2),
    )
    E = {k: np.clip(raw[k] / scale[k], 0, 1) for k in EMOTIONS}
    E["incertidumbre"] = E["incertidumbre"] * (1 - 0.7 * E["fuerza"])
    E["esperanza"] = E["esperanza"] * (1 - 0.6 * E["miedo"])
    E["enigma"] = E["enigma"] * (1 - 0.5 * E["fuerza"])
    return E


def sections(A, F, E, fps, min_len=12.0):
    """Divide el track en secciones (liberaciones, comienzos de tensión, silencios, cambios fuertes
    de timbre) y resume las emociones de cada una."""
    n = A["n"]
    cut = {0, n}
    cut |= set(np.where(A["rise"])[0].tolist())
    closed = F["cerrado"] > 0.5
    cut |= set((np.where(np.diff(closed.astype(int)) == 1)[0] + 1).tolist())
    nov = gaussian_filter1d(F["sorpresa"], fps)
    peaks = np.where((nov > 0.6) & (nov >= maximum_filter1d(nov, int(16 * fps))))[0]
    cut |= set(peaks.tolist())
    cuts = sorted(cut)
    merged = [cuts[0]]
    for c in cuts[1:]:
        if c - merged[-1] >= min_len * fps or c == n:
            merged.append(c)
        elif A["rise"][c] or (0 < c < n and closed[c] and not closed[c - 1]):
            merged[-1] = c                                   # los cortes del track mandan sobre los de timbre
    if merged[-1] != n:
        merged.append(n)
    out = []
    for a, b in zip(merged[:-1], merged[1:]):
        if b - a < fps:
            continue
        m = {k: float(E[k][a:b].mean()) for k in EMOTIONS}
        top = sorted(m, key=m.get, reverse=True)
        out.append(dict(start=a / fps, end=b / fps, emociones=m,
                        principales=[k for k in top[:2] if m[k] >= 0.2] or top[:1],
                        tension=bool(closed[a:b].mean() > 0.5), liberacion=bool(A["rise"][a])))
    return out


def dna(A, F, E, fps, bpm):
    """Medidas del track entero, absolutas para poder comparar tracks (base del ADN)."""
    n = A["n"]
    closed = F["cerrado"] > 0.5
    zones = np.diff(np.r_[0, closed.astype(int), 0])
    lens = (np.where(zones == -1)[0] - np.where(zones == 1)[0]) / fps
    nov = gaussian_filter1d(F["sorpresa"], fps)
    peaks = np.where((nov > 0.6) & (nov >= maximum_filter1d(nov, int(16 * fps))))[0]
    raw = dict(
        planitud=float(np.mean(A["flatness"])),
        cambios_por_min=len(peaks) / (n / fps / 60),
        cerrado=float(closed.mean()),
        tension_larga=float(lens.max()) if len(lens) else 0.0,
        kicks_por_beat=len(A["onsets"]) / max(1.0, n / fps * bpm / 60),
        centroide_hz=float(np.exp(np.mean(np.log(A["centroid"] + 1)))),
        capas=float(np.mean(F["capas"])),
        repeticion=float(np.mean(F["repeticion_abs"])),
        bpm=bpm,
    )
    return raw


# los ejes del ADN, en 0..1; las anclas están calibradas con los tracks de tracks/
DNA_AXES = ("caos", "vacio", "voragine", "energia", "luz", "densidad", "hipnosis", "aspereza")


def dna_axes(raw):
    """Medidas del track -> los ejes de su sistema estelar."""
    c = lambda x: float(np.clip(x, 0, 1))
    return dict(
        caos=c(0.6 * (raw["cambios_por_min"] - 0.5) / 1.5 + 0.4 * (1 - (raw["repeticion"] - 0.2) / 0.4)),
        vacio=c(0.6 * raw["cerrado"] + 0.4 * (1 - raw["capas"] / 5)),
        voragine=c(0.6 * raw["tension_larga"] / 80 + 0.4 * raw["cerrado"]),
        energia=c(0.7 * (raw["kicks_por_beat"] - 0.4) / 0.6 + 0.3 * (1 - raw["cerrado"])),
        luz=c((raw["centroide_hz"] - 200) / 250),
        densidad=c(raw["capas"] / 5),
        hipnosis=c((raw["repeticion"] - 0.2) / 0.4),
        aspereza=c((raw["planitud"] - 0.02) / 0.05),
    )


def star_system(ax):
    """Propuesta: qué varía en el sistema estelar de cada track según su ADN.
    Devuelve (parámetro, valor para este track, eje que lo define, qué expresa)."""
    L = lambda a, b, t: a + (b - a) * t
    vio = [int(round(L(a, b, ax["luz"]))) for a, b in ((86, 150), (36, 95), (150, 230))]
    fav = sorted([("pulsares y estrellas", ax["hipnosis"]), ("jets", ax["energia"]),
                  ("lunas solitarias", ax["vacio"]), ("cometas y meteoros", ax["caos"])],
                 key=lambda x: -x[1])[:2]
    return [
        ("Masa del agujero", f"radio {L(0.06, 0.11, ax['voragine']):.3f} · lente ×{L(0.8, 1.4, ax['voragine']):.2f}",
         "vorágine", "más atracción: un agujero más grande que curva más el espacio"),
        ("Inclinación del disco", f"{L(12, 34, ax['caos']):.0f}°", "caos",
         "orden: disco plano y estable; caos: inclinado, inquieto"),
        ("Velocidad orbital", f"×{L(0.7, 1.35, ax['energia']):.2f}", "energía",
         "cuánto empuja el track: órbitas lentas o vertiginosas"),
        ("Densidad de materia", f"×{L(0.7, 1.3, ax['densidad']):.2f}", "densidad",
         "cantidad de partículas, órbitas y segmentos"),
        ("Turbulencia", f"ondas ×{L(0.5, 1.8, ax['caos']):.2f}", "caos",
         "órbitas que se deforman y ondas en el espacio"),
        ("Estrellas de fondo", f"{int(L(400, 1400, ax['vacio']))} estrellas", "vacío",
         "cuánto espacio profundo se ve: la calma del vacío"),
        ("Luz", f"resplandor ×{L(0.7, 1.35, ax['luz']):.2f}", "luz",
         "intensidad del bloom y del halo del horizonte"),
        ("Tono del violeta", "#{:02x}{:02x}{:02x}".format(*vio), "luz",
         "dentro de la paleta: violeta profundo (oscuro) ↔ lavanda (luminoso)"),
        ("Proporción de acento", f"{L(10, 30, ax['energia']):.0f} %", "energía",
         "cuánto violeta frente al blanco"),
        ("Grano", f"×{L(0.6, 1.8, ax['aspereza']):.2f}", "aspereza",
         "textura: limpio y liso, o áspero como el ruido del track"),
        ("Estela", f"×{L(0.7, 1.6, ax['vacio']):.2f}", "vacío",
         "cuánto dura el recuerdo de lo que pasó"),
        ("Cámara", f"distancia ×{L(1.0, 1.25, ax['vacio']):.2f} · temblor ×{L(0.6, 1.4, ax['energia']):.2f}",
         "vacío · energía", "lejos y quieta en el vacío; cerca y golpeada con energía"),
        ("Fenómenos preferidos", " y ".join(f for f, _ in fav), "hipnosis · energía · vacío · caos",
         "qué fenómenos del espacio tienden a aparecer en este sistema"),
        ("Ritmo del viaje", f"{'rápido' if ax['energia'] > 0.6 else 'pausado'}", "energía",
         "cuántos escalones sube el viaje por liberación"),
    ]


def emotional_map(A, fps, bpm):
    """Todo lo de la capa emocional de un track: rasgos, emociones, secciones y ADN."""
    F = features(A, fps, bpm)
    E = emotions(F, fps)
    raw = dna(A, F, E, fps, bpm)
    ax = dna_axes(raw)
    return dict(features=F, emotions=E, sections=sections(A, F, E, fps), dna=raw, axes=ax,
                system=star_system(ax))
