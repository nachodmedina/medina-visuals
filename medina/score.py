"""La partitura visual: todo lo que el dibujo necesita saber de cada cuadro.

Se arma una sola vez a partir del análisis, recorriendo el track en orden (el movimiento se
acumula: avance del túnel, giro, mutaciones, remolino, estilo del viaje). Después cualquier
cuadro se dibuja solo con su entrada, así que tramos, cuadros sueltos y render en paralelo
salen iguales sin "rebobinar". Es el contrato entre la escucha y el dibujo (y con el futuro
render en GPU).

Por cuadro (FrameState):
    preset   estilo vigente (nombre en PRESETS)
    st       movimiento acumulado: ph (avance), rot (giro), mut (mutaciones), wt (grosores),
             tw (torsión del remolino), rh (tamaño del horizonte), glow (destello de la materia)
    c        lectura del momento: low, mid, high, kick, chaos, tens, closed, silent, chapter,
             hat, hat_n, nk (kicks nuevos)
    kc       kicks acumulados
Para todo el track: accent (color de acento por cuadro), camera (zoom, roll, dx, dy por cuadro).
"""
import numpy as np

from .audio import envelope
from .noise import hash32
from .presets import ACCENTS, PALETTES, PRESETS

_ST_KEYS = ("ph", "rot", "mut", "wt", "tw", "rh", "glow")


class FrameState:
    __slots__ = ("preset", "st", "c", "kc")

    def __init__(self, preset, st, c, kc):
        self.preset, self.st, self.c, self.kc = preset, st, c, kc


class Score:
    def __init__(self, fps, seed, preset, analysis, frames, accent, camera, s):
        self.fps, self.seed, self.preset = fps, seed, preset
        self.analysis = analysis      # señales del track (analysis.analyze)
        self.frames = frames          # list[FrameState], uno por cuadro
        self.accent = accent          # (n, 3) uint8
        self.camera = camera          # dict de arrays (n,) o None
        self.s = s                    # escala de salida usada para el temblor de cámara (alto / 1080)

    @property
    def n(self):
        return len(self.frames)


class _Motion:
    """Movimiento compartido por todos los estilos. Se acumula cuadro a cuadro."""

    def __init__(self, A, fps, seed):
        self.A, self.fps, self.seed = A, fps, seed
        self.st = dict(ph=0.0, rot=0.0, kc=0, ch=-1, rel=0, mut=0, tw=0.35, rh=0.08, glow=0.0)

    def step(self, i, kc):
        A, fps, st = self.A, self.fps, self.st
        c = dict(low=float(A["low"][i]), mid=float(A["mid"][i]), high=float(A["high"][i]),
                 kick=float(A["kick"][i]), chaos=float(A["chaos"][i]), tens=float(A["tension"][i]),
                 closed=(not A["sub_on"][i]) or bool(A["break_zone"][i]),
                 silent=bool(A["silent"][i]) or bool(A["breath"][i]), chapter=int(A["chapter"][i]))
        c["hat"] = float(A["hat"][i])
        c["hat_n"] = int(A["hat_n"][i])
        fd = float(A["fade"][i])
        if fd > 0.05:                           # la desaparición final se lee como tensión
            c["closed"] = True
            c["tens"] = max(c["tens"], min(1.0, fd))
        if c["chapter"] != st["ch"]:            # capítulo nuevo: nuevo azar
            st["ch"] = c["chapter"]
            st["g"] = np.random.default_rng([self.seed, 202, c["chapter"]])
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
        st["kc"] = kc
        st["rot"] += (0.05 + 0.35 * c["mid"] + 0.3 * c["chaos"]) / fps
        # torsión del remolino y tamaño del horizonte: siguen a la tensión (suavizados)
        tens_c = c["tens"] if c["closed"] else 0.0
        st["tw"] += (0.35 + 2.2 * tens_c - st["tw"]) * 0.15
        st["rh"] += (0.07 + 0.10 * tens_c + 0.03 * c["low"] - st["rh"]) * 0.25
        c["nk"] = nk
        return st, c


class _Journey:
    """El viaje: elige el estilo según la historia del track. Capítulo -> mundo; dentro del
    mundo sube de a un escalón por liberación. Solo cambia en liberaciones o capítulos."""

    def __init__(self, A, fps, preset):
        self.A, self.fps = A, fps
        self.worlds = PRESETS[preset]["worlds"]
        self.n_ch = int(A["chapter"].max()) + 1
        self.cur, self.ch, self.n_rise, self.k, self.ch_i, self.t_change = None, -1, 0, 0, 0, 0

    def update(self, i):
        A, worlds, n_ch = self.A, self.worlds, self.n_ch
        ch = int(A["chapter"][i])
        if n_ch > 1:
            world = worlds[min(ch, len(worlds) - 1)] if ch < n_ch - 1 or n_ch <= len(worlds) \
                else worlds[-1]
        else:
            world = worlds[0] + worlds[1]
        chaos = float(A["chaos"][i])
        change = self.cur is None or ch != self.ch or bool(A["rise"][i])
        if change:
            if ch != self.ch:
                self.n_rise = 0
                self.ch_i = i
                self.k = 0
            elif A["rise"][i]:
                if i - self.ch_i < 5 * self.fps:      # el drop pegado al inicio del capítulo no cuenta
                    return self.cur
                self.n_rise += 1
            if world is worlds[0] or n_ch == 1:
                # sube de a un escalón por liberación, solo si el track se abrió lo suficiente;
                # sin capítulos, la escalera es completa (partículas -> espacio)
                if n_ch == 1:
                    target = min(len(world) - 1, int(chaos * len(world)))
                else:
                    target = 0 if chaos < 0.35 else (1 if chaos < 0.75 else 2)
                stale = (i - self.t_change) > 60 * self.fps
                if A["rise"][i] and (target > self.k or stale):
                    self.k += 1
                k = min(self.k, len(world) - 1)
            else:
                k = self.n_rise % len(world)          # el segundo mundo evoluciona por drop
            name = world[k]
            if name != self.cur:
                self.t_change = i
                self.cur = name
            self.ch = ch
        return self.cur


def _accents(A, palette):
    """Color de acento de cada cuadro según la paleta."""
    n = A["n"]
    pal = PALETTES[palette]
    if isinstance(pal, list):                     # alterna por kick dentro de cada liberación
        m = A["rel_kick"].astype(np.int64)
        return np.array(pal, np.uint8)[np.where(m >= 0, m % len(pal), 0)]
    if pal is not None:
        return np.broadcast_to(np.asarray(pal, np.uint8), (n, 3)).copy()
    return np.array(ACCENTS, np.uint8)[A["accent"].astype(np.int64)]


def _camera(A, fps, seed, s):
    """Cámara, función pura del cuadro.
    zoom: empuja hacia adentro con la tensión (sobre todo al final) y suelta de golpe en la
    liberación, con un rebote; en lo abierto respira lento. kick: golpe de zoom + empujón.
    giro: deriva lenta (más amplia con el caos) y un giro seco en cada liberación."""
    n = A["n"]
    g = np.random.default_rng([seed, 909])
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
    ang = hash32(np.maximum(last, 0), np.zeros(n, np.int64) + 31, seed) * 2 * np.pi
    sh = 15.0 * s * kamp
    dx, dy = sh * np.cos(ang), sh * np.sin(ang)
    # nervio: temblor chico en los hats al final de la tensión
    nerv = np.clip((T - 0.35) / 0.65, 0, 1) * (T > 0)
    hang = hash32(A["hat_n"].astype(np.int64), np.zeros(n, np.int64) + 57, seed) * 2 * np.pi
    hs_ = 5.0 * s * A["hat"] * nerv
    dx, dy = dx + hs_ * np.cos(hang), dy + hs_ * np.sin(hang)
    return dict(zoom=zoom, roll=roll, dx=dx, dy=dy)


def build_score(A, preset="viaje", fps=30, seed=0, s=1.0, camera=True, palette="violeta"):
    """Análisis -> partitura. `s` = alto de salida / 1080 (escala del temblor de cámara)."""
    journey = _Journey(A, fps, preset) if PRESETS[preset]["style"] == "journey" else None
    motion = _Motion(A, fps, seed)
    onset_set = set(A["onsets"].tolist())
    frames = []
    kc = 0
    for i in range(A["n"]):
        if i in onset_set:
            kc += 1
        name = journey.update(i) if journey is not None else preset
        st, c = motion.step(i, kc)
        frames.append(FrameState(name, {k: st[k] for k in _ST_KEYS}, c, kc))
    return Score(fps, seed, preset, A, frames, _accents(A, palette),
                 _camera(A, fps, seed, s) if camera else None, s)
