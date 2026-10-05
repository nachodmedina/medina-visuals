"""Estilos del viaje. Cada uno dibuja un cuadro en la grilla como índices de paleta
(0 negro, 1 blanco, 2 acento, 3-5 blancos tenues, 6-7 acento tenue), a partir del contexto
del cuadro (preset, movimiento, lectura del momento) y de la lente del agujero negro."""
import numpy as np

from .noise import hash32


class FrameCtx:
    """Contexto de un cuadro para el dibujo. La lente (lens) completa lensL/lensT/lens_box/ripple."""
    __slots__ = ("P", "st", "c", "kc", "i", "roll", "box", "seed", "fps", "sys",
                 "lensL", "lensT", "lens_box", "ripple")

    def __init__(self, P, st, c, kc, i, roll, box, seed, fps, system=None):
        from .system import StarSystem
        self.P, self.st, self.c, self.kc, self.i = P, st, c, kc, i
        self.roll, self.box, self.seed, self.fps = roll, box, seed, fps
        self.sys = system or StarSystem.neutral()
        self.lensL = self.lensT = self.lens_box = self.ripple = None


# ------------------------------------------------------------- tensión ----
def nerv(c):
    """Nervio de la tensión: 0 hasta la mitad del tramo filtrado, 1 al final."""
    if not c["closed"] or c["silent"]:
        return 0.0
    return float(np.clip((c["tens"] - 0.35) / 0.65, 0, 1))


def thin(c):
    """En la tensión, lo que queda se va afinando."""
    return 1.0 - 0.7 * c["tens"] if c["closed"] else 1.0


def erode(ids, c, salt=0):
    """Tensión sin marcos: cada elemento tiene un umbral fijo; a medida que sube la tensión se
    van apagando hasta quedar pocos. Al final del tramo lo que sobrevive titila entre hats, y en
    cada hat vuelve un instante parte de lo erosionado."""
    if not c["closed"] or c["silent"]:
        return True
    nv = nerv(c)
    e = 0.88 * c["tens"] ** 1.2 * (1 - 0.65 * nv * c["hat"])
    keep = hash32(ids + 7919, np.zeros_like(ids) + 3 + salt, 11) >= e
    if nv > 0:
        keep &= hash32(ids + 104729, np.zeros_like(ids) + 5 + salt, c["hat_n"]) \
            >= 0.5 * nv * (1 - c["hat"])
    return keep


def levels(I, acc):
    """Intensidad continua -> tonos de la paleta (blanco o acento, en 4 niveles)."""
    lev = np.zeros(I.shape, np.uint8)
    w = ~acc
    for thr, j in ((0.06, 3), (0.22, 4), (0.45, 5), (0.75, 1)):
        lev[(I > thr) & w] = j
    for thr, j in ((0.06, 6), (0.30, 7), (0.70, 2)):
        lev[(I > thr) & acc] = j
    return lev


# -------------------------------------------------------------- la lente ----
def lens(grid, fr):
    """Lente gravitacional: cerca del horizonte se ve lo que está 'detrás' del agujero,
    apretado en un anillo muy denso alrededor del borde (lente puntual: fuente = r - El²/r,
    con El elegido para que justo en el borde la fuente sea casi el centro), y transición suave
    a espacio plano hacia 6 horizontes. También arma la torsión (la espiral general más un
    remolino que se aprieta pegado al borde) y la onda gravitacional de cada liberación.
    Devuelve log(r) curvado; se calcula solo en la zona central."""
    st, lf = fr.st, fr.sys.lens
    Re = float(st["rh"]) * 1.12
    s0 = 0.06 * Re                             # lo que se ve pegado al borde viene de casi el centro
    El = float(np.sqrt(Re * Re - s0 * Re))
    L = grid.logRR.copy()
    Tw = st["tw"] * grid.SWIRL
    half = int(6 * Re * lf * grid.unit) + 2
    cy, cx = grid.hh // 2, grid.ww // 2
    sl = (slice(max(0, cy - half), cy + half), slice(max(0, cx - half), cx + half))
    R = grid.RR[sl]
    src = R - El * El / np.maximum(R, 1e-4)
    w = np.clip((6 * Re * lf - R) / (4 * Re * lf), 0, 1)
    w = w * w * (3 - 2 * w)
    r_eff = np.maximum(R + w * (src - R), s0)
    L[sl] = np.log(r_eff + 1e-4)
    glow = float(st.get("glow", 0.0))
    Tw[sl] += (1.0 + 1.2 * st["tw"] + 3.0 * glow) * lf * w * (Re / np.maximum(R, Re)) ** 2   # en el drop se retuerce más
    # onda gravitacional: en cada liberación sale del horizonte una onda que estira y
    # comprime el espacio a su paso (dura lo que el destello, ~2.5 s)
    fr.ripple = None
    if glow > 0.03:
        ts = -2.5 * np.log(glow)                   # segundos desde la liberación
        d = grid.RR - (Re + 0.9 * ts)
        rip = ((0.07 * fr.sys.turbulence) * glow ** 0.5 * np.sin(2 * np.pi * d / 0.16) * np.exp(-(d / 0.22) ** 2)).astype(np.float32)
        L = L + np.log1p(rip)
        fr.ripple = 1 + rip
    fr.lensL, fr.lensT = L, Tw
    fr.lens_box = (sl, (r_eff / np.maximum(R, 1e-4)).astype(np.float32))
    return L


# ------------------------------------------------------------- estilos ----
def draw_particles(grid, fr):
    """Partículas sobre el túnel: una celda por (anillo, sector); viajan hacia afuera con el
    avance del túnel y crecen con la distancia. Cada celda recuerda si está encendida
    (se renueva cada 8 kicks) para que el viaje se perciba; en la tensión titila con los hats."""
    P, st, c, kc, sy = fr.P, fr.st, fr.c, fr.kc, fr.sys
    N = int(round(P.get("N", 48) * (1 + 0.5 * c["chaos"])))
    Kd = P.get("Kd", 7.5)
    u = Kd * lens(grid, fr) - st["ph"] * (Kd / 5.0)
    ri = np.floor(u).astype(np.int32)
    fr_ = u - ri
    a = np.mod((grid.TH + 0.35 * st["rot"] + fr.roll + fr.lensT)
               * (1 / (2 * np.pi)), 1.0) * N
    ai = np.floor(a).astype(np.int32)
    fa = a - ai
    nv = nerv(c)
    tkey = c["hat_n"] if nv > 0 else kc // 8
    r1 = hash32(ri + 5000, ai + 5000, tkey)
    r2 = hash32(ri + 9000, ai + 1000, kc // 8 + 7)
    p = P.get("p", 0.2)
    energy = 0.5 * c["low"] + 0.5 * c["mid"]
    if c["closed"]:
        pr = p * P.get("dens", 0.7) * sy.matter * 0.35 * ((1 - 0.9 * c["tens"]) * (1 - 0.5 * nv * (1 - c["hat"]))
                                                          + 1.6 * nv * c["hat"])      # fogonazo en cada hat
    else:
        pr = p * P.get("dens", 0.7) * sy.matter * (0.4 + 0.9 * energy + 0.8 * c["kick"])
    lit = r1 < pr
    flick = P.get("flicker", 0)
    if flick:
        lit &= hash32(ri + 77, ai + 33, fr.i // flick) < 0.7
    th_ = thin(c)
    th_ += (1 - th_) * nv * c["hat"]                # en el fogonazo vuelven a tamaño pleno
    sz = (P.get("dot", 0.30) + 0.06 * c["kick"]) * th_ + 0.08
    half = sz / 2
    # polvo: la perspectiva las agranda hacia afuera, pero nunca más que unos pocos píxeles
    # (nada de cuadrados grandes); celdas en píxeles de grilla, radial y angular
    cr = np.maximum(grid.RR * (grid.unit / Kd), 1e-3)
    ca = np.maximum(grid.RR * (grid.unit * 2 * np.pi / N), 1e-3)
    cap = 0.5 * P.get("px", 3.0) * grid.gs
    hr, ha = np.minimum(half, cap / cr), np.minimum(half, cap / ca)
    w = c.get("warp", 0.0)
    if w > 0:                                       # velocidad de la luz: se estiran en líneas hacia donde se viaja
        hr = np.minimum(0.5, hr + 0.45 * w)
    dr, da = np.abs(fr_ - 0.5), np.abs(fa - 0.5)
    inside = (dr < hr) & (da < ha)
    if P.get("plus"):                               # destellos en cruz, finos y cortos (como los de las estrellas)
        arm = min(0.5, half + 0.12 + 0.08 * c["kick"])
        ar, aa = np.minimum(arm, 7 * grid.gs / cr), np.minimum(arm, 7 * grid.gs / ca)
        tr_, ta = np.minimum(0.05, 0.5 * grid.gs / cr), np.minimum(0.05, 0.5 * grid.gs / ca)
        inside |= (((dr < ar) & (da < ta)) | ((da < aa) & (dr < tr_))) & (r1 < pr * 0.5)
    on = lit & inside
    acc = on & (r2 < (0.08 + 0.2 * c["chaos"]) * sy.accent)
    # profundidad: la mayoría tenues, pocas brillantes; el kick las enciende (no las agranda)
    r3 = hash32(ri + 13000, ai + 3000, kc // 8)
    I = np.where(on, 0.25 + 0.75 * r3 ** 2.2 + 0.3 * c["kick"], 0).astype(np.float32)
    return levels(I, acc)


def draw_arcs(grid, fr):
    """Anillos siempre partidos en arcos irregulares; cada anillo gira en sentido contrario al
    vecino, a su propia velocidad, y engrana medio segmento en cada kick.
    Pocos, gruesos, mucho negro; hueco en el centro (nada de diana)."""
    P, st, c, kc = fr.P, fr.st, fr.c, fr.kc
    K = P.get("K", 3.2)
    u = K * lens(grid, fr) - st["ph"]
    n = np.floor(u)
    f = u - n
    ni = n.astype(np.int32)
    # tabla por anillo (ids de anillo en un rango chico): sentido, velocidad, segmentos
    ids = np.arange(-64, 64, dtype=np.int32)
    hsp = hash32(ids + 300, np.zeros_like(ids) + 1, 0)
    hsg = hash32(ids + 600, np.zeros_like(ids) + 2, st["mut"])
    nseg = np.floor(5 + (5 + 8 * c["chaos"]) * hsg).astype(np.int32)       # 5..18 segmentos
    sgn = np.where(ids % 2 == 0, 1.0, -1.0)
    ang = sgn * (st["rot"] * (0.6 + 1.2 * hsp) * 2.2) + kc * np.pi / nseg * sgn
    ang = (ang + fr.roll).astype(np.float32)
    k = (ni + 64) % 128
    a = np.mod((grid.TH + ang[k] + fr.lensT) * (1 / (2 * np.pi)), 1.0) * nseg[k]
    sg = np.floor(a).astype(np.int32)
    fa = a - sg
    hseg = hash32(ni + 900, sg, st["mut"])
    keep = hseg < (0.55 + 0.15 * c["chaos"])
    keep &= (fa > 0.04) & (fa < 0.96)                                      # corte seco entre arcos
    w = st["wt"][ni % 32] * (P.get("dens", 0.7) * fr.sys.matter)
    if c["closed"]:
        on = (f < np.clip((0.08 + 0.10 * c["mid"]) * w * thin(c), 0.015, 0.5)) & erode(ni, c)
    else:
        duty = np.clip((0.22 + 0.30 * c["kick"]) * w, 0.05, 0.85)
        duty = np.minimum(duty, 0.30 * K / (grid.RR + 0.3))                # los de afuera no se engordan de más
        on = f < duty
    on &= keep
    acc = on & (hash32(ni + 1200, sg, kc // 8) < (0.14 + 0.22 * c["chaos"]) * fr.sys.accent)
    return on.astype(np.uint8) + acc.astype(np.uint8)


def _orbits(grid, fr, Lr, ang_src, K, lw_px, wl_scale=None):
    """Materia en órbita: carriles en log(r) (en la tensión se contraen hacia el agujero) con
    estelas que giran a velocidad orbital (las de adentro, más rápido; el kick las estira).
    Nada homogéneo: anillos densos y huecos, la mayoría tenues y pocas brillantes, más brillo
    cerca del agujero, grosores distintos, ondas espirales que deforman las órbitas en conjunto,
    y cada estela con la cabeza brillante y la cola que se apaga.
    Devuelve (intensidad 0..1, acento, id de carril)."""
    st, c, kc = fr.st, fr.c, fr.kc
    tens_c = c["tens"] if c["closed"] else 0.0
    scale = 1 - 0.35 * tens_c
    u = K * (Lr - np.float32(np.log(scale)))
    # ondas de densidad espirales (colectivas: todas las órbitas se deforman juntas)
    g = np.random.default_rng([fr.seed, 2024])
    f1, f2, f3 = g.uniform(0, 2 * np.pi, 3)
    tb = fr.sys.turbulence
    u = u + (0.18 * tb) * np.sin(2 * ang_src + f1 + 0.15 * st["rot"]) + (0.1 * tb) * np.sin(3 * ang_src - f2 - 0.1 * st["rot"])
    li = np.floor(u).astype(np.int32)
    fr_ = u - li
    ids = np.arange(-96, 96, dtype=np.int32)
    z = np.zeros_like(ids)
    r_l = np.exp((ids + 0.5) / K) * scale
    om = (0.25 / np.maximum(r_l, 0.04)) ** 1.2
    spd = 0.6 + 0.8 * hash32(ids + 40, z + 4, 0)
    lane_ang = (st["rot"] * 2.0 * om * spd).astype(np.float32)
    nseg = np.floor(1 + 6 * hash32(ids + 70, z + 6, st["mut"])).astype(np.int32)
    dens = np.clip(0.5 + 0.5 * np.sin(ids * 0.37 + f3) * np.sin(ids * 0.11 + f1), 0, 1)   # anillos y huecos
    bri = (0.18 + 0.95 * hash32(ids + 2000, z + 7, 0) ** 2.4) * (0.55 + 0.6 * dens) \
        * np.clip(0.3 / np.maximum(r_l, 0.02), 0.35, 2.2)                                   # tenues; cerca del agujero, mucho brillo
    thick = 0.35 + 0.8 * hash32(ids + 2200, z + 8, 0) ** 1.5
    k = np.clip(li + 96, 0, 191)
    a = np.mod((ang_src + lane_ang[k]) * (1 / (2 * np.pi)), 1.0) * nseg[k]
    si = np.floor(a).astype(np.int32)
    fa = a - si
    hl = hash32(li + 300, si, st["mut"])
    ln = np.minimum((0.08 + 0.8 * hl ** 1.5) * (1 + 1.4 * c["kick"]), 0.97)
    seg_on = (fa < ln) & (hash32(li + 500, si, st["mut"] // 2)
                          < (0.3 + 0.5 * dens[k]) * (1 + 0.3 * c["chaos"] + 0.8 * st["glow"]) * fr.sys.matter)
    wl = lw_px / (grid.hh / 2) * K / (grid.RR + 1e-3) if wl_scale is None else wl_scale
    wl = wl * thick[k]
    if c["closed"]:
        wl = wl * max(0.45, thin(c))
    on = (np.abs(fr_ - 0.5) < wl) & seg_on
    if c["closed"]:
        on &= erode(li, c, 4)
    q = np.minimum(fa / np.maximum(ln, 1e-3), 1)                                             # fuera del segmento no se usa
    along = (1 - q) ** 1.6 * np.clip(q / 0.07, 0, 1)                                          # cabeza que se funde -> cola
    I = np.where(on, bri[k] * along * (0.65 + 0.6 * c["kick"]) * (1 + 4.0 * st["glow"]), 0).astype(np.float32)
    if c["closed"]:
        I *= 0.7 + 0.5 * nerv(c) * c["hat"]
    acc = on & (hash32(li + 800, np.zeros_like(li) + 8, kc // 16) < (0.12 + 0.2 * c["chaos"]) * fr.sys.accent)
    return I, acc, li


def draw_streaks(grid, fr):
    """Estelas: materia en órbita alrededor del agujero, curvada por la lente."""
    K = fr.P.get("K", 14.0)
    L = lens(grid, fr)
    I, acc, _ = _orbits(grid, fr, L, grid.TH + fr.lensT + fr.roll, K, 0.55 * grid.tg)
    return levels(I, acc)


def draw_disk(grid, fr):
    """Disco de acreción visto casi de canto, en diagonal: una banda de estelas que cruza el
    agujero; la lente dobla la parte de atrás alrededor del horizonte (la imagen de Gargantua).
    El lado que se acerca, blanco y más brillante; el que se aleja, en el color de acento."""
    P, st = fr.P, fr.st
    K = P.get("K", 16.0)
    si_ = fr.sys.incl * P.get("open", 1.0)
    lens(grid, fr)
    sl, ratio = fr.lens_box
    XR, YR, RHO, PHI = grid.disk_coords(P.get("tilt", fr.sys.tilt), si_)
    rho, phi = RHO.copy(), PHI.copy()
    xs, ys = XR[sl] * ratio, YR[sl] * ratio / si_
    rho[sl] = np.hypot(xs, ys)
    phi[sl] = np.arctan2(ys, xs)
    if fr.ripple is not None:
        rho *= fr.ripple
    Re = float(st["rh"]) * 1.12
    I, acc, li = _orbits(grid, fr, np.log(rho + 1e-4), phi + fr.roll, K, 0.5 * grid.tg,
                         wl_scale=np.float32(0.10))
    I *= (rho > 2.0 * Re) & (rho < 2.2)
    I *= np.clip(1.25 - 0.6 * rho + 0.8 * st["glow"], 0.15, 1.0)  # hacia afuera, más tenue (en el drop se enciende entero)
    I *= 1.9 * np.clip(1 - 0.45 * XR, 0.55, 1.5)                  # doppler: brilla más el lado que se acerca
    # lado que se aleja (derecha) en acento; cada órbita cambia de color en un punto distinto
    doppler = hash32(li + 1500, np.zeros_like(li) + 9, 0) < np.clip(0.5 + 0.9 * XR, 0, 1)
    return levels(I, (I > 0) & (doppler | acc))


STYLES = {"dots": draw_particles, "arcs": draw_arcs, "streaks": draw_streaks, "disk": draw_disk}
