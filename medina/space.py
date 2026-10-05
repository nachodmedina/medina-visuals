"""Lo que comparten todos los estilos: el espacio profundo (estrellas), el silencio, los
fenómenos de cada capa del track y el agujero negro."""
import numpy as np

from .noise import hash32

PHENOMENA = ["cometa", "pulsar", "luna", "estrella", "meteoros", "jets"]


def box(idx, y0, y1, x0, x1, col):
    """Rectángulo recortado a la pantalla (sin índices negativos que 'dan la vuelta')."""
    h, w = idx.shape
    y0, y1 = max(0, int(y0)), min(h, int(y1))
    x0, x1 = max(0, int(x0)), min(w, int(x1))
    if y1 > y0 and x1 > x0:
        idx[y0:y1, x0:x1] = col


def line(idx, x0, y0, x1, y1, th, col):
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


def silence(idx, grid, c):
    """En el silencio (y en la respiración antes de cada liberación): negro y una sola línea."""
    if c["silent"]:
        idx[:] = 0
        h = idx.shape[0] // 2
        idx[h:h + grid.tg, :] = 1
    return idx


class Starfield:
    """Espacio profundo: estrellas lejanas, casi quietas, que rodean el agujero muy despacio.
    Se ven más en el vacío (tensión, intro) y casi nada cuando el track está lleno. Solo
    encienden píxeles negros: siempre quedan detrás de todo. Con los hats, algunas destellan.
    A la velocidad de la luz se recorre el espacio: cada estrella tiene su profundidad, se
    acerca (se abre hacia afuera, cada vez más rápido) y se estira en una línea hacia el centro."""

    def __init__(self, seed, n=900):
        n = int(n)
        g = np.random.default_rng([seed, 1717])
        self.n = n
        self.r = (2.1 * np.sqrt(g.random(n))).astype(np.float32)
        self.a = (g.random(n) * 2 * np.pi).astype(np.float32)
        self.big = g.random(n) < 0.12
        self.h = g.random(n).astype(np.float32)
        self.z = np.random.default_rng([seed, 1718]).uniform(0.35, 1.0, n).astype(np.float32)   # profundidad

    def draw(self, idx, grid, fr):
        st, c, i, n = fr.st, fr.c, fr.i, self.n
        w, tr = float(c.get("warp", 0.0)), float(st.get("tr", 0.0))
        void = 1.0 if c["closed"] else 0.25
        thr = 0.25 + 0.6 * void
        if w > 0:
            thr += 0.6 * w                                   # a la velocidad de la luz se ven casi todas
        vis = hash32(np.arange(n), np.zeros(n, np.int64) + 91, i // 9) < thr
        vis &= self.h < 0.75 + 0.25 * void
        t = i / fr.fps
        Re = float(st["rh"]) * 1.12
        r = self.r
        El = float(np.sqrt(Re * Re - 0.06 * Re * Re))
        if tr or w:                  # viajando: cada estrella a su profundidad, vuelven a entrar por el fondo
            z = self.Z0 + np.mod(self.z - self.Z0 - tr, self.ZS)
            R = self.r * self.z                              # distancia al eje del viaje (fija)
            r = R / z
        rl = (r + np.sqrt(r * r + 4 * El * El)) / 2          # lente puntual: la imagen se aparta del horizonte
        a = self.a + 0.012 * t + 0.08 * st["tw"] * (0.3 / (r + 0.05)) + fr.roll
        hh2 = grid.unit
        xs = (grid.ww / 2 + rl * np.cos(a) * hh2).astype(np.int32)
        ys = (grid.hh / 2 + rl * np.sin(a) * hh2).astype(np.int32)
        px = max(1, int(round(grid.gs)))                    # 1 píxel de salida
        if w > 0.02:
            self._streaks(idx, grid, fr, np.where(vis)[0], z, R, El, t, w)
        for x, y_, big in zip(xs[vis], ys[vis], self.big[vis]):
            sz = px * (2 if big else 1)
            if 0 <= x < grid.ww - sz and 0 <= y_ < grid.hh - sz:
                sub = idx[y_:y_ + sz, x:x + sz]
                sub[sub == 0] = 1
        # destellos: con cada hat algunas estrellas se encienden en cruz (más en el vacío)
        if c["hat"] > 0.3:
            gl = vis & (hash32(np.arange(n), np.zeros(n, np.int64) + 93, c["hat_n"]) < 0.012 + 0.03 * void)
            arm = int((4 + 14 * c["hat"]) * grid.gs)
            for x, y_ in zip(xs[gl], ys[gl]):
                if arm < x < grid.ww - arm and arm < y_ < grid.hh - arm:
                    for sub in (idx[y_:y_ + px, x - arm:x + arm + px], idx[y_ - arm:y_ + arm + px, x:x + px]):
                        sub[sub == 0] = 1
        return idx

    Z0, ZS = 0.35, 0.65              # profundidades del viaje (de cerca a lejos)

    def _streaks(self, idx, grid, fr, sel, z, R, El, t, w):
        """La estela de cada estrella: de donde estaba hace un instante (más al fondo) a donde
        está ahora; más larga cuanto más cerca pasa. Fina y tenue, con la punta más brillante."""
        st = fr.st
        zs, Rs = z[sel], R[sel]
        s = 0.16 * w                                         # recorrido de la estela, en profundidad
        unit = grid.unit
        L_px = Rs * (1 / zs - 1 / (zs + s)) * unit
        K = int(min(900, max(2, float(L_px.max()) + 2)))
        tau = np.linspace(0, 1, K, dtype=np.float32)[None, :]
        r = (Rs[:, None] / (zs[:, None] + tau * s)).ravel()
        rl = (r + np.sqrt(r * r + 4 * El * El)) / 2
        a = np.repeat(self.a[sel], K) + 0.012 * t + 0.08 * st["tw"] * (0.3 / (r + 0.05)) + fr.roll
        xs = (grid.ww / 2 + rl * np.cos(a) * unit).astype(np.int32)
        ys = (grid.hh / 2 + rl * np.sin(a) * unit).astype(np.int32)
        lev = np.broadcast_to(np.where(tau < 0.2, 5, np.where(tau < 0.55, 4, 3)).astype(np.uint8),
                              (len(sel), K)).ravel()
        hh, ww = idx.shape
        m = (xs >= 0) & (xs < ww) & (ys >= 0) & (ys < hh)
        xs, ys, lev = xs[m], ys[m], lev[m]
        free = idx[ys, xs] == 0
        idx[ys[free], xs[free]] = lev[free]


class Phenomena:
    """Un fenómeno del espacio por cada capa que entró al track (cometa, pulsar, luna, estrella,
    meteoros, jets), reaccionando a SU actividad, no al kick. La capa más reciente va en color de
    acento; las viejas, solo en sus golpes fuertes. Qué fenómeno le toca a cada capa, posiciones
    y trayectorias salen de la semilla del audio (reproducibles)."""

    def __init__(self, seed, prefer=()):
        self.seed = seed
        kinds = list(np.random.default_rng([seed, 404]).permutation(PHENOMENA))
        # los fenómenos que prefiere el sistema estelar van primero (les tocan las primeras capas)
        self.kinds = [k for k in prefer if k in kinds] + [k for k in kinds if k not in prefer]

    def draw(self, idx, grid, fr, layers):
        Ly = layers
        if not Ly or not Ly["order"]:
            return idx
        i, seed = fr.i, self.seed
        hh, ww = idx.shape
        active = [k for k in Ly["order"] if Ly["on"][i, k]]
        if not active:
            return idx
        newest = max(active, key=lambda k: Ly["first"][k])
        kinds = self.kinds
        R0 = grid.unit                               # unidad: media altura (a la distancia de la cámara)
        cx, cy = ww / 2, hh / 2
        fine = max(1, int(round(grid.gs)))            # 1 píxel de salida
        bold = 2 * fine
        t = i / fr.fps
        rh = float(fr.st["rh"]) * 1.12
        P = lambda r, ang: (cx + r * np.cos(ang) * R0, cy + r * np.sin(ang) * R0)
        # sin repetir tipo: el orden de entrada elige el fenómeno
        for rank, k in enumerate(Ly["order"]):
            if k not in active:
                continue
            a = float(Ly["act"][i, k])
            since = (i - Ly["first"][k]) / fr.fps
            if since < 1.0:
                a = max(a, 1.0)                    # presentación: entra fuerte
            thr = 0.45 if k == newest else 0.9      # las capas viejas solo en sus golpes fuertes
            if a < thr:
                continue
            a = min(a, 1.3)
            col = 2 if k == newest else 1
            kind = kinds[rank % len(kinds)]
            gk = np.random.default_rng([seed, 606, k])      # constantes de esta capa
            if kind == "cometa":                   # cruza la pantalla; cola que se alarga con la capa
                per = 7.0 + 4.0 * gk.random()
                ev, ph = int(t // per), (t % per) / per
                ge = np.random.default_rng([seed, 607, k, ev])
                th0 = ge.uniform(0, 2 * np.pi)
                th1 = th0 + np.pi + ge.uniform(-0.7, 0.7)
                (x0, y0), (x1, y1) = P(2.0, th0), P(2.0, th1)
                hx, hy = x0 + (x1 - x0) * ph, y0 + (y1 - y0) * ph
                L = np.hypot(x1 - x0, y1 - y0) + 1e-6
                tl = (0.15 + 0.35 * a) * R0
                line(idx, hx, hy, hx - (x1 - x0) / L * tl, hy - (y1 - y0) / L * tl, fine, col)
                box(idx, hy - bold, hy + bold, hx - bold, hx + bold, col)
            elif kind == "pulsar":                 # punto lejano que titila con dos haces que giran
                px_, py_ = P(gk.uniform(0.7, 1.4), gk.uniform(0, 2 * np.pi))
                sz = int(fine * (1 + 2 * a))
                box(idx, py_ - sz, py_ + sz, px_ - sz, px_ + sz, col)
                ang = t * 2 * np.pi * (0.35 + 0.3 * gk.random())
                bl = (0.06 + 0.22 * a) * R0
                for sgn in (1, -1):
                    line(idx, px_, py_, px_ + sgn * np.cos(ang) * bl, py_ + sgn * np.sin(ang) * bl, fine, col)
            elif kind == "luna":                   # cuerpo chico en órbita; se esconde detrás del agujero
                Ro = gk.uniform(0.35, 0.65)
                ang = t * 0.9 * (0.25 / Ro) ** 1.5 + gk.uniform(0, 2 * np.pi)
                mx, my = cx + Ro * np.cos(ang) * R0, cy + Ro * np.sin(ang) * 0.35 * R0
                behind = np.sin(ang) < 0 and abs(mx - cx) < rh * R0 * 1.3
                if not behind:
                    sz = int(fine * (2 + 3 * a))
                    box(idx, my - sz, my + sz, mx - sz, mx + sz, col)
            elif kind == "estrella":               # estrella fija que se enciende en cruz
                sx_, sy_ = P(gk.uniform(0.5, 1.5), gk.uniform(0, 2 * np.pi))
                arm = (0.02 + 0.12 * a) * R0
                line(idx, sx_ - arm, sy_, sx_ + arm, sy_, fine, col)
                line(idx, sx_, sy_ - arm, sx_, sy_ + arm, fine, col)
                box(idx, sy_ - fine, sy_ + 2 * fine, sx_ - fine, sx_ + 2 * fine, col)
            elif kind == "meteoros":               # estelas cortas que caen hacia el agujero
                gm = np.random.default_rng([seed, 608, k, int(t // 0.5)])
                for _ in range(1 + int(3 * a)):
                    r0, ang = gm.uniform(0.8, 1.9), gm.uniform(0, 2 * np.pi)
                    ln = gm.uniform(0.05, 0.16) * (0.6 + 0.6 * a)
                    (x0, y0), (x1, y1) = P(r0, ang), P(max(r0 - ln, 0.05), ang + 0.04)
                    line(idx, x0, y0, x1, y1, fine, col)
            elif kind == "jets":                   # chorros polares del agujero: se alargan con la capa
                jl = (0.12 + 0.55 * a) * R0
                e0 = rh * R0 * 1.05
                for sgn in (1, -1):
                    line(idx, cx, cy + sgn * e0, cx, cy + sgn * (e0 + jl), fine, col)
        return idx


def horizon(idx, grid, fr):
    """El agujero negro: disco negro en la grilla (late con los graves y crece con la tensión).
    Devuelve también dónde cae en la salida (x, y, radio, con el encuadre de la cámara) para el
    resplandor difuso que agrega el post."""
    rh = float(fr.st["rh"]) * 1.12
    half = int(rh * grid.unit) + 2
    cy, cx = grid.hh // 2, grid.ww // 2
    sl = (slice(max(0, cy - half), cy + half), slice(max(0, cx - half), cx + half))
    idx[sl][grid.RR[sl] < rh] = 0
    x0, y0, x1, y1 = fr.box or (0, 0, grid.ww, grid.hh)
    sx, sy = grid.tw / (x1 - x0), grid.th / (y1 - y0)
    return idx, ((cx - x0) * sx, (cy - y0) * sy, rh * grid.unit * sy)
