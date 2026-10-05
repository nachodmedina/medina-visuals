"""El sistema estelar: sale del ADN del track; el neutro reproduce el motor sin ADN."""
import numpy as np
from pytest import approx

from medina.emotion import DNA_AXES
from medina.output import draw_still
from medina.space import PHENOMENA, Phenomena
from medina.system import PREFER, StarSystem, describe
from tests.conftest import SEED, small_renderer

LO = dict.fromkeys(DNA_AXES, 0.0)
HI = dict.fromkeys(DNA_AXES, 1.0)


def test_neutro_son_los_valores_de_siempre():
    s = StarSystem.neutral()
    assert (s.hole, s.lens, s.tilt, s.incl, s.orbit, s.matter, s.stars) == (0.07, 1.0, -24.0, 0.21, 1.0, 1.0, 900)
    assert (s.light, s.grain, s.trail, s.distance, s.shake, s.accent) == (1.0,) * 6
    assert s.violet == (125, 55, 200) and s.phenomena == () and not s.fast_journey and s.warp == 0


def test_rangos_del_adn():
    lo, hi = StarSystem.from_axes(LO), StarSystem.from_axes(HI)
    assert lo.hole == approx(0.055) and hi.hole == approx(0.12)  # vorágine: más masa
    assert lo.incl < hi.incl                                  # y la elipse del disco se abre
    assert -8 == lo.tilt > hi.tilt == -40                    # caos: disco más inclinado
    assert (lo.stars, hi.stars) == (300, 1800)               # vacío: más espacio profundo
    assert (lo.distance, hi.distance) == (approx(0.95), approx(1.3))
    assert lo.violet < hi.violet                             # luz: del violeta profundo al lavanda
    assert not lo.fast_journey and hi.fast_journey
    assert len(describe(LO)) == len(describe(HI)) == 15


def test_fenomenos_preferidos():
    ax = dict(LO, vacio=1.0, caos=0.8)
    s = StarSystem.from_axes(ax)
    assert s.phenomena == PREFER["vacio"] + PREFER["caos"]
    assert set(s.phenomena) <= set(PHENOMENA)
    ph = Phenomena(SEED, s.phenomena)
    assert ph.kinds[:3] == list(s.phenomena) and sorted(ph.kinds) == sorted(PHENOMENA)
    assert Phenomena(SEED).kinds == Phenomena(SEED, ()).kinds   # sin preferencias: el orden de la semilla


def test_el_sintetico_tiene_su_sistema(adn_score):
    assert set(adn_score.dna) == set(DNA_AXES)
    assert adn_score.system == StarSystem.from_axes(adn_score.dna)
    assert adn_score.system != StarSystem.neutral()
    assert (adn_score.accent == adn_score.system.violet).all()      # su tono de violeta
    assert set(adn_score.emotions) >= {"miedo", "fuerza", "soledad"}


def test_el_sistema_cambia_la_imagen(score, adn_score):
    a = draw_still(small_renderer(score), 50 * score.fps)
    b = draw_still(small_renderer(adn_score), 50 * adn_score.fps)
    assert not np.array_equal(a, b)


def test_velocidad_de_la_luz(synth, score, adn_score):
    """El salto es el escape: después de la liberación, nunca en la tensión; el neutro no la tiene."""
    A, fps = synth["A"], synth["fps"]
    assert score.warp is None and all(f.st["tr"] == 0 for f in score.frames)
    w = adn_score.warp
    assert w.shape == (A["n"],) and 0 <= w.min() and w.max() <= 1
    closed = ~A["sub_on"].astype(bool) | A["break_zone"].astype(bool)
    r = int(round(synth["ex"]["release"] * fps))
    tens = closed & (np.arange(A["n"]) < r)
    assert w[r - 5 * fps:r].max() < 0.01 and w[tens].mean() < 0.05  # en la tensión se frena (y al final, nada)
    assert w[r + fps:r + 3 * fps].mean() > 0.5                       # el salto
    assert w[r + 3 * fps:r + 3 * fps + 1] > w[r + 15 * fps]         # y se asienta en el crucero
    tr = [f.st["tr"] for f in adn_score.frames]
    assert tr[r - 1] == tr[r - 5 * fps] and tr[r + 3 * fps] > tr[r] + 0.5 * 2 * 0.9 * 0.5   # quieto en la tensión; viaja al escapar
