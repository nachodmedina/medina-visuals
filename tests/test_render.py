"""El dibujo: cuadros bien formados, mucho negro, la respiración a negro, cuadros sueltos
independientes, tramos iguales al video completo y cuadros de referencia."""
import os

import numpy as np
import pytest
from PIL import Image

from medina.output import draw_still
from tests.conftest import small_renderer

GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "golden")
# momentos del sintético: intro abierta, final de la tensión, 2 cuadros después de la liberación, capa nueva
GOLDEN = {"intro": 5.0, "tension": 38.0, "liberacion": 40.47, "capa": 50.0}


def golden_frames(score):
    R = small_renderer(score)
    return {k: draw_still(R, int(round(t * score.fps))) for k, t in GOLDEN.items()}


def test_cuadro_bien_formado_y_oscuro(any_score):
    f = draw_still(small_renderer(any_score), 50 * any_score.fps)
    assert f.shape == (180, 320, 3) and f.dtype == np.uint8
    lum = f.mean(-1)
    assert np.median(lum) < 12              # mucho negro
    assert (lum > 128).mean() > 0.001       # pero algo encendido


def test_la_respiracion_es_negro(synth, any_score):
    b = np.where(synth["A"]["breath"])[0]
    f = draw_still(small_renderer(any_score), int(b[len(b) // 2]))
    lum = f.mean(-1)
    assert (lum > 40).mean() < 0.03         # solo la línea (y la firma, apenas)


def test_cuadros_sueltos_independientes(any_score):
    R = small_renderer(any_score)
    solo = draw_still(R, 1500)
    draw_still(R, 1200)                      # otro cuadro antes no debe cambiar nada
    again = draw_still(R, 1500)
    assert np.array_equal(solo, again)


def test_tramo_igual_al_video_completo(any_score):
    """Un tramo arranca con la estela precalentada 3 s (o más, si el sistema la alarga):
    debe coincidir con el video completo."""
    R = small_renderer(any_score)
    i = 45 * any_score.fps
    frame = np.zeros((180, 320, 3), np.uint8)
    for j in range(0, i + 1):                # video completo desde el cuadro 0
        R.draw(frame, j)
    full = frame.copy()
    chunk = draw_still(R, i)
    d = np.abs(full.astype(int) - chunk.astype(int))
    assert d.max() <= 2 and (d > 0).mean() < 0.001


@pytest.fixture(scope="module")
def golden(score):
    return golden_frames(score)


def test_sin_post(any_score):
    f = draw_still(small_renderer(any_score, look="seco"), 50 * any_score.fps)
    assert f.shape == (180, 320, 3)


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_cuadros_de_referencia(golden, name):
    """Compara contra los PNG de tests/golden (generados con tests/make_golden.py).
    Tolerancia chica para diferencias numéricas entre versiones de numpy/scipy."""
    path = os.path.join(GOLDEN_DIR, f"{name}.png")
    if not os.path.exists(path):
        pytest.skip("sin cuadros de referencia: correr python -m tests.make_golden")
    ref = np.asarray(Image.open(path)).astype(int)
    d = np.abs(golden[name].astype(int) - ref)
    assert d.mean() < 0.05 and (d > 3).mean() < 0.001, f"{name}: media {d.mean():.4f}, máx {d.max()}"
