"""El motor de GPU: se ve igual que el de la CPU, los cuadros sueltos son independientes y un
tramo sale igual que el video completo. Si no hay moderngl o contexto de OpenGL, se saltean."""
import numpy as np
import pytest

from medina.output import draw_still
from tests.conftest import small_renderer

pytest.importorskip("moderngl")

MOMENTS = (5.0, 38.0, 40.47, 45.0, 50.0)       # intro, final de la tensión, liberación, drop


def gpu_renderer(score, look="luz"):
    from medina.gpu import GPURenderer
    try:
        return GPURenderer(score, 320, 180, ss=2, look=look)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"sin contexto de OpenGL: {e}")


@pytest.mark.parametrize("t", MOMENTS)
def test_se_ve_igual_que_la_cpu(any_score, t):
    i = int(round(t * any_score.fps))
    a = draw_still(small_renderer(any_score), i)
    b = draw_still(gpu_renderer(any_score), i)
    d = np.abs(a.astype(int) - b.astype(int))
    assert d.mean() < 3.0 and (d.max(-1) > 60).mean() < 0.005, f"media {d.mean():.2f}"


def test_cuadros_sueltos_independientes(any_score):
    R = gpu_renderer(any_score)
    solo = draw_still(R, 1500)
    draw_still(R, 1200)
    assert np.array_equal(solo, draw_still(R, 1500))


def test_tramo_igual_al_video_completo(any_score):
    R = gpu_renderer(any_score)
    i = 45 * any_score.fps
    frame = np.zeros((180, 320, 3), np.uint8)
    for j in range(0, i + 1):
        R.draw(frame, j)
    full = frame.copy()
    d = np.abs(full.astype(int) - draw_still(R, i).astype(int))
    assert d.max() <= 2 and (d > 0).mean() < 0.001


def test_sin_post(any_score):
    f = draw_still(gpu_renderer(any_score, look="seco"), 50 * any_score.fps)
    assert f.shape == (180, 320, 3) and f.mean() > 0


# ------------------------------------------------------------- el agujero negro en 3D ----
@pytest.fixture(scope="module")
def hole_score(synth):
    from medina.score import build_score
    from tests.conftest import FPS, SEED
    return build_score(synth["A"], "agujero", FPS, SEED, system="adn", bpm=synth["bpm"])


def test_agujero_oscuro_y_con_luz(hole_score):
    R = gpu_renderer(hole_score)
    f = draw_still(R, 50 * hole_score.fps)
    lum = f.mean(-1)
    assert np.median(lum) < 15 and (lum > 100).mean() > 0.0005        # mucho negro, algo de luz
    # el centro (la sombra) es negro absoluto
    assert f[80:100, 150:170].max() < 10


def test_agujero_silencio_es_negro(synth, hole_score):
    b = np.where(synth["A"]["breath"])[0]
    f = draw_still(gpu_renderer(hole_score), int(b[len(b) // 2]))
    assert ((f.mean(-1) > 40).mean()) < 0.03


def test_agujero_tramo_igual_al_video_completo(hole_score):
    R = gpu_renderer(hole_score)
    i = 45 * hole_score.fps
    frame = np.zeros((180, 320, 3), np.uint8)
    for j in range(0, i + 1):
        R.draw(frame, j)
    d = np.abs(frame.astype(int) - draw_still(R, i).astype(int))
    assert d.max() <= 2 and (d > 0).mean() < 0.001


def test_dilatacion_del_tiempo_vuelve_a_la_musica(synth, hole_score):
    """En la liberación el tiempo de la imagen se detiene, y después acelera hasta recuperar lo perdido."""
    R = gpu_renderer(hole_score)
    hm = R._hole_motion()
    fps, r = hole_score.fps, int(np.where(synth["A"]["rise"])[0][0])
    assert hm["speed"][r + 6] < 0.3                                   # quieto
    assert hm["echo_k"][r + 6] > 0.2                                  # solo el eco de luz se mueve
    tt = np.cumsum(hm["speed"]) / fps
    t = np.arange(len(tt)) / fps
    assert abs(tt[r + 5 * fps] - t[r + 5 * fps]) < 0.05               # vuelve al tiempo de la música
