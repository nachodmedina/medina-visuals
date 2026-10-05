import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from medina.analysis import analyze  # noqa: E402
from medina.audio import detect_bpm  # noqa: E402
from medina.render import Renderer  # noqa: E402
from medina.score import build_score  # noqa: E402
from tests.synth import expected, make_track, write_wav  # noqa: E402

FPS = 30
SEED = 7


def listen_synth(folder):
    """El track sintético escuchado por el motor, igual que un track real (desde el archivo)."""
    y = make_track()
    path = os.path.join(folder, "synth.wav")
    write_wav(path, y)
    bpm = detect_bpm(path)
    A = analyze(y, FPS, (bpm - 3, bpm + 3))
    return dict(y=y, path=path, bpm=bpm, A=A, ex=expected(), fps=FPS)


@pytest.fixture(scope="session")
def synth(tmp_path_factory):
    return listen_synth(str(tmp_path_factory.mktemp("synth")))


@pytest.fixture(scope="session")
def score(synth):
    """El motor sin ADN (sistema neutro): contra esta partitura se comparan los cuadros de referencia."""
    return build_score(synth["A"], "viaje", FPS, SEED, system="neutro")


@pytest.fixture(scope="session")
def adn_score(synth):
    """El sistema estelar que sale del ADN del sintético."""
    return build_score(synth["A"], "viaje", FPS, SEED, system="adn", bpm=synth["bpm"])


@pytest.fixture(scope="session", params=["neutro", "adn"])
def any_score(request):
    """Para lo que tiene que valer con cualquier sistema estelar."""
    return request.getfixturevalue("score" if request.param == "neutro" else "adn_score")


def small_renderer(score, look="luz"):
    """Render chico (320x180, con supersampling) para tests rápidos."""
    return Renderer(score, 320, 180, ss=2, look=look)
