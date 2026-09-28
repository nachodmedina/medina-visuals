"""La partitura: determinista, el viaje cambia solo en liberaciones, cámara y paletas acotadas."""
import pickle

import numpy as np

from medina.presets import RED, VIOLET
from medina.score import build_score
from tests.conftest import FPS, SEED


def _key(fs):
    st = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in fs.st.items()}
    return fs.preset, st, fs.c, fs.kc


def test_misma_pieza_misma_partitura(synth, score):
    other = build_score(synth["A"], "viaje", FPS, SEED)
    assert [_key(a) for a in score.frames] == [_key(b) for b in other.frames]
    for k in score.camera:
        assert np.array_equal(score.camera[k], other.camera[k])


def test_se_puede_guardar_y_recuperar(score):
    back = pickle.loads(pickle.dumps(score))
    assert [_key(a) for a in back.frames] == [_key(b) for b in score.frames]


def test_el_viaje_cambia_solo_en_liberaciones(synth, score):
    rise = set(np.where(synth["A"]["rise"])[0].tolist())
    names = [f.preset for f in score.frames]
    changes = [i for i in range(1, len(names)) if names[i] != names[i - 1]]
    assert set(changes) <= rise


def test_camara_acotada(score):
    C = score.camera
    assert 1.0 <= C["zoom"].min() and C["zoom"].max() <= 1.35
    assert np.abs(C["roll"]).max() <= np.deg2rad(6)
    assert np.hypot(C["dx"], C["dy"]).max() <= 25          # píxeles de 1080p


def test_paletas(synth):
    A = synth["A"]
    v = build_score(A, "viaje", FPS, SEED, palette="violeta").accent
    assert (v == VIOLET).all()
    vr = build_score(A, "viaje", FPS, SEED, palette="violeta_rojo").accent
    is_red = (vr == RED).all(1)
    assert ((vr == VIOLET).all(1) | is_red).all()
    assert is_red.any() and (A["rel_kick"][is_red] >= 0).all()   # rojo solo dentro de las liberaciones


def test_sin_camara(synth):
    assert build_score(synth["A"], "viaje", FPS, SEED, camera=False).camera is None
