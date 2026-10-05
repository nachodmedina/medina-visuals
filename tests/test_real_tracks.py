"""Integración con los tracks reales de tracks/ (no están en el repo: si faltan, se saltean).
Los valores fueron validados a mano contra el espectrograma y el sub medido del audio."""
import os

import numpy as np
import pytest

from medina.analysis import ANALYSIS_SR, analyze
from medina.audio import detect_bpm, load_audio

TRACKS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tracks")
VALIDATED = {
    "Untitled6 12-Audio.wav":   dict(bpm=138.00, rises=[1670, 2088, 3340, 4996]),
    "Untitled 9-Audio.wav":     dict(bpm=142.02, rises=[1622, 3649, 6908, 8111]),
    "Untitled999 12-Audio.wav": dict(bpm=136.00, rises=[423, 1694, 2118, 2541, 2965, 4658, 5082, 5930]),
    # 0:16 y 4:34 son golpes sueltos de kick dentro del filtrado: no son liberaciones
    "Untitled13 14-Audio.wav":  dict(bpm=138.00, rises=[836, 1253, 1827, 2088, 2505, 2922, 3340, 5009, 5844,
                                                        6261, 7096, 8349]),
}


@pytest.mark.parametrize("name", sorted(VALIDATED))
def test_track_real(name):
    path = os.path.join(TRACKS_DIR, name)
    if not os.path.exists(path):
        pytest.skip(f"no está {name} en tracks/")
    v = VALIDATED[name]
    bpm = detect_bpm(path)
    assert abs(bpm - v["bpm"]) < 0.05
    A = analyze(load_audio(path, ANALYSIS_SR), 30, (bpm - 3, bpm + 3))
    assert np.where(A["rise"])[0].tolist() == v["rises"]
    for r in v["rises"]:
        assert A["tension"][r] == 0              # la tensión termina en la liberación
        assert A["breath"][r - 1]                # y el beat anterior es la respiración
