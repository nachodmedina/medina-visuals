"""La escucha: sobre un track sintético con una historia conocida (ver tests/synth.py)."""
import numpy as np
import pytest

from medina.analysis import analyze
from medina.audio import mute_report
from tests.synth import BEAT, BPM, SR, beat_times, expected, make_track


def _rise(synth):
    r = np.where(synth["A"]["rise"])[0]
    assert len(r) == 1, f"se esperaba una liberación, hay {len(r)}"
    return int(r[0])


def test_tempo(synth):
    assert abs(synth["bpm"] - synth["ex"]["bpm"]) < 0.05


def test_kicks_en_los_tramos_con_sub(synth):
    A, fps, ex = synth["A"], synth["fps"], synth["ex"]
    open_beats = [bt for bt in beat_times() if not (ex["filter_start"] <= bt < ex["release"])]
    frames = np.round(np.array(open_beats) * fps).astype(int)
    hits = sum(np.min(np.abs(A["onsets"] - f)) <= 1 for f in frames)
    assert hits >= 0.95 * len(frames)


def test_liberacion_en_el_beat_que_vuelve_el_sub(synth):
    assert abs(_rise(synth) - synth["ex"]["release"] * synth["fps"]) <= 1


def test_la_tension_crece_y_termina_en_la_liberacion(synth):
    A, fps, ex = synth["A"], synth["fps"], synth["ex"]
    r = _rise(synth)
    closed = (~A["sub_on"]) | A["break_zone"]
    start = int(np.argmax(closed[int(5 * fps):])) + int(5 * fps)   # primer tramo cerrado después de la intro
    assert abs(start / fps - ex["filter_start"]) < 1.0
    assert A["tension"][r - 1] > 0.95 and A["tension"][r] == 0
    assert np.all(np.diff(A["tension"][start:r]) >= 0)
    assert not closed[r:r + 2 * fps].any()


def test_respiracion_es_el_beat_anterior(synth):
    A, fps = synth["A"], synth["fps"]
    r = _rise(synth)
    b = np.where(A["breath"])[0]
    assert b[-1] == r - 1
    assert abs(len(b) - BEAT * fps) <= 1


def test_hats_en_corcheas_y_nada_antes(synth):
    A, fps, ex = synth["A"], synth["fps"], synth["ex"]
    on = np.diff(np.r_[0, A["hat_n"]]) > 0
    assert on[:int(ex["hats_from"] * fps)].sum() == 0
    rate = on[10 * fps:60 * fps].sum() / 50
    assert abs(rate - 2 / BEAT) < 0.5                   # dos por beat


def test_capa_nueva(synth):
    fps, ex = synth["fps"], synth["ex"]
    first = synth["A"]["layers"]["first"] / fps
    assert np.min(np.abs(first - ex["layer_at"])) < 2.0


def test_audio_mudo():
    assert mute_report(np.zeros(SR * 30, np.float32), SR)
    assert mute_report(np.full(SR * 30, 0.1, np.float32), SR) is None


@pytest.mark.xfail(strict=True, reason="falla conocida: el control de graves mira un solo cuadro; un kick "
                   "que cae justo antes del centro del cuadro no se cuenta (ver README, pendientes)")
def test_kicks_en_cualquier_fase_del_cuadro():
    t0 = 8 / 30 - 0.008                     # cada kick cae un poco antes del centro de un cuadro
    A = analyze(make_track(t0=t0), 30, (BPM - 3, BPM + 3))
    ex = expected(t0)
    open_beats = [bt for bt in beat_times(t0) if not (ex["filter_start"] <= bt < ex["release"])]
    frames = np.round(np.array(open_beats) * 30).astype(int)
    hits = sum(len(A["onsets"]) and np.min(np.abs(A["onsets"] - f)) <= 1 for f in frames)
    assert hits >= 0.95 * len(frames)
