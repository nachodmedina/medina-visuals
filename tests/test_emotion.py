"""La capa emocional sobre el track sintético: la tensión se siente distinta de la liberación."""
import numpy as np

from medina import emotion as em


def test_mapa_completo_y_acotado(synth):
    M = em.emotional_map(synth["A"], synth["fps"], synth["bpm"])
    n = synth["A"]["n"]
    for k in em.EMOTIONS:
        assert M["emotions"][k].shape == (n,)
        assert 0 <= M["emotions"][k].min() and M["emotions"][k].max() <= 1
    assert set(M["axes"]) == set(em.DNA_AXES) and all(0 <= v <= 1 for v in M["axes"].values())
    assert len(M["system"]) >= 10
    secs = M["sections"]
    assert secs[0]["start"] == 0 and abs(secs[-1]["end"] - n / synth["fps"]) < 1e-6
    assert all(abs(a["end"] - b["start"]) < 1e-6 for a, b in zip(secs[:-1], secs[1:]))


def test_la_tension_no_se_siente_como_la_liberacion(synth):
    E = em.emotional_map(synth["A"], synth["fps"], synth["bpm"])["emotions"]
    fps, r = synth["fps"], int(round(synth["ex"]["release"] * synth["fps"]))
    late = slice(r - 6 * fps, r - fps)            # final de la tensión
    after = slice(r + 2 * fps, r + 12 * fps)      # después de la liberación
    assert E["fuerza"][after].mean() > E["fuerza"][late].mean() + 0.2
    dark = np.maximum(E["miedo"], E["incertidumbre"])
    assert dark[late].mean() > dark[after].mean() + 0.1
    assert E["esperanza"][r:r + 4 * fps].mean() > E["esperanza"][late].mean()   # alivio en la liberación
