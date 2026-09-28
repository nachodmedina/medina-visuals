"""Arma la página del mapa emocional con los tracks de tracks/ (para validar la capa emocional).

    .venv/bin/python tools/mapa_emocional.py            # -> renders/mapa-emocional.html

La página publicada guarda las respuestas en su base de datos (colección "respuestas").
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from medina import emotion as em  # noqa: E402
from medina.analysis import ANALYSIS_SR, analyze, key_name  # noqa: E402
from medina.audio import detect_bpm, load_audio  # noqa: E402

FPS = 30


def track_data(path, key):
    name = os.path.splitext(os.path.basename(path))[0]
    bpm = detect_bpm(path)
    A = analyze(load_audio(path, ANALYSIS_SR), FPS, (bpm - 3, bpm + 3))
    M = em.emotional_map(A, FPS, bpm)
    n = A["n"]
    per_s = lambda x: [round(float(v), 2) for v in x[:(n // FPS) * FPS].reshape(-1, FPS).mean(1)]
    k = int(np.bincount(A["key"]).argmax())
    tono, camelot = key_name(k)
    return dict(
        key=key, name=name, bpm=round(bpm, 2), dur=round(n / FPS, 1), tono=tono, camelot=camelot,
        releases=[round(r / FPS, 2) for r in np.where(A["rise"])[0]],
        curves={**{e: per_s(M["emotions"][e]) for e in em.EMOTIONS},
                "energia": per_s(M["features"]["energia"]), "tension": per_s(M["features"]["tension"])},
        sections=[dict(start=round(s["start"], 2), end=round(s["end"], 2), principales=s["principales"],
                       valores={e: round(v, 2) for e, v in s["emociones"].items()},
                       tension=s["tension"], liberacion=s["liberacion"]) for s in M["sections"]],
        axes={a: round(v, 2) for a, v in M["axes"].items()},
        system=[dict(parametro=a, valor=b, eje=c, expresa=d) for a, b, c, d in M["system"]],
    )


def main():
    tracks = sorted(glob.glob(os.path.join(ROOT, "tracks", "*.wav")))
    if not tracks:
        sys.exit("No hay tracks en tracks/.")
    # ids estables (las respuestas guardadas los usan) y el mismo orden que la página publicada
    keys = {"Untitled6 12-Audio": "u6", "Untitled 9-Audio": "u9", "Untitled999 12-Audio": "u999"}
    order = list(keys)
    base = lambda p: os.path.splitext(os.path.basename(p))[0]
    tracks.sort(key=lambda p: (order.index(base(p)) if base(p) in order else len(order), base(p)))
    data = dict(emotions=list(em.EMOTIONS), axes=list(em.DNA_AXES), tracks=[])
    for i, p in enumerate(tracks):
        name = os.path.splitext(os.path.basename(p))[0]
        data["tracks"].append(track_data(p, keys.get(name, f"t{i}")))
        print("  ", name)
    tpl = open(os.path.join(ROOT, "tools", "mapa_emocional.html")).read()
    out = os.path.join(ROOT, "renders", "mapa-emocional.html")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        fh.write(tpl.replace("__DATA__", json.dumps(data, ensure_ascii=False)))
    print("  ->", out)


if __name__ == "__main__":
    main()
