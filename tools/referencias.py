"""Referencias para verificar que un cambio no altera la imagen (refactors, paso a GPU).

    .venv/bin/python tools/referencias.py guardar renders/_ref/base [adn|neutro]   # con la versión aprobada
    .venv/bin/python tools/referencias.py comparar renders/_ref/base               # después del cambio
    .venv/bin/python tools/referencias.py comparar renders/_ref/base gpu           # el motor de GPU, con tolerancia

Por cada track de tracks/: el análisis, cuadros sueltos (inicio, 20/40/60/80 %, cada liberación
+0.3 s y la respiración anterior) y un tramo de 3 s alrededor de la primera liberación.
Todo a 960x540 con el look luz. Informa diferencias exactas por cuadro.
La referencia recuerda con qué sistema estelar se guardó (adn por defecto; las viejas, neutro)
y `comparar` usa ese mismo.
"""
import glob
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from medina.analysis import ANALYSIS_SR, analyze  # noqa: E402
from medina.audio import detect_bpm, load_audio, seed_from_audio  # noqa: E402
from medina.output import draw_still  # noqa: E402
from medina.render import Renderer  # noqa: E402
from medina.score import build_score  # noqa: E402

W, H, FPS = 960, 540, 30


def moments(A):
    n = A["n"]
    at = {f"{p:02d}pct": int(n * p / 100) for p in (1, 20, 40, 60, 80)}
    for j, r in enumerate(np.where(A["rise"])[0]):
        at[f"lib{j}"] = int(min(n - 1, r + 0.3 * FPS))
        b = np.where(A["breath"][:r])[0]
        if len(b):
            at[f"resp{j}"] = int(b[len(b) // 2])
    return at


# el motor de GPU no es idéntico bit a bit: alcanza con que se vea igual
GPU_MEAN, GPU_FAR = 2.0, 0.5          # diferencia media (de 255) y % de píxeles con más de 60 de diferencia


def snapshot(path, system="adn", motor="cpu"):
    y = load_audio(path, ANALYSIS_SR)
    bpm = detect_bpm(path)
    seed = seed_from_audio(y)
    A = analyze(y, FPS, (bpm - 3, bpm + 3))
    score = build_score(A, "viaje", FPS, seed, system=system, bpm=bpm)
    if motor == "gpu":
        from medina.gpu import GPURenderer
        R = GPURenderer(score, W, H)
    else:
        R = Renderer(score, W, H)
    frames = {k: draw_still(R, i).copy() for k, i in moments(A).items()}
    rises = np.where(A["rise"])[0]
    chunk = []
    if len(rises):
        R.post.reset() if R.post else None
        f = np.zeros((H, W, 3), np.uint8)
        for i in range(max(0, rises[0] - 2 * FPS), min(A["n"], rises[0] + FPS)):
            chunk.append(R.draw(f, i).copy())
    signals = {k: v for k, v in A.items() if isinstance(v, np.ndarray)}
    return dict(bpm=bpm, seed=seed, system=system, signals=signals, frames=frames, chunk=np.array(chunk))


def _save(snap, path):
    arrays = {f"s_{k}": v for k, v in snap["signals"].items()}
    arrays.update({f"f_{k}": v for k, v in snap["frames"].items()})
    np.savez_compressed(path, bpm=snap["bpm"], seed=snap["seed"], system=snap["system"], chunk=snap["chunk"],
                        **arrays)


def _load(path):
    z = np.load(path)
    return dict(bpm=float(z["bpm"]), seed=int(z["seed"]), chunk=z["chunk"],
                system=str(z["system"]) if "system" in z.files else "neutro",
                signals={k[2:]: z[k] for k in z.files if k.startswith("s_")},
                frames={k[2:]: z[k] for k in z.files if k.startswith("f_")})


def _compare(ref, snap, motor="cpu"):
    bad = [k for k, v in ref["signals"].items() if k not in snap["signals"] or not np.array_equal(v, snap["signals"][k])]
    lines = [f"señales distintas: {bad}"] if bad else []
    if motor == "gpu":                      # con tolerancia: que se vea igual
        worst = (0.0, 0.0, "")
        for k, f in list(ref["frames"].items()) + [(f"tramo{j}", c) for j, c in enumerate(ref["chunk"])]:
            g = snap["frames"][k] if k in snap["frames"] else snap["chunk"][int(k[5:])]
            d = np.abs(f.astype(int) - g.astype(int))
            mean, far = d.mean(), 100 * (d.max(-1) > 60).mean()
            worst = max(worst, (mean, far, k))
            if mean > GPU_MEAN or far > GPU_FAR:
                lines.append(f"cuadro {k}: diferencia media {mean:.2f}, {far:.2f}% de píxeles con más de 60")
        return lines, f"equivalente (peor cuadro: {worst[2]}, media {worst[0]:.2f}, {worst[1]:.2f}% > 60)"
    for k, f in ref["frames"].items():
        if k not in snap["frames"]:
            lines.append(f"cuadro {k}: falta")
            continue
        d = np.abs(f.astype(int) - snap["frames"][k].astype(int))
        if d.max():
            lines.append(f"cuadro {k}: máx {d.max()}/255, {100 * (d.max(-1) > 0).mean():.3f}% de píxeles")
    if ref["chunk"].shape != snap["chunk"].shape:
        lines.append("tramo: distinta cantidad de cuadros")
    elif len(ref["chunk"]):
        d = np.abs(ref["chunk"].astype(int) - snap["chunk"].astype(int)).reshape(len(ref["chunk"]), -1)
        if d.max():
            lines.append(f"tramo: {(d.max(1) > 0).sum()}/{len(d)} cuadros distintos, máx {d.max()}/255")
    return lines, "idéntico"


def _job(args):
    cmd, folder, track, system, motor = args
    name = os.path.splitext(os.path.basename(track))[0]
    path = os.path.join(folder, f"{name}.npz")
    if cmd == "guardar":
        snap = snapshot(track, system)
        _save(snap, path)
        for k, f in snap["frames"].items():
            Image.fromarray(f).save(os.path.join(folder, f"{name}_{k}.png"))
        return name, [f"{len(snap['frames'])} cuadros + tramo de {len(snap['chunk'])}"], True
    if not os.path.exists(path):
        return name, ["no hay referencia guardada"], False
    ref = _load(path)
    lines, ok_msg = _compare(ref, snapshot(track, ref["system"], motor), motor)
    return name, lines or [ok_msg], not lines


def main():
    if len(sys.argv) not in (3, 4) or sys.argv[1] not in ("guardar", "comparar") \
            or sys.argv[3:] not in ([], ["adn"], ["neutro"], ["gpu"]) or (sys.argv[1], sys.argv[3:]) == ("guardar", ["gpu"]):
        sys.exit(__doc__)
    from concurrent.futures import ProcessPoolExecutor
    cmd, folder = sys.argv[1], sys.argv[2]
    motor = "gpu" if sys.argv[3:] == ["gpu"] else "cpu"
    system = sys.argv[3] if len(sys.argv) == 4 and motor == "cpu" else "adn"
    tracks = sorted(glob.glob(os.path.join(ROOT, "tracks", "*.wav")))
    if not tracks:
        sys.exit("No hay tracks en tracks/.")
    os.makedirs(folder, exist_ok=True)
    with ProcessPoolExecutor(max_workers=min(len(tracks), max(1, (os.cpu_count() or 2) // 3))) as ex:
        results = list(ex.map(_job, [(cmd, folder, t, system, motor) for t in tracks]))
    for name, lines, ok in results:
        print(f"  {name}: {lines[0]}")
        for ln in lines[1:]:
            print(f"      {ln}")
    if cmd == "comparar":
        ok_all = all(ok for _, _, ok in results)
        print("RESULTADO:", ("todo equivalente" if motor == "gpu" else "todo idéntico") if ok_all
              else "hay diferencias (ver arriba)")


if __name__ == "__main__":
    main()
