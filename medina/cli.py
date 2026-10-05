"""Línea de comandos: escuchar el track, armar la partitura y exportar."""
import argparse
import os
import pickle
import shutil
import sys
import time

from .analysis import ANALYSIS_SR, analyze, key_name
from .audio import detect_bpm, load_audio, mute_report, seed_from_audio
from .output import encoder, render_parallel, render_stills, render_video
from .presets import LOOKS, PALETTES, PRESETS
from .render import Renderer
from .score import build_score


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="brutal_viz.py",
                                 description="Visuales audio-reactivas de MED1NA: el track decide el viaje.")
    ap.add_argument("audio", nargs="?", help="archivo de audio (wav, aiff, flac, mp3...)")
    ap.add_argument("-o", "--output", default="visual.mp4", help="video de salida (o base de los PNG con --stills)")
    ap.add_argument("--preset", default=None, choices=sorted(PRESETS),
                    help="agujero (por defecto con GPU: el agujero negro en 3D), viaje (el track elige el "
                         "estilo; por defecto sin GPU) o un estilo fijo")
    ap.add_argument("--look", default="luz", choices=sorted(LOOKS),
                    help="acabado reactivo: luz, o seco = sin post-proceso")
    ap.add_argument("--paleta", default="violeta", choices=sorted(PALETTES),
                    help="violeta = negro/blanco/violeta; violeta_rojo = alterna con rojo por kick en las "
                         "liberaciones; rgb = negro/blanco/rojo con RGB en las liberaciones")
    ap.add_argument("--res", default="1920x1080", help="ej: 1920x1080, 2560x1440")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--ss", type=int, default=2,
                    help="supersampling (2 = bordes limpios; 1 = píxel duro a media resolución)")
    ap.add_argument("--sistema", default="adn", choices=["adn", "neutro"],
                    help="adn = cada track tiene su sistema estelar según su ADN; neutro = el motor sin ADN")
    ap.add_argument("--motor", default="gpu", choices=["cpu", "gpu"],
                    help="gpu = shaders en la placa de video (unas 20 veces más rápido; si no hay moderngl u "
                         "OpenGL, usa la CPU); cpu = numpy")
    ap.add_argument("--sin-warp", dest="warp", action="store_false",
                    help="sin velocidad de la luz (el salto al escapar del agujero en las liberaciones)")
    ap.add_argument("--no-cam", dest="cam", action="store_false",
                    help="sin movimiento de cámara (zoom, giro, golpe en el kick)")
    ap.add_argument("--chunk", default=None,
                    help="renderizar solo un tramo 'desde-hasta' en segundos; la lectura y el viaje "
                         "siguen siendo los del track completo")
    ap.add_argument("--stills", default=None,
                    help="en vez de video, guardar cuadros PNG en estos segundos, ej: 150,167.2,180")
    ap.add_argument("--jobs", type=int, default=0,
                    help="procesos en paralelo para renders largos (0 = automático, 1 = uno solo)")
    ap.add_argument("--no-audio", action="store_true", help="exportar solo el video")
    ap.add_argument("--start", type=float, default=0, help="segundo del archivo en que empieza la lectura")
    ap.add_argument("--duration", type=float, default=0, help="duración a leer en segundos (0 = todo)")
    ap.add_argument("--bpm", default="auto", help="rango de tempo, ej: 140-142 (por defecto se detecta solo)")
    ap.add_argument("--seed", type=int, default=None,
                    help="variación visual; por defecto sale del propio audio (estable por track)")
    ap.add_argument("--title", default="MED1NA", help="la firma, abajo a la derecha")
    ap.add_argument("--font", default=None, help="fuente .ttf/.otf para la firma")
    ap.add_argument("--encoder-preset", dest="x264", default="veryfast", help="preset x264 (ultrafast..slow)")
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--list-presets", action="store_true", help="mostrar los presets y salir")
    ap.add_argument("--show-keys", action="store_true", help="mostrar la tonalidad detectada a lo largo del track")
    ap.add_argument("--score", default=None, help=argparse.SUPPRESS)      # uso interno (--jobs)
    ap.add_argument("--quiet", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.audio is None and not a.list_presets:
        ap.error("falta el archivo de audio")
    return a


def resolve_engine(a):
    """El agujero negro necesita la GPU: sin ella, el motor de la CPU y el viaje."""
    from .gpu import available
    if a.motor == "gpu" and not a.score and not available():
        print("  sin GPU (moderngl / OpenGL): uso el motor de la CPU", flush=True)
        a.motor = "cpu"
    if a.preset is None:
        a.preset = "agujero" if a.motor == "gpu" else "viaje"
    if PRESETS[a.preset]["style"] == "hole" and a.motor != "gpu":
        sys.exit("El agujero negro en 3D necesita el motor de GPU (--motor gpu).")


def listen(a, fps):
    """Leer el track y armar la partitura (imprime lo que el motor escuchó)."""
    t0 = time.time()
    print("Analizando audio...", flush=True)
    y = load_audio(a.audio, ANALYSIS_SR, a.start, a.duration)
    if len(y) == 0:
        sys.exit("No pude leer audio de ese archivo.")
    msg = mute_report(y, ANALYSIS_SR)
    if msg:
        sys.exit(msg)
    if a.bpm == "auto":
        bpm = detect_bpm(a.audio)
        print(f"  tempo detectado: {bpm:.2f} BPM", flush=True)
        lo_b, hi_b = bpm - 1, bpm + 1
    else:
        lo_b, hi_b = (float(v) for v in a.bpm.split("-"))
        bpm = (lo_b + hi_b) / 2
    if a.seed is None:
        a.seed = seed_from_audio(y)
    A = analyze(y, fps, (lo_b - 2, hi_b + 2))
    Ly = A.get("layers")
    if Ly and Ly["order"]:
        print("  capas que entran: " + " · ".join(
            f"{int(Ly['first'][k] / fps) // 60}:{int(Ly['first'][k] / fps) % 60:02d} "
            f"(~{Ly['center'][k]:.0f} Hz)" for k in Ly["order"]), flush=True)
    print(f"  capítulos: {int(A['chapter'].max()) + 1} · liberaciones: {int(A['rise'].sum())} · "
          f"sub filtrado: {100 * (1 - A['sub_on'].mean()):.0f}% del tiempo", flush=True)
    print(f"  {A['n'] / fps / 60:.1f} min · {len(A['onsets'])} kicks · {time.time() - t0:.1f}s", flush=True)
    score = build_score(A, a.preset, fps, a.seed, camera=a.cam, palette=a.paleta, system=a.sistema, bpm=bpm,
                        warp=a.warp)
    if score.dna is not None:
        from .system import describe
        print("  ADN: " + " · ".join(f"{k} {v:.2f}" for k, v in score.dna.items()), flush=True)
        rows = [f"{p.lower()} {v}" for p, v, _, _ in describe(score.dna)]
        print("  sistema estelar: " + " · ".join(rows[:7]) + "\n                   " + " · ".join(rows[7:]),
              flush=True)
    return score


def main(argv=None):
    a = parse_args(argv)
    if a.list_presets:
        for k, v in PRESETS.items():
            print(f"  {k:14s} {v}")
        return
    if not shutil.which("ffmpeg"):
        sys.exit("No encuentro ffmpeg. Instalalo (brew install ffmpeg / winget install ffmpeg) y reintentá.")
    W, H = (int(v) for v in a.res.lower().split("x"))
    t0 = time.time()

    if a.score:                                   # tramo de un render en paralelo
        with open(a.score, "rb") as fh:
            score = pickle.load(fh)
    else:
        resolve_engine(a)
        score = listen(a, a.fps)
    fps, n = score.fps, score.n

    if a.show_keys:
        prev = None
        for i in range(0, n, fps * 15):
            k = int(score.analysis["key"][i])
            if k != prev:
                name, cam = key_name(k)
                t = int(a.start + i / fps)
                print(f"  {t // 60:02d}:{t % 60:02d}  {name:4s} ({cam})")
                prev = k

    i0, i1 = 0, n
    if a.chunk:
        c0, c1 = (float(v) for v in a.chunk.split("-"))
        i0, i1 = int(round(c0 * fps)), min(n, int(round(c1 * fps)))
    stills = None
    if a.stills:
        stills = sorted({min(n - 1, int(round(float(v) * fps))) for v in a.stills.split(",")})
        i0, i1 = stills[0], stills[-1] + 1
    t_start = a.start + i0 / fps
    t_len = (i1 - i0) / fps

    jobs = a.jobs or max(1, min(8, (os.cpu_count() or 2) - 2))
    R = None
    if a.motor == "gpu":
        try:
            from .gpu import GPURenderer
            R = GPURenderer(score, W, H, a.title, a.font, a.ss, a.look)
            jobs = 1                              # la GPU ya trabaja en paralelo
        except Exception as e:  # noqa: BLE001  (sin moderngl o sin contexto de OpenGL)
            if PRESETS[score.preset]["style"] == "hole":
                sys.exit(f"El agujero negro en 3D necesita la GPU ({e}).")
            print(f"  sin GPU ({e}): uso el motor de la CPU", flush=True)
    if stills is None and not a.score and jobs > 1 and (i1 - i0) >= jobs * 10 * fps:
        worker = ["--res", a.res, "--start", str(a.start), "--title", a.title, "--encoder-preset", a.x264,
                  "--crf", str(a.crf), "--look", a.look, "--ss", str(a.ss), "--motor", "cpu"] \
            + (["--font", a.font] if a.font else [])
        try:
            render_parallel(score, a.output, a.audio, i0, i1, jobs, t_start, t_len, worker, a.no_audio)
        except RuntimeError as e:
            sys.exit(str(e))
        print(f"Listo: {a.output}  ({(time.time() - t0) / 60:.1f} min)")
        return

    if R is None:
        R = Renderer(score, W, H, a.title, a.font, a.ss, a.look)
    if stills is not None:
        render_stills(R, stills, os.path.splitext(a.output)[0], a.start)
    else:
        enc = encoder(a.output, W, H, fps, None if a.no_audio else a.audio, t_start, t_len, a.x264, a.crf)
        render_video(R, i0, i1, enc, a.quiet)
    if not a.quiet:
        print(f"\nListo: {a.output}  ({(time.time() - t0) / 60:.1f} min)")
