"""Exportar: video (con el audio del tramo), cuadros sueltos en PNG y render en paralelo."""
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
from PIL import Image

WARM_S = 3.0          # la estela depende de los cuadros anteriores: se dibujan ~3 s antes, sin exportar


def encoder(output, W, H, fps, audio=None, t_start=0.0, t_len=0.0, x264="veryfast", crf=18):
    """ffmpeg recibiendo cuadros rgb24 por stdin -> mp4 (h264, con el audio del tramo si hay)."""
    cmd = ["ffmpeg", "-y", "-v", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-"]
    if audio is None:
        cmd += ["-c:v", "libx264", "-preset", x264, "-crf", str(crf), "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", output]
    else:
        cmd += ["-ss", f"{t_start:.3f}", "-t", f"{t_len:.3f}", "-i", audio, "-map", "0:v", "-map", "1:a",
                "-c:v", "libx264", "-preset", x264, "-crf", str(crf), "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "320k", "-shortest", "-movflags", "+faststart", output]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def render_video(renderer, i0, i1, enc, quiet=False):
    """Dibuja los cuadros [i0, i1) (con la estela precalentada) y los manda al encoder."""
    g = renderer.grid
    warm = int(WARM_S * g.fps) if renderer.post is not None else 0
    frame = np.zeros((g.H, g.W, 3), np.uint8)
    n_out = i1 - i0
    k = 0
    t1 = last_print = time.time()
    try:
        for i in range(max(0, i0 - warm), i1):
            renderer.draw(frame, i)
            if i < i0:
                continue
            enc.stdin.write(frame.tobytes())
            k += 1
            now = time.time()
            if not quiet and (now - last_print > 5 or i == i1 - 1):
                rate = k / (now - t1)
                eta = (n_out - k) / rate
                print(f"\r  render {100 * k / n_out:5.1f}%  {rate:5.1f} fps  ETA {int(eta // 60)}m{int(eta % 60):02d}s ",
                      end="", flush=True)
                last_print = now
    except BrokenPipeError:
        pass
    enc.stdin.close()
    enc.wait()


def render_stills(renderer, stills, base, start=0.0):
    """Guarda un PNG por cada cuadro de `stills` (con la estela precalentada)."""
    g = renderer.grid
    warm = int(WARM_S * g.fps) if renderer.post is not None else 0
    frame = np.zeros((g.H, g.W, 3), np.uint8)
    todo = sorted({i for s in stills for i in range(max(0, s - warm), s + 1)})
    keep = set(stills)
    saved = []
    for i in todo:
        renderer.draw(frame, i)
        if i in keep:
            ts = start + i / g.fps
            path = f"{base}_{int(ts // 60)}m{ts % 60:05.2f}s.png"
            Image.fromarray(frame).save(path)
            saved.append(path)
    return saved


def render_parallel(score, output, audio, i0, i1, jobs, t_start, t_len, worker_args, no_audio=False):
    """Parte el render en `jobs` tramos que se dibujan en paralelo (cada uno con la partitura
    completa y la estela precalentada) y los une sin recodificar."""
    fps = score.fps
    tmp = tempfile.mkdtemp(prefix=".medina_", dir=os.path.dirname(os.path.abspath(output)))
    try:
        cache = os.path.join(tmp, "score.pkl")
        with open(cache, "wb") as fh:
            pickle.dump(score, fh)
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env = dict(os.environ, PYTHONPATH=root + os.pathsep + os.environ.get("PYTHONPATH", ""))
        cmd = [sys.executable, "-m", "medina", audio, "--score", cache,
               "--no-audio", "--jobs", "1", "--quiet"] + list(worker_args)
        cuts = np.linspace(i0, i1, jobs + 1).round().astype(int)
        parts, procs = [], []
        for k in range(jobs):
            parts.append(os.path.join(tmp, f"part{k:02d}.mp4"))
            procs.append(subprocess.Popen(cmd + ["-o", parts[-1], "--chunk",
                                                 f"{cuts[k] / fps}-{cuts[k + 1] / fps}"], env=env))
        print(f"  render en {jobs} procesos...", flush=True)
        t1 = time.time()
        while any(p.poll() is None for p in procs):
            time.sleep(5)
            done = sum(p.poll() is not None for p in procs)
            print(f"\r  tramos listos {done}/{jobs}  ({int(time.time() - t1)}s)", end="", flush=True)
        print()
        if any(p.returncode for p in procs):
            sys.exit("Falló algún tramo del render en paralelo.")
        lst = os.path.join(tmp, "parts.txt")
        with open(lst, "w") as fh:
            fh.writelines(f"file '{p}'\n" for p in parts)
        cat = ["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst]
        if no_audio:
            cat += ["-c", "copy", "-movflags", "+faststart", output]
        else:
            cat += ["-ss", f"{t_start:.3f}", "-t", f"{t_len:.3f}", "-i", audio, "-map", "0:v", "-map", "1:a",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "320k", "-shortest", "-movflags", "+faststart",
                    output]
        subprocess.run(cat, check=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
