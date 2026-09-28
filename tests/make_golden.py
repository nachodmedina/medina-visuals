"""Regenera los cuadros de referencia de tests/golden a partir del track sintético.

Usarlo solo cuando un cambio de look es intencional:
    .venv/bin/python -m tests.make_golden
"""
import os
import tempfile

from PIL import Image

from medina.score import build_score
from tests.conftest import FPS, SEED, listen_synth
from tests.test_render import GOLDEN_DIR, golden_frames


def main():
    with tempfile.TemporaryDirectory() as tmp:
        A = listen_synth(tmp)["A"]
    frames = golden_frames(build_score(A, "viaje", FPS, SEED))
    os.makedirs(GOLDEN_DIR, exist_ok=True)
    for name, f in frames.items():
        Image.fromarray(f).save(os.path.join(GOLDEN_DIR, f"{name}.png"))
        print("  ", os.path.join(GOLDEN_DIR, f"{name}.png"))


if __name__ == "__main__":
    main()
