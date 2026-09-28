#!/usr/bin/env python3
"""Visuales audio-reactivas de MED1NA: el motor escucha el track y conduce el viaje.

    python3 brutal_viz.py tracks/mi_track.wav -o renders/mi_track.mp4 --preset viaje

El código está en el paquete `medina/` (también se puede correr como `python3 -m medina`).
"""
from medina.cli import main

if __name__ == "__main__":
    main()
