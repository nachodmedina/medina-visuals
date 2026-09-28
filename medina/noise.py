"""Azar reproducible: la misma entrada da siempre el mismo valor (misma pieza = mismo video)."""
import numpy as np


def hash32(a, b, t):
    """Ruido determinístico por celda: enteros (a, b, t) -> valores en [0, 1)."""
    h = (a.astype(np.uint32) * np.uint32(73856093)) ^ (b.astype(np.uint32) * np.uint32(19349663)) \
        ^ np.uint32((int(t) * 83492791) & 0xFFFFFFFF)
    h ^= h >> np.uint32(13)
    h *= np.uint32(0x5bd1e995)
    h ^= h >> np.uint32(15)
    return (h & np.uint32(0xFFFFFF)).astype(np.float32) / float(0xFFFFFF)
