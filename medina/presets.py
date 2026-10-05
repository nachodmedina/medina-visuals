"""El lenguaje visual: presets (estilos y el viaje), looks (acabado) y paletas."""
import numpy as np

BLACK = np.array([0, 0, 0], np.uint8)
WHITE = np.array([255, 255, 255], np.uint8)
RED = np.array([255, 0, 0], np.uint8)
GREEN = np.array([0, 255, 0], np.uint8)
BLUE = np.array([0, 0, 255], np.uint8)
VIOLET = np.array([125, 55, 200], np.uint8)
ACCENTS = [RED, GREEN, BLUE]              # paleta rgb: rota por kick en las liberaciones

# violeta = negro / blanco / violeta; violeta_rojo = base violeta y en cada liberación alterna
# con rojo por kick (32 kicks); rgb = negro / blanco / rojo, con rojo -> verde -> azul por kick
# en las liberaciones
PALETTES = {"violeta": VIOLET, "violeta_rojo": [VIOLET, RED], "rgb": None}

# Cada preset = un estilo + parámetros. Todos leen las mismas señales del track
# (kick, tensión, liberación, caos, hats, capas); el preset solo cambia el lenguaje.
PRESETS = {
    # partículas sobre el túnel (polvo): viajan hacia la cámara
    "brillos":       dict(style="dots", N=88, Kd=13, p=0.03, flicker=4, plus=True, dot=0.2),
    "puntos":        dict(style="dots", N=48, Kd=7.5, p=0.22),
    # el disco (inclinación y apertura: del sistema estelar); el abierto es el mismo visto más desde arriba
    "disco_abierto": dict(style="disk", K=16.0, open=2.2, glitch=False),
    "disco":         dict(style="disk", K=20.0, glitch=False),
    # viaje: el track decide el estilo. Antes del primer silencio, mundo de partículas (polvo que
    # viene hacia la cámara); después, el disco en diagonal, cada vez más de canto. Solo cambia en
    # las liberaciones. Nada de anillos ni círculos: lo que se ve tiene que parecer parte del espacio.
    "viaje":         dict(style="journey",
                          worlds=[["brillos", "puntos"],
                                  ["disco_abierto", "disco"]]),
}

# Post-proceso reactivo. Cada efecto lee una señal del track:
#   bloom     <- liberaciones: se enciende en el rise y se apaga en unos segundos
#   estela    <- tensión: feedback que se alarga mientras el sub está filtrado
#   grano     <- caos: el track se abre -> más grano
#   fantasma  <- kicks fuertes: un eco en el color de acento se desplaza hacia afuera
#   viñeta    <- fija; no se cierra con la tensión (nada de iris)
LOOKS = {
    # bloom=(base, pico en la liberación); trail_hl=(vida media abierta, en tensión máx) en s;
    # trail_g = brillo del recuerdo (lo vivo siempre va al frente); black = punto de negro
    "luz":  dict(bloom=(0.10, 0.75), bloom_tau=3.5, bloom_sig=(8, 32), bloom_mix=0.15,
                 trail_hl=(0.012, 0.40), trail_g=0.55, grain=(0.012, 0.045), grain_px=1,
                 ghost_px=5, vignette=0.30, black=0.05),
    "seco": None,                          # sin post-proceso
}
