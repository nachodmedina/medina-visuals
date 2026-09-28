# medina-visuals

Motor de visuales audio-reactivas para los tracks de techno de **MED1NA**.

El motor escucha el track y decide solo: detecta el tempo, los kicks, los tramos en tensión, las liberaciones y los elementos que van entrando, y con eso conduce un viaje visual. No se configura nada a mano por track: el mismo track siempre produce el mismo video, porque la semilla del azar sale del propio audio.

## La idea

El video es un **viaje en el espacio hacia un agujero negro**. Cada estado del track tiene su lectura:

| El track… | En el video… |
|---|---|
| filtra el sub (tensión) | la **atracción**: todo cae en espiral hacia el horizonte, que crece |
| suelta el sub / vuelve el kick (liberación) | el **escape**: todo sale disparado y la materia se enciende |
| suena pleno | el viaje por el espacio: el túnel avanza hacia la cámara |
| se vacía | la **calma del vacío**: negro profundo y estrellas lejanas |
| se desvanece al final | se cruza el horizonte: todo cae adentro |

Estética: minimalismo oscuro y duro, mucho negro. Paleta negro / blanco / violeta. MED1NA chico abajo a la derecha: letras negras con un filo tenue, que curvan el espacio que pasa detrás, como el agujero.

## Instalación

Requisitos: Python 3.10+ y [ffmpeg](https://ffmpeg.org) en el PATH (`brew install ffmpeg`).

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Uso

```bash
# video completo (en paralelo, usa varios núcleos)
.venv/bin/python brutal_viz.py tracks/mi_track.wav -o renders/mi_track.mp4 --preset viaje

# para YouTube conviene 1440p: le da más calidad de compresión al grano y a los negros
.venv/bin/python brutal_viz.py tracks/mi_track.wav -o renders/mi_track_1440p.mp4 --preset viaje --res 2560x1440

# preview de un tramo (en segundos): el análisis y la animación son los del track completo
.venv/bin/python brutal_viz.py tracks/mi_track.wav -o renders/preview.mp4 --preset viaje --chunk 140-170

# cuadros sueltos en PNG
.venv/bin/python brutal_viz.py tracks/mi_track.wav -o renders/cuadro.png --preset viaje --stills 95,150,167.2
```

Flujo recomendado: primero un fragmento de 10–30 s con `--chunk` y algunos `--stills`; el video completo, solo de la versión aprobada.

Si el audio viene casi mudo (típico de una exportación con pistas en solo), el motor frena y avisa.

### Opciones principales

| Opción | Qué hace |
|---|---|
| `--preset viaje` | el track elige el estilo según su historia (ver abajo); `--list-presets` muestra todos |
| `--look luz \| atmosfera \| seco` | acabado reactivo (por defecto `luz`); `seco` = sin post-proceso |
| `--paleta violeta \| violeta_rojo \| rgb` | `violeta` (por defecto); `violeta_rojo` alterna con rojo por kick en las liberaciones; `rgb` es la paleta original negro/blanco/rojo con RGB en las liberaciones |
| `--res 2560x1440` | resolución de salida |
| `--chunk desde-hasta` | renderiza solo ese tramo |
| `--stills s1,s2,…` | guarda cuadros PNG en vez de video |
| `--jobs N` | procesos en paralelo (0 = automático; `--jobs 4` calienta menos la máquina) |
| `--ss 2` | supersampling (2 = bordes limpios; 1 = píxel duro a media resolución) |
| `--no-cam` | sin movimiento de cámara |
| `--no-audio` | exporta solo el video |

## Cómo funciona

### 1. Lectura del track (`analyze`)

- **Tempo** por autocorrelación (precisión de 0,01 BPM) y **beats** por programación dinámica; los **kicks** son los beats con energía en 35–130 Hz.
- **Sub presente / filtrado** (25–60 Hz, relativo al sub pleno): los cortes parciales también cuentan, porque la tensión se arma filtrando el sub, no cortando el kick.
- **Liberaciones**: vuelve el sub o vuelve el kick después de ≥ 8 s. Se anclan al primer kick que ya trae el sub entero, medido sin suavizar. Dos liberaciones a menos de 2 s cuentan como una.
- **Tensión**: progreso 0 → 1 dentro de cada tramo filtrado; termina exactamente en la liberación.
- **Caos**: curva lenta de agudos (el track "se abre").
- **Hats**: ataques de 5–11 kHz; cargan la tensión en su último tramo.
- **Silencios** (capítulos) y **fade** final.
- **Capas**: NMF sobre bandas de 300 Hz–11 kHz para detectar cuándo entra cada elemento.

Cada cambio en la lectura se valida en todos los tracks de `tracks/` contra el espectrograma (`ffmpeg … showspectrumpic`).

### 2. El viaje (`--preset viaje`)

El estilo cambia solo en las liberaciones, de a un escalón:

**brillos → puntos → puntos densos → arcos → estelas → disco**

- **Partículas** (brillos, puntos): viven sobre un túnel radial y viajan hacia la cámara.
- **Arcos**: anillos fragmentados que giran en sentidos opuestos y engranan con el kick.
- **Estelas**: materia en órbita, más rápida cerca del agujero; el kick las estira. La mayoría son tenues y brillan más cerca del horizonte, con anillos densos y huecos.
- **Disco**: el disco de acreción visto en diagonal, casi de canto (tipo Gargantua); la lente dobla la parte de atrás alrededor del horizonte.

En todos los estilos:

- **El agujero negro**: disco negro con un resplandor difuso que late con los kicks y los hats. A su alrededor, una **lente gravitacional** comprime lo que está detrás en un anillo denso y un remolino tuerce todo en espiral.
- **Estrellas lejanas** de fondo, con destellos en cruz en los hats.
- **Fenómenos del espacio**: cada capa que entra suma uno (cometa, pulsar, luna en órbita, estrella que se enciende, meteoros, jets polares). Reacciona a la actividad de *esa* capa; la más nueva va en color de acento.
- **Tensión**: erosión (los elementos se apagan de a uno) y, al final del tramo, lo que sobrevive titila con los hats. El último beat antes de la liberación es una **respiración** a negro.
- **Liberación**: la materia se enciende entera, el remolino se retuerce y sale del horizonte una **onda gravitacional** que deforma el espacio a su paso durante unos segundos.

### 3. Acabado

- **Acabado B**: se dibuja a 2× y se baja con Lanczos, para tener bordes limpios sin perder dureza.
- **Post-proceso reactivo** (`--look luz`):
  - bloom que se enciende en las liberaciones;
  - estela (feedback) que se alarga con la tensión;
  - grano que crece con el caos;
  - un eco del color de acento en los kicks fuertes;
  - viñeta fija;
  - punto de negro.
- **Cámara**: zoom que empuja con la tensión y suelta en la liberación, giro lento con un paso seco en cada liberación, y golpe de zoom con temblor en cada kick.

## Archivos

| Archivo | |
|---|---|
| `brutal_viz.py` | el motor (lectura, estilos, post-proceso, render) |
| `master.py` | master técnico aparte: EQ, M/S, compresión de bus, limitador con objetivo de LUFS y true peak (necesita `pyloudnorm` y `pedalboard`) |
| `tracks/`, `renders/` | locales, fuera del repo |

## Próximo paso

Pasar el render a GPU (moderngl / shaders): 3D real con profundidad, niebla y cámara, y vista previa en tiempo real.
