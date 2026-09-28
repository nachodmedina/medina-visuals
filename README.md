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
.venv/bin/pip install -r requirements-dev.txt     # para correr los tests
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

Flujo recomendado: primero un fragmento de 10–30 s con `--chunk` y algunos `--stills`; el video completo, solo de la versión aprobada. Un tramo o un cuadro suelto salen iguales que en el video completo (la estela se precalienta 3 s antes).

También se puede correr como módulo: `.venv/bin/python -m medina …`.

Si el audio viene casi mudo (típico de una exportación con pistas en solo), el motor frena y avisa.

### Opciones principales

| Opción | Qué hace |
|---|---|
| `--preset viaje` | el track elige el estilo según su historia (ver abajo); `--list-presets` muestra todos |
| `--look luz \| seco` | acabado reactivo (por defecto `luz`); `seco` = sin post-proceso |
| `--paleta violeta \| violeta_rojo \| rgb` | `violeta` (por defecto); `violeta_rojo` alterna con rojo por kick en las liberaciones; `rgb` es la paleta original negro/blanco/rojo con RGB en las liberaciones |
| `--res 2560x1440` | resolución de salida |
| `--chunk desde-hasta` | renderiza solo ese tramo |
| `--stills s1,s2,…` | guarda cuadros PNG en vez de video |
| `--jobs N` | procesos en paralelo (0 = automático; `--jobs 4` calienta menos la máquina) |
| `--ss 2` | supersampling (2 = bordes limpios; 1 = píxel duro a media resolución) |
| `--no-cam` | sin movimiento de cámara |
| `--no-audio` | exporta solo el video |
| `--show-keys` | muestra la tonalidad detectada a lo largo del track (con código Camelot) |

## Cómo funciona

El motor trabaja en tres capas: **escuchar** el track, escribir una **partitura** con lo que pasa en cada cuadro y **dibujar** cada cuadro a partir de su entrada en la partitura.

### 1. Lectura del track (`medina/analysis.py`)

- **Tempo** por autocorrelación (precisión de 0,01 BPM) y **beats** por programación dinámica; los **kicks** son los beats con energía en 35–130 Hz.
- **Sub presente / filtrado** (25–60 Hz, relativo al sub pleno): los cortes parciales también cuentan, porque la tensión se arma filtrando el sub, no cortando el kick.
- **Liberaciones**: vuelve el sub o vuelve el kick después de ≥ 8 s. Se anclan al primer kick que ya trae el sub entero, medido sin suavizar. Dos liberaciones a menos de 2 s cuentan como una.
- **Tensión**: progreso 0 → 1 dentro de cada tramo filtrado; termina exactamente en la liberación.
- **Caos**: curva lenta de agudos (el track "se abre").
- **Hats**: ataques de 5–11 kHz; cargan la tensión en su último tramo.
- **Silencios** (capítulos) y **fade** final.
- **Capas**: NMF sobre bandas de 300 Hz–11 kHz para detectar cuándo entra cada elemento.

Cada cambio en la lectura se valida en todos los tracks de `tracks/` contra el espectrograma (`ffmpeg … showspectrumpic`).

### 2. La partitura (`medina/score.py`)

Se arma una sola vez recorriendo el track en orden, porque el movimiento se acumula. Para cada cuadro guarda:
- el estilo vigente;
- el movimiento (avance del túnel, giro, mutaciones, remolino, tamaño del horizonte, destello de la materia);
- la lectura del momento (energía, kick, tensión, caos, hats…);
- la cámara y el color de acento.

Después cualquier cuadro se dibuja solo con su entrada. Por eso los tramos, los cuadros sueltos y el render en paralelo salen iguales sin "rebobinar", y por eso la partitura es el contrato con el futuro render en GPU. No depende de la resolución de salida.

### 3. El viaje (`--preset viaje`)

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

### 4. Acabado

- **Acabado B**: se dibuja a 2× y se baja con Lanczos, para tener bordes limpios sin perder dureza.
- **Post-proceso reactivo** (`--look luz`):
  - bloom que se enciende en las liberaciones;
  - estela (feedback) que se alarga con la tensión;
  - grano que crece con el caos;
  - un eco del color de acento en los kicks fuertes;
  - viñeta fija;
  - punto de negro.
- **Cámara**: zoom que empuja con la tensión y suelta en la liberación, giro lento con un paso seco en cada liberación, y golpe de zoom con temblor en cada kick.

## Código

| Archivo | |
|---|---|
| `brutal_viz.py` | el comando de siempre (llama al paquete `medina`) |
| `medina/audio.py` | leer el archivo, tempo, beats, control de audio mudo, semilla del audio |
| `medina/analysis.py` | escuchar el track: kicks, sub, tensión, liberaciones, respiración, caos, hats, silencios, capas, tonalidad |
| `medina/score.py` | la partitura: el viaje, el movimiento, la cámara y el color de cada cuadro |
| `medina/emotion.py` | la capa emocional (en calibración): qué emoción transmite cada momento (incertidumbre, miedo, esperanza, enigma, soledad, fuerza, vulnerabilidad), las secciones del track y su ADN, del que sale su sistema estelar. Todavía no cambia el video |
| `medina/styles.py` | los estilos (partículas, arcos, estelas, disco) y la lente gravitacional |
| `medina/space.py` | lo común a todos: estrellas, silencio, fenómenos de las capas, agujero negro |
| `medina/post.py`, `medina/logo.py` | acabado reactivo y firma |
| `medina/render.py` | arma cada cuadro a partir de la partitura (grilla, supersampling, cámara) |
| `medina/output.py`, `medina/cli.py` | video, tramos, cuadros sueltos, render en paralelo y línea de comandos |
| `medina/presets.py` | presets, looks y paletas |
| `master.py` | master técnico aparte: EQ, M/S, compresión de bus, limitador con objetivo de LUFS y true peak (necesita `pyloudnorm` y `pedalboard`) |
| `tracks/`, `renders/` | locales, fuera del repo |

El modo viejo para sets (espectro, video del celular, barras, túnel) y los estilos que el viaje ya no usa quedaron en la etiqueta `v0.1-antes-de-modularizar`.

## Tests

```bash
.venv/bin/python -m pytest -q
```

- **Track sintético** (`tests/synth.py`): tiene una historia conocida (kick con sub, un tramo filtrado, la liberación en un beat exacto, hats en corcheas y un sinte que entra), así que los tests no necesitan tu música. Verifican el tempo, los kicks, la liberación, la tensión, la respiración, los hats, las capas, la partitura (determinista, el viaje cambia solo en liberaciones, cámara y paletas acotadas) y el dibujo (mucho negro, respiración a negro, cuadros sueltos independientes, tramos iguales al video completo). También hay cuadros de referencia en `tests/golden/`: si un cambio de look es intencional, se regeneran con `.venv/bin/python -m tests.make_golden`.
- **Tus tracks** (`tests/test_real_tracks.py`): si están en `tracks/`, se verifican el tempo y las liberaciones validadas contra el espectrograma. Si no están, se saltean.

Para verificar que un cambio no altera la imagen de tus tracks (por ejemplo, el paso a GPU):

```bash
.venv/bin/python tools/referencias.py guardar renders/_ref/base     # antes, con la versión aprobada
.venv/bin/python tools/referencias.py comparar renders/_ref/base    # después
```

## Pendientes conocidos

- **Kicks que caen justo antes del centro de un cuadro.** El control de "¿hay graves en este beat?" mira un solo cuadro, y un kick corto que cae un poco antes del centro del cuadro no se cuenta. En los tracks actuales no pasa: se detecta el 100 % de los kicks en los tramos con sub. Pero en un tempo que dura un número entero de cuadros (por ejemplo, 120 BPM = 15 cuadros por beat) podrían perderse todos. El test `test_kicks_en_cualquier_fase_del_cuadro` lo documenta como falla conocida.
  - **Arreglo probado:** mirar ±1 cuadro, como ya hace el flujo del kick.
  - **Qué cambia:** cuenta también los kicks de tus tramos filtrados (+12, +58 y +31 en los tres tracks), y en `Untitled 9` agrega una liberación falsa a mitad de la tensión larga (3:28).
  - **Qué falta:** ajustar también la regla de "vuelve el kick" (que exija que vuelva el sub) y validarlo visualmente.

## Próximo paso

Pasar el render a GPU (moderngl / shaders): 3D real con profundidad, niebla y cámara, y vista previa en tiempo real.
