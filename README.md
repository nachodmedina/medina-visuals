# medina-visuals

Motor de visuales audio-reactivas para los tracks de techno de **MED1NA**.

El motor escucha el track y decide solo: detecta el tempo, los kicks, los tramos en tensión, las liberaciones y los elementos que van entrando, y con eso conduce un viaje visual. No se configura nada a mano por track: el mismo track siempre produce el mismo video, porque la semilla del azar sale del propio audio.

## La idea

El video es **un agujero negro**, y todo gira a su alrededor: oscuro, denso, pesado, hipnótico. Cada estado del track tiene su lectura:

| El track… | En el video… |
|---|---|
| filtra el sub (tensión) | la **atracción**: la cámara cae hacia el agujero, la sombra crece, el cielo se apaga |
| suelta el sub / vuelve el kick (liberación) | el **escape**: el tiempo se detiene un beat (solo un eco de luz da la vuelta al borde), vuelve, el espacio se contrae a la velocidad de la luz y sale una onda gravitacional |
| suena pleno | la cámara orbita el agujero; el cielo, curvado por la gravedad, fluye a su alrededor |
| el kick | el agujero late, como un corazón pesado |
| se vacía | la **calma del vacío**: negro profundo y estrellas lejanas |

Todos los tracks comparten ese lenguaje, pero cada uno tiene **su propio agujero**, que sale de su ADN: la masa, la luz del borde, el cielo, la galaxia del fondo, el movimiento.

Estética: minimalismo oscuro y duro, mucho negro. Paleta negro / blanco / violeta. MED1NA chico abajo a la derecha: letras negras con un filo tenue, que curvan el espacio que pasa detrás, como el agujero.

## Instalación

Requisitos: Python 3.10+ y [ffmpeg](https://ffmpeg.org) en el PATH (`brew install ffmpeg`).

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -r requirements-dev.txt     # para correr los tests
```

### En Windows (por ejemplo, una PC con placa NVIDIA)

1. Instalar [Python 3.13](https://www.python.org/downloads/) (marcar "Add python.exe to PATH"), [Git](https://git-scm.com/download/win), ffmpeg (`winget install ffmpeg` en PowerShell) y el driver de NVIDIA al día.
2. Clonar el repo (es privado: Git pide iniciar sesión en GitHub):
   ```powershell
   git clone https://github.com/nachodmedina/medina-visuals.git
   cd medina-visuals
   ```
3. Crear el entorno e instalar:
   ```powershell
   py -3.13 -m venv .venv
   .venv\Scripts\pip install -r requirements.txt -r requirements-dev.txt
   ```
4. Copiar los tracks a `tracks\` (no están en el repo).
5. Probar: `.venv\Scripts\python -m pytest -q`. Los tests de GPU tienen que pasar (si se saltean, no hay OpenGL: revisar el driver).
6. Renderizar: `.venv\Scripts\python brutal_viz.py "tracks\mi track.wav" -o "renders\mi track.mp4" --res 2560x1440`

## Uso

```bash
# video completo: el agujero negro (con GPU; sin GPU, el viaje)
.venv/bin/python brutal_viz.py tracks/mi_track.wav -o renders/mi_track.mp4

# para YouTube conviene 1440p: le da más calidad de compresión al grano y a los negros
.venv/bin/python brutal_viz.py tracks/mi_track.wav -o renders/mi_track_1440p.mp4 --res 2560x1440

# preview de un tramo (en segundos): el análisis y la animación son los del track completo
.venv/bin/python brutal_viz.py tracks/mi_track.wav -o renders/preview.mp4 --chunk 140-170

# cuadros sueltos en PNG
.venv/bin/python brutal_viz.py tracks/mi_track.wav -o renders/cuadro.png --stills 95,150,167.2

# el viaje anterior (polvo y disco), en vez del agujero
.venv/bin/python brutal_viz.py tracks/mi_track.wav -o renders/mi_track.mp4 --preset viaje
```

Flujo recomendado: primero un fragmento de 10–30 s con `--chunk` y algunos `--stills`; el video completo, solo de la versión aprobada. Un tramo o un cuadro suelto salen iguales que en el video completo (la estela se precalienta 3 s antes).

También se puede correr como módulo: `.venv/bin/python -m medina …`.

Si el audio viene casi mudo (típico de una exportación con pistas en solo), el motor frena y avisa.

### Opciones principales

| Opción | Qué hace |
|---|---|
| `--preset agujero \| viaje` | `agujero` (por defecto con GPU): el agujero negro en 3D; `viaje`: polvo y disco, el track elige el estilo (por defecto sin GPU); `--list-presets` muestra todos |
| `--look luz \| seco` | acabado reactivo (por defecto `luz`); `seco` = sin post-proceso |
| `--paleta violeta \| violeta_rojo \| rgb` | `violeta` (por defecto); `violeta_rojo` alterna con rojo por kick en las liberaciones; `rgb` es la paleta original negro/blanco/rojo con RGB en las liberaciones |
| `--res 2560x1440` | resolución de salida |
| `--chunk desde-hasta` | renderiza solo ese tramo |
| `--stills s1,s2,…` | guarda cuadros PNG en vez de video |
| `--jobs N` | procesos en paralelo (0 = automático; `--jobs 4` calienta menos la máquina) |
| `--ss 2` | supersampling (2 = bordes limpios; 1 = píxel duro a media resolución) |
| `--motor gpu \| cpu` | `gpu` (por defecto): shaders en la placa de video, unas 20 veces más rápido (necesita moderngl y OpenGL 3.3+; si no están, usa la CPU); `cpu`: numpy |
| `--sistema adn \| neutro` | `adn` (por defecto): cada track tiene su propio sistema estelar según su ADN; `neutro`: el motor sin ADN, igual para todos |
| `--sin-warp` | sin velocidad de la luz |
| `--no-cam` | sin movimiento de cámara |
| `--no-audio` | exporta solo el video |
| `--show-keys` | muestra la tonalidad detectada a lo largo del track (con código Camelot) |

## Cómo funciona

El motor trabaja en tres capas: **escuchar** el track, escribir una **partitura** con lo que pasa en cada cuadro y **dibujar** cada cuadro a partir de su entrada en la partitura.

### 1. Lectura del track (`medina/analysis.py`)

- **Tempo** por autocorrelación (precisión de 0,01 BPM) y **beats** por programación dinámica; los **kicks** son los beats con energía en 35–130 Hz.
- **Sub presente / filtrado** (25–60 Hz, relativo al sub pleno): los cortes parciales también cuentan, porque la tensión se arma filtrando el sub, no cortando el kick.
- **Liberaciones**: vuelve el sub o vuelve el kick después de ≥ 8 s y se queda (un golpe suelto de kick dentro del filtrado no es un drop). Se anclan al primer kick que ya trae el sub entero, medido sin suavizar. Dos liberaciones a menos de 2 s cuentan como una.
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

### 3. El agujero negro (`--preset agujero`, `medina/shaders/hole.frag`)

Cada píxel es un rayo de luz que se curva por la gravedad (geometría de Schwarzschild): lo que cae es negro absoluto; lo que roza la esfera de fotones deja la luz del borde; lo que escapa ve el cielo, curvado.

- **La luz del borde**: un eclipse finísimo, más intenso de un lado, que ondula con el beat del track; late con el kick y titila con los hats al final de la tensión.
- **El cielo**: estrellas y una banda galáctica (polvo y estrellas lejanas) fijas en el espacio. Cerca del agujero se doblan, se duplican y se cierran en anillos: el espacio-tiempo deformado.
- **La cámara con masa**: orbita el agujero (acelera en los drops, casi se detiene en las tensiones); todo se mueve con inercia, nada a saltos.
- **La tensión**: la cámara cae hacia el agujero, la sombra crece, el cielo se apaga.
- **La liberación**, en secuencia: el impacto; el **tiempo se detiene** un beat (solo un **eco de luz** da la vuelta al borde); el tiempo vuelve (acelera un momento para recuperar lo perdido, así la imagen sigue con la música), el espacio se contrae a la **velocidad de la luz** (aberración relativista) y sale una **onda gravitacional** que cruza todo. Más intenso cuanto más larga fue la tensión.
- **Cada track, su agujero** (de su ADN): la masa y la atracción (vorágine), el borde más grueso (densidad) o más ondulado (caos), blanco o violeta (luz), dónde cae su lado encendido, cuánto cielo hay (vacío), la orientación, el polvo y el brillo de su galaxia, la velocidad de la órbita y del latido (energía).

Necesita el motor de GPU.

### El viaje (`--preset viaje`)

El estilo cambia solo en las liberaciones, de a un escalón:

**brillos → puntos → disco abierto → disco**

Todo lo que se ve tiene que parecer parte del espacio: nada de anillos ni formas geométricas marcadas. El kick pega en la luz (enciende), no en la forma (casi no agranda ni estira).

- **Partículas** (brillos, puntos): viven sobre un túnel radial y viajan hacia la cámara. Tienen profundidad: la mayoría tenues, pocas brillantes.
- **Disco**: el disco de acreción en diagonal (tipo Gargantua): materia en órbita, más rápida cerca del agujero; la mayoría tenue, más brillante cerca del horizonte, con anillos densos y huecos. La lente dobla la parte de atrás alrededor del horizonte. Primero se ve **abierto** (más desde arriba) y después **casi de canto**. Su inclinación y su apertura son las del sistema estelar del track.

En todos los estilos:

- **El agujero negro**: disco negro con un resplandor difuso que late con los kicks y los hats. A su alrededor, una **lente gravitacional** comprime lo que está detrás en un anillo denso y un remolino tuerce todo en espiral.
- **Estrellas lejanas** de fondo, con destellos en cruz en los hats.
- **Fenómenos del espacio**: cada capa que entra suma uno (cometa, pulsar, luna en órbita, estrella que se enciende, meteoros, jets polares). Reacciona a la actividad de *esa* capa; la más nueva va en color de acento.
- **Tensión**: erosión (los elementos se apagan de a uno) y, al final del tramo, lo que sobrevive titila con los hats. El último beat antes de la liberación es una **respiración** a negro.
- **Liberación**: la materia se enciende entera, el remolino se retuerce y sale del horizonte una **onda gravitacional** que deforma el espacio a su paso durante unos segundos.
- **Velocidad de la luz**: la liberación es el escape del agujero. El espacio salta a la velocidad de la luz: las estrellas se abren hacia afuera estiradas en líneas finas, el túnel se acelera y las partículas se estiran hacia donde se viaja. El salto es más alto cuanto más larga fue la tensión, se sostiene un compás y se asienta en un crucero suave mientras dura la fuerza; en la tensión (la atracción) se frena del todo.

### El sistema estelar de cada track (`medina/system.py`)

Todos los tracks hablan el mismo idioma, pero cada uno tiene su propio sistema. La capa emocional (`medina/emotion.py`) mide el **ADN** del track en ocho ejes (caos, vacío, vorágine, energía, luz, densidad, hipnosis, aspereza) y de ahí salen:

| Parámetro | Eje | Qué expresa |
|---|---|---|
| masa del agujero (radio y lente) | vorágine | más atracción: un agujero más grande que curva más el espacio |
| disco: inclinación y apertura | caos · vorágine | orden: plano y estable; caos: inclinado, inquieto; más masa: la elipse se abre |
| velocidad orbital | energía | órbitas lentas o vertiginosas |
| densidad de materia | densidad | partículas, órbitas y segmentos |
| turbulencia | caos | órbitas que se deforman, ondas en el espacio |
| estrellas de fondo | vacío | cuánto espacio profundo se ve |
| luz y tono del violeta | luz | resplandor; violeta profundo ↔ lavanda |
| proporción de acento | energía | cuánto violeta frente al blanco |
| grano | aspereza | limpio o áspero como el ruido del track |
| estela | vacío | cuánto dura el recuerdo de lo que pasó |
| cámara (distancia y temblor) | vacío · energía | lejos y quieta en el vacío; cerca y golpeada con energía |
| fenómenos preferidos | hipnosis · energía · vacío · caos | pulsares, jets, lunas solitarias, cometas… |
| ritmo del viaje | energía | cuántos escalones sube por liberación |
| velocidad de la luz | energía | cuánto se estira el espacio al escapar |

El sistema `neutro` (`--sistema neutro`) reproduce exactamente el motor sin ADN.

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

### El motor de GPU (`--motor gpu`)

La misma partitura, dibujada con shaders (OpenGL vía moderngl): la GPU calcula la escena píxel por píxel (polvo o disco, lente, onda, horizonte, silencio) y todo el acabado (estela, fantasma, bloom, resplandor del horizonte, viñeta, punto de negro, grano). La CPU sigue dibujando lo disperso (estrellas y fenómenos, en capas que la GPU combina en el mismo orden), la firma y el glitch. El azar es el mismo (la misma función de hash, el mismo banco de grano), así que cada track conserva su imagen.

No es idéntico bit a bit (la GPU redondea distinto), pero se ve igual: `tools/referencias.py comparar renders/_ref/base gpu` lo verifica contra las referencias de tus tracks, con tolerancia. A 1440p dibuja unos 30 cuadros por segundo (la CPU, menos de uno por proceso).

## Código

| Archivo | |
|---|---|
| `brutal_viz.py` | el comando de siempre (llama al paquete `medina`) |
| `medina/audio.py` | leer el archivo, tempo, beats, control de audio mudo, semilla del audio |
| `medina/analysis.py` | escuchar el track: kicks, sub, tensión, liberaciones, respiración, caos, hats, silencios, capas, tonalidad |
| `medina/score.py` | la partitura: el viaje, el movimiento, la cámara y el color de cada cuadro |
| `medina/emotion.py` | la capa emocional: qué emoción transmite cada momento (incertidumbre, miedo, esperanza, enigma, soledad, fuerza, vulnerabilidad), las secciones del track y su ADN. La fuerza marca la velocidad de la luz |
| `medina/system.py` | el sistema estelar de cada track, a partir de su ADN |
| `medina/styles.py` | los estilos (partículas y disco) y la lente gravitacional |
| `medina/space.py` | lo común a todos: estrellas, silencio, fenómenos de las capas, agujero negro |
| `medina/post.py`, `medina/logo.py` | acabado reactivo y firma |
| `medina/render.py` | arma cada cuadro a partir de la partitura (grilla, supersampling, cámara) |
| `medina/gpu.py`, `medina/shaders/` | el motor de GPU: la escena y el acabado en shaders GLSL |
| `medina/output.py`, `medina/cli.py` | video, tramos, cuadros sueltos, render en paralelo y línea de comandos |
| `medina/presets.py` | presets, looks y paletas |
| `master.py` | master técnico aparte: EQ, M/S, compresión de bus, limitador con objetivo de LUFS y true peak (necesita `pyloudnorm` y `pedalboard`) |
| `tracks/`, `renders/` | locales, fuera del repo |

El modo viejo para sets (espectro, video del celular, barras, túnel) y los estilos que el viaje ya no usa quedaron en la etiqueta `v0.1-antes-de-modularizar`. Los arcos y las órbitas vistas de frente (que se descartaron por no parecer parte del espacio) quedaron en el commit `b68817f`.

## Tests

```bash
.venv/bin/python -m pytest -q
```

- **Track sintético** (`tests/synth.py`): tiene una historia conocida (kick con sub, un tramo filtrado, la liberación en un beat exacto, hats en corcheas y un sinte que entra), así que los tests no necesitan tu música. Verifican el tempo, los kicks, la liberación, la tensión, la respiración, los hats, las capas, la partitura (determinista, el viaje cambia solo en liberaciones, cámara y paletas acotadas), el sistema estelar y la velocidad de la luz, y el dibujo con el sistema neutro y con el del ADN (mucho negro, respiración a negro, cuadros sueltos independientes, tramos iguales al video completo). También hay cuadros de referencia en `tests/golden/`: si un cambio de look es intencional, se regeneran con `.venv/bin/python -m tests.make_golden`.
- **Motor de GPU** (`tests/test_gpu.py`): se ve igual que el de la CPU (con tolerancia), cuadros sueltos independientes y tramos iguales al video completo. Si no hay moderngl u OpenGL, se saltean.
- **Tus tracks** (`tests/test_real_tracks.py`): si están en `tracks/`, se verifican el tempo y las liberaciones validadas contra el espectrograma. Si no están, se saltean.

Para verificar que un cambio no altera la imagen de tus tracks (por ejemplo, el paso a GPU):

```bash
.venv/bin/python tools/referencias.py guardar renders/_ref/base [adn|neutro]   # antes, con la versión aprobada
.venv/bin/python tools/referencias.py comparar renders/_ref/base               # después (con el mismo sistema)
```

## Pendientes conocidos

- **Kicks que caen justo antes del centro de un cuadro.** El control de "¿hay graves en este beat?" mira un solo cuadro, y un kick corto que cae un poco antes del centro del cuadro no se cuenta. En los tracks actuales no pasa: se detecta el 100 % de los kicks en los tramos con sub. Pero en un tempo que dura un número entero de cuadros (por ejemplo, 120 BPM = 15 cuadros por beat) podrían perderse todos. El test `test_kicks_en_cualquier_fase_del_cuadro` lo documenta como falla conocida.
  - **Arreglo probado:** mirar ±1 cuadro, como ya hace el flujo del kick.
  - **Qué cambia:** cuenta también los kicks de tus tramos filtrados (+12, +58 y +31 en los tres tracks), y en `Untitled 9` agrega una liberación falsa a mitad de la tensión larga (3:28).
  - **Qué falta:** validarlo visualmente. La regla de "vuelve el kick" ya exige que el kick se quede (un golpe suelto no cuenta), lo que evitaba dos liberaciones falsas en `Untitled13`; falta probar si con eso alcanza para la falsa de `Untitled 9`.

## Próximo paso

Sobre el motor de GPU: pasar también las estrellas y los fenómenos a la GPU (hoy son lo que más tarda), vista previa en tiempo real con el audio, y el agujero negro en 3D (raymarching, con la curvatura real de la luz).
