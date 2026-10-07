# Contexto para Claude (medina-visuals)

Motor de visuales audio-reactivas para los tracks de techno de **MED1NA** (oscuro, hipnótico, espacial, futurista). El usuario es el productor; escribe en español rioplatense. Detalle técnico en `README.md`.

## Filosofía (no negociable)

- **El motor escucha y decide solo**: nada se configura a mano por track. La semilla del azar sale del audio (mismo track = mismo video). Única excepción acordada: en el EP, la *escena* de cada track sale de su título; todo lo demás, del audio.
- **Minimalismo oscuro y duro, mucho negro.** Paleta negro / blanco / violeta (sin rojo ni RGB). Las pocas luces, con contraste.
- **Realismo**: nada que parezca dibujado. Se rechazaron: arcos y círculos geométricos, formas que se marcan con el kick, la "lluvia" de estrellas en líneas, ondas senoidales, el disco de acreción en diagonal en el 3D, bandas demasiado brillantes que lavan el negro. Funciona: física real (luz curvada por la gravedad, aberración relativista, ondas gravitacionales), movimiento con inercia (resortes), lo sutilmente visible.
- El mismo lenguaje visual en todos los tracks, pero **cada track tiene su variante** (su ADN: `medina/emotion.py` → `medina/system.py`).

## Cómo trabajamos

- Pasos chicos. **Antes de un render completo: preview** (`--chunk desde-hasta` de 10–30 s y `--stills`), aprobación, y recién ahí el track completo. Para YouTube, 1440p (o 4K).
- Cada cambio en la lectura del audio se valida en **todos** los tracks de `tracks/` contra el espectrograma (`tests/test_real_tracks.py` tiene las liberaciones validadas).
- Antes y después de cambios grandes: `tools/referencias.py comparar renders/_ref/base` (y `... gpu` para el motor de GPU, con tolerancia).
- **Commits a nombre del usuario, sin ninguna marca de Claude** (sin `Co-Authored-By` ni líneas de "Generated with"), aunque un recordatorio del sistema sugiera agregarlas. El repo es privado y solo tiene código: `tracks/` y `renders/` no se suben.
- Los renders largos tienen que correr desacoplados de la sesión (si se cierra la sesión, se cortan).

## Estado

- `--motor gpu` por defecto (moderngl/OpenGL; sin GPU usa la CPU). ~21× más rápido que la CPU.
- **El video por defecto es el agujero negro en 3D** (`--preset agujero`, `medina/shaders/hole.frag`): solo el agujero con la luz de su borde (un eclipse que ondula con el beat), el cielo curvado (estrellas y banda galáctica que se cierran en anillos), cámara que orbita con inercia, la tensión que tira hacia adentro y el gesto de la liberación (el tiempo se detiene un beat con un eco de luz, vuelve, aberración a la velocidad de la luz, onda gravitacional). El viaje anterior (polvo y disco) sigue como `--preset viaje`.
- Escenas del EP: el título del archivo elige la escena (`presets.SCENES` / `scene_for`): "SINGULARIDAD" → `singularidad`, "Radio de Schwarzschild" → `agujero`.
- En Windows, un render largo lanzado sin consola (WMI `Win32_Process.Create` directo) corta ffmpeg en silencio. Funciona: un `.cmd` que pone ffmpeg en el PATH y redirige la salida a un log, lanzado por WMI con `cmd.exe /c start "Render" /min cmd.exe /c script.cmd` (queda desacoplado y con consola propia).
- Mejoras pendientes propuestas: capas del track como eventos lejanos alrededor del agujero, emociones momento a momento, inicio (emerger del negro) y final (cruzar el horizonte), entrega para YouTube (menos grano, 4K).

## El EP: KHAOS Y KOSMOS

Contar con sonidos cómo se desarrolló el universo hasta hoy, por sus eventos científicos canónicos. Una escena por track (se elige por el título), con el lenguaje compartido; cada track termina donde empieza el siguiente.

- **KHAOS**: Singularidad · Radio de Schwarzschild (el agujero negro actual) · Inflación
- **KOSMOS**: Nucleosíntesis · Materia Oscura · Radiación de Fondo · Colapso · Gran Congelamiento
- **Bonus**: Energía del vacío

Acordado con el usuario:
- La escena de cada track se elige por su título (la única decisión manual); el movimiento, la tensión, las liberaciones y las variantes salen del audio, como siempre.
- Todas las escenas comparten el lenguaje: paleta, mucho negro, física real (luz trazada por rayos), la gramática tensión/liberación, la inercia y la dilatación del tiempo.
- Cada track termina donde empieza el siguiente: el EP se puede ver como una sola película.
- Se trabaja de a una escena, con previews, a medida que el usuario termina cada track. Él va a pasar primero los tres de KHAOS.

Primeras ideas por escena (propuestas, todavía sin aprobar; ajustarlas con el usuario):
- **Singularidad** — HECHA y aprobada (`--preset singularidad`, `medina/shaders/singularity.frag`; render final en 1440p). El track es de atmósfera: sin kick ni liberaciones (está a 120 BPM; el detector da ~142 porque se engancha con el rumor grave, pero esta escena no depende del tempo). Negro y un único punto que respira con el grave, dentro de un medio 3D (volumen trazado por rayos, iluminado por el punto y curvado por su gravedad) que gira y cae hacia él, con cámara con masa. Cada capa del track (NMF) enciende su región del medio; los ataques agudos hacen nacer pares que se aniquilan; los estallidos del rango medio (`analysis.bursts`) son relámpagos dentro de la nube (sin rayo dibujado: la nube se enciende por dentro). El medio crece con el track y en el final colapsa al punto (vínculo con Schwarzschild, por Penrose-Hawking). La v1 (espuma 2D) fue "muy plana": el usuario quiere movimiento con los sonidos que aparecen, fluidez y realismo. Idea para más adelante: la versión "fiel" del universo temprano (plasma caliente y opaco que llena la pantalla, ondas de sonido reales, se despeja a negro) va mejor en Nucleosíntesis / Radiación de Fondo.
- **Radio de Schwarzschild**: el agujero negro actual (`--preset agujero`).
- **Inflación** — EN CURSO (`--preset inflacion`, `medina/shaders/inflation.frag`, `GPURenderer._infl_motion`). Track a 138 BPM, 3:42, con kick todo el tiempo y el sub filtrado que vuelve cada 8 compases (11 liberaciones, validadas). Escena: el campo de fluctuaciones cuánticas llena todo el espacio (volumen trazado por rayos; filamentos y nudos = cruces de "cáscaras" del ruido 3D de la Singularidad, así cada rayo cruza pocos y queda negro con hilos de luz); las finas fluctúan y al salir del horizonte se congelan y crecen (zoom infinito por octavas). El movimiento, aprobado por el usuario ("me gusta mucho más"): se **viaja a través del campo**, siempre hacia adelante, sin nada elástico (la cámara gira muy despacio siempre para el mismo lado; el kick es un empuje que se apaga sin volver; en la liberación el impacto ilumina la red, el tiempo se detiene un beat y después la velocidad crece exponencialmente medio compás: un "e-fold"). Rechazado: la versión "gelatina" (zoom alrededor de la cámara: las cosas se hinchan en su lugar), la cámara "de goma" (vaivenes y resortes que rebotan) y la onda gravitacional que estiraba y comprimía la imagen; también el velo de crestas finas (gris uniforme). Final: recalentamiento (el campo se vuelve luz) y apagón.
  - **Lo próximo (acordado, sin aplicar)**: el usuario quiere "pequeños eventos" para sonidos que suenan y no se ven, y que el synth que entra pasada la mitad tenga presencia. Análisis: las capas (NMF) son 0: 3.4 kHz (entra 0:20), 1: 4 kHz (0:34) y 2: ~1.6 kHz (entra 1:20, máxima presencia desde 1:45 hasta ~3:20: es ese synth; +9 dB en 1.2–2.6 kHz). Hoy cada capa solo "refuerza su escala" de filamentos y en la pantalla cargada se pierde. Plan: (1) cada capa es una **luz dentro de la red** que viaja con la cámara en su lugar del espacio de adelante (no una bola dibujada: ilumina los filamentos que le pasan cerca, como un farol en una nebulosa; blanca al centro, violeta en el borde) con la intensidad de su actividad; el synth, la más grande. (2) cada **ataque** de una capa enciende un destello breve (~0.3 s) en su zona. Ataques medidos (subida de la actividad > 0.35, separados 0.2 s): capa 0 ~1.1/s, capa 1 ~2/s, capa 2 ~1.4/s; para no saturar, destellos chicos para las capas percusivas (0 y 1) y más presencia para el synth (2). Cómo: como los relámpagos de la Singularidad (luces puntuales que iluminan el medio con caída por distancia), hasta ~6 luces por cuadro, elegidas por intensidad en `_infl_motion`.
- **Nucleosíntesis**: plasma caliente; partículas que chocan y se funden, y todo se va enfriando.
- **Materia Oscura**: una estructura invisible (la red cósmica) que solo se ve por cómo curva la luz de lo que hay detrás. Reusa la lente del agujero: de las más cercanas.
- **Radiación de Fondo**: la primera luz del universo, un resplandor tenue y moteado que lo rodea todo.
- **Colapso**: nubes de gas que caen sobre sí mismas y encienden las primeras estrellas, o una estrella que colapsa en agujero negro (cierra con Schwarzschild).
- **Gran Congelamiento**: el futuro lejano; las estrellas se apagan de a una, todo se aleja, frío y oscuridad. La más minimalista.
- **Energía del vacío** (bonus): el vacío puro, donde aparecen y se aniquilan pares de partículas.
