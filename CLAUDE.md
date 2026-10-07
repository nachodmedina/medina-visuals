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
- **Inflación** — HECHA y aprobada; falta el render final en 1440p (lo hace el usuario desde la PC) (`--preset inflacion`, `medina/shaders/inflation.frag`, `GPURenderer._infl_motion`). Track a 138 BPM, 3:42, con kick todo el tiempo y el sub filtrado que vuelve cada 8 compases (11 liberaciones, validadas). Escena: el campo de fluctuaciones cuánticas llena todo el espacio (volumen trazado por rayos; filamentos y nudos = cruces de "cáscaras" del ruido 3D de la Singularidad, así cada rayo cruza pocos y queda negro con hilos de luz); las finas fluctúan y al salir del horizonte se congelan y crecen (zoom infinito por octavas). El movimiento, aprobado por el usuario ("me gusta mucho más"): se **viaja a través del campo**, siempre hacia adelante, sin nada elástico (la cámara gira muy despacio siempre para el mismo lado; el kick es un empuje que se apaga sin volver; en la liberación el impacto ilumina la red, el tiempo se detiene un beat y después la velocidad crece exponencialmente medio compás: un "e-fold"). Rechazado: la versión "gelatina" (zoom alrededor de la cámara: las cosas se hinchan en su lugar), la cámara "de goma" (vaivenes y resortes que rebotan) y la onda gravitacional que estiraba y comprimía la imagen; también el velo de crestas finas (gris uniforme). Final: recalentamiento (el campo se vuelve luz) y apagón.
  - **Los sonidos como luces (aprobado: "me gusta")**: el usuario pidió "pequeños eventos" para sonidos que suenan y no se ven, y presencia para el synth que entra a 1:20 (capa 2, ~1.6 kHz; no es un colchón: golpes rítmicos de ~2 notas/s). Cada capa (NMF) es una **luz dentro de la red** que viaja con la cámara en su lugar del espacio de adelante (gira muy despacio, siempre para el mismo lado) con la presencia lenta de la capa; cada **ataque** (`analysis.layer_attacks`: la actividad sube > 0.35 en ~0.1 s con la capa sonando y sin silencio) enciende un **destello** breve fijo en el espacio, cerca de su luz (la cámara le pasa de largo; su vida corre con el tiempo de la imagen, así se congela en la liberación). No se ve la fuente: la luz ilumina los filamentos y su **entorno** (el campo tiene una densidad "iluminable", tubos 2.6× más anchos que los filamentos, que solo se ve si algo la ilumina; solo con los filamentos finos la luz no se notaba ni ×30). Blanca cerca, violeta al alejarse; caída `1/(1+q²)²`. El tamaño de cada luz sale de lo grave de su sonido (más grave, más grande: el synth domina sin configurar nada) y pesa todo: las capas chicas solo encienden sus ataques más fuertes (fuerza × peso ≥ 0.3) y cada luz necesita un respiro entre destellos (el synth, ~1 beat). Hasta 6 luces por cuadro (`_infl_lights`), `LANT_K, FLASH_K = 7, 15`. Con las luces el usuario pidió "un poco menos" de densidad: filamentos más finos y menos trama fina (`INFL_THR, INFL_FINE = 0.032, (0.14, 0.55)`): la red ocupa un tercio menos de pantalla. Medido: lo que marca el ritmo de la imagen es el kick y la trama fina, no los destellos.
- **Nucleosíntesis**: plasma caliente; partículas que chocan y se funden, y todo se va enfriando.
- **Materia Oscura**: una estructura invisible (la red cósmica) que solo se ve por cómo curva la luz de lo que hay detrás. Reusa la lente del agujero: de las más cercanas.
- **Radiación de Fondo**: la primera luz del universo, un resplandor tenue y moteado que lo rodea todo.
- **Colapso**: nubes de gas que caen sobre sí mismas y encienden las primeras estrellas, o una estrella que colapsa en agujero negro (cierra con Schwarzschild).
- **Gran Congelamiento**: el futuro lejano; las estrellas se apagan de a una, todo se aleja, frío y oscuridad. La más minimalista.
- **Energía del vacío** (bonus): el vacío puro, donde aparecen y se aniquilan pares de partículas.
