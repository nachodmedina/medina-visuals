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
- Mejoras pendientes propuestas: capas del track como eventos lejanos alrededor del agujero, emociones momento a momento, inicio (emerger del negro) y final (cruzar el horizonte), entrega para YouTube (menos grano, 4K).

## El EP: KHAOS Y KOSMOS

Contar con sonidos cómo se desarrolló el universo hasta hoy, por sus eventos científicos canónicos. Una escena por track (se elige por el título), con el lenguaje compartido; cada track termina donde empieza el siguiente.

- **KHAOS**: Singularidad · Radio de Schwarzschild (el agujero negro actual) · Inflación
- **KOSMOS**: Nucleosíntesis · Materia Oscura · Radiación de Fondo · Colapso · Gran Congelamiento
- **Bonus**: Energía del vacío

Se trabaja de a una escena, a medida que el usuario termina cada track.
