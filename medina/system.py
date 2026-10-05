"""El sistema estelar de cada track: los parámetros visuales que salen de su ADN.

Todos los tracks hablan el mismo idioma (el viaje, el agujero, la paleta), pero cada uno tiene
su propio sistema: más o menos masa, un disco más plano o más inclinado, órbitas más lentas o
vertiginosas, más o menos materia, estrellas, luz, grano... El sistema `neutro` (los valores por
defecto) reproduce el motor sin ADN.
"""
from dataclasses import dataclass

L = lambda a, b, t: a + (b - a) * t

# qué fenómenos del espacio prefiere cada rasgo del ADN
PREFER = dict(hipnosis=("pulsar", "estrella"), energia=("jets",), vacio=("luna",), caos=("cometa", "meteoros"))
PREFER_LABEL = dict(hipnosis="pulsares y estrellas", energia="jets", vacio="lunas solitarias", caos="cometas y meteoros")


@dataclass(frozen=True)
class StarSystem:
    hole: float = 0.07             # radio base del horizonte
    lens: float = 1.0              # alcance de la lente y del remolino pegado al borde
    tilt: float = -24.0            # inclinación del disco en grados (negativo: diagonal que sube)
    incl: float = 0.21             # apertura del disco (0.16 = casi de canto; 0.32 = elipse abierta)
    orbit: float = 1.0             # velocidad orbital (giro de todo el sistema)
    matter: float = 1.0            # densidad de materia (partículas y órbitas)
    turbulence: float = 1.0        # ondas espirales y onda gravitacional
    stars: int = 900               # estrellas de fondo
    light: float = 1.0             # resplandor: bloom y halo del horizonte
    violet: tuple = (125, 55, 200)  # tono del violeta (dentro de la paleta)
    accent: float = 1.0            # proporción de acento (× la de siempre)
    grain: float = 1.0             # grano
    trail: float = 1.0             # duración de la estela
    distance: float = 1.0          # distancia de la cámara (más lejos: todo más chico)
    shake: float = 1.0             # temblor de cámara en los kicks
    phenomena: tuple = ()          # fenómenos preferidos, en orden
    fast_journey: bool = False     # el viaje sube un escalón en cada liberación
    warp: float = 0.0              # velocidad de la luz al escapar del agujero (0 = nunca)

    @classmethod
    def neutral(cls):
        return cls()

    @classmethod
    def from_axes(cls, ax):
        """ADN (ejes 0..1 de emotion.dna_axes) -> sistema estelar."""
        top = sorted(PREFER, key=lambda k: -ax[k])[:2]
        return cls(
            hole=L(0.055, 0.12, ax["voragine"]),
            lens=L(0.75, 1.5, ax["voragine"]),
            tilt=-L(8, 40, ax["caos"]),
            incl=L(0.16, 0.32, ax["voragine"]),
            orbit=L(0.6, 1.5, ax["energia"]),
            matter=L(0.6, 1.4, ax["densidad"]),
            turbulence=L(0.4, 2.0, ax["caos"]),
            stars=int(L(300, 1800, ax["vacio"])),
            light=L(0.7, 1.4, ax["luz"]),
            violet=tuple(int(round(L(a, b, ax["luz"]))) for a, b in ((86, 150), (36, 95), (150, 230))),
            accent=L(10, 30, ax["energia"]) / 20,
            grain=L(0.5, 2.0, ax["aspereza"]),
            trail=L(0.7, 1.8, ax["vacio"]),
            distance=L(0.95, 1.3, ax["vacio"]),
            shake=L(0.6, 1.4, ax["energia"]),
            phenomena=tuple(p for k in top for p in PREFER[k]),
            fast_journey=ax["energia"] > 0.6,
            warp=L(0.75, 1.25, ax["energia"]),
        )


def describe(ax):
    """El sistema de un ADN en palabras: (parámetro, valor, eje que lo define, qué expresa)."""
    s = StarSystem.from_axes(ax)
    fav = " y ".join(PREFER_LABEL[k] for k in sorted(PREFER, key=lambda k: -ax[k])[:2])
    return [
        ("Masa del agujero", f"radio {s.hole:.3f} · lente ×{s.lens:.2f}", "vorágine",
         "más atracción: un agujero más grande que curva más el espacio"),
        ("Disco", f"inclinación {-s.tilt:.0f}° · apertura {s.incl:.2f}", "caos · vorágine",
         "orden: plano y estable; caos: inclinado, inquieto; más masa: la elipse se abre"),
        ("Velocidad orbital", f"×{s.orbit:.2f}", "energía", "cuánto empuja el track: órbitas lentas o vertiginosas"),
        ("Densidad de materia", f"×{s.matter:.2f}", "densidad", "cantidad de partículas, órbitas y segmentos"),
        ("Turbulencia", f"ondas ×{s.turbulence:.2f}", "caos", "órbitas que se deforman y ondas en el espacio"),
        ("Estrellas de fondo", f"{s.stars} estrellas", "vacío", "cuánto espacio profundo se ve: la calma del vacío"),
        ("Luz", f"resplandor ×{s.light:.2f}", "luz", "intensidad del bloom y del halo del horizonte"),
        ("Tono del violeta", "#{:02x}{:02x}{:02x}".format(*s.violet), "luz",
         "dentro de la paleta: violeta profundo (oscuro) ↔ lavanda (luminoso)"),
        ("Proporción de acento", f"{s.accent * 20:.0f} %", "energía", "cuánto violeta frente al blanco"),
        ("Grano", f"×{s.grain:.2f}", "aspereza", "textura: limpio y liso, o áspero como el ruido del track"),
        ("Estela", f"×{s.trail:.2f}", "vacío", "cuánto dura el recuerdo de lo que pasó"),
        ("Cámara", f"distancia ×{s.distance:.2f} · temblor ×{s.shake:.2f}", "vacío · energía",
         "lejos y quieta en el vacío; cerca y golpeada con energía"),
        ("Fenómenos preferidos", fav, "hipnosis · energía · vacío · caos",
         "qué fenómenos del espacio tienden a aparecer en este sistema"),
        ("Ritmo del viaje", "rápido" if s.fast_journey else "pausado", "energía",
         "cuántos escalones sube el viaje por liberación"),
        ("Velocidad de la luz", f"salto ×{s.warp:.2f}", "energía",
         "al escapar del agujero el espacio se estira: estrellas en líneas, túnel acelerado"),
    ]
