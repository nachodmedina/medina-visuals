#version 330
// Singularidad: el estado previo a que todo se forme.
// - El punto: toda la materia y la energía en un lugar sin tamaño. Se ve como una fuente puntual a
//   través de una óptica real (núcleo que satura y resplandor de ley de potencia; sin anillos) y
//   respira con el grave del track.
// - El medio: un volumen que lo rodea (cada píxel es un rayo que lo atraviesa), iluminado por el punto
//   (la luz cae con la distancia al cuadrado) y curvado por su gravedad (los rayos se doblan cerca).
//   Es un fluido: gira más rápido cerca que lejos, cae hacia el punto y se agita con turbulencia.
// - Cada sonido que aparece en el track enciende y agita su región del medio: lo grave lejos y
//   amplio, lo agudo cerca del punto y más blanco.
// - Fluctuaciones cuánticas: pares que nacen de la nada, se separan apenas y se aniquilan.
// - Relámpagos: como una tormenta vista de lejos, la nube se enciende desde adentro un instante (no
//   se ve el rayo, solo la luz de su canal en el medio). Los disparan los estallidos del track.
// - El medio crece con el track (más extenso y más denso) y en el final cae todo hacia el punto.

uniform vec2 out_size;
uniform float aspect, focal;
uniform vec2 px;                       // un píxel de salida en unidades de pantalla (media altura = 1)
uniform vec3 cam, fw, rt, up;          // la cámara (con masa: la mueve la CPU)
uniform sampler3D noise;               // ruido 3D periódico (4 canales independientes)
uniform vec3 swirl_ax;                 // eje del giro del medio
uniform float swirl, flow, turb, warp_k, lens_k;
uniform float L0, med_k, sigma, ly_k;  // luz del punto, brillo del medio, absorción, brillo de las capas
uniform float core_k, halo_k, core_px, halo_px;
uniform float tint;
uniform vec3 accent;
uniform int n_ly;
uniform vec4 ly_d[8];                  // capas: dirección (xyz) y actividad
uniform vec4 ly_p[8];                  // capas: radio, ancho angular, blancura, -
uniform float env_r, dens_k;          // extensión y densidad del medio (crece con el track)
uniform int n_fl;
uniform vec4 fl_a[2];                  // relámpagos: un extremo del canal (xyz) e intensidad
uniform vec4 fl_b[2];                  // relámpagos: el otro extremo (xyz)
uniform vec3 fl_col;
uniform int n_ev;
uniform vec4 ev_a[48];                 // pares: posición en pantalla (x, y), eje, brillo
uniform vec2 ev_b[48];                 // pares: fase de vida 0..1, separación (pantalla)
out vec4 color;

float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }

vec3 rotate(vec3 p, vec3 k, float a) {           // Rodrigues
    float c = cos(a), s = sin(a);
    return p * c + cross(k, p) * s + k * dot(k, p) * (1.0 - c);
}

// una fuente puntual vista por una óptica: núcleo gaussiano (en píxeles) + resplandor de ley de potencia
float source(vec2 d, float sig_px, float glare_px) {
    float rp = length(d / px.y);
    return exp(-rp * rp / (2.0 * sig_px * sig_px)) + 0.035 * pow(1.0 + rp * rp / (glare_px * glare_px), -1.25);
}

// densidad del medio en p (y la posición en el fluido, para reusar)
float density(vec3 p, float r) {
    vec3 d = p / r;
    // giro diferencial (más rápido cerca) y caída hacia el punto (más rápida cerca)
    // la materia, en coordenadas log-radiales: cae hacia el punto sin llegar nunca (cada vez más
    // lenta, como cerca de un horizonte) y su forma no se degrada con el tiempo
    vec3 dr = rotate(d, swirl_ax, -swirl * (1.0 + 0.9 * exp(-r / 0.7)));
    vec3 q = dr * 1.6 + vec3(0.37, 0.71, 0.59) * (1.1 * log(r) + flow);
    // turbulencia: el fluido se deforma con un campo de ruido que evoluciona
    vec3 w = texture(noise, q * 0.045 + vec3(0.013, 0.007, 0.011) * turb).xyz - 0.5;
    q += warp_k * w;
    float a = 0.5, f = 0.085, s = 0.0;
    for (int o = 0; o < 4; o++) {
        float n = texture(noise, q * f + vec3(0.31, 0.17, 0.73) * float(o) + vec3(0.0, 0.0, 0.004) * turb).w;
        s += a * pow(1.0 - abs(2.0 * n - 1.0), 5.0);           // velos finos (crestas del ruido)
        f *= 2.07;
        a *= 0.55;
    }
    float env = exp(-r / env_r) * smoothstep(0.015, 0.09, r);
    return dens_k * pow(s / 0.8, 1.6) * env;
}

void main() {
    vec2 uv = (gl_FragCoord.xy / out_size * 2.0 - 1.0) * vec2(aspect, 1.0);
    vec3 v = normalize(fw * focal + rt * uv.x - up * uv.y);
    vec3 p = cam;
    vec3 violet = mix(vec3(1.0), accent, tint);
    vec3 col = vec3(0.0);
    float T = 1.0;
    p += v * 0.15 * hash(gl_FragCoord.xy);                    // sin bandas: cada rayo arranca distinto
    for (int k = 0; k < 140; k++) {
        float r = length(p);
        if (r < 0.006) break;                                   // cae en el punto
        if (r > 4.2 && dot(p, v) > 0.0) break;                  // se fue
        float dt = clamp(0.03 + 0.05 * r, 0.025, 0.12);
        v = normalize(v - lens_k * p / (r * r * r + 1e-4) * dt);   // la gravedad curva la luz
        float dn = density(p, r);
        if (dn > 0.002) {
            vec3 d = p / r;
            vec3 em = violet * med_k * L0 / (r * r + 0.03);     // el punto ilumina el medio
            // los sonidos encienden solo lo denso de su región (los filamentos, no una niebla)
            vec3 eml = vec3(0.0);
            for (int j = 0; j < n_ly; j++) {
                float act = ly_d[j].w;
                if (act < 0.01) continue;
                vec4 lp = ly_p[j];
                float ang = exp(-(1.0 - dot(d, ly_d[j].xyz)) / lp.y);
                float rad = exp(-pow((r - lp.x) / (0.3 * lp.x + 0.1), 2.0));
                eml += mix(violet, vec3(1.0), lp.z) * act * ang * rad;
            }
            // relámpagos: la luz del canal se dispersa en el medio y cae con la distancia
            for (int j = 0; j < n_fl; j++) {
                vec3 a = fl_a[j].xyz, ab = fl_b[j].xyz - a;
                float h = clamp(dot(p - a, ab) / dot(ab, ab), 0.0, 1.0);
                vec3 dd = p - a - ab * h;
                float d2 = dot(dd, dd);
                em += fl_col * fl_a[j].w * 2.0 * exp(-sqrt(d2) / 0.25) / (d2 + 0.03);   // se extingue en el medio
            }
            col += T * (dn * em + ly_k * dn * dn * eml) * dt;
            T *= exp(-sigma * dn * dt);
            if (T < 0.02) break;
        }
        p += v * dt;
    }
    col = 1.0 - exp(-1.15 * col);                             // altas luces suaves (los relámpagos no queman)
    // el punto (lo que hay delante lo vela apenas)
    vec3 o = -cam;
    float z = dot(o, fw);
    if (z > 0.0) {
        vec2 sp = vec2(dot(o, rt), -dot(o, up)) * focal / z;
        float rr = length(uv - sp);
        float s = source(uv - sp, core_px, 6.0);
        float tv = mix(1.0, T, 0.5);
        col += tv * (vec3(1.0) * core_k * s + mix(vec3(1.0), violet, 0.5) * halo_k
               * pow(1.0 + pow(rr / (halo_px * px.y), 2.0), -1.1));
    }
    // los pares que nacen y se aniquilan
    for (int k = 0; k < n_ev; k++) {
        vec4 e = ev_a[k];
        float q = ev_b[k].x;
        float sep = ev_b[k].y * sin(3.14159265 * q);                  // se separan y vuelven a juntarse
        vec2 ax = vec2(cos(e.z), sin(e.z));
        float life = e.w * pow(sin(3.14159265 * q), 0.6);
        float flash = e.w * 1.6 * exp(-pow((q - 0.97) / 0.04, 2.0)); // la aniquilación
        vec2 d0 = uv - e.xy;
        float b = life * (source(d0 - ax * sep, 1.0, 2.5) + source(d0 + ax * sep, 1.0, 2.5))
                + flash * source(d0, 1.3, 3.5);
        col += mix(vec3(1.0), violet, 0.35) * b;
    }
    color = vec4(clamp(col, 0.0, 1.0), 1.0);
}
