#version 330
// Inflación: el espacio se estira exponencialmente y las fluctuaciones cuánticas crecen hasta
// volverse la estructura del universo.
// - El campo llena todo el espacio (cada píxel es un rayo que lo atraviesa): negro, con filamentos y
//   nudos de luz (las crestas del ruido, la misma materia que la Singularidad).
// - Se viaja a través del campo, siempre hacia adelante y cada vez más rápido (la CPU mueve la
//   cámara, sin nada elástico). Además el espacio se estira: cada octava de fluctuaciones se agranda
//   (las nuevas aparecen finas; las grandes se pierden). Las finas, por dentro del horizonte,
//   fluctúan; al salir del horizonte se congelan y crecen quietas: así nace la estructura.
// - Cada sonido del track enciende su escala: lo grave, los filamentos grandes; lo agudo, lo fino.
// - Cada sonido es además una luz dentro de la red (y cada ataque, un destello breve): no se ve la
//   fuente, solo los filamentos que le pasan cerca, como un farol en una nebulosa (blancos cerca,
//   violetas al alejarse).
// - El recalentamiento: al final, el campo se vuelve luz y se apaga.

uniform vec2 out_size;
uniform float aspect, focal;
uniform vec3 fw, rt, up, cam;            // la cámara: orientación y posición (viaja a través del campo)
uniform sampler3D noise;                 // ruido 3D periódico (el mismo de la Singularidad)
uniform float zf;                        // parte fraccionaria del estiramiento (en octavas)
uniform int zi;                          // parte entera (identidad de cada octava)
uniform float zt;                        // estiramiento total (reloj de las fluctuaciones)
uniform float base_f, hz_off;            // frecuencia de la octava más gruesa; cruce del horizonte
uniform float dens_k, thr, sigma, depth, expo;
uniform float fine_k, coarse_k, kick, tint, light, reheat, fade_k;
uniform vec3 accent;
uniform float pxs;                       // un píxel de salida en unidades de pantalla (antialias)
uniform int n_ly;
uniform vec4 ly[4];                      // capas: escala (0 grueso .. 1 fino), actividad, blancura, -
uniform int n_lt;
uniform vec4 lt_a[6];                    // luces: posición (coordenadas de la cámara) e intensidad
uniform vec4 lt_b[6];                    // luces: radio, -, -, -
out vec4 color;

const int NO = 4;

float hashf(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
float hash1(int n) {
    uint h = uint(n) * 0x9E3779B1u;
    h ^= h >> 15u; h *= 0x85EBCA77u; h ^= h >> 13u;
    return float(h & 0xFFFFFFu) / 16777215.0;
}
vec3 off(int J) { return vec3(hash1(J * 3 + 1), hash1(J * 3 + 2), hash1(J * 3 + 3)) * 7.0; }
// cáscara: 1 sobre la superficie donde el ruido vale 0.5, 0 lejos. El cruce de dos cáscaras es un
// filamento (una línea en 3D); el de tres, un nudo. Así cada rayo cruza pocos: negro con hilos de luz.
float shell(float n, float w) { return max(0.0, 1.0 - abs(n - 0.5) / w); }

// densidad (x), blancura (y) y entorno de los filamentos (z: solo se ve si algo lo ilumina) del campo
// en un punto (en coordenadas de la cámara)
vec3 field(vec3 pc, float t) {
    vec3 pw = cam + rt * pc.x + up * pc.y + fw * pc.z;
    float s = 0.0, white = 0.0, halo = 0.0;
    for (int j = 0; j < NO; j++) {
        float lvl = float(j) - zf;                       // -1..NO-1 (más alto = más fino)
        float x = (lvl + 1.0) / float(NO);               // 0 grueso .. 1 fino
        float w = smoothstep(0.0, 0.22, x) * (1.0 - smoothstep(0.82, 1.0, x));
        // antialias: si los filamentos de esta escala quedan más finos que un píxel, se desvanecen
        float fpx = 0.0016 / (base_f * exp2(lvl)) * focal / max(t, 0.05) / pxs;
        w *= smoothstep(0.7, 1.8, fpx);
        if (w < 0.001) continue;
        int J = j + zi;
        // las finas fluctúan; al cruzar el horizonte su reloj se detiene (se congelan)
        float tau = min(zt, float(J) + hz_off);
        vec4 n = texture(noise, pw * base_f * exp2(lvl) + off(J) + vec3(0.0, 0.0, 0.11 * tau));
        float wd = thr;                                  // ancho de los filamentos (la tensión los afina)
        float fil = shell(n.w, wd) * shell(n.x, wd);
        float node = shell(n.w, 1.4 * wd) * shell(n.x, 1.4 * wd) * shell(n.y, 1.4 * wd);
        float boost = 1.0;
        for (int k = 0; k < n_ly; k++) boost += ly[k].y * 1.5 * exp(-pow((x - ly[k].x) / 0.2, 2.0));
        float sc = w * mix(coarse_k, fine_k, x);
        float a = sc * (fil * fil + 3.0 * node) * boost;
        s += a;
        white += a * x;
        float hw = shell(n.w, 2.6 * wd) * shell(n.x, 2.6 * wd);
        halo += sc * hw * hw;
    }
    return vec3(s, white / (s + 1e-4), halo);
}

void main() {
    vec2 uv = (gl_FragCoord.xy / out_size * 2.0 - 1.0) * vec2(aspect, 1.0);
    vec3 dir = normalize(vec3(uv.x, -uv.y, focal));
    vec3 violet = mix(vec3(1.0), accent, tint);
    vec3 col = vec3(0.0);
    float T = 1.0;
    float t = 0.04 + 0.04 * hashf(gl_FragCoord.xy);          // sin bandas: cada rayo arranca distinto
    for (int k = 0; k < 96; k++) {
        if (t > depth) break;
        float dt = 0.018 + 0.03 * t;
        vec3 fd = field(dir * t, t);
        float dn = fd.x * dens_k;
        float dl = fd.z * dens_k * float(n_lt > 0);
        if (dn > 0.0001) {
            vec3 c = mix(violet, vec3(1.0), clamp(fd.y * 1.3, 0.0, 1.0));
            // lo que pasa pegado a la cámara no se vuelve una mancha: aparece desde un poco más lejos
            col += T * dn * c * exp(-t / (0.5 * depth)) * smoothstep(0.04, 0.25, t) * dt * (1.0 + 0.6 * kick);
        }
        if (dl > 0.0001) {
            // las luces: iluminan los filamentos y su entorno cerca de ellas (cae rápido con la distancia)
            vec3 pc = dir * t;
            for (int j = 0; j < n_lt; j++) {
                vec3 dd = pc - lt_a[j].xyz;
                float q2 = dot(dd, dd) / (lt_b[j].x * lt_b[j].x);
                vec3 lc = mix(vec3(1.0), violet, smoothstep(0.1, 1.5, q2));
                col += T * dl * lc * lt_a[j].w / ((1.0 + q2) * (1.0 + q2)) * exp(-t / depth) * dt;
            }
        }
        if (dn > 0.0001) {
            T *= exp(-sigma * dn * dt);
            if (T < 0.03) break;
        }
        t += dt;
    }
    col = 1.0 - exp(-expo * col);
    // el recalentamiento: el campo se vuelve luz (un resplandor que llena todo) y se apaga
    col += reheat * mix(violet, vec3(1.0), 0.5) * (0.25 + 0.75 * col);
    color = vec4(clamp(col * light * fade_k, 0.0, 1.0), 1.0);
}
