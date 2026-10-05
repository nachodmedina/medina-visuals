#version 330
// El agujero negro en 3D. Cada píxel es un rayo de luz que viaja hacia atrás desde la cámara y se
// curva por la gravedad (geometría de Schwarzschild, en radios de Schwarzschild: el horizonte es
// r = 1, la esfera de fotones r = 1.5, la sombra se ve de radio ~2.6).
// - Lo que cae es negro absoluto.
// - Lo que roza la esfera de fotones deja la luz del borde: un eclipse finísimo, más intenso de un
//   lado, que ondula con el track (y el eco de luz que da la vuelta en cada liberación).
// - Lo que se escapa ve el cielo, curvado por el agujero: estrellas y una banda galáctica fijas en
//   el espacio (cerca del borde se doblan, se duplican y se cierran en anillos).
// - A la velocidad de la luz (aberración relativista) todo se junta hacia el agujero y se enciende.
// - La onda gravitacional estira y comprime el espacio-tiempo a su paso.

uniform vec2 out_size;
uniform vec4 box;
uniform vec2 grid;
uniform float dist, tg, roll, tilt;
uniform int silent, closed;
uniform float camD, elev, azim, focal;   // cámara: distancia (radios de Schwarzschild), elevación, órbita
uniform float glow, kick, tens, hat, ring_k, light, warp, star_px, t_s;
uniform int hat_n, seed;
uniform vec3 accent;                     // 0..1
// el carácter del agujero de cada track (sale de su ADN)
uniform float ring_w, ring_amp, ring_ang, ring_tint, star_dens;
uniform vec3 band_n;                     // la banda galáctica: su plano (normal), ancho, brillo, polvo
uniform float band_w, band_k, band_dust;
// el gesto de la liberación
uniform float wave_r, wave_amp;          // la onda gravitacional (radio en pantalla, amplitud)
uniform float echo_ang, echo_k;          // el eco de luz que da la vuelta al borde
out vec4 color;

const float TAU = 6.283185307179586;
const float PI = 3.141592653589793;

float hash32(int a, int b, uint t) {
    uint h = (uint(a) * 73856093u) ^ (uint(b) * 19349663u) ^ (t * 83492791u);
    h ^= h >> 13u;
    h *= 0x5bd1e995u;
    h ^= h >> 15u;
    return float(h & 0xFFFFFFu) / 16777215.0;
}

float nerv() {
    if (closed == 0 || silent == 1) return 0.0;
    return clamp((tens - 0.35) / 0.65, 0.0, 1.0);
}

// ruido de valores 3D (para el polvo de la banda)
float h3(ivec3 c) { return hash32(c.x + 1973 * c.z, c.y - 9277 * c.z, uint(seed) + uint(c.z & 1023)); }
float vnoise(vec3 p) {
    vec3 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    ivec3 c = ivec3(i);
    float n000 = h3(c), n100 = h3(c + ivec3(1, 0, 0)), n010 = h3(c + ivec3(0, 1, 0)), n110 = h3(c + ivec3(1, 1, 0));
    float n001 = h3(c + ivec3(0, 0, 1)), n101 = h3(c + ivec3(1, 0, 1)), n011 = h3(c + ivec3(0, 1, 1)), n111 = h3(c + ivec3(1, 1, 1));
    return mix(mix(mix(n000, n100, f.x), mix(n010, n110, f.x), f.y), mix(mix(n001, n101, f.x), mix(n011, n111, f.x), f.y), f.z);
}
float fbm(vec3 p) {
    float a = 0.5, s = 0.0;
    for (int o = 0; o < 5; o++) { s += a * vnoise(p); p *= 2.03; a *= 0.5; }
    return s;
}

// la banda galáctica (fija en el espacio): estrellas lejanas sin resolver y polvo oscuro
float galaxy(vec3 d) {
    float bd = dot(d, band_n);
    float core = exp(-pow(bd / band_w, 2.0));
    if (core < 0.002) return 0.0;
    float glowb = fbm(d * 5.0 + 1.7);
    float lanes = smoothstep(0.42, 0.7, fbm(d * 11.0 + 7.3)) * band_dust;      // carriles de polvo
    float grainb = vnoise(d * 160.0);                                          // estrellas sin resolver
    return band_k * core * (0.3 + 0.7 * glowb) * (1.0 - 0.75 * lanes) * (0.75 + 0.5 * grainb);
}

// estrellas fijas en el espacio (cuadrícula sobre las caras de un cubo), más densas en la banda;
// con los hats algunas se encienden y se apagan suave
float stars(vec3 d) {
    vec3 ad = abs(d);
    int face;
    vec2 uv;
    float m;
    if (ad.x >= ad.y && ad.x >= ad.z) { face = d.x > 0.0 ? 0 : 1; uv = d.yz / ad.x; m = ad.x; }
    else if (ad.y >= ad.z) { face = d.y > 0.0 ? 2 : 3; uv = d.xz / ad.y; m = ad.y; }
    else { face = d.z > 0.0 ? 4 : 5; uv = d.xy / ad.z; m = ad.z; }
    const float CS = 0.0045;
    vec2 cc = floor(uv / CS);
    float dens = star_dens * (0.5 + 2.5 * exp(-pow(dot(d, band_n) / (band_w * 1.6), 2.0)));
    float acc = 0.0;
    for (int j = -1; j <= 1; j++) for (int i = -1; i <= 1; i++) {
        ivec2 q = ivec2(cc) + ivec2(i, j);
        if (hash32(q.x + 7919 * face, q.y + 4049, uint(seed)) > 0.035 * dens) continue;
        vec2 sp = (vec2(q) + vec2(hash32(q.x, q.y + 31 * face, 11u), hash32(q.y, q.x + 17 * face, 13u))) * CS;
        float ang = length(uv - sp) * m * m / star_px;                     // en píxeles de salida
        float b = 0.12 + 0.88 * pow(hash32(q.x + 5, q.y + 9 + face, 7u), 5.0);
        if (hash32(q.x + 31, q.y + 77 + face, uint(hat_n)) < 0.05) b *= 1.0 + 3.0 * hat * smoothstep(0.1, 0.5, hat);
        acc += b * exp(-ang * ang / 0.55);
    }
    return acc;
}

void main() {
    vec2 g = box.xy + gl_FragCoord.xy / out_size * (box.zw - box.xy);
    if (silent == 1) {                                          // negro y una sola línea
        float h = floor(grid.y / 2.0);
        color = vec4(vec3((g.y >= h && g.y < h + tg) ? 1.0 : 0.0), 1.0);
        return;
    }
    float r0 = grid.y / 2.0;
    float xx = (g.x - grid.x / 2.0) / r0 * dist;
    float yy = (g.y - grid.y / 2.0) / r0 * dist;
    float al = radians(tilt) + roll;                            // orientación del eclipse y giro de cámara
    float sx = xx * cos(al) + yy * sin(al);
    float sy = -xx * sin(al) + yy * cos(al);
    // onda gravitacional: cuando el tiempo vuelve a correr, sale del agujero y cruza todo
    if (wave_amp > 0.0) {
        float R = length(vec2(sx, sy));
        float d = R - wave_r;
        float rip = wave_amp * sin(TAU * d / 0.25) * exp(-(d / 0.35) * (d / 0.35));
        sx *= 1.0 + rip;
        sy *= 1.0 + rip;
    }
    // la cámara mira al agujero y orbita a su alrededor
    vec3 C = camD * vec3(sin(azim) * cos(elev), sin(elev), -cos(azim) * cos(elev));
    vec3 fw = normalize(-C);
    vec3 rt = normalize(cross(vec3(0.0, 1.0, 0.0), fw));
    vec3 up = cross(fw, rt);
    vec3 p = C;
    vec3 v = normalize(fw * focal + rt * sx - up * sy);
    // aberración relativista: a casi la velocidad de la luz, lo que se ve adelante viene de un ángulo
    // mucho más abierto (el cielo y el agujero se juntan al centro) y brilla más
    float beta = 0.6 * smoothstep(0.3, 1.0, warp);
    float boost = 1.0;
    if (beta > 0.001) {
        float c1 = dot(v, fw);
        float c0 = (c1 - beta) / (1.0 - beta * c1);
        vec3 perp = v - fw * c1;
        float pl = length(perp);
        if (pl > 1e-5) v = fw * c0 + perp / pl * sqrt(max(1.0 - c0 * c0, 0.0));
        boost = clamp(pow((1.0 - beta * c0) / sqrt(1.0 - beta * beta), -1.5), 0.3, 4.0);
    }
    // el rayo, curvado por la gravedad
    vec3 L = cross(p, v);
    float h2 = dot(L, L);
    float rmin = 1e9;
    bool captured = false;
    for (int k = 0; k < 500; k++) {
        float r = length(p);
        rmin = min(rmin, r);
        if (r < 1.0) { captured = true; break; }
        if (r > camD + 40.0 && dot(p, v) > 0.0) break;          // se escapó
        float dt = clamp(0.06 * r * r / (r + 2.0), 0.015, 2.0);
        v += -1.5 * h2 * p / pow(r, 5.0) * dt;
        p += v * dt;
    }
    vec3 col = vec3(0.0);
    if (!captured) {
        // la luz del borde: más intensa de un lado (eclipse), ondula con el beat del track, late con
        // el kick, se enciende en la liberación y al final de la tensión titila con los hats
        float tha = atan(sy, sx);
        float wob = 0.5 * sin(3.0 * tha + 0.9 * t_s) + 0.3 * sin(5.0 * tha - 1.3 * t_s) + 0.2 * sin(9.0 * tha + 2.1 * t_s);
        float x = (rmin - 1.5 - ring_amp * wob) / (ring_w * (1.0 + 0.5 * glow));
        float shape = 0.8 * exp(-x * x) + 0.22 * exp(-abs(x) / 5.0);
        float side = 0.5 + 0.5 * cos(tha - ring_ang);
        float flow = (0.6 + 0.4 * sin(4.0 * tha - 1.7 * t_s + 1.3 * sin(2.0 * tha + 0.7 * t_s))) / max(light, 0.75);
        float ring = ring_k * (0.25 + 0.75 * side) * flow * (1.0 + 1.5 * kick + 2.0 * glow + 1.5 * nerv() * hat) * shape;
        col += ring * mix(vec3(1.0), accent, ring_tint * (1.0 - 0.6 * side));
        // el eco de luz: un destello que da una vuelta al borde mientras el tiempo está quieto
        if (echo_k > 0.0) {
            float da = mod(tha - echo_ang + PI, TAU) - PI;
            float e = exp(-da * da / 0.03) + (da < 0.0 ? 0.55 * exp(da / 0.9) : 0.0);
            col += echo_k * 2.2 * e * shape * vec3(1.0);
        }
        // el cielo, curvado por el agujero (la gravedad magnifica lo que pasa cerca)
        vec3 dd = normalize(v);
        float att = 1.0 - 0.5 * (closed == 1 ? tens : 0.0);     // la atracción apaga el cielo
        float s = stars(dd) * boost * att * (1.0 + 2.0 * exp(-(rmin - 1.5) / 1.2));
        col += vec3(min(s, 1.2));
        col += mix(vec3(1.0), accent, 0.35) * galaxy(dd) * att * min(boost, 1.8);
    }
    color = vec4(clamp(col * light, 0.0, 1.0), 1.0);
}
