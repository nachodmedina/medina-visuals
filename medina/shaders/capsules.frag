#version 330
// Cápsulas: píldoras de dos mitades que caen en columnas sobre el negro.
// - Cada cápsula es un objeto 3D trazado por rayos (intersección exacta con dos cápsulas: el cuerpo
//   y la tapa, un poco más ancha, que monta sobre él; de ahí la costura).
// - El color es una película delgada (la física de una pompa de jabón): la luz que se refleja arriba
//   y abajo de la película interfiere, y según el espesor y el ángulo unas longitudes de onda se
//   refuerzan y otras se anulan. Las bandas se proyectan a la paleta (violeta / lavanda / blanco).
//   Cuando la película se adelgaza (la tensión) se vuelve negra, como una pompa antes de romperse.
// - La píldora roja: después de cada liberación cae una, entre las violetas (la única excepción a la
//   paleta, solo en esta escena).
// - En la liberación algunas se abren: las dos mitades se separan (copas abiertas: por el corte se ve
//   el interior) y sueltan polvo (dust.vert; la profundidad de acá lo tapa detrás de las cápsulas).
// - La CPU integra la caída y el giro (con inercia); cada celda saca su forma, su eje y su película
//   de un hash, así cada cuadro se dibuja solo.

uniform vec2 out_size;
uniform float aspect, focal;
uniform float D;                          // profundidad del plano de las cápsulas
uniform float sp;                         // separación entre columnas (y entre filas)
uniform float rad, hlen;                  // radio y medio largo del cuerpo (en unidades de sp)
uniform float vac;                        // fracción de celdas vacías
uniform float disorder;                   // 0 alineadas .. 1 cada una para su lado
uniform float fall, tumble;               // caída y giro integrados
uniform float d0, d_rng, thin;            // espesor de la película (nm), su variación; adelgazamiento 0..1
uniform float wave_r, wave_amp;           // onda de color de la liberación (radio en pantalla, amplitud)
uniform float bright, light, fade_k;
uniform vec3 accent;
uniform int seed;
uniform int n_ev;
uniform vec4 ev[8];                       // eventos: columna, fila, intensidad, edad (s)
uniform int n_red;
uniform vec2 red[4];                      // las píldoras rojas: columna, fila
uniform int n_op;
uniform vec4 op[6];                       // las que se abren: columna, fila, edad (s), fuerza
uniform float zfar;                       // la profundidad que se escribe (la misma del polvo)
out vec4 color;

uint hashu(uint x) {
    x ^= x >> 16u; x *= 0x7FEB352Du; x ^= x >> 15u; x *= 0x846CA68Bu; x ^= x >> 16u;
    return x;
}
float h01(int c, int r, int k) {
    uint h = hashu(uint(c) * 0x9E3779B1u ^ hashu(uint(r) * 0x85EBCA77u ^ uint(seed) * 0xC2B2AE3Du ^ uint(k) * 0x27D4EB2Fu));
    return float(h & 0xFFFFFFu) / 16777215.0;
}
vec3 hdir(int c, int r, int k) {
    float z = 2.0 * h01(c, r, k) - 1.0, a = 6.2831853 * h01(c, r, k + 1);
    return vec3(sqrt(1.0 - z * z) * cos(a), sqrt(1.0 - z * z) * sin(a), z);
}
vec3 rotate(vec3 p, vec3 k, float a) {           // Rodrigues
    float c = cos(a), s = sin(a);
    return p * c + cross(k, p) * s + k * dot(k, p) * (1.0 - c);
}

// rayo contra cápsula (segmento pa-pb, radio ra): distancia o -1 (Inigo Quilez)
float capsule(vec3 ro, vec3 rd, vec3 pa, vec3 pb, float ra) {
    vec3 ba = pb - pa, oa = ro - pa;
    float baba = dot(ba, ba), bard = dot(ba, rd), baoa = dot(ba, oa);
    float rdoa = dot(rd, oa), oaoa = dot(oa, oa);
    float a = baba - bard * bard;
    float b = baba * rdoa - baoa * bard;
    float c = baba * oaoa - baoa * baoa - ra * ra * baba;
    float h = b * b - a * c;
    if (h >= 0.0) {
        float t = (-b - sqrt(h)) / a;
        float y = baoa + t * bard;
        if (y > 0.0 && y < baba) return t;
        vec3 oc = (y <= 0.0) ? oa : ro - pb;
        b = dot(rd, oc);
        c = dot(oc, oc) - ra * ra;
        h = b * b - c;
        if (h > 0.0) return -b - sqrt(h);
    }
    return -1.0;
}

// media cápsula abierta: cerrada en e (el centro de su media esfera), cortada por el plano que pasa
// por q (normal de e hacia q). Por el corte se ve el interior (la salida del rayo). Devuelve t.
float cup(vec3 ro, vec3 rd, vec3 e, vec3 q, float R, out float inner, out vec3 segb) {
    vec3 n = normalize(q - e);
    segb = q + n * 1.5 * R;                                       // el cilindro sigue más allá del corte
    inner = 0.0;
    float t = capsule(ro, rd, e, segb, R);
    if (t < 0.0) return -1.0;
    if (dot(ro + rd * t - q, n) <= 0.0) return t;                 // afuera, antes del corte
    float T = 30.0;                                                // la salida: el primer cruce al revés
    float tb = capsule(ro + rd * T, -rd, e, segb, R);
    if (tb < 0.0) return -1.0;
    float tf = T - tb;
    if (dot(ro + rd * tf - q, n) <= 0.0) { inner = 1.0; return tf; }
    return -1.0;
}

// reflectancia de una película delgada (laca, n = 1.5) a tres longitudes de onda, normalizada
vec3 film(float cos_i, float d) {
    const float nf = 1.5;
    float sin_t = sqrt(max(0.0, 1.0 - cos_i * cos_i)) / nf;
    float cos_t = sqrt(max(0.0, 1.0 - sin_t * sin_t));
    float r = (1.0 - nf) / (1.0 + nf), r2 = r * r;
    vec3 lam = vec3(650.0, 532.0, 450.0);
    vec3 dl = 4.0 * 3.14159265 * nf * d * cos_t / lam;
    vec3 R = 2.0 * r2 * (1.0 - cos(dl)) / (1.0 + r2 * r2 - 2.0 * r2 * cos(dl));
    return R / (4.0 * r2 / ((1.0 + r2) * (1.0 + r2)));           // 0..1 (el máximo de la película)
}

void main() {
    vec2 uv = (gl_FragCoord.xy / out_size * 2.0 - 1.0) * vec2(aspect, 1.0);
    vec3 ro = vec3(0.0);
    vec3 rd = normalize(vec3(uv.x, -uv.y, focal));               // (la fila 0 del cuadro es la de arriba)
    vec3 p0 = rd * (D / rd.z);                                    // donde el rayo cruza el plano
    int c0 = int(floor(p0.x / sp + 0.5));
    float best = 1e9;
    vec3 hp, hn, ha, hb;
    int hc = 0, hr = 0;
    float hcap = 0.0, hred = 0.0, hin = 0.0;
    for (int dc = -1; dc <= 1; dc++) {
        int c = c0 + dc;
        float vc = 0.8 + 0.4 * h01(c, 0, 11);                     // cada columna cae a su ritmo
        float Y = fall * vc + sp * h01(c, 0, 12);
        int r0 = int(floor((p0.y + Y) / sp + 0.5));
        for (int dr = -1; dr <= 1; dr++) {
            int r = r0 + dr;
            bool is_red = false;
            for (int k = 0; k < n_red; k++) is_red = is_red || (int(red[k].x) == c && int(red[k].y) == r);
            if (!is_red && h01(c, r, 1) < vac) continue;          // celda vacía
            float age = -1.0, imp = 0.0;                          // ¿se abrió?
            for (int k = 0; k < n_op; k++)
                if (int(op[k].x) == c && int(op[k].y) == r) { age = op[k].z; imp = op[k].w; }
            float s = sp * (is_red ? 1.0 : 0.9 + 0.2 * h01(c, r, 2));
            vec3 ctr = vec3((float(c) + 0.24 * (h01(c, r, 3) - 0.5)) * sp,
                            float(r) * sp - Y + 0.2 * sp * (h01(c, r, 4) - 0.5),
                            is_red ? D : D + 0.5 * sp * (h01(c, r, 5) - 0.5));
            // el eje: alrededor de una dirección común (orden) o cualquiera (caos), y gira
            vec3 a0 = normalize(mix(vec3(0.6, 0.8, -0.1), hdir(c, r, 6), disorder));
            vec3 ax = hdir(c, r, 8);
            float w = (0.5 + h01(c, r, 10)) * (h01(c, r, 9) < 0.5 ? -1.0 : 1.0);
            vec3 a = rotate(a0, ax, tumble * w + 6.2831853 * h01(c, r, 7));
            if (is_red) a = normalize(vec3(a.xy / max(length(a.xy), 1e-3), 0.35 * a.z));   // la roja, de costado
            float L = hlen * s, R = rad * s;
            float t, cap = 0.0, inner = 0.0;
            vec3 sa, sb;
            if (age < 0.0) {
                // cerrada: el cuerpo y la tapa (apenas más ancha, monta sobre el cuerpo)
                vec3 pa = ctr - a * L, pb = ctr + a * L, pm = ctr - a * 0.06 * L, pe = ctr + a * 0.1 * L;
                float t1 = capsule(ro, rd, pa, pe, R);
                float t2 = capsule(ro, rd, pm, pb, R * 1.03);
                t = t1; sa = pa; sb = pe;
                if (t2 > 0.0 && (t1 < 0.0 || t2 < t1)) { t = t2; cap = 1.0; sa = pm; sb = pb; }
            } else {
                // abierta: las mitades se separan a lo largo del eje y se tuercen apenas
                float sep = L * (1.1 * (1.0 - exp(-age / 0.35)) + 0.12 * age);
                float ang = 0.9 * imp * (1.0 - exp(-age / 0.7)) + 0.15 * age;
                vec3 perp = normalize(cross(a, vec3(0.0, 0.0, 1.0)) + vec3(1e-4, 0.0, 0.0));
                vec3 ab = rotate(a, perp, ang), ac = rotate(a, perp, -ang);
                float i1, i2;
                vec3 b1, b2;
                vec3 e1 = ctr - ab * (L + sep), e2 = ctr + ac * (L + sep);
                float t1 = cup(ro, rd, e1, ctr + ab * (0.1 * L - sep), R, i1, b1);
                float t2 = cup(ro, rd, e2, ctr + ac * (sep - 0.06 * L), R * 1.03, i2, b2);
                t = t1; inner = i1; sa = e1; sb = b1;
                if (t2 > 0.0 && (t1 < 0.0 || t2 < t1)) { t = t2; cap = 1.0; inner = i2; sa = e2; sb = b2; }
            }
            if (t > 0.0 && t < best) {
                best = t;
                hp = ro + rd * t;
                hc = c; hr = r; hcap = cap; hred = is_red ? 1.0 : 0.0; hin = inner;
                ha = sa; hb = sb;
                vec3 ba = hb - ha, pa_ = hp - ha;
                float hh = clamp(dot(pa_, ba) / dot(ba, ba), 0.0, 1.0);
                hn = normalize(pa_ - hh * ba) * (inner > 0.5 ? -1.0 : 1.0);
            }
        }
    }
    vec3 col = vec3(0.0);
    vec3 key = normalize(vec3(-0.35, 0.75, -0.55));
    if (best < 1e8 && hin > 0.5) {
        // el interior de una mitad abierta: en sombra, sin película
        float dif = 0.15 + 0.85 * max(0.0, dot(hn, key));
        col = (hred > 0.5 ? vec3(0.35, 0.02, 0.03) : accent * 0.5) * 0.35 * dif * bright;
    } else if (best < 1e8) {
        vec3 v = -rd;
        float cos_i = clamp(dot(hn, v), 0.0, 1.0);
        // el espesor: propio de cada mitad, varía a lo largo de la cápsula; la tensión lo adelgaza
        vec3 axis = normalize(hb - ha);
        float along = dot(hp - ha, axis) / max(length(hb - ha), 1e-4);
        float d = d0 + d_rng * (h01(hc, hr, 20 + int(hcap)) - 0.5) + 90.0 * (along - 0.5);
        // la onda de color de la liberación: un frente que sale del centro de la pantalla
        vec2 sc = hp.xy * focal / hp.z;
        float wv = wave_amp * exp(-pow((length(sc) - wave_r) / 0.35, 2.0));
        d += 260.0 * wv;
        float ev_k = 0.0;
        for (int k = 0; k < n_ev; k++) {
            if (int(ev[k].x) == hc && int(ev[k].y) == hr) {
                float age = ev[k].w;
                float e = ev[k].z * exp(-age / 0.3);
                d += 300.0 * e * (1.0 - exp(-age / 0.05));        // las bandas corren por la cápsula
                ev_k += e;
            }
        }
        d *= exp(-2.6 * thin);                                    // la tensión: hasta la película negra
        vec3 F = film(cos_i, max(d, 0.0));
        // a la paleta: cada longitud de onda de la película va a un tono del violeta del track (la roja,
        // lila; la verde, lavanda pálida; la azul, violeta azulado): las bandas se funden entre tonos
        vec3 lilac = mix(accent, vec3(1.0, 0.8, 1.0), 0.55);
        vec3 pale = mix(accent, vec3(1.0), 0.75);
        vec3 bviol = mix(accent, vec3(0.3, 0.35, 1.0), 0.45);
        vec3 fc = mat3(lilac, pale, bviol) * F * 0.6;
        // la roja: la misma película, en rojos (carmín, rojo pálido, rojo profundo)
        if (hred > 0.5) fc = mat3(vec3(0.95, 0.07, 0.1), vec3(1.0, 0.45, 0.42), vec3(0.45, 0.0, 0.04)) * F * 0.75;
        // la luz: suave desde arriba y adelante; el contorno se apaga hacia el negro (sin filo)
        float dif = 0.25 + 0.75 * max(0.0, dot(hn, key));
        float limb = smoothstep(0.0, 0.55, cos_i);
        col = fc * dif * limb * (1.0 + 0.25 * pow(1.0 - cos_i, 2.0)) * bright * (1.0 + 1.5 * ev_k);
        col += vec3(1.0) * 0.08 * pow(max(0.0, dot(reflect(-v, hn), key)), 24.0) * bright;
    }
    gl_FragDepth = best < 1e8 ? clamp(hp.z / zfar, 0.0, 1.0) : 1.0;
    color = vec4(clamp(col * light * fade_k, 0.0, 1.0), 1.0);
}
