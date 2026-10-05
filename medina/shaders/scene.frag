#version 330
// La escena de un cuadro, píxel por píxel: el estilo (polvo o disco) con la lente del agujero,
// la onda gravitacional, las estrellas y fenómenos que dibuja la CPU, el horizonte y el silencio.
// Las mismas cuentas que medina/styles.py y medina/space.py, en coordenadas de la grilla.
// Convención: la fila 0 de cada textura es la fila 0 del arreglo de numpy (sin espejar).

uniform vec2 out_size;        // tamaño de este render (supersampling x salida)
uniform vec4 box;             // encuadre de la cámara en píxeles de grilla (x0, y0, x1, y1)
uniform vec2 grid;            // ww, hh de la grilla completa
uniform float dist, unit, gs, tg, roll;
uniform int style, silent, closed;
uniform float rh, tw_, glow, rot, ph;
uniform float kick, chaos, tens, hat, low, mid, warp;
uniform int mut, kc, hat_n, frame_i;
uniform float lf, turb, matter, accent_k;
// disco
uniform float incl, tilt, K;
uniform vec3 f;
uniform sampler2D lanes;      // 192 x 5: giro, segmentos, densidad, brillo, grosor de cada carril
// polvo
uniform float N, Kd, p_, dens_, dot_, pxcap;
uniform int flick, plus_;
// capas de la CPU y paleta
uniform usampler2D stars_tex; // estrellas (solo donde hay negro)
uniform usampler2D phen_tex;  // fenómenos (tapan)
uniform int has_phen;
uniform vec3 pal[8];
out vec4 color;

const float TAU = 6.283185307179586;

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

float thin_() { return closed == 1 ? 1.0 - 0.7 * tens : 1.0; }

bool erode(int ids, int salt) {
    if (closed == 0 || silent == 1) return true;
    float nv = nerv();
    float e = 0.88 * pow(tens, 1.2) * (1.0 - 0.65 * nv * hat);
    bool keep = hash32(ids + 7919, 3 + salt, 11u) >= e;
    if (nv > 0.0) keep = keep && hash32(ids + 104729, 5 + salt, uint(hat_n)) >= 0.5 * nv * (1.0 - hat);
    return keep;
}

int levels(float I, bool acc) {
    if (!acc) {
        if (I > 0.75) return 1;
        if (I > 0.45) return 5;
        if (I > 0.22) return 4;
        if (I > 0.06) return 3;
    } else {
        if (I > 0.70) return 2;
        if (I > 0.30) return 7;
        if (I > 0.06) return 6;
    }
    return 0;
}

// lente gravitacional: log(r) curvado, torsión, cuánto se acerca la fuente y la onda de la liberación
void lens(float R, out float L, out float Tw, out float ratio, out float ripple) {
    float Re = rh * 1.12;
    float s0 = 0.06 * Re;
    float El = sqrt(Re * Re - s0 * Re);
    float w = clamp((6.0 * Re * lf - R) / (4.0 * Re * lf), 0.0, 1.0);
    w = w * w * (3.0 - 2.0 * w);
    float src = R - El * El / max(R, 1e-4);
    float r_eff = max(R + w * (src - R), s0);
    L = log(r_eff + 1e-4);
    float q = Re / max(R, Re);
    Tw = tw_ * (0.3 / (R + 0.05)) + (1.0 + 1.2 * tw_ + 3.0 * glow) * lf * w * q * q;
    ratio = r_eff / max(R, 1e-4);
    ripple = 1.0;
    if (glow > 0.03) {
        float ts = -2.5 * log(glow);
        float d = R - (Re + 0.9 * ts);
        float rip = (0.07 * turb) * sqrt(glow) * sin(TAU * d / 0.16) * exp(-(d / 0.22) * (d / 0.22));
        L += log(1.0 + rip);
        ripple = 1.0 + rip;
    }
}

// polvo sobre el túnel (draw_particles)
int dots(float R, float TH, float L, float Tw) {
    float u = Kd * L - ph * (Kd / 5.0);
    float rif = floor(u);
    int ri = int(rif);
    float fr_ = u - rif;
    float a = mod((TH + 0.35 * rot + roll + Tw) * (1.0 / TAU), 1.0) * N;
    float aif = floor(a);
    int ai = int(aif);
    float fa = a - aif;
    float nv = nerv();
    uint tkey = nv > 0.0 ? uint(hat_n) : uint(kc / 8);
    float r1 = hash32(ri + 5000, ai + 5000, tkey);
    float r2 = hash32(ri + 9000, ai + 1000, uint(kc / 8 + 7));
    float energy = 0.5 * low + 0.5 * mid;
    float pr;
    if (closed == 1)
        pr = p_ * dens_ * matter * 0.35 * ((1.0 - 0.9 * tens) * (1.0 - 0.5 * nv * (1.0 - hat)) + 1.6 * nv * hat);
    else
        pr = p_ * dens_ * matter * (0.4 + 0.9 * energy + 0.8 * kick);
    bool lit = r1 < pr;
    if (flick > 0) lit = lit && hash32(ri + 77, ai + 33, uint(frame_i / flick)) < 0.7;
    float th_ = thin_();
    th_ += (1.0 - th_) * nv * hat;
    float half_ = ((dot_ + 0.06 * kick) * th_ + 0.08) / 2.0;
    float cr = max(R * (unit / Kd), 1e-3);
    float ca = max(R * (unit * TAU / N), 1e-3);
    float hr = min(half_, pxcap / cr);
    float ha = min(half_, pxcap / ca);
    if (warp > 0.0) hr = min(0.5, hr + 0.45 * warp);
    float dr = abs(fr_ - 0.5), da = abs(fa - 0.5);
    bool inside = dr < hr && da < ha;
    if (plus_ == 1) {
        float arm = min(0.5, half_ + 0.12 + 0.08 * kick);
        float ar = min(arm, 7.0 * gs / cr), aa = min(arm, 7.0 * gs / ca);
        float tr_ = min(0.05, 0.5 * gs / cr), ta = min(0.05, 0.5 * gs / ca);
        inside = inside || ((((dr < ar) && (da < ta)) || ((da < aa) && (dr < tr_))) && (r1 < pr * 0.5));
    }
    bool on = lit && inside;
    bool acc = on && r2 < (0.08 + 0.2 * chaos) * accent_k;
    float r3 = hash32(ri + 13000, ai + 3000, uint(kc / 8));
    float I = on ? 0.25 + 0.75 * pow(r3, 2.2) + 0.3 * kick : 0.0;
    return levels(I, acc);
}

float lane(int k, int row) { return texelFetch(lanes, ivec2(k, row), 0).r; }

// disco de acreción en diagonal (draw_disk + _orbits)
int disk(float xx, float yy, float ratio, float ripple) {
    float Re = rh * 1.12;
    float al = radians(tilt);
    float XR = xx * cos(al) + yy * sin(al);
    float YR = -xx * sin(al) + yy * cos(al);
    float xs = XR * ratio, ys = YR * ratio / incl;
    float rho = sqrt(xs * xs + ys * ys) * ripple;
    float ang = atan(ys, xs) + roll;
    float tens_c = closed == 1 ? tens : 0.0;
    float scale = 1.0 - 0.35 * tens_c;
    float u = K * (log(rho + 1e-4) - log(scale));
    u += (0.18 * turb) * sin(2.0 * ang + f.x + 0.15 * rot) + (0.1 * turb) * sin(3.0 * ang - f.y - 0.1 * rot);
    float lif = floor(u);
    int li = int(lif);
    float fr_ = u - lif;
    int k = clamp(li + 96, 0, 191);
    float a = mod((ang + lane(k, 0)) * (1.0 / TAU), 1.0) * lane(k, 1);
    float sif = floor(a);
    int si = int(sif);
    float fa = a - sif;
    float hl = hash32(li + 300, si, uint(mut));
    float ln = min((0.08 + 0.8 * pow(hl, 1.5)) * (1.0 + 1.4 * kick), 0.97);
    bool seg_on = fa < ln && hash32(li + 500, si, uint(mut / 2))
                  < (0.3 + 0.5 * lane(k, 2)) * (1.0 + 0.3 * chaos + 0.8 * glow) * matter;
    float wl = 0.10 * lane(k, 4);
    if (closed == 1) wl *= max(0.45, thin_());
    bool on = abs(fr_ - 0.5) < wl && seg_on;
    if (closed == 1) on = on && erode(li, 4);
    float q = min(fa / max(ln, 1e-3), 1.0);
    float along = pow(1.0 - q, 1.6) * clamp(q / 0.07, 0.0, 1.0);
    float I = on ? lane(k, 3) * along * (0.65 + 0.6 * kick) * (1.0 + 4.0 * glow) : 0.0;
    if (closed == 1) I *= 0.7 + 0.5 * nerv() * hat;
    bool acc = on && hash32(li + 800, 8, uint(kc / 16)) < (0.12 + 0.2 * chaos) * accent_k;
    I *= (rho > 2.0 * Re && rho < 2.2) ? 1.0 : 0.0;
    I *= clamp(1.25 - 0.6 * rho + 0.8 * glow, 0.15, 1.0);
    I *= 1.9 * clamp(1.0 - 0.45 * XR, 0.55, 1.5);
    bool doppler = hash32(li + 1500, 9, 0u) < clamp(0.5 + 0.9 * XR, 0.0, 1.0);
    return levels(I, I > 0.0 && (doppler || acc));
}

void main() {
    vec2 g = box.xy + gl_FragCoord.xy / out_size * (box.zw - box.xy);   // píxeles de grilla
    int idx;
    if (silent == 1) {                         // negro y una sola línea
        float h = floor(grid.y / 2.0);
        idx = (g.y >= h && g.y < h + tg) ? 1 : 0;
    } else {
        float r0 = grid.y / 2.0;
        float xx = (g.x - grid.x / 2.0) / r0 * dist;
        float yy = (g.y - grid.y / 2.0) / r0 * dist;
        float R = sqrt(xx * xx + yy * yy);
        float L, Tw, ratio, ripple;
        lens(R, L, Tw, ratio, ripple);
        idx = style == 0 ? dots(R, atan(yy, xx), L, Tw) : disk(xx, yy, ratio, ripple);
        ivec2 gp = clamp(ivec2(floor(g)), ivec2(0), ivec2(grid) - 1);
        if (idx == 0) idx = int(texelFetch(stars_tex, gp, 0).r);
        if (has_phen == 1) {
            uint pv = texelFetch(phen_tex, gp, 0).r;
            if (pv != 0u) idx = int(pv);
        }
        if (R < rh * 1.12) idx = 0;             // el horizonte se traga todo
    }
    color = vec4(pal[idx], 1.0);
}
