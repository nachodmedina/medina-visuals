#version 330
// estela (lo vivo al frente, lo anterior como recuerdo tenue) y fantasma en el color de acento
uniform sampler2D frame;      // el cuadro (0..1)
uniform sampler2D acc_prev;   // memoria de la estela
uniform int fresh;            // 1: sin memoria (primer cuadro)
uniform float decay, trail_g;
uniform int ghost_px;
uniform vec3 accent;          // 0..1
uniform ivec2 size;           // W, H
uniform int has_hole;
uniform vec3 hole;            // cx, cy, r en píxeles de salida
layout(location = 0) out vec4 y_out;
layout(location = 1) out vec4 acc_out;

vec3 trailed(ivec2 p) {
    vec3 x = texelFetch(frame, p, 0).rgb;
    if (fresh == 1) return x;
    return max(x, texelFetch(acc_prev, p, 0).rgb * decay * trail_g);
}

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec3 x = texelFetch(frame, p, 0).rgb;
    vec3 acc = fresh == 1 ? x : max(x, texelFetch(acc_prev, p, 0).rgb * decay);
    vec3 xt = trailed(p);
    vec3 y = xt;
    if (ghost_px >= 1) {                       // lo encendido, desplazado hacia afuera
        float k = 1.0 + float(ghost_px) / (float(size.x) / 2.0);
        vec2 c = vec2(size) / 2.0;
        ivec2 q = ivec2(clamp(roundEven(c + (vec2(p) + 0.5 - c) / k - 0.5), vec2(0.0), vec2(size - 1)));
        vec3 xq = trailed(q);
        float m = max(xq.r, max(xq.g, xq.b));
        y = max(xt, (0.85 * m) * accent);
    }
    if (has_hole == 1) {                       // la estela también cae adentro del agujero
        float ext = hole.z * 3.2 + 8.0;
        int x0 = max(0, int(hole.x - ext)), x1 = min(size.x, int(hole.x + ext) + 1);
        int y0 = max(0, int(hole.y - ext)), y1 = min(size.y, int(hole.y + ext) + 1);
        if (p.x >= x0 && p.x < x1 && p.y >= y0 && p.y < y1) {
            float d = length(vec2(p) - hole.xy + 0.5);
            acc *= clamp((d - hole.z * 0.985) / (0.02 * hole.z + 0.8), 0.0, 1.0);
        }
    }
    y_out = vec4(y, 1.0);
    acc_out = vec4(acc, 1.0);
}
