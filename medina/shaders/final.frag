#version 330
// bloom (screen), horizonte difuso tipo Gargantua, viñeta, punto de negro y grano
uniform sampler2D ytex;
uniform sampler2D bl0, bl1;   // bloom en dos escalas (1/4 de resolución)
uniform float bloom, bloom_mix;
uniform ivec2 size;
uniform int has_hole;
uniform vec3 hole;            // cx, cy, r
uniform float amp, s_, vignette, black;
uniform vec3 accent;
uniform isampler2D grain_tex;
uniform ivec2 grain_off;      // (ox, oy)
uniform float grain;
out vec4 color;
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec3 y = texelFetch(ytex, p, 0).rgb;
    if (bloom > 0.01) {
        vec2 uv = (vec2(p) + 0.5) / vec2(size);
        vec3 up = (1.0 - bloom_mix) * texture(bl0, uv).rgb + bloom_mix * texture(bl1, uv).rgb;
        y = y + min(bloom * up, 1.0) * (1.0 - y);
    }
    if (has_hole == 1) {
        float r = hole.z;
        float ext = r * 3.2 + 8.0;
        int x0 = max(0, int(hole.x - ext)), x1 = min(size.x, int(hole.x + ext) + 1);
        int y0 = max(0, int(hole.y - ext)), y1 = min(size.y, int(hole.y + ext) + 1);
        if (p.x >= x0 && p.x < x1 && p.y >= y0 && p.y < y1) {
            float d = length(vec2(p) - hole.xy + 0.5);
            float o = max(d - r, 0.0);
            float core = 0.55 * exp(-pow(o / (1.2 * s_ + 0.012 * r), 2.0));
            float halo = amp * exp(-o / (0.28 * r + 2.0));
            float mx = clamp(o / (0.5 * r + 1.0), 0.0, 1.0);
            vec3 col = (1.0 - mx) + mx * accent;
            vec3 g = (core + halo) * col;
            float inside = clamp((d - r * 0.985) / (0.02 * r + 0.8), 0.0, 1.0);
            y = (y + min(g, 1.0) * (1.0 - y)) * inside;
        }
    }
    vec2 v = (vec2(p) - vec2(size) / 2.0) / (vec2(size) / 2.0);
    float e = clamp((length(v) / sqrt(2.0) - 0.3) / 0.7, 0.0, 1.0);
    y *= 1.0 - vignette * e * e * (3.0 - 2.0 * e);
    if (black > 0.0) y = max(y - black, 0.0) * (1.0 / (1.0 - black));
    float nz = float(texelFetch(grain_tex, p + grain_off, 0).r) * (grain / 40.0);
    float lum = (y.r + y.g + y.b) / 3.0;
    y = y + nz * (0.12 + 0.88 * min(lum, 1.0));
    color = vec4(clamp(y, 0.0, 1.0), 1.0);
}
