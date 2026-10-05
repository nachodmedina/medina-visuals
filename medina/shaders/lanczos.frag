#version 330
// supersampling -> salida con Lanczos-3, en una dirección por pasada (como PIL: primero horizontal,
// después vertical, con el intermedio en 8 bits)
uniform sampler2D img;
uniform ivec2 dir;            // (1, 0) u (0, 1)
uniform float scale;          // píxeles de entrada por píxel de salida
uniform ivec2 offset;         // dónde empieza el rectángulo de salida
out vec4 color;
const float PI = 3.141592653589793;
float sinc(float x) { return x == 0.0 ? 1.0 : sin(PI * x) / (PI * x); }
float lanczos(float x) { return (x >= -3.0 && x < 3.0) ? sinc(x) * sinc(x / 3.0) : 0.0; }
int idot(ivec2 a, ivec2 b) { return a.x * b.x + a.y * b.y; }
void main() {
    ivec2 o = ivec2(gl_FragCoord.xy) - offset;
    int n = idot(textureSize(img, 0), dir);
    int xo = idot(o, dir);
    ivec2 other = o * (ivec2(1) - dir);       // la coordenada que no cambia en esta pasada
    float fs = max(scale, 1.0);
    float support = 3.0 * fs;
    float center = (float(xo) + 0.5) * scale;
    int xmin = max(int(center - support + 0.5), 0);
    int xmax = min(int(center + support + 0.5), n);
    vec4 acc = vec4(0.0);
    float ww = 0.0;
    for (int x = xmin; x < xmax; x++) {
        float w = lanczos((float(x) - center + 0.5) / fs);
        acc += w * texelFetch(img, other + dir * x, 0);
        ww += w;
    }
    color = acc / ww;
}
