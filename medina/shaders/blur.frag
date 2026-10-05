#version 330
// gaussiana separable (como scipy gaussian_filter, modo reflect)
uniform sampler2D img;
uniform ivec2 dir;
uniform int radius;
uniform float w[128];
out vec4 color;
int refl(int i, int n) {
    if (i < 0) i = -i - 1;
    if (i >= n) i = 2 * n - i - 1;
    return clamp(i, 0, n - 1);
}
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    ivec2 n = textureSize(img, 0);
    vec4 acc = vec4(0.0);
    for (int t = -radius; t <= radius; t++) {
        ivec2 q = p + dir * t;
        q = ivec2(refl(q.x, n.x), refl(q.y, n.y));
        acc += w[abs(t)] * texelFetch(img, q, 0);
    }
    color = acc;
}
