#version 330
// bloom: la imagen a 1/4 (promedio de bloques 4 x 4)
uniform sampler2D img;
out vec4 color;
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy) * 4;
    vec4 acc = vec4(0.0);
    for (int j = 0; j < 4; j++)
        for (int i = 0; i < 4; i++)
            acc += texelFetch(img, p + ivec2(i, j), 0);
    color = acc / 16.0;
}
