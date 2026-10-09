#version 330
// Una mota de polvo: un disco suave (la luz que dispersa), sumada a lo que hay.
in vec3 v_col;
in float v_bri;
out vec4 color;
void main() {
    vec2 q = gl_PointCoord * 2.0 - 1.0;
    float a = exp(-3.0 * dot(q, q));
    if (a < 0.02) discard;
    color = vec4(v_col * v_bri * a, 1.0);
}
