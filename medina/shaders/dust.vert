#version 330
// El polvo de las cápsulas que se abren: cada partícula es un punto en el espacio de la cámara
// (la misma proyección que capsules.frag: la fila 0 del cuadro es la de arriba). La profundidad es la
// misma que escriben las cápsulas, así el polvo que queda detrás de una se tapa.
in vec3 pos;
in vec3 col;
in float size;                 // diámetro en píxeles de la escena
in float bri;
uniform float aspect, focal, zfar;
out vec3 v_col;
out float v_bri;
void main() {
    gl_Position = vec4(pos.x * focal / aspect, -pos.y * focal, (2.0 * pos.z / zfar - 1.0) * pos.z, pos.z);
    gl_PointSize = max(size, 1.0);
    v_col = col;
    v_bri = bri * min(1.0, size * size);       // lo más chico que un píxel: menos luz, no más grande
}
