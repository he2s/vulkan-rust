#version 450

layout(location = 0) in vec4 v_color;
layout(location = 1) in vec2 v_uv;

layout(location = 0) out vec4 f_color;

layout(binding = 0) uniform sampler2D u_texture;

void main() {
    // Sample texture and multiply by vertex color
    vec4 tex_color = texture(u_texture, v_uv);

    // egui uses sRGB textures, output is also sRGB
    f_color = v_color * tex_color;
}
