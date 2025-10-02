#version 450

layout(location = 0) in vec2 a_pos;
layout(location = 1) in vec2 a_uv;
layout(location = 2) in vec4 a_color;

layout(location = 0) out vec4 v_color;
layout(location = 1) out vec2 v_uv;

layout(push_constant) uniform PushConstants {
    vec2 screen_size;
} push;

void main() {
    // Convert from screen coordinates to clip space [-1, 1]
    vec2 pos = a_pos / push.screen_size;
    pos = pos * 2.0 - 1.0;
    // Note: NOT flipping Y - egui already provides coordinates in Vulkan convention

    gl_Position = vec4(pos, 0.0, 1.0);
    v_color = a_color;
    v_uv = a_uv;
}
