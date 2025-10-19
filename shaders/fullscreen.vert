#version 450

// Push constants matching your PushConstants struct
layout(push_constant) uniform PushConstants {
    float time;
    uint mouse_x;
    uint mouse_y;
    uint mouse_pressed;
    float note_velocity;
    float pitch_bend;
    float cc1;
    float cc74;
    uint note_count;
    uint last_note;
    float osc_ch1;
    float osc_ch2;
    uint render_w;
    uint render_h;
} pc;

// Vertex inputs (from buffer) - matches Rust GeometryMode::Trivial
layout(location = 0) in vec2 in_position;
layout(location = 1) in vec2 in_uv;

// Outputs to fragment shader
layout(location = 0) out vec2 frag_uv;
layout(location = 1) out float vertex_energy;
layout(location = 2) out vec3 world_pos;

void main() {
    // Use vertex data from buffer
    gl_Position = vec4(in_position, 0.0, 1.0);

    // Pass UV to fragment shader
    frag_uv = in_uv;

    // Compute vertex energy based on distance from center
    vertex_energy = 1.0 - length(in_position) * 0.5;
    vertex_energy = clamp(vertex_energy, 0.0, 1.0);

    // Compute world position (for shaders that need it)
    world_pos = vec3(in_position, 0.0);
}