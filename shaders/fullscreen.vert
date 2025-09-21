#version 450

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
    float bpm;
    float time_to_next_beat;
    float time_since_last_beat;
    uint beats_per_bar;
} pc;

// Vertex attributes
layout(location = 0) in vec2 in_position;
layout(location = 1) in vec2 in_uv;

// Instance attributes
layout(location = 2) in vec2 instance_offset;
layout(location = 3) in vec2 instance_scale;
layout(location = 4) in float instance_rotation_cos;
layout(location = 5) in float instance_rotation_sin;
layout(location = 6) in uint instance_color_index;

layout(location = 0) out vec2 frag_uv;
layout(location = 1) out flat uint frag_instance_id;
layout(location = 2) out vec2 frag_screen_pos;

void main() {
    // Use pre-computed rotation values and add time-based animation
    float time_factor = pc.time * 0.5;
    float time_cos = cos(time_factor);
    float time_sin = sin(time_factor);

    // Combine base rotation with time animation using rotation addition formulas
    // cos(a + b) = cos(a)cos(b) - sin(a)sin(b)
    // sin(a + b) = sin(a)cos(b) + cos(a)sin(b)
    float c = instance_rotation_cos * time_cos - instance_rotation_sin * time_sin;
    float s = instance_rotation_sin * time_cos + instance_rotation_cos * time_sin;

    // Fast 2D rotation using direct multiplication (avoid mat2 construction)
    vec2 scaled_pos = in_position * instance_scale;
    vec2 rotated_pos = vec2(
        scaled_pos.x * c - scaled_pos.y * s,
        scaled_pos.x * s + scaled_pos.y * c
    );

    // Apply instance offset with optional OSC modulation
    vec2 final_pos = rotated_pos + (instance_offset * (1.0 + 9.0 * pc.osc_ch1));

    gl_Position = vec4(final_pos, 0.0, 1.0);
    frag_uv = in_uv;
    frag_instance_id = gl_InstanceIndex;
    frag_screen_pos = final_pos;
}