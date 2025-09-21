#version 450

// Simple fragment shader for basic geometry (non-weird mode)

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
    float bpm;
    float time_to_next_beat;
} pc;

// Standard inputs from geometry shader
layout(location = 0) in vec2 frag_uv;
layout(location = 1) in vec2 frag_screen_pos;
layout(location = 2) in float frag_distance_from_center;

// Output color
layout(location = 0) out vec4 out_color;

// Simple color palette
vec3 simple_palette(float t) {
    t += pc.time * 0.1;

    vec3 a = vec3(0.5, 0.5, 0.5);
    vec3 b = vec3(0.5, 0.5, 0.5);
    vec3 c = vec3(1.0, 1.0, 1.0);
    vec3 d = vec3(0.0, 0.33, 0.67) + vec3(pc.osc_ch1, pc.osc_ch2, pc.note_velocity) * 0.2;

    return a + b * cos(6.28318 * (c * t + d));
}

void main() {
    vec2 uv = frag_uv;
    float edge_distance = frag_distance_from_center;

    // Simple distance-based falloff
    float center_factor = 1.0 - smoothstep(0.0, 1.0, edge_distance);

    // Base color from simple palette
    vec3 base_color = simple_palette(edge_distance + pc.time * 0.05);

    // Audio-reactive intensity
    float audio_intensity = pc.note_velocity + pc.osc_ch1 * 0.5 + pc.osc_ch2 * 0.5;
    base_color *= 0.7 + audio_intensity * 0.5;

    // Beat pulse effect
    float beat_pulse = 1.0 - pc.time_to_next_beat;
    if (beat_pulse > 0.8) {
        base_color += vec3(0.3, 0.2, 0.1) * (beat_pulse - 0.8) * 5.0;
    }

    // Simple glow
    float glow = exp(-edge_distance * 2.0) * 0.3;
    base_color += glow;

    // Final intensity
    base_color *= center_factor;

    // Simple alpha with smooth edges
    float alpha = smoothstep(1.0, 0.0, edge_distance);
    alpha = max(alpha, glow);

    out_color = vec4(base_color, alpha);
}