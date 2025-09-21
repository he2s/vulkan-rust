#version 450

// Simple geometry shader - basic audio-reactive triangles
layout(points) in;
layout(triangle_strip, max_vertices = 3) out;

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

// Input from vertex shader
layout(location = 0) in vec2 vs_uv[];
layout(location = 1) in vec2 vs_screen_pos[];

// Output to fragment shader (simple version - no weird effects)
layout(location = 0) out vec2 frag_uv;
layout(location = 1) out vec2 frag_screen_pos;
layout(location = 2) out float frag_distance_from_center;

void main() {
    // Get the input point position
    vec4 center = gl_in[0].gl_Position;

    // Simple size calculation based on audio/MIDI parameters
    float beat_pulse = 1.0 - pc.time_to_next_beat;
    float size = 0.08 + pc.note_velocity * 0.04 + pc.osc_ch1 * 0.02 + beat_pulse * 0.02;

    // Simple rotation based on time
    float rotation = pc.time * 0.5 + center.x * 2.0 + center.y * 1.5;
    float cos_r = cos(rotation);
    float sin_r = sin(rotation);

    // Create a simple triangle
    vec2 vertices[3] = vec2[3](
        vec2(0.0, 1.0),           // Top
        vec2(-0.866, -0.5),       // Bottom left
        vec2(0.866, -0.5)         // Bottom right
    );

    for (int i = 0; i < 3; i++) {
        // Apply rotation
        vec2 rotated = vec2(
            vertices[i].x * cos_r - vertices[i].y * sin_r,
            vertices[i].x * sin_r + vertices[i].y * cos_r
        );

        // Scale and position
        vec2 final_offset = rotated * size;

        gl_Position = center + vec4(final_offset, 0.0, 0.0);
        frag_uv = vs_uv[0] + final_offset * 5.0;
        frag_screen_pos = vs_screen_pos[0] + final_offset;
        frag_distance_from_center = length(final_offset);

        EmitVertex();
    }

    EndPrimitive();
}