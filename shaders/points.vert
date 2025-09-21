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
    float bpm;
    float time_to_next_beat;
} pc;

layout(location = 0) out vec2 frag_uv;
layout(location = 1) out vec2 frag_screen_pos;

// Generate points in a grid pattern
void main() {
    // Create a grid of points based on gl_VertexIndex
    int points_per_row = 20;
    int row = gl_VertexIndex / points_per_row;
    int col = gl_VertexIndex % points_per_row;

    // Map to screen coordinates (-1 to 1)
    float x = (col / float(points_per_row - 1)) * 2.0 - 1.0;
    float y = (row / float(points_per_row - 1)) * 2.0 - 1.0;

    // Add some animation based on time, audio, and beat timing
    float wave = sin(pc.time * 2.0 + x * 3.0 + y * 3.0) * 0.1;
    float beat_influence = (1.0 - pc.time_to_next_beat) * 0.15; // Beat-synchronized movement
    float audio_influence = pc.osc_ch1 * 0.2 + pc.note_velocity * 0.1 + beat_influence;

    vec2 pos = vec2(x, y) + vec2(wave * audio_influence);

    gl_Position = vec4(pos, 0.0, 1.0);

    // Pass UV and screen position to geometry shader
    frag_uv = vec2(col, row) / float(points_per_row - 1);
    frag_screen_pos = pos;
}