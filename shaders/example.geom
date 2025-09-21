#version 450

// Geometry shader that takes points and creates triangles
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
} pc;

// Input from vertex shader
layout(location = 0) in vec2 vs_uv[];
layout(location = 1) in vec2 vs_screen_pos[];

// Output to fragment shader
layout(location = 0) out vec2 frag_uv;
layout(location = 1) out vec2 frag_screen_pos;
layout(location = 2) out float frag_distance_from_center;

void main() {
    // Get the input point position
    vec4 center = gl_in[0].gl_Position;

    // Calculate triangle size based on audio/MIDI parameters
    float size = 0.1 + pc.note_velocity * 0.05 + pc.osc_ch1 * 0.02;

    // Create a triangle from the input point
    // Triangle is oriented to create an interesting visual pattern

    // First vertex - top
    gl_Position = center + vec4(0.0, size, 0.0, 0.0);
    frag_uv = vs_uv[0] + vec2(0.5, 1.0);
    frag_screen_pos = vs_screen_pos[0] + vec2(0.0, size);
    frag_distance_from_center = length(vec2(0.0, size));
    EmitVertex();

    // Second vertex - bottom left
    gl_Position = center + vec4(-size * 0.866, -size * 0.5, 0.0, 0.0);
    frag_uv = vs_uv[0] + vec2(0.0, 0.0);
    frag_screen_pos = vs_screen_pos[0] + vec2(-size * 0.866, -size * 0.5);
    frag_distance_from_center = length(vec2(-size * 0.866, -size * 0.5));
    EmitVertex();

    // Third vertex - bottom right
    gl_Position = center + vec4(size * 0.866, -size * 0.5, 0.0, 0.0);
    frag_uv = vs_uv[0] + vec2(1.0, 0.0);
    frag_screen_pos = vs_screen_pos[0] + vec2(size * 0.866, -size * 0.5);
    frag_distance_from_center = length(vec2(size * 0.866, -size * 0.5));
    EmitVertex();

    EndPrimitive();
}