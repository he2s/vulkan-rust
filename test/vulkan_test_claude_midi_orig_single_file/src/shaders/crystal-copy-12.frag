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
} pc;

layout(location = 0) in vec2 frag_uv;
layout(location = 1) in vec2 frag_screen_pos;
layout(location = 0) out vec4 out_color;

const float PI = 3.14159265359;
const float TAU = PI * 2.0;

// Hash functions
float hash(float n) {
    return fract(sin(n) * 43758.5453123);
}
float hash2(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
}

// Neon flickering grid pattern
float neonGrid(vec2 p, float t) {
    float intensity = (pc.note_velocity + pc.cc1 + pc.cc74 + pc.osc_ch1 + pc.osc_ch2) / 5.0;
    p *= 3.0 + intensity * 5.0;
    p += vec2(hash(floor(p.x + p.y)), hash(floor(p.y - p.x))) * 2.0;

    float line = step(0.02 + 0.05 * intensity, mod(p.x + p.y + t, 0.1));
    float flicker = sin(t * 10.0 + hash(floor(p.x * 10.0 + p.y))) * 0.5 + 0.5;

    return mix(line, 1.0 - line, flicker);
}

void main() {
    vec2 uv = frag_uv;
    vec2 resolution = vec2(pc.render_w, pc.render_h);

    float t = pc.time * 2.0 + sin(pc.time) * 0.5;

    // Create chaotic motion
    vec2 p = (uv - 0.5) * resolution.x / resolution.y * 2.0;
    p += vec2(sin(t + p.y * 3.0), cos(t + p.x * 3.0)) * 0.5;

    // Generate neon grid pattern
    float pattern = neonGrid(p, t);

    // Color mutation for neon effect
    vec3 color = vec3(0.0);
    color.r = pattern * 1.2 + sin(t * 1.5 + pattern * 10.0) * 0.2;
    color.g = pattern * 0.8 + cos(t * 1.7 + pattern * 10.0) * 0.2;
    color.b = pattern * 1.0 + sin(t * 1.3 + pattern * 12.0) * 0.2;

    // Add some chaotic flickers
    color += vec3(
        sin(t * 20.0 + hash(floor(p.x * 5.0))) * 0.2,
        cos(t * 25.0 + hash(floor(p.y * 5.0))) * 0.2,
        sin(t * 30.0 + hash(floor(p.x + p.y))) * 0.2
    );

    // Final adjustments for a neon glow
    color = pow(color, vec3(1.2));
    color = clamp(color, 0.0, 2.0);

    // Output with glow
    out_color = vec4(color, 1.0);
}
