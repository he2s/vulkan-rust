#version 450

// Push constants (must match vertex shader)
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
layout(location = 0) in vec2 fragCoord;
layout(location = 1) in float pattern_intensity;
layout(location = 2) in vec2 uv;

// Output
layout(location = 0) out vec4 outColor;

// Simple hash for dithering
float hash21(vec2 p) {
    p = fract(p * vec2(234.34, 435.345));
    p += dot(p, p + 34.23);
    return fract(p.x * p.y);
}

void main() {
    // Calculate distance from center
    float dist = length(fragCoord);

    // Create concentric circles modulated by vertex distortion
    float circles = sin(dist * 20.0 - pc.time * 2.0 + pattern_intensity * 10.0);
    circles = smoothstep(-0.1, 0.1, circles);

    // Grid pattern
    vec2 grid = sin(fragCoord * 15.0 + pattern_intensity * 5.0);
    float grid_pattern = step(0.0, grid.x * grid.y);

    // Combine patterns
    float pattern = mix(circles, grid_pattern, 0.5 + pc.cc1 * 0.5);

    // Modulate with vertex shader's pattern intensity
    pattern *= (0.5 + pattern_intensity);

    // Add some noise/dithering for organic feel
    float noise = hash21(fragCoord * 100.0 + pc.time);
    pattern += noise * 0.1;

    // MIDI reactive threshold
    float threshold = 0.5 + pc.note_velocity * 0.3 - pc.pitch_bend * 0.2;

    // Sharp black and white with smooth edges
    float bw = smoothstep(threshold - 0.02, threshold + 0.02, pattern);

    // Invert based on mouse press or high MIDI activity
    if (pc.mouse_pressed > 0 || pc.note_count > 3) {
        bw = 1.0 - bw;
    }

    // Final color output
    outColor = vec4(vec3(bw), 1.0);
}