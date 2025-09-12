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

layout(location = 0) in vec2 fragCoord;
layout(location = 1) in float dimension;
layout(location = 2) in vec2 uv;

layout(location = 0) out vec4 outColor;

const float PHI = 1.618033988749;

float hyperdimensional_pattern(vec2 p, float d) {
    float pattern = 0.0;

    // Multiple dimensional slices interfering
    for (int i = 0; i < 7; i++) {
        float slice = float(i) * PHI;
        vec2 offset = vec2(cos(slice), sin(slice)) * d;

        // Each dimension has its own frequency
        float freq = 10.0 + float(i) * PHI * 3.0;
        float wave = sin(dot(p + offset, vec2(1.0, PHI)) * freq + pc.time * float(i + 1));

        // Interference pattern
        pattern += wave * pow(0.7, float(i));
    }

    // Quantize to create sharp edges
    pattern = floor(pattern * 5.0 + 0.5) / 5.0;

    return pattern;
}

void main() {
    // Base hyperdimensional pattern
    float pattern = hyperdimensional_pattern(fragCoord, dimension);

    // Kaleidoscope mirrors create fractal boundaries
    float mirror_edges = abs(sin(fragCoord.x * 20.0 * (1.0 + pc.cc1))) *
    abs(cos(fragCoord.y * 20.0 * (1.0 + pc.cc74)));
    mirror_edges = step(0.9, mirror_edges);

    // Penrose tiling influence
    float penrose = sin(length(fragCoord) * PHI * 10.0 + dimension * 5.0);
    penrose = step(0.0, penrose);

    // Mix patterns based on MIDI
    float final_pattern = pattern;
    final_pattern = mix(final_pattern, penrose, pc.note_velocity * 0.5);
    final_pattern = mix(final_pattern, mirror_edges, pc.osc_ch1);

    // Dimensional rifts (inverted zones)
    float rift = step(0.8, sin(dimension * 50.0 + pc.time * 5.0));
    final_pattern = mix(final_pattern, 1.0 - final_pattern, rift);

    // Sharp black and white with dimension-based threshold
    float threshold = 0.5 + pc.pitch_bend * 0.3 + sin(dimension * 10.0) * 0.2;
    float bw = step(threshold, final_pattern);

    // Flash on note changes
    if (pc.note_count > 0) {
        float flash = sin(pc.time * 50.0) * 0.5 + 0.5;
        bw = mix(bw, 1.0 - bw, flash * pc.note_velocity * 0.3);
    }

    outColor = vec4(1.0);
}