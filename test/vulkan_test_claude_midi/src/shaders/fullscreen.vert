#version 450

layout(push_constant) uniform PushConstants {
    float time;
    uint mouse_x;
    uint mouse_y;
    uint mouse_pressed;
    float note_velocity;
    float pitch_bend;
    float cc1;          // Modulation wheel
    float cc74;         // Filter cutoff
    uint note_count;
    uint last_note;
} pc;

vec2 positions[3] = vec2[](
    vec2(-1.0, -1.0),
    vec2( 3.0, -1.0),
    vec2(-1.0,  3.0)
);

vec2 uvs[3] = vec2[](
    vec2(0.0, 1.0),
    vec2(2.0, 1.0),
    vec2(0.0, -1.0)
);

layout(location = 0) out vec2 fragUV;
layout(location = 1) out float vertexEnergy;
layout(location = 2) out vec3 worldPos;

// Convert MIDI note to frequency for vertex calculations
float noteToFreq(uint note) {
    return 440.0 * pow(2.0, (float(note) - 69.0) / 12.0);
}

// Multi-octave noise function
float noise(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
}

float fbm(vec2 p) {
    float value = 0.0;
    float freq = 1.0;
    float amp = 0.5;

    for(int i = 0; i < 4; i++) {
        value += amp * (noise(p * freq) - 0.5);
        freq *= 2.0;
        amp *= 0.5;
    }
    return value;
}

void main() {
    vec2 pos = positions[gl_VertexIndex];
    vec2 uv = uvs[gl_VertexIndex];

    // Calculate musical frequency for vertex displacement
    float freq = noteToFreq(pc.last_note);
    float freq_norm = freq / 2000.0; // Normalize frequency

    // Create complex vertex displacement based on multiple inputs
    float time_freq = pc.time * freq_norm * 0.1;

    // MIDI-driven vertex waves - multiple harmonics
    vec2 displacement = vec2(0.0);

    // Fundamental frequency displacement
    displacement += 0.05 * pc.note_velocity * vec2(
        sin(time_freq + pos.x * 10.0),
        cos(time_freq + pos.y * 8.0)
    );

    // Harmonic series (2nd, 3rd, 5th harmonics)
    displacement += 0.03 * pc.note_velocity * vec2(
        sin(time_freq * 2.0 + pos.x * 15.0),
        sin(time_freq * 3.0 + pos.y * 12.0)
    );
    displacement += 0.02 * pc.note_velocity * vec2(
        sin(time_freq * 5.0 + pos.x * 25.0),
        cos(time_freq * 5.0 + pos.y * 20.0)
    );

    // Pitch bend creates space-time curvature
    float bend_factor = pc.pitch_bend * 0.3;
    vec2 bent_pos = pos;
    float dist_from_center = length(pos);
    bent_pos += normalize(pos) * bend_factor * sin(dist_from_center * 5.0 + pc.time);

    // Modulation wheel (CC1) controls geometric morphing
    float morph = pc.cc1;
    vec2 morph_offset = vec2(
        morph * sin(pc.time + pos.x * 8.0) * 0.1,
        morph * cos(pc.time + pos.y * 6.0) * 0.1
    );

    // Filter cutoff (CC74) controls fractal-like displacement
    float cutoff = pc.cc74;
    vec2 fractal_disp = fbm(pos * 3.0 + pc.time * 0.5) * cutoff * 0.08 * vec2(1.0, 1.0);

    // Combine all displacements
    pos += displacement + morph_offset + fractal_disp;
    bent_pos += displacement + morph_offset + fractal_disp;

    // Final position with pitch bend warping
    vec2 final_pos = mix(pos, bent_pos, abs(pc.pitch_bend));

    gl_Position = vec4(final_pos, 0.0, 1.0);
    fragUV = uv;

    // Pass energy level to fragment shader
    vertexEnergy = pc.note_velocity * float(pc.note_count) / 10.0;
    worldPos = vec3(final_pos, freq_norm);
}