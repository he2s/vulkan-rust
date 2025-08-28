#version 450

// Push constants - must match the vertex shader
layout(push_constant) uniform PushConstants {
    float time;
    uint mouse_x;
    uint mouse_y;
    uint mouse_pressed;
    // MIDI data
    float note_velocity;
    float pitch_bend;
    float cc1;          // Modulation wheel
    float cc74;         // Filter cutoff
    uint note_count;
    uint last_note;
} pc;

layout(location = 0) in vec2 fragUV;
layout(location = 0) out vec4 outColor;

// Convert MIDI note to frequency (for visual mapping)
float noteToFreq(uint note) {
    return 440.0 * pow(2.0, (float(note) - 69.0) / 12.0);
}

void main() {
    vec2 uv = fragUV;
    
    // Convert mouse position to UV coordinates
    vec2 mouse_uv = vec2(float(pc.mouse_x) / 800.0, float(pc.mouse_y) / 600.0);
    float mouse_dist = length(uv - mouse_uv);
    
    // MIDI-reactive base colors
    float note_freq = noteToFreq(pc.last_note);
    float freq_factor = note_freq / 1000.0; // Normalize frequency
    
    // Create colors that respond to the played note
    vec3 base_color = vec3(
        0.5 + 0.5 * sin(freq_factor * 2.0 + pc.time),
        0.5 + 0.5 * sin(freq_factor * 3.0 + pc.time * 1.3),
        0.5 + 0.5 * sin(freq_factor * 4.0 + pc.time * 0.7)
    );
    
    // Velocity-based intensity
    float intensity = 1.0 + pc.note_velocity * 2.0;
    vec3 color = base_color * intensity;
    
    // Pitch bend warping effect
    vec2 bent_uv = uv;
    bent_uv.x += pc.pitch_bend * 0.2 * sin(uv.y * 10.0);
    bent_uv.y += pc.pitch_bend * 0.1 * cos(uv.x * 8.0);
    
    // Modulation wheel creates swirling patterns
    float mod_swirl = pc.cc1 * 5.0;
    float angle = atan(bent_uv.y - 0.5, bent_uv.x - 0.5) + pc.time * mod_swirl;
    float radius = length(bent_uv - 0.5);
    
    // CC74 (filter cutoff) controls color filtering
    float cutoff = pc.cc74;
    color.r *= 0.5 + 0.5 * cutoff;
    color.g *= 0.3 + 0.7 * cutoff;
    color.b *= cutoff;
    
    // Create pulsing effect based on number of notes pressed
    float pulse = 1.0 + 0.3 * sin(pc.time * 8.0) * float(pc.note_count) / 10.0;
    color *= pulse;
    
    // Add spiral patterns when notes are playing
    if (pc.note_count > 0) {
        float spiral = sin(radius * 30.0 - angle * 3.0 + pc.time * 10.0);
        color += spiral * 0.2 * pc.note_velocity;
    }
    
    // Mouse interaction (as before)
    if (pc.mouse_pressed == 1) {
        float ripple = sin(mouse_dist * 50.0 - pc.time * 10.0) * 0.5 + 0.5;
        color += ripple * 0.3;
    } else {
        color += 0.2 / (mouse_dist * 10.0 + 1.0);
    }
    
    // Add some noise/texture
    float noise = sin(bent_uv.x * 100.0) * sin(bent_uv.y * 100.0) * 0.05;
    color += noise * pc.note_velocity;
    
    outColor = vec4(color, 1.0);
}
