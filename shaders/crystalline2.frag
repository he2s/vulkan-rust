// Example Fragment Shader using OSC values
// Save as: shaders/osc_example.frag

#version 450

layout(location = 0) out vec4 outColor;

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
    uint render_w;
    uint render_h;
    // OSC parameters
    float osc_a_x;
    float osc_a_y;
    float osc_b_x;
    float osc_b_y;
    float osc_trig;
    float _padding[3];
} pc;

void main() {
    vec2 uv = gl_FragCoord.xy / vec2(pc.render_w, pc.render_h);
    
    // Use OSC A position as a center point
    vec2 centerA = vec2(pc.osc_a_x, 1.0 - pc.osc_a_y);
    
    // Use OSC B position as a second center point
    vec2 centerB = vec2(pc.osc_b_x, 1.0 - pc.osc_b_y);
    
    // Calculate distances from both centers
    float distA = distance(uv, centerA);
    float distB = distance(uv, centerB);
    
    // Create animated waves based on distance
    float waveA = sin(distA * 20.0 - pc.time * 3.0) * 0.5 + 0.5;
    float waveB = sin(distB * 20.0 + pc.time * 2.0) * 0.5 + 0.5;
    
    // Mix the waves
    float wave = mix(waveA, waveB, 0.5);
    
    // Apply trigger as a flash/pulse effect
    float triggerPulse = pc.osc_trig * sin(pc.time * 10.0) * 0.5;
    
    // Create color based on OSC positions and waves
    vec3 color = vec3(
        wave * pc.osc_a_x + triggerPulse,
        wave * pc.osc_a_y,
        wave * pc.osc_b_x * (1.0 - pc.osc_trig * 0.5)
    );
    
    // Add MIDI influence if present
    color += vec3(pc.note_velocity * 0.3, pc.cc1 * 0.2, pc.cc74 * 0.2);
    
    // Add radial gradient from both points
    float gradA = 1.0 - smoothstep(0.0, 0.5, distA);
    float gradB = 1.0 - smoothstep(0.0, 0.5, distB);
    color *= 1.0 + (gradA + gradB) * 0.5;
    
    // Trigger creates a strobe effect
    if (pc.osc_trig > 0.5) {
        color = mix(color, vec3(1.0), 0.3 * sin(pc.time * 30.0));
    }
    
    outColor = vec4(color, 1.0);
}