#version 450

// Push constants - shared between vertex and fragment shader
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

// Generate a full-screen triangle using vertex ID
vec2 positions[3] = vec2[](
    vec2(-1.0, -1.0),  // Bottom-left
    vec2( 3.0, -1.0),  // Bottom-right (extends beyond screen)
    vec2(-1.0,  3.0)   // Top-left (extends beyond screen)
);

// UV coordinates for the triangle
vec2 uvs[3] = vec2[](
    vec2(0.0, 1.0),    // Bottom-left
    vec2(2.0, 1.0),    // Bottom-right
    vec2(0.0, -1.0)    // Top-left
);

layout(location = 0) out vec2 fragUV;

void main() {
    vec2 pos = positions[gl_VertexIndex];
    
    // MIDI-controlled vertex displacement
    float displacement = pc.note_velocity * 0.1;
    float bend_effect = pc.pitch_bend * 0.05;
    float mod_effect = pc.cc1 * 0.03;
    
    // Create ripple effects based on MIDI input
    float wave = sin(pc.time * 4.0 + pos.x * 20.0) * displacement;
    pos.x += wave + bend_effect;
    pos.y += sin(pc.time * 3.0 + pos.y * 15.0) * mod_effect;
    
    gl_Position = vec4(pos, 0.0, 1.0);
    fragUV = uvs[gl_VertexIndex];
}
