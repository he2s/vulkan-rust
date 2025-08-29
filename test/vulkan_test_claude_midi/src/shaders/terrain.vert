#version 450

layout(push_constant) uniform PushConstants {
    float time;
    uint  mouse_x;
    uint  mouse_y;
    uint  mouse_pressed;
    float note_velocity;
    float pitch_bend;
    float cc1;
    float cc74;
    uint  note_count;
    uint  last_note;
} pc;

layout(location = 0) out vec2 fragUV;
layout(location = 1) out float vertexEnergy;
layout(location = 2) out vec3 worldPos;

const vec2 positions[3] = vec2[](
    vec2(-1.0, -1.0),
    vec2( 3.0, -1.0),
    vec2(-1.0,  3.0)
);

const vec2 uvs[3] = vec2[](
    vec2(0.0, 1.0),
    vec2(2.0, 1.0),
    vec2(0.0, -1.0)
);

void main() {
    vec2 pos = positions[gl_VertexIndex];
    vec2 uv = uvs[gl_VertexIndex];

    // Simple pass-through with minimal distortion
    // The main effect will be in the fragment shader
    float energy = pc.note_velocity * 0.8 + pc.cc1 * 0.2;

    // Subtle breathing effect
    float breathe = 1.0 + 0.02 * sin(pc.time * 2.0) * energy;
    pos *= breathe;

    gl_Position = vec4(pos, 0.0, 1.0);
    fragUV = uv;
    vertexEnergy = energy;
    worldPos = vec3(pos, 0.0);
}