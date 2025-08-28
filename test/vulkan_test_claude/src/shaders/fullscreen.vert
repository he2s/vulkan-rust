#version 450

// Push constants - shared between vertex and fragment shader
layout(push_constant) uniform PushConstants {
    float time;
    uint mouse_x;
    uint mouse_y; 
    uint mouse_pressed;
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
    
    // Add some vertex animation based on time
    pos += 0.05 * sin(pc.time * 2.0 + pos.x * 10.0) * vec2(0.1, 0.1);
    
    gl_Position = vec4(pos, 0.0, 1.0);
    fragUV = uvs[gl_VertexIndex];
}
