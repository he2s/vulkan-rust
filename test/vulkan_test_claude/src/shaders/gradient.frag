#version 450

// Push constants - must match the vertex shader
layout(push_constant) uniform PushConstants {
    float time;
    uint mouse_x;
    uint mouse_y;
    uint mouse_pressed;
} pc;

layout(location = 0) in vec2 fragUV;
layout(location = 0) out vec4 outColor;

void main() {
    // Normalize UV coordinates to [0, 1]
    vec2 uv = fragUV;
    
    // Convert mouse position to UV coordinates (assuming 800x600 window)
    vec2 mouse_uv = vec2(float(pc.mouse_x) / 800.0, float(pc.mouse_y) / 600.0);
    
    // Distance from mouse cursor
    float mouse_dist = length(uv - mouse_uv);
    
    // Time-based animation
    float wave = sin(pc.time * 3.0);
    
    // Create animated gradient
    vec3 color = vec3(
        0.5 + 0.5 * sin(uv.x * 3.14159 + pc.time),           // Red channel with time
        0.5 + 0.5 * sin(uv.y * 3.14159 + pc.time * 1.3),    // Green channel
        0.5 + 0.5 * sin((uv.x + uv.y) * 3.14159 + pc.time * 0.7)  // Blue channel
    );
    
    // Mouse interaction effect
    if (pc.mouse_pressed == 1) {
        // Create ripple effect when mouse is pressed
        float ripple = sin(mouse_dist * 50.0 - pc.time * 10.0) * 0.5 + 0.5;
        color += ripple * 0.3;
    } else {
        // Subtle glow around mouse cursor
        color += 0.2 / (mouse_dist * 10.0 + 1.0);
    }
    
    // Add some animation based on distance from center
    float center_dist = length(uv - 0.5);
    color *= 0.8 + 0.2 * sin(center_dist * 20.0 + pc.time * 5.0);
    
    outColor = vec4(color, 1.0);
}
