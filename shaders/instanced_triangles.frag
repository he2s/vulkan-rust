#version 450

// Inputs from vertex shader
layout(location = 0) in vec2 frag_uv;
layout(location = 1) in vec4 frag_color;
layout(location = 2) in float frag_intensity;
layout(location = 3) in float frag_distance_from_center;

// Output color
layout(location = 0) out vec4 out_color;

void main() {
    // Create a smooth falloff from center to edge
    float falloff = 1.0 - smoothstep(0.0, 1.0, frag_distance_from_center);

    // Apply intensity and falloff
    vec4 final_color = frag_color;
    final_color.rgb *= frag_intensity * falloff;
    final_color.a *= falloff;

    // Add some glow effect
    float glow = exp(-frag_distance_from_center * 2.0);
    final_color.rgb += glow * 0.2;

    // Ensure minimum alpha for visibility
    final_color.a = max(final_color.a, 0.1 * frag_intensity);

    out_color = final_color;
}