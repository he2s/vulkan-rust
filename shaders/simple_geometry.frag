#version 450

// Beautiful enhanced fragment shader with sophisticated visual effects

// Push constants matching your PushConstants struct
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
    float bpm;
    float time_to_next_beat;
} pc;

// Standard inputs from geometry shader
layout(location = 0) in vec2 frag_uv;
layout(location = 1) in vec2 frag_screen_pos;
layout(location = 2) in float frag_distance_from_center;

// Output color
layout(location = 0) out vec4 out_color;

#define PI 3.14159265359
#define TAU 6.28318530718

// Enhanced color palettes
vec3 aurora_palette(float t) {
    vec3 a = vec3(0.5, 0.5, 0.5);
    vec3 b = vec3(0.5, 0.5, 0.5);
    vec3 c = vec3(2.0, 1.0, 0.0);
    vec3 d = vec3(0.5, 0.20, 0.25);
    return a + b * cos(TAU * (c * t + d));
}

vec3 sunset_palette(float t) {
    vec3 a = vec3(0.8, 0.5, 0.4);
    vec3 b = vec3(0.2, 0.4, 0.2);
    vec3 c = vec3(2.0, 1.0, 1.0);
    vec3 d = vec3(0.0, 0.25, 0.25);
    return a + b * cos(TAU * (c * t + d));
}

vec3 cosmic_palette(float t) {
    vec3 a = vec3(0.5, 0.5, 0.5);
    vec3 b = vec3(0.5, 0.5, 0.5);
    vec3 c = vec3(1.0, 1.0, 0.5);
    vec3 d = vec3(0.8, 0.9, 0.3);
    return a + b * cos(TAU * (c * t + d));
}

// Smooth noise function
float noise(vec2 p) {
    return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}

// Smooth interpolated noise
float smooth_noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);

    float a = noise(i);
    float b = noise(i + vec2(1.0, 0.0));
    float c = noise(i + vec2(0.0, 1.0));
    float d = noise(i + vec2(1.0, 1.0));

    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

// Fractal Brownian Motion
float fbm(vec2 p, int octaves) {
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;

    for (int i = 0; i < octaves; i++) {
        value += amplitude * smooth_noise(p * frequency);
        amplitude *= 0.5;
        frequency *= 2.0;
    }

    return value;
}

// Rotating UV coordinates
vec2 rotate(vec2 uv, float angle) {
    float c = cos(angle);
    float s = sin(angle);
    return mat2(c, -s, s, c) * uv;
}

void main() {
    vec2 uv = frag_uv;
    vec2 center_uv = uv - 0.5;
    float edge_distance = frag_distance_from_center;

    // Time variables for different animation speeds
    float slow_time = pc.time * 0.2;
    float med_time = pc.time * 0.5;
    float fast_time = pc.time * 1.2;

    // Audio reactivity
    float audio_intensity = pc.note_velocity * 2.0 + pc.osc_ch1 + pc.osc_ch2;
    float pitch_factor = pc.pitch_bend * 0.5 + 0.5;

    // Complex geometric patterns
    vec2 rotated_uv = rotate(center_uv, slow_time + audio_intensity * 0.5);
    float spiral = atan(rotated_uv.y, rotated_uv.x) + length(rotated_uv) * 8.0 - fast_time * 2.0;
    float spiral_pattern = sin(spiral) * 0.5 + 0.5;

    // Layered noise for organic feel
    float noise_layer1 = fbm(uv * 4.0 + vec2(slow_time, med_time), 4);
    float noise_layer2 = fbm(uv * 8.0 - vec2(med_time * 0.7, slow_time * 1.3), 3);
    float combined_noise = (noise_layer1 + noise_layer2 * 0.5) * 0.66;

    // Distance-based rings with audio reactivity
    float ring_freq = 12.0 + audio_intensity * 8.0;
    float rings = sin((edge_distance + combined_noise * 0.3) * ring_freq - fast_time * 4.0);
    rings = pow(max(0.0, rings), 2.0);

    // Pulsing center
    float center_pulse = 1.0 - smoothstep(0.0, 0.4 + audio_intensity * 0.3, edge_distance);
    center_pulse *= sin(fast_time * 3.0 + audio_intensity * 5.0) * 0.5 + 0.5;

    // Color mixing based on different factors
    float color_param1 = edge_distance + combined_noise * 0.5 + slow_time * 0.3;
    float color_param2 = spiral_pattern + pitch_factor;
    float color_param3 = rings + center_pulse;

    // Blend multiple palettes
    vec3 color1 = aurora_palette(color_param1);
    vec3 color2 = sunset_palette(color_param2);
    vec3 color3 = cosmic_palette(color_param3);

    // Mix palettes based on audio and time
    vec3 base_color = mix(color1, color2, sin(med_time + audio_intensity) * 0.5 + 0.5);
    base_color = mix(base_color, color3, rings * 0.6);

    // Enhanced glow effects
    float inner_glow = exp(-edge_distance * 3.0) * (1.5 + audio_intensity);
    float outer_glow = exp(-edge_distance * 1.0) * 0.3;
    base_color += inner_glow * vec3(0.8, 0.6, 1.2);
    base_color += outer_glow * vec3(0.4, 0.8, 1.0);

    // Beat synchronization
    float beat_pulse = 1.0 - pc.time_to_next_beat;
    if (beat_pulse > 0.7) {
        float pulse_intensity = (beat_pulse - 0.7) * 3.33;
        base_color += pulse_intensity * vec3(1.0, 0.8, 0.2);
        base_color *= 1.0 + pulse_intensity * 0.5;
    }

    // Chromatic aberration effect near edges
    float aberration = smoothstep(0.6, 1.0, edge_distance) * 0.02;
    vec2 red_offset = vec2(aberration, 0.0);
    vec2 blue_offset = vec2(-aberration, 0.0);

    // Depth and dimension
    float depth_factor = 1.0 - edge_distance * 0.7;
    base_color *= depth_factor;

    // Particle-like sparkles
    float sparkle_noise = noise(uv * 200.0 + fast_time * 10.0);
    if (sparkle_noise > 0.98 && edge_distance < 0.8) {
        base_color += vec3(2.0, 2.0, 1.5) * (sparkle_noise - 0.98) * 50.0;
    }

    // Enhanced alpha with soft falloff and glow contribution
    float alpha = 1.0 - smoothstep(0.0, 1.2, edge_distance);
    alpha = max(alpha, inner_glow * 0.3);
    alpha = max(alpha, rings * 0.4);
    alpha *= 0.9;

    // Final color enhancement
    base_color = pow(base_color, vec3(0.9)); // Slightly less gamma
    base_color = mix(base_color, base_color * base_color, 0.3); // Slight contrast boost

    out_color = vec4(base_color, alpha);
}