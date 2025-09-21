#version 450

// Enhanced fragment shader with antialiasing for weird geometry shapes

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

// Inputs from geometry shader
layout(location = 0) in vec2 frag_uv;
layout(location = 1) in vec2 frag_screen_pos;
layout(location = 2) in float frag_distance_from_center;
layout(location = 3) in float frag_weird_factor;
layout(location = 4) in vec3 frag_color_mod;

// Output color
layout(location = 0) out vec4 out_color;

// Advanced noise functions for effects
float hash21(vec2 p) {
    p = fract(p * vec2(233.34, 851.73));
    p += dot(p, p + 23.45);
    return fract(p.x * p.y);
}

float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);

    float a = hash21(i);
    float b = hash21(i + vec2(1.0, 0.0));
    float c = hash21(i + vec2(0.0, 1.0));
    float d = hash21(i + vec2(1.0, 1.0));

    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

// Fractal Brownian Motion for complex textures
float fbm(vec2 p) {
    float f = 0.0;
    f += 0.5000 * noise(p); p *= 2.02;
    f += 0.2500 * noise(p); p *= 2.03;
    f += 0.1250 * noise(p); p *= 2.01;
    f += 0.0625 * noise(p);
    return f / 0.9375;
}

// Advanced anti-aliasing using analytical derivatives
float aa_step(float edge, float x, float width) {
    float d = fwidth(x) * width;
    return smoothstep(edge - d, edge + d, x);
}

// Enhanced color palette with psychedelic shifts
vec3 psychedelic_palette(float t) {
    t += pc.time * 0.1 + frag_weird_factor * 0.3;

    vec3 a = vec3(0.5, 0.5, 0.5);
    vec3 b = vec3(0.5, 0.5, 0.5);
    vec3 c = vec3(1.0, 1.0, 1.0) + frag_color_mod * 0.5;
    vec3 d = vec3(0.0, 0.33, 0.67) + vec3(pc.osc_ch1, pc.osc_ch2, pc.note_velocity) * 0.3;

    return a + b * cos(6.28318 * (c * t + d));
}

// Iridescent color shifting
vec3 iridescence(vec2 uv, float shift) {
    float angle = atan(uv.y, uv.x) + shift;
    float dist = length(uv);

    float r = 0.5 + 0.5 * sin(angle * 3.0 + pc.time * 2.0 + dist * 10.0);
    float g = 0.5 + 0.5 * sin(angle * 3.0 + pc.time * 2.0 + dist * 10.0 + 2.094);
    float b = 0.5 + 0.5 * sin(angle * 3.0 + pc.time * 2.0 + dist * 10.0 + 4.188);

    return vec3(r, g, b);
}

// Holographic interference patterns
float holographic_pattern(vec2 uv) {
    vec2 grid = uv * 50.0 + pc.time * vec2(1.0, -0.7);
    float pattern1 = sin(grid.x) * sin(grid.y);

    grid = uv * 30.0 + pc.time * vec2(-0.5, 1.2);
    float pattern2 = sin(grid.x + grid.y) * cos(grid.x - grid.y);

    return pattern1 * 0.6 + pattern2 * 0.4;
}

// Chromatic aberration effect
vec3 chromatic_aberration(vec2 uv, float strength) {
    vec2 offset = normalize(uv) * strength * 0.01;

    float r = noise(uv + offset);
    float g = noise(uv);
    float b = noise(uv - offset);

    return vec3(r, g, b);
}

void main() {
    vec2 uv = frag_uv;
    vec2 screen_uv = frag_screen_pos;

    // Distance-based effects
    float edge_distance = frag_distance_from_center;
    float center_factor = 1.0 - edge_distance;

    // Anti-aliased edge detection
    float edge_aa = aa_step(0.8, edge_distance, 2.0);
    float inner_aa = 1.0 - aa_step(0.2, edge_distance, 1.5);

    // Base color from psychedelic palette
    vec3 base_color = psychedelic_palette(edge_distance + pc.time * 0.1);

    // Apply color modulation from geometry shader
    base_color *= frag_color_mod;

    // Iridescent shifting based on weird factor
    vec3 irid_color = iridescence(uv, frag_weird_factor * 3.14159);
    base_color = mix(base_color, irid_color, frag_weird_factor * 0.3);

    // Audio-reactive effects
    float beat_pulse = 1.0 - pc.time_to_next_beat;
    float audio_intensity = pc.osc_ch1 + pc.osc_ch2 + pc.note_velocity;

    // Noise-based texture overlay
    vec2 noise_coord = screen_uv * 20.0 + pc.time * 0.5;
    float texture_noise = fbm(noise_coord) * 0.3;
    base_color += texture_noise * audio_intensity;

    // Holographic interference
    float holo = holographic_pattern(screen_uv) * 0.15;
    base_color += vec3(holo) * beat_pulse;

    // Chromatic aberration on edges
    vec3 chroma = chromatic_aberration(uv, frag_weird_factor * audio_intensity);
    base_color = mix(base_color, chroma, edge_aa * 0.4);

    // Energy pulses from center
    float pulse_rings = sin(edge_distance * 20.0 - pc.time * 10.0) * 0.5 + 0.5;
    pulse_rings *= exp(-edge_distance * 2.0); // Fade with distance
    base_color += vec3(pulse_rings) * beat_pulse * 0.3;

    // Weird factor intensity modulation
    if (frag_weird_factor > 1.5) {
        // Ultra weird mode - fractal interference
        vec2 fractal_coord = uv * 10.0 + pc.time * 0.2;
        float fractal = fbm(fractal_coord) * fbm(fractal_coord * 2.0);
        base_color += vec3(fractal) * 0.5;

        // Strobing effect
        float strobe = step(0.5, sin(pc.time * 20.0 * audio_intensity));
        base_color *= 1.0 + strobe * 0.3;
    } else if (frag_weird_factor > 0.5) {
        // Medium weird mode - flowing patterns
        vec2 flow = vec2(sin(pc.time + uv.x * 5.0), cos(pc.time + uv.y * 5.0)) * 0.1;
        vec3 flow_color = psychedelic_palette(noise(uv + flow) + pc.time * 0.2);
        base_color = mix(base_color, flow_color, 0.3);
    }

    // Final intensity and glow
    float intensity = inner_aa * (1.0 + audio_intensity * 0.5);
    base_color *= intensity;

    // Outer glow with anti-aliasing
    float glow = exp(-edge_distance * 3.0) * (1.0 - edge_aa);
    vec3 glow_color = psychedelic_palette(pc.time * 0.3) * glow * 0.8;
    base_color += glow_color;

    // Beat-synchronized flash
    if (beat_pulse > 0.8) {
        float flash = (beat_pulse - 0.8) * 5.0;
        base_color += vec3(1.0, 0.8, 0.6) * flash * 0.3;
    }

    // Temporal anti-aliasing using dithering
    vec2 screen_pos = gl_FragCoord.xy;
    float dither = hash21(screen_pos + pc.time) * 0.02 - 0.01;
    base_color += dither;

    // HDR tone mapping with enhanced saturation
    base_color = base_color / (base_color + vec3(1.0));
    base_color = pow(base_color, vec3(0.8)); // Gamma correction with boost

    // Final alpha with smooth edges
    float alpha = 1.0 - edge_aa;
    alpha *= 0.8 + inner_aa * 0.2; // Ensure minimum visibility
    alpha = max(alpha, glow * 0.5); // Glow contributes to alpha

    out_color = vec4(base_color, alpha);
}