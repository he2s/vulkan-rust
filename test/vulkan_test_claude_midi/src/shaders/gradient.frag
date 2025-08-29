#version 450

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
} pc;

layout(location = 0) in vec2 fragUV;
layout(location = 1) in float vertexEnergy;
layout(location = 2) in vec3 worldPos;
layout(location = 0) out vec4 outColor;

// Mathematical constants
const float PI = 3.14159265359;
const float TAU = 6.28318530718;

// Convert MIDI note to frequency
float noteToFreq(uint note) {
    return 440.0 * pow(2.0, (float(note) - 69.0) / 12.0);
}

// Convert frequency to color using harmonic series
vec3 freqToColor(float freq) {
    // Map frequency to hue using musical intervals
    float hue = mod(log2(freq / 440.0) * 12.0, 12.0) / 12.0; // Chromatic scale

    // Convert hue to RGB
    vec3 rgb = clamp(abs(mod(hue * 6.0 + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0);
    return rgb;
}

// Complex noise functions
float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), f.x),
               mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), f.x), f.y);
}

// Fractal Brownian Motion
float fbm(vec2 p, int octaves) {
    float value = 0.0;
    float freq = 1.0;
    float amp = 0.5;

    for(int i = 0; i < octaves; i++) {
        value += amp * noise(p * freq);
        freq *= 2.0;
        amp *= 0.5;
    }
    return value;
}

// Mandelbrot-like fractal
float mandelbrot(vec2 c, int maxIter) {
    vec2 z = vec2(0.0);
    int iter = 0;

    for(int i = 0; i < maxIter; i++) {
        if(dot(z, z) > 4.0) break;
        z = vec2(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + c;
        iter++;
    }

    return float(iter) / float(maxIter);
}

// Particle system simulation
vec3 particleField(vec2 uv, vec2 mouse_uv) {
    vec3 particles = vec3(0.0);
    float mouse_influence = 1.0 / (length(uv - mouse_uv) + 0.1);

    // Create multiple particle layers
    for(int layer = 0; layer < 3; layer++) {
        float layer_speed = (float(layer) + 1.0) * 0.5;
        vec2 layer_offset = vec2(float(layer) * 100.0);

        for(int i = 0; i < 8; i++) {
            vec2 particle_pos = layer_offset + vec2(float(i) * 50.0, sin(float(i)) * 30.0);

            // Animate particles with MIDI influence
            particle_pos.x += pc.time * layer_speed * (1.0 + pc.note_velocity);
            particle_pos.y += sin(pc.time * 2.0 + float(i)) * 0.3 * pc.cc1;
            particle_pos = mod(particle_pos, 400.0) - 200.0;

            // Convert to UV space
            particle_pos = particle_pos / 200.0;

            // Mouse attraction/repulsion
            vec2 mouse_force = normalize(particle_pos - mouse_uv) * mouse_influence * 0.1;
            if(pc.mouse_pressed == 1) {
                particle_pos -= mouse_force; // Attraction
            } else {
                particle_pos += mouse_force * 0.5; // Gentle repulsion
            }

            float dist = length(uv - particle_pos);
            float size = 0.05 + pc.note_velocity * 0.1;

            if(dist < size) {
                float intensity = (size - dist) / size;
                vec3 particle_color = freqToColor(noteToFreq(pc.last_note) * (float(i) + 1.0));
                particles += particle_color * intensity * intensity;
            }
        }
    }

    return particles;
}

void main() {
    vec2 uv = fragUV;
    vec2 mouse_uv = vec2(float(pc.mouse_x) / 800.0, float(pc.mouse_y) / 600.0);

    // Base musical color from current note
    vec3 base_color = freqToColor(noteToFreq(pc.last_note));

    // Create harmonic color palette
    vec3 harmonic_colors = vec3(0.0);
    float[] harmonics = float[](1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);

    for(int i = 0; i < 8; i++) {
        float harmonic_freq = noteToFreq(pc.last_note) * harmonics[i];
        vec3 harmonic_color = freqToColor(harmonic_freq);
        float harmonic_weight = pc.note_velocity / (harmonics[i] * harmonics[i]);
        harmonic_colors += harmonic_color * harmonic_weight;
    }
    harmonic_colors = normalize(harmonic_colors);

    // Fractal background based on pitch bend and filter cutoff
    vec2 fractal_uv = uv * 2.0 - 1.0;
    fractal_uv *= 1.0 + pc.pitch_bend * 0.5; // Pitch bend zooms
    fractal_uv += vec2(pc.time * 0.1, sin(pc.time * 0.3) * 0.2);

    float fractal = mandelbrot(fractal_uv * pc.cc74 * 2.0, 32);
    vec3 fractal_color = mix(base_color, harmonic_colors, fractal) * fractal;

    // Multi-layer noise patterns
    float noise1 = fbm(uv * 4.0 + pc.time * 0.5, 4) * pc.cc1;
    float noise2 = fbm(uv * 8.0 - pc.time * 0.3, 3) * pc.cc74;
    float noise3 = noise(uv * 16.0 + pc.time * 1.2) * pc.note_velocity;

    vec3 noise_pattern = vec3(noise1, noise2, noise3) * 0.3;

    // Particle system
    vec3 particles = particleField(uv, mouse_uv);

    // Energy rings around mouse when pressed
    vec3 mouse_rings = vec3(0.0);
    if(pc.mouse_pressed == 1) {
        float mouse_dist = length(uv - mouse_uv);
        for(int ring = 1; ring <= 3; ring++) {
            float ring_radius = float(ring) * 0.1 + pc.time * 0.2;
            float ring_thickness = 0.02;
            if(abs(mouse_dist - ring_radius) < ring_thickness) {
                float ring_intensity = (ring_thickness - abs(mouse_dist - ring_radius)) / ring_thickness;
                mouse_rings += harmonic_colors * ring_intensity * pc.note_velocity;
            }
        }
    }

    // Pitch bend creates spatial distortion visualization
    vec2 bent_uv = uv;
    float bend_amount = pc.pitch_bend * 0.5;
    bent_uv.x += bend_amount * sin(uv.y * PI * 4.0 + pc.time * 2.0);
    bent_uv.y += bend_amount * cos(uv.x * PI * 3.0 + pc.time * 1.5);

    vec3 distortion_lines = vec3(0.0);
    float line_pattern = sin(bent_uv.x * 50.0) * sin(bent_uv.y * 50.0);
    distortion_lines = base_color * smoothstep(0.7, 1.0, line_pattern) * abs(pc.pitch_bend) * 0.5;

    // Combine all effects
    vec3 final_color = fractal_color * (1.0 + vertexEnergy);
    final_color += noise_pattern;
    final_color += particles;
    final_color += mouse_rings;
    final_color += distortion_lines;

    // Global intensity modulation based on note count
    float global_intensity = 1.0 + float(pc.note_count) * 0.2;
    final_color *= global_intensity;

    // Time-based color cycling
    final_color += sin(pc.time + final_color.rgb) * 0.1;

    // Ensure we don't blow out the colors
    final_color = clamp(final_color, 0.0, 2.0);

    outColor = vec4(final_color, 1.0);
}