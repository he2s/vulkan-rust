#version 450

// Push constants matching your Rust PushConstants struct
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
} pc;

// Output to fragment shader
layout(location = 0) out vec2 fragCoord;
layout(location = 1) out float pattern_intensity;
layout(location = 2) out vec2 uv;

// Constants for the generative pattern
const float PI = 3.14159265359;
const float TAU = 6.28318530718;

// Hash function for pseudo-random values
float hash(float n) {
    return fract(sin(n) * 43758.5453123);
}

// 2D rotation matrix
mat2 rot2D(float angle) {
    float s = sin(angle);
    float c = cos(angle);
    return mat2(c, -s, s, c);
}

void main() {
    // Generate vertex ID-based coordinates for fullscreen triangle
    vec2 vertices[3] = vec2[3](
    vec2(-1.0, -1.0),
    vec2( 3.0, -1.0),
    vec2(-1.0,  3.0)
    );

    vec2 base_pos = vertices[gl_VertexIndex];

    // Calculate UV coordinates
    uv = base_pos * 0.5 + 0.5;

    // Mouse influence (normalized)
    vec2 mouse = vec2(float(pc.mouse_x), float(pc.mouse_y)) / vec2(float(pc.render_w), float(pc.render_h));
    mouse = mouse * 2.0 - 1.0; // Convert to [-1, 1] range

    // Time-based animation
    float t = pc.time * 0.5;
    float pulse = sin(t * 2.0) * 0.5 + 0.5;

    // MIDI-reactive parameters
    float midi_energy = pc.note_velocity * 2.0 + pc.cc1;
    float midi_morph = pc.pitch_bend * 0.5 + 0.5;
    float osc_influence = (pc.osc_ch1 + pc.osc_ch2) * 0.5;

    // Create multiple layers of movement
    vec2 displacement = vec2(0.0);

    // Layer 1: Circular waves emanating from center
    float dist_from_center = length(base_pos);
    float wave1 = sin(dist_from_center * 10.0 - t * 3.0 + midi_energy * PI);
    displacement += normalize(base_pos) * wave1 * 0.1 * pc.note_velocity;

    // Layer 2: Spiral distortion
    float angle = atan(base_pos.y, base_pos.x);
    float spiral = sin(angle * 5.0 + dist_from_center * 8.0 - t * 2.0);
    vec2 spiral_disp = vec2(cos(angle + spiral), sin(angle + spiral)) * 0.05;
    displacement += spiral_disp * (0.5 + midi_morph);

    // Layer 3: Grid distortion based on vertex ID
    float grid_x = sin(float(gl_VertexIndex) * 1.618 + t) * 0.1;
    float grid_y = cos(float(gl_VertexIndex) * 2.718 - t * 1.3) * 0.1;
    displacement += vec2(grid_x, grid_y) * pc.cc74;

    // Layer 4: Mouse interaction - attract/repel vertices
    if (pc.mouse_pressed > 0) {
        vec2 to_mouse = mouse - base_pos;
        float mouse_dist = length(to_mouse);
        float mouse_influence = exp(-mouse_dist * 2.0) * 0.3;
        displacement += normalize(to_mouse) * mouse_influence * sin(t * 10.0);
    }

    // Layer 5: Audio reactive breathing
    float breathing = sin(t + pc.cc1 * TAU) * 0.05;
    displacement += base_pos * breathing * (1.0 + osc_influence);

    // Layer 6: Fractal-like recursive displacement
    for (int i = 0; i < 3; i++) {
        float scale = pow(2.0, float(i));
        vec2 offset = vec2(
        sin(t * scale + float(i) * 1.234),
        cos(t * scale * 0.7 + float(i) * 2.345)
        ) * 0.02 / scale;
        displacement += offset * midi_energy;
    }

    // Layer 7: Note-based geometric transforms
    float note_norm = float(pc.last_note) / 127.0;
    mat2 rotation = rot2D(note_norm * TAU + t * 0.5);
    displacement = rotation * displacement;

    // Apply Perlin-noise-like distortion
    float noise_scale = 5.0 + sin(t) * 2.0;
    float noise_x = sin(base_pos.x * noise_scale + t) * cos(base_pos.y * noise_scale - t * 0.7);
    float noise_y = cos(base_pos.x * noise_scale - t * 1.3) * sin(base_pos.y * noise_scale + t);
    displacement += vec2(noise_x, noise_y) * 0.03 * (1.0 + pc.note_count * 0.1);

    // Final vertex position with bounded displacement
    vec2 final_pos = base_pos + displacement * 0.5;

    // Ensure we still cover the full screen (clamp displacement magnitude)
    float max_displacement = 0.3;
    if (length(displacement) > max_displacement) {
        displacement = normalize(displacement) * max_displacement;
        final_pos = base_pos + displacement;
    }

    // Calculate pattern intensity for fragment shader
    pattern_intensity = length(displacement) * 3.0;
    pattern_intensity += wave1 * 0.5;
    pattern_intensity += spiral * 0.3;
    pattern_intensity = clamp(pattern_intensity, 0.0, 1.0);

    // Mix with MIDI/audio values for more variation
    pattern_intensity = mix(pattern_intensity, pc.note_velocity, 0.3);
    pattern_intensity = mix(pattern_intensity, abs(pc.pitch_bend), 0.2);

    // Output final vertex position
    gl_Position = vec4(final_pos, 0.0, 1.0);

    // Pass through coordinates for fragment shader
    fragCoord = final_pos;
}