#version 450

// EXPERIMENTAL WEIRD GEOMETRY SHADER - Creates complex audio-reactive forms
layout(points) in;
layout(triangle_strip, max_vertices = 32) out; // Increased for complex shapes

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

// Input from vertex shader
layout(location = 0) in vec2 vs_uv[];
layout(location = 1) in vec2 vs_screen_pos[];

// Output to fragment shader
layout(location = 0) out vec2 frag_uv;
layout(location = 1) out vec2 frag_screen_pos;
layout(location = 2) out float frag_distance_from_center;
layout(location = 3) out float frag_weird_factor;
layout(location = 4) out vec3 frag_color_mod;

// Advanced noise functions
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

// Fractal noise
float fbm(vec2 p) {
    float f = 0.0;
    f += 0.5000 * noise(p); p *= 2.02;
    f += 0.2500 * noise(p); p *= 2.03;
    f += 0.1250 * noise(p); p *= 2.01;
    f += 0.0625 * noise(p);
    return f / 0.9375;
}

void main() {
    vec4 center = gl_in[0].gl_Position;
    vec2 screen_center = vs_screen_pos[0];

    // Audio-reactive parameters
    float beat_pulse = 1.0 - pc.time_to_next_beat;
    float audio_chaos = pc.osc_ch1 + pc.osc_ch2 + pc.note_velocity;
    float pitch_warp = pc.pitch_bend * 2.0 - 1.0;

    // Weird factor based on multiple inputs
    float weird_base = sin(pc.time * 0.5 + center.x * 10.0) * cos(pc.time * 0.3 + center.y * 8.0);
    float weird_factor = weird_base + audio_chaos + beat_pulse;

    // Noise-based distortions
    vec2 noise_coord = screen_center * 5.0 + pc.time * 0.2;
    float chaos_noise = fbm(noise_coord) * 2.0 - 1.0;
    float temporal_noise = noise(vec2(pc.time * 2.0, dot(screen_center, vec2(1.0)))) * 2.0 - 1.0;

    // Dynamic size with extreme variations
    float base_size = 0.05 + pc.note_velocity * 0.1;
    float size_mod = 1.0 + sin(weird_factor * 6.28) * 0.5 + chaos_noise * 0.3;
    float final_size = base_size * size_mod * (1.0 + beat_pulse * 2.0);

    // Determine shape complexity based on audio
    int num_sides = int(3.0 + audio_chaos * 8.0 + abs(pitch_warp) * 4.0);
    num_sides = clamp(num_sides, 3, 15);

    // Color modulation for fragment shader
    vec3 color_mod = vec3(
        0.5 + sin(pc.time + weird_factor) * 0.5,
        0.5 + cos(pc.time * 1.3 + chaos_noise) * 0.5,
        0.5 + sin(pc.time * 0.7 + temporal_noise) * 0.5
    );

    // WEIRD SHAPE GENERATION - Multiple possible forms
    if (weird_factor > 1.5) {
        // FRACTAL SPIKES MODE
        for (int i = 0; i <= num_sides; i++) {
            float angle = float(i) * 6.28318 / float(num_sides);
            float spike_length = final_size * (1.0 + sin(angle * 3.0 + pc.time * 4.0) * 0.5);
            spike_length *= 1.0 + chaos_noise * 0.4;

            // Inner vertex (center-ish with jitter)
            vec2 inner_offset = vec2(cos(angle), sin(angle)) * final_size * 0.3;
            inner_offset += vec2(temporal_noise, chaos_noise) * 0.01;

            gl_Position = center + vec4(inner_offset, 0.0, 0.0);
            frag_uv = vs_uv[0] + inner_offset * 10.0;
            frag_screen_pos = screen_center + inner_offset;
            frag_distance_from_center = length(inner_offset);
            frag_weird_factor = weird_factor;
            frag_color_mod = color_mod;
            EmitVertex();

            // Outer spike vertex
            vec2 outer_offset = vec2(cos(angle), sin(angle)) * spike_length;
            outer_offset += vec2(
                sin(angle * 5.0 + pc.time * 3.0) * final_size * 0.2,
                cos(angle * 7.0 + pc.time * 2.0) * final_size * 0.2
            );

            gl_Position = center + vec4(outer_offset, 0.0, 0.0);
            frag_uv = vs_uv[0] + outer_offset * 5.0;
            frag_screen_pos = screen_center + outer_offset;
            frag_distance_from_center = length(outer_offset);
            frag_weird_factor = weird_factor * 1.5;
            frag_color_mod = color_mod * 1.2;
            EmitVertex();
        }

    } else if (weird_factor > 0.5) {
        // TWISTED POLYGON MODE
        float twist_amount = pitch_warp * 3.14159 + pc.time * 0.5;

        // Center vertex
        gl_Position = center;
        frag_uv = vs_uv[0];
        frag_screen_pos = screen_center;
        frag_distance_from_center = 0.0;
        frag_weird_factor = weird_factor;
        frag_color_mod = color_mod;
        EmitVertex();

        for (int i = 0; i <= num_sides; i++) {
            float angle = float(i) * 6.28318 / float(num_sides);
            float twisted_angle = angle + twist_amount * sin(angle * 2.0);

            float radius_mod = 1.0 + sin(angle * 4.0 + pc.time * 2.0) * 0.3;
            radius_mod += chaos_noise * 0.2;

            vec2 offset = vec2(cos(twisted_angle), sin(twisted_angle)) * final_size * radius_mod;

            gl_Position = center + vec4(offset, 0.0, 0.0);
            frag_uv = vs_uv[0] + offset * 8.0;
            frag_screen_pos = screen_center + offset;
            frag_distance_from_center = length(offset);
            frag_weird_factor = weird_factor;
            frag_color_mod = color_mod;
            EmitVertex();
        }

    } else {
        // CHAOTIC TRIANGLE MODE with heavy distortion
        vec2 vertex_offsets[3] = vec2[3](
            vec2(0.0, 1.0),
            vec2(-0.866, -0.5),
            vec2(0.866, -0.5)
        );

        for (int i = 0; i < 3; i++) {
            vec2 base_offset = vertex_offsets[i] * final_size;

            // Add multiple layers of distortion
            base_offset += vec2(
                sin(pc.time * 3.0 + base_offset.x * 20.0) * final_size * 0.3,
                cos(pc.time * 2.5 + base_offset.y * 15.0) * final_size * 0.3
            );

            base_offset += vec2(chaos_noise, temporal_noise) * final_size * 0.5;
            base_offset *= 1.0 + sin(weird_factor * 10.0) * 0.2;

            gl_Position = center + vec4(base_offset, 0.0, 0.0);
            frag_uv = vs_uv[0] + base_offset * 12.0;
            frag_screen_pos = screen_center + base_offset;
            frag_distance_from_center = length(base_offset);
            frag_weird_factor = weird_factor;
            frag_color_mod = color_mod;
            EmitVertex();
        }
    }

    EndPrimitive();
}