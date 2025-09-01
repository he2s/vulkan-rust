#version 450

// Push constants
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

layout(location = 0) in vec2 frag_uv;
layout(location = 1) in vec2 frag_screen_pos;

layout(location = 0) out vec4 out_color;

const float PI = 3.14159265359;
const float TAU = PI * 2.0;

// === GLITCH HASH FUNCTIONS ===

float hash(float n) { return fract(sin(n) * 1e4); }
float hash2(vec2 p) { return fract(1e4 * sin(17.0 * p.x + p.y * 0.1) * (0.1 + abs(sin(p.y * 13.0 + p.x)))); }

// === DIGITAL CORRUPTION ===

float digitalNoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);

    float a = hash2(i);
    float b = hash2(i + vec2(1.0, 0.0));
    float c = hash2(i + vec2(0.0, 1.0));
    float d = hash2(i + vec2(1.0, 1.0));

    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
}

// === CORRUPTED GRID ===

float corruptedGrid(vec2 p, float corruption_level) {
    // Base grid with glitch distortions
    vec2 corrupted_p = p;

    // Digital corruption effects
    if (corruption_level > 0.3) {
        // Pixel sorting artifacts
        float sort_strength = (corruption_level - 0.3) * 5.0;
        corrupted_p.x += floor(corrupted_p.y * 10.0) * sort_strength * 0.1;

        // Data moshing
        float mosh_time = floor(pc.time * 30.0 * corruption_level);
        corrupted_p += vec2(
            hash(mosh_time + corrupted_p.x * 100.0) - 0.5,
            hash(mosh_time + corrupted_p.y * 100.0) - 0.5
        ) * sort_strength * 0.2;
    }

    // Bitcrushing simulation
    float bit_depth = max(1.0, 8.0 - corruption_level * 6.0);
    corrupted_p = floor(corrupted_p * bit_depth) / bit_depth;

    // Multiple grid layers with different corruptions
    float grid_result = 0.0;

    for (int i = 0; i < 4; i++) {
        float layer_scale = 1.0 + float(i) * corruption_level * 2.0;
        vec2 layer_p = corrupted_p * layer_scale;

        // Scanline corruption
        if (mod(floor(layer_p.y * 20.0 + pc.time * 60.0), 3.0) == 0.0) {
            layer_p.x += sin(pc.time * 50.0 + layer_p.y) * corruption_level;
        }

        // Channel separation
        vec2 r_offset = vec2(corruption_level * 0.1, 0.0);
        vec2 g_offset = vec2(0.0, corruption_level * 0.05);
        vec2 b_offset = vec2(-corruption_level * 0.08, corruption_level * 0.03);

        float r_grid = step(0.5, fract(layer_p.x + r_offset.x) * fract(layer_p.y + r_offset.y));
        float g_grid = step(0.5, fract(layer_p.x + g_offset.x) * fract(layer_p.y + g_offset.y));
        float b_grid = step(0.5, fract(layer_p.x + b_offset.x) * fract(layer_p.y + b_offset.y));

        grid_result += (r_grid + g_grid + b_grid) / 3.0 / float(i + 1);
    }

    // Add digital artifacts
    float artifact = digitalNoise(corrupted_p * 50.0 + pc.time * 10.0);
    if (corruption_level > 0.5) {
        grid_result = mix(grid_result, artifact, (corruption_level - 0.5) * 2.0);
    }

    return fract(grid_result);
}

// === DATABENDING EFFECTS ===

vec3 databend(vec3 color, vec2 pos, float intensity) {
    vec3 bent = color;

    // Buffer overflow simulation
    if (intensity > 0.6) {
        float overflow = (intensity - 0.6) * 2.5;
        bent.r = fract(bent.r + overflow * hash(pos.x * 1000.0));
        bent.g = fract(bent.g + overflow * hash(pos.y * 1000.0 + 100.0));
        bent.b = fract(bent.b + overflow * hash((pos.x + pos.y) * 1000.0 + 200.0));
    }

    // Memory corruption
    float memory_corrupt = floor(pc.time * 20.0 * intensity + pos.x * 50.0) * 0.01;
    bent = fract(bent + memory_corrupt);

    // Bit shifting
    if (intensity > 0.4) {
        float shift_amount = floor(intensity * 8.0);
        bent = floor(bent * 255.0);
        bent.r = mod(bent.r * pow(2.0, shift_amount), 256.0) / 255.0;
        bent.g = mod(bent.g / pow(2.0, shift_amount), 256.0) / 255.0;
        bent.b = mod(bent.b * pow(2.0, shift_amount * 0.5), 256.0) / 255.0;
    }

    return bent;
}

void main() {
    vec2 fragCoord = frag_uv * vec2(float(pc.render_w), float(pc.render_h));
    vec2 resolution = vec2(float(pc.render_w), float(pc.render_h));

    // GLITCH TIME - stuttering and frame drops
    float audio_intensity = (pc.note_velocity + pc.cc1 + pc.cc74 + pc.osc_ch1 + pc.osc_ch2) / 5.0;
    float glitch_time = pc.time;

    // Frame stuttering based on audio
    if (audio_intensity > 0.5) {
        float stutter_rate = 30.0 * (audio_intensity - 0.5);
        glitch_time = floor(pc.time * stutter_rate) / stutter_rate;
    }

    // Frame drops and skips
    float frame_chaos = sin(pc.time * 60.0) * audio_intensity;
    if (frame_chaos > 0.7) {
        glitch_time += (frame_chaos - 0.7) * 10.0;
    }

    // CORRUPT POSITION CALCULATION
    vec2 base_p = fragCoord.xy / (20.0 + pc.note_velocity * 100.0);

    // Buffer underrun simulation
    vec2 p = base_p;
    float buffer_corruption = audio_intensity;

    if (buffer_corruption > 0.3) {
        // Memory access violations
        p += vec2(
            hash(floor(pc.time * 50.0)) - 0.5,
            hash(floor(pc.time * 50.0) + 1000.0) - 0.5
        ) * (buffer_corruption - 0.3) * 5.0;
    }

    // SCANLINE ARTIFACTS
    float scanline_y = mod(fragCoord.y + glitch_time * 200.0, 1.0 + audio_intensity * 20.0);
    if (scanline_y < 0.1 * audio_intensity) {
        p.x += sin(pc.time * 100.0) * audio_intensity * 0.5;
    }

    // RGB CHANNEL SEPARATION
    vec2 separation = vec2(audio_intensity * 0.02, 0.0);

    vec2 screen_center = (fragCoord.xy - resolution.xy / 2.0) / resolution.x / 1.5;

    // Mouse creates system crashes
    vec2 mouse_pos = vec2(float(pc.mouse_x) / float(pc.render_w),
                          float(pc.mouse_y) / float(pc.render_h)) - 0.5;
    mouse_pos.x *= float(pc.render_w) / float(pc.render_h);

    float corruption_level = audio_intensity;

    // Mouse click creates buffer overflow
    if (pc.mouse_pressed != 0) {
        float mouse_corruption = 3.0 * exp(-length(screen_center - mouse_pos) * 1.0);
        corruption_level = min(corruption_level + mouse_corruption, 2.0);
    }

    // Pitch bend creates memory leaks
    corruption_level += abs(pc.pitch_bend) * 1.5;

    // GLITCHED BOUNDARY
    float boundary = screen_center.x + 0.1 * screen_center.y;

    // Add glitch artifacts to boundary
    boundary += digitalNoise(screen_center * 20.0 + glitch_time * 5.0) * corruption_level * 0.3;

    // Random boundary jumps
    if (hash(floor(glitch_time * 10.0)) > 0.8) {
        boundary += (hash(floor(glitch_time * 10.0) + 100.0) - 0.5) * corruption_level;
    }

    // RENDER DIFFERENT AREAS
    vec3 final_color = vec3(0.0);

    if (boundary > -corruption_level * 0.3) {
        // CORRUPTED GRID AREA
        float grid_r = corruptedGrid(p + separation, corruption_level);
        float grid_g = corruptedGrid(p, corruption_level);
        float grid_b = corruptedGrid(p - separation, corruption_level);

        vec3 grid_color = vec3(grid_r, grid_g, grid_b);

        // MIDI creates system errors
        if (pc.note_count > 0) {
            float error_code = float(pc.last_note);
            vec3 error_color = vec3(
                step(0.5, fract(error_code * 0.1)),
                step(0.5, fract(error_code * 0.13)),
                step(0.5, fract(error_code * 0.17))
            ) * pc.note_velocity;

            // Blue screen of death effect
            if (pc.note_velocity > 0.8) {
                error_color = mix(error_color, vec3(0.0, 0.0, 1.0), 0.7);
            }

            grid_color = mix(grid_color, error_color, corruption_level * 0.5);
        }

        final_color = grid_color;
    } else {
        // EXTREME CORRUPTION ZONE
        vec4 cc = vec4(0.0);
        float total = 0.0;

        // Sampling with memory violations
        float samp = 15.0 + corruption_level * 50.0;
        float radius = length(screen_center) * (30.0 + corruption_level * 200.0);

        for (float t = -samp; t <= samp; t++) {
            float percent = t / samp;
            float weight = 1.0 - abs(percent);

            // Memory access with buffer overflow
            float u = t / 30.0;
            if (corruption_level > 0.7) {
                u += (hash(t + glitch_time * 100.0) - 0.5) * (corruption_level - 0.7) * 5.0;
            }

            vec2 sample_pos = p + dir * skew;

            // Memory corruption during sampling
            if (hash(sample_pos.x * 100.0 + glitch_time * 20.0) > 0.9) {
                sample_pos += vec2(
                    hash(sample_pos.x * 1000.0) - 0.5,
                    hash(sample_pos.y * 1000.0) - 0.5
                ) * corruption_level * 0.5;
            }

            vec4 samplev = vec4(
                corruptedGrid(sample_pos + vec2(0.03, 0.0), corruption_level),
                corruptedGrid(sample_pos + radius * vec2(0.005, 0.00), corruption_level * 1.1),
                corruptedGrid(sample_pos + radius * vec2(0.007, 0.00), corruption_level * 0.9),
                1.0
            );

            // Bit errors during sampling
            if (corruption_level > 0.6) {
                samplev.rgb = floor(samplev.rgb * 16.0) / 16.0; // Quantization

                // Random bit flips
                if (hash(u * 50.0 + glitch_time * 25.0) > 0.95) {
                    samplev.r = 1.0 - samplev.r;
                }
                if (hash(u * 50.0 + glitch_time * 25.0 + 100.0) > 0.95) {
                    samplev.g = 1.0 - samplev.g;
                }
                if (hash(u * 50.0 + glitch_time * 25.0 + 200.0) > 0.95) {
                    samplev.b = 1.0 - samplev.b;
                }
            }

            cc += samplev * weight;
            total += weight;
        }

        vec4 blurred = cc / total;

        // EXTREME CORRUPTION POST-PROCESSING
        vec3 corrupted_blur = blurred.rgb;

        // Databending
        corrupted_blur = databend(corrupted_blur, screen_center, corruption_level);

        // Compression artifacts
        if (corruption_level > 0.4) {
            float block_size = max(1.0, 8.0 - corruption_level * 6.0);
            vec2 block_coord = floor(fragCoord / block_size) * block_size;
            float block_hash = hash2(block_coord);

            // JPEG-like artifacts
            corrupted_blur += (block_hash - 0.5) * (corruption_level - 0.4) * 2.0;
        }

        // Buffer underrun darkness
        float darkness = length(screen_center) * (1.0 + corruption_level * 3.0);
        corrupted_blur -= darkness * vec3(0.8, 0.9, 1.0);

        final_color = corrupted_blur;
    }

    // GLOBAL CORRUPTION EFFECTS

    // System crash colors
    if (pc.note_count > 3) {
        // Multiple notes = system overload
        float overload = float(pc.note_count) / 10.0;
        vec3 crash_color = vec3(1.0, 0.0, 0.0) * overload; // Red screen of death
        final_color = mix(final_color, crash_color, overload * 0.3);
    }

    // OSC creates interference
    if (pc.osc_ch1 > 0.1) {
        // Horizontal interference lines
        float line_y = mod(fragCoord.y + pc.time * 50.0 * pc.osc_ch1, 4.0);
        if (line_y < 1.0) {
            final_color.r += pc.osc_ch1 * 0.5;
            final_color.g *= 1.0 - pc.osc_ch1 * 0.3;
        }
    }

    if (pc.osc_ch2 > 0.1) {
        // Vertical interference lines
        float line_x = mod(fragCoord.x + sin(pc.time * 30.0) * 10.0, 3.0);
        if (line_x < 1.0) {
            final_color.b += pc.osc_ch2 * 0.4;
            final_color.rg *= 1.0 - pc.osc_ch2 * 0.2;
        }
    }

    // Pitch bend creates signal noise
    if (abs(pc.pitch_bend) > 0.05) {
        float noise_intensity = abs(pc.pitch_bend);
        vec3 signal_noise = vec3(
            digitalNoise(fragCoord * 0.5 + pc.time * 20.0),
            digitalNoise(fragCoord * 0.7 + pc.time * 25.0 + vec2(100.0)),
            digitalNoise(fragCoord * 0.3 + pc.time * 15.0 + vec2(200.0))
        ) * noise_intensity * 0.3;

        final_color += signal_noise;
    }

    // FINAL SYSTEM CORRUPTION

    // CRT phosphor decay simulation
    float phosphor_decay = 1.0 - corruption_level * 0.2;
    final_color *= phosphor_decay;

    // VHS tracking errors
    if (corruption_level > 0.5) {
        float tracking_error = (corruption_level - 0.5) * 2.0;
        float track_line = mod(fragCoord.y + sin(pc.time * 10.0) * 50.0, 100.0);
        if (track_line < 5.0) {
            final_color *= 1.0 - tracking_error * 0.8;
            final_color.r += tracking_error * 0.5;
        }
    }

    // Interlacing artifacts
    if (mod(fragCoord.y, 2.0) < 1.0) {
        final_color *= 1.0 - corruption_level * 0.1;
    }

    // Ensure some visibility in extreme corruption
    final_color = max(final_color, vec3(0.02, 0.01, 0.03));

    // Prevent complete white-out from corruption
    final_color = min(final_color, vec3(1.5));

    out_color = vec4(final_color, 1.0);
} dir = vec2(
                hash(u * 1000.0 + glitch_time * 10.0),
                hash(u * 1000.0 + glitch_time * 10.0 + 500.0)
            ) - 0.5;
            dir = normalize(dir) * (0.01 + corruption_level * 0.05);

            float skew = percent * radius;

            // Stack overflow simulation
            if (corruption_level > 0.8) {
                skew *= 1.0 + sin(percent * 50.0 + glitch_time * 30.0) * (corruption_level - 0.8) * 10.0;
            }

            vec2