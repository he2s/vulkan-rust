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

// === EXTREME HASH FUNCTIONS ===

float hash(float n) { return fract(sin(n) * 43758.5453123); }
float hash2(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123); }

vec3 hash3(vec3 p) {
    p = vec3(dot(p, vec3(127.1, 311.7, 74.7)),
             dot(p, vec3(269.5, 183.3, 246.1)),
             dot(p, vec3(113.5, 271.9, 124.6)));
    return -1.0 + 2.0 * fract(sin(p) * 43758.5453123);
}

// === EXTREME NOISE ===

float noise3d(vec3 x) {
    vec3 p = floor(x);
    vec3 f = fract(x);
    f = f * f * (3.0 - 2.0 * f);

    float n = p.x + p.y * 57.0 + 113.0 * p.z;
    return mix(mix(mix(hash(n + 0.0), hash(n + 1.0), f.x),
                   mix(hash(n + 57.0), hash(n + 58.0), f.x), f.y),
               mix(mix(hash(n + 113.0), hash(n + 114.0), f.x),
                   mix(hash(n + 170.0), hash(n + 171.0), f.x), f.y), f.z);
}

// === INSANE GRID FUNCTION ===

float psychedelicGrid(vec2 p, float chaos_level) {
    // Multiple overlapping grid layers with different orientations
    float grid_sum = 0.0;

    for (int i = 0; i < 8; i++) {
        float layer_chaos = chaos_level * float(i + 1);

        // Wildly rotating orientations
        vec2 orient = normalize(vec2(
            sin(pc.time * (2.0 + float(i)) + layer_chaos * 10.0),
            cos(pc.time * (1.5 + float(i)) + layer_chaos * 8.0)
        ));

        // Chaotic scaling
        float scale = 0.5 + layer_chaos * 5.0;
        scale *= 1.0 + sin(pc.time * 3.0 + float(i)) * layer_chaos;

        // Warped position
        vec2 warped_p = p;
        warped_p += vec2(
            sin(p.y * scale + pc.time * 4.0) * layer_chaos,
            cos(p.x * scale + pc.time * 3.0) * layer_chaos
        );

        vec2 perp = vec2(orient.y, -orient.x);

        float g = mod(
            floor(scale * dot(warped_p, orient)) +
            floor(scale * dot(warped_p, perp)),
            2.0 + float(i % 3)
        );

        grid_sum += g / float(i + 1);
    }

    return fract(grid_sum);
}

// === FRACTAL DISTORTION ===

vec2 fractalDistort(vec2 p, float intensity) {
    vec2 distorted = p;

    for (int i = 0; i < 5; i++) {
        float scale = pow(2.0, float(i));
        vec2 noise_pos = distorted * scale + pc.time * (1.0 + float(i) * 0.5);

        distorted += vec2(
            noise3d(vec3(noise_pos, pc.time)) * intensity / scale,
            noise3d(vec3(noise_pos + 100.0, pc.time + 50.0)) * intensity / scale
        );
    }

    return distorted;
}

void main() {
    vec2 fragCoord = frag_uv * vec2(float(pc.render_w), float(pc.render_h));
    vec2 resolution = vec2(float(pc.render_w), float(pc.render_h));

    // EXTREME time scaling with feedback loops
    float audio_intensity = (pc.note_velocity + pc.cc1 + pc.cc74 + pc.osc_ch1 + pc.osc_ch2) / 5.0;
    float time_chaos = pc.time * (0.1 + audio_intensity * 10.0);
    time_chaos += sin(pc.time * 20.0) * audio_intensity * 2.0;

    // INSANE position calculations
    vec2 base_p = fragCoord.xy / (10.0 + pc.note_velocity * 200.0);

    // Multiple time offsets creating feedback
    vec2 p = base_p + vec2(
        -time_chaos + pc.osc_ch1 * 10.0 + sin(pc.time * 15.0) * pc.cc1 * 3.0,
        time_chaos + pc.osc_ch2 * 8.0 + cos(pc.time * 12.0) * pc.cc74 * 4.0
    );

    // EXTREME fractal distortion
    float distortion_intensity = 0.1 + audio_intensity * 2.0;
    p = fractalDistort(p, distortion_intensity);

    // Screen center with chaotic offset
    vec2 q = (fragCoord.xy - (resolution.xy / 2.0)) / resolution.x / 1.5;

    // Mouse creates reality-warping effects
    vec2 mouse_pos = vec2(float(pc.mouse_x) / float(pc.render_w),
                          float(pc.mouse_y) / float(pc.render_h)) - 0.5;
    mouse_pos.x *= float(pc.render_w) / float(pc.render_h);

    // CHAOS LEVEL calculation
    float chaos_level = audio_intensity;
    if (pc.mouse_pressed != 0) {
        float mouse_chaos = 2.0 * exp(-length(q - mouse_pos) * 2.0);
        chaos_level += mouse_chaos;
    }

    // Pitch bend creates reality breaks
    chaos_level += abs(pc.pitch_bend) * 3.0;

    // Note count multiplies chaos
    chaos_level *= 1.0 + float(pc.note_count) * 0.5;

    // EXTREME boundary calculation
    float boundary = q.x + 0.1 * q.y;
    boundary += sin(q.x * 20.0 + pc.time * 10.0) * chaos_level * 0.3;
    boundary += cos(q.y * 15.0 + pc.time * 8.0) * chaos_level * 0.2;
    boundary += noise3d(vec3(q * 10.0, pc.time * 2.0)) * chaos_level * 0.5;

    if (boundary > -chaos_level * 0.5) {
        // PSYCHEDELIC GRID AREA
        vec4 c = vec4(psychedelicGrid(p, chaos_level));

        // EXTREME color mutations
        vec3 base_color = c.rgb;

        // Multiple color layers with different frequencies
        for (int i = 0; i < 6; i++) {
            float freq = float(i + 1) * 2.0;
            vec3 color_layer = 0.5 + 0.5 * sin(vec3(
                pc.time * freq + base_color.r * 10.0,
                pc.time * freq * 1.3 + base_color.g * 12.0,
                pc.time * freq * 0.8 + base_color.b * 8.0
            ));

            base_color = mix(base_color, color_layer, chaos_level / 6.0);
        }

        // MIDI note creates color explosions
        if (pc.note_count > 0) {
            float note_hue = float(pc.last_note) / 127.0;
            vec3 note_explosion = vec3(
                sin(note_hue * TAU * 3.0 + pc.time * 20.0),
                sin(note_hue * TAU * 5.0 + pc.time * 25.0),
                sin(note_hue * TAU * 7.0 + pc.time * 30.0)
            ) * pc.note_velocity * 2.0;
            base_color += note_explosion;
        }

        // OSC creates color storms
        base_color.r += pc.osc_ch1 * sin(pc.time * 50.0 + q.x * 30.0) * 2.0;
        base_color.g += pc.osc_ch2 * cos(pc.time * 40.0 + q.y * 25.0) * 2.0;
        base_color.b += abs(pc.pitch_bend) * sin(pc.time * 60.0) * 1.5;

        out_color = vec4(base_color, 1.0);
    } else {
        // EXTREME BLUR AREA
        vec4 cc = vec4(0.0);
        float total = 0.0;

        // INSANE sample count
        float samp = 20.0 + chaos_level * 100.0;

        // EXTREME radius calculation
        float radius = length(q) * (50.0 + chaos_level * 500.0);
        radius *= 1.0 + sin(pc.time * 10.0) * chaos_level;

        for (float t = -samp; t <= samp; t++) {
            float percent = t / samp;
            float weight = 1.0 - abs(percent);
            float u = t / 50.0;

            // CHAOTIC direction generation
            vec2 chaos_offset = vec2(
                noise3d(vec3(u, pc.time * 0.1, 0.0)),
                noise3d(vec3(u, pc.time * 0.1, 100.0))
            ) * chaos_level * 0.1;

            vec2 dir = vec2(
                fract(sin(537.3 * (u + 0.5) + pc.time * chaos_level)),
                fract(sin(523.7 * (u + 0.25) + pc.time * chaos_level * 1.3))
            ) + chaos_offset;

            dir = normalize(dir) * (0.01 + chaos_level * 0.1);

            float skew = percent * radius;
            skew += sin(percent * 20.0 + pc.time * 15.0) * chaos_level * 10.0;

            // EXTREME multi-dimensional sampling
            vec2 sample_pos = p + dir * skew;

            vec4 samplev = vec4(
                psychedelicGrid(sample_pos + vec2(0.03, 0.0) * (1.0 + chaos_level), chaos_level),
                psychedelicGrid(sample_pos + radius * vec2(0.005, 0.00) * (1.0 + chaos_level), chaos_level * 1.2),
                psychedelicGrid(sample_pos + radius * vec2(0.007, 0.00) * (1.0 + chaos_level), chaos_level * 0.8),
                1.0
            );

            // Color mutation during sampling
            samplev.rgb += 0.2 * sin(vec3(
                u * 100.0 + pc.time * 30.0,
                u * 120.0 + pc.time * 25.0,
                u * 80.0 + pc.time * 35.0
            ));

            cc += samplev * weight;
            total += weight;
        }

        vec4 blurred = cc / total;

        // EXTREME darkening with oscillations
        float darken_factor = 0.5 + chaos_level * 5.0;
        darken_factor *= 1.0 + sin(length(q) * 50.0 + pc.time * 20.0) * chaos_level;

        vec4 darkening = length(q) * vec4(1.0, 1.0, 1.0, 1.0) * darken_factor;

        vec4 final_blur = blurred - darkening;

        // EXTREME color enhancement
        if (pc.note_count > 0) {
            float note_hue = float(pc.last_note) / 127.0;
            vec3 note_madness = vec3(
                sin(note_hue * TAU * 10.0 + pc.time * 50.0),
                sin(note_hue * TAU * 12.0 + pc.time * 60.0),
                sin(note_hue * TAU * 8.0 + pc.time * 40.0)
            ) * pc.note_velocity * chaos_level;
            final_blur.rgb += note_madness;
        }

        // OSC creates color explosions
        final_blur.r += pc.osc_ch1 * sin(pc.time * 100.0 + length(q) * 50.0) * chaos_level;
        final_blur.g += pc.osc_ch2 * cos(pc.time * 80.0 + atan(q.y, q.x) * 20.0) * chaos_level;
        final_blur.b += abs(pc.pitch_bend) * sin(pc.time * 120.0) * chaos_level * 2.0;

        // Mouse creates reality tears
        if (pc.mouse_pressed != 0) {
            float mouse_dist = length(q - mouse_pos);
            float reality_tear = 5.0 * exp(-mouse_dist * 1.0);
            final_blur.rgb += vec3(reality_tear) * sin(vec3(
                pc.time * 200.0,
                pc.time * 180.0 + PI,
                pc.time * 220.0 + PI * 0.5
            ));
        }

        out_color = final_blur;
    }

    // EXTREME post-processing
    vec3 final_color = out_color.rgb;

    // Color channel feedback loops
    final_color.r += sin(final_color.g * 20.0 + pc.time * 30.0) * chaos_level * 0.2;
    final_color.g += cos(final_color.b * 25.0 + pc.time * 25.0) * chaos_level * 0.2;
    final_color.b += sin(final_color.r * 30.0 + pc.time * 35.0) * chaos_level * 0.2;

    // Global chaos intensity
    final_color *= 1.0 + chaos_level * 2.0;

    // Chromatic aberration on steroids
    vec2 offset = (frag_screen_pos * 0.02 + sin(pc.time * 10.0) * 0.01) * chaos_level;
    final_color.r += sin(pc.time * 100.0 + offset.x * 50.0) * 0.1;
    final_color.b += cos(pc.time * 80.0 + offset.y * 40.0) * 0.1;

    // Ensure some visibility in the madness
    final_color = max(final_color, vec3(0.05) * chaos_level);

    // Clamp to prevent complete white-out
    final_color = min(final_color, vec3(3.0));

    out_color = vec4(final_color, 1.0);
}