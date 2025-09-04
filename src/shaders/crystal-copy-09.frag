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

// Simple hash
float hash(float n) {
    return fract(sin(n) * 43758.5453123);
}

float hash2(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
}

// Crazy grid function
float crazyGrid(vec2 p) {
    // Audio intensity
    float intensity = (pc.note_velocity + pc.cc1 + pc.cc74 + pc.osc_ch1 + pc.osc_ch2) / 5.0;

    // Multiple rotating grids
    float result = 0.0;

    for (int i = 0; i < 5; i++) {
        float layer = float(i + 1);

        // Crazy rotation based on time and audio
        float angle = pc.time * layer * (2.0 + intensity * 10.0) + pc.pitch_bend * TAU;
        mat2 rotation = mat2(cos(angle), -sin(angle), sin(angle), cos(angle));

        vec2 rotated_p = rotation * p;

        // Scale based on audio and OSC
        float scale = layer * (1.0 + intensity * 5.0);
        rotated_p *= scale;

        // Warping
        rotated_p += vec2(
            sin(rotated_p.y + pc.time * 5.0) * intensity,
            cos(rotated_p.x + pc.time * 3.0) * intensity
        );

        // Grid calculation
        vec2 grid_pos = fract(rotated_p) - 0.5;
        float grid = step(0.1 + intensity * 0.3, max(abs(grid_pos.x), abs(grid_pos.y)));

        result += grid / layer;
    }

    return fract(result);
}

void main() {
    vec2 fragCoord = frag_uv * vec2(float(pc.render_w), float(pc.render_h));
    vec2 resolution = vec2(float(pc.render_w), float(pc.render_h));

    // Audio intensity
    float intensity = (pc.note_velocity + pc.cc1 + pc.cc74 + pc.osc_ch1 + pc.osc_ch2) / 5.0;

    // Crazy time with stutters
    float crazy_time = pc.time * (0.5 + intensity * 5.0);
    if (intensity > 0.7) {
        crazy_time = floor(crazy_time * 30.0) / 30.0; // Stuttering
    }

    // Position with extreme movement
    vec2 p = fragCoord.xy / (30.0 + pc.note_velocity * 200.0);
    p += vec2(
        -crazy_time + pc.osc_ch1 * 5.0,
        crazy_time + pc.osc_ch2 * 5.0
    );

    // Screen center
    vec2 q = (fragCoord.xy - resolution.xy / 2.0) / resolution.x / 1.5;

    // Mouse position
    vec2 mouse = vec2(float(pc.mouse_x) / float(pc.render_w),
                      float(pc.mouse_y) / float(pc.render_h)) - 0.5;
    mouse.x *= float(pc.render_w) / float(pc.render_h);

    // Boundary with chaos
    float boundary = q.x + 0.1 * q.y;
    boundary += sin(q.x * 20.0 + crazy_time * 10.0) * intensity * 0.5;
    boundary += cos(q.y * 15.0 + crazy_time * 8.0) * intensity * 0.3;

    // Mouse effect
    if (pc.mouse_pressed != 0) {
        float mouse_dist = length(q - mouse);
        boundary += 2.0 * exp(-mouse_dist * 3.0);
    }

    vec3 final_color;

    if (boundary > -intensity * 0.5) {
        // Grid area with crazy colors
        float grid = crazyGrid(p);

        vec3 base_color = vec3(grid);

        // Crazy color mutations
        base_color.r += sin(crazy_time * 20.0 + grid * 10.0) * intensity;
        base_color.g += cos(crazy_time * 25.0 + grid * 15.0) * intensity;
        base_color.b += sin(crazy_time * 30.0 + grid * 8.0) * intensity;

        // MIDI note colors
        if (pc.note_count > 0) {
            float note_hue = float(pc.last_note) / 127.0;
            vec3 note_color = vec3(
                sin(note_hue * TAU),
                sin(note_hue * TAU + TAU/3.0),
                sin(note_hue * TAU + 2.0*TAU/3.0)
            ) * pc.note_velocity;
            base_color += note_color;
        }

        // OSC colors
        base_color.r += pc.osc_ch1 * sin(crazy_time * 50.0);
        base_color.g += pc.osc_ch2 * cos(crazy_time * 40.0);
        base_color.b += abs(pc.pitch_bend) * sin(crazy_time * 60.0);

        final_color = base_color;
    } else {
        // Blur area
        vec3 blur_color = vec3(0.0);
        float total = 0.0;

        float samples = 30.0 + intensity * 50.0;
        float radius = length(q) * (100.0 + intensity * 300.0);

        for (float i = 0.0; i < samples; i += 1.0) {
            float angle = (i / samples) * TAU;
            float dist = (i / samples) * radius;

            vec2 offset = vec2(cos(angle), sin(angle)) * dist * 0.01;
            vec2 sample_pos = p + offset;

            float sample_grid = crazyGrid(sample_pos);
            float weight = 1.0 - (i / samples);

            blur_color += vec3(sample_grid) * weight;
            total += weight;
        }

        blur_color /= total;

        // Darken based on distance
        float darkness = length(q) * (1.5 + intensity * 2.0);
        blur_color -= darkness;

        // Color enhancement
        blur_color.r += pc.osc_ch1 * sin(crazy_time * 100.0) * intensity;
        blur_color.g += pc.osc_ch2 * cos(crazy_time * 80.0) * intensity;
        blur_color.b += abs(pc.pitch_bend) * sin(crazy_time * 120.0) * intensity;

        final_color = blur_color;
    }

    // Global effects
    final_color *= 1.0 + intensity * 2.0;

    // Scanlines
    float scanline = sin(fragCoord.y * 2.0 + crazy_time * 50.0) * 0.1 * intensity;
    final_color += scanline;

    // Chromatic aberration
    if (intensity > 0.5) {
        final_color.r += sin(crazy_time * 200.0) * (intensity - 0.5);
        final_color.b += cos(crazy_time * 180.0) * (intensity - 0.5);
    }

    // Ensure visibility
    final_color = max(final_color, vec3(0.01));
    final_color = min(final_color, vec3(2.0));

    out_color = vec4(final_color, 1.0);
}