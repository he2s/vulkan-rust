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

void main() {
    vec2 uv = (frag_uv - 0.5) * 2.0;
    float aspect = float(pc.render_w) / float(pc.render_h);
    uv.x *= aspect;

    // Audio intensity
    float audio = (pc.note_velocity + pc.osc_ch1 + pc.osc_ch2) * 0.5;

    // Warped time
    float t = pc.time * (1.0 + audio * 3.0);

    // Neon tube formula - weird minimal oscillations
    float tube = sin(uv.x * 10.0 - t * 5.0) * cos(uv.y * 8.0 + t * 3.0);
    tube += sin(length(uv) * 15.0 - t * 4.0 + pc.pitch_bend * TAU);
    tube = abs(tube);

    // Mouse warp
    if (pc.mouse_pressed != 0) {
        vec2 mouse = vec2(float(pc.mouse_x), float(pc.mouse_y)) / vec2(float(pc.render_w), float(pc.render_h));
        mouse = (mouse - 0.5) * 2.0;
        mouse.x *= aspect;
        float mdist = length(uv - mouse);
        tube += exp(-mdist * 5.0) * 2.0;
    }

    // Neon glow layers
    float glow = 0.0;
    glow += 1.0 / (tube * 40.0 + 0.1);  // Bright core
    glow += 0.5 / (tube * 80.0 + 0.5);  // Mid glow
    glow += 0.2 / (tube * 200.0 + 1.0); // Outer halo

    // MIDI note color shift
    float hue = pc.note_count > 0 ? float(pc.last_note) / 127.0 : 0.5;
    hue += t * 0.1;

    // Weird neon colors
    vec3 color;
    color.r = glow * (0.5 + 0.5 * sin(hue * TAU + pc.osc_ch1 * TAU));
    color.g = glow * (0.5 + 0.5 * sin(hue * TAU + TAU/3.0 - pc.osc_ch2 * TAU));
    color.b = glow * (0.5 + 0.5 * sin(hue * TAU + 2.0*TAU/3.0 + audio * PI));

    // Feedback flicker
    color *= 1.0 + sin(t * 100.0 * (1.0 + audio * 10.0)) * audio * 0.3;

    // CC modulation for extra weirdness
    if (pc.cc1 > 0.0 || pc.cc74 > 0.0) {
        float cc_mod = (pc.cc1 + pc.cc74) * 0.5;
        color = mix(color, color.bgr, cc_mod);
        color += vec3(0.0, cc_mod, cc_mod * 0.5) * glow * 0.5;
    }

    // Ensure neon brightness
    color = clamp(color, vec3(0.0), vec3(3.0));

    out_color = vec4(color, 1.0);
}