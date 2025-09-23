#version 450

layout(push_constant) uniform PushConstants {
    float time;
    uint  mouse_x;
    uint  mouse_y;
    uint  mouse_pressed;
    float note_velocity;
    float pitch_bend;
    float cc1;
    float cc74;
    uint  note_count;
    uint  last_note;
    uint  render_w;
    uint  render_h;
} pc;

layout(location = 0) in  vec2 fragCoord;
layout(location = 0) out vec4 fragColor;

void main() {
    vec2 resolution = vec2(float(max(1000, 1u)), float(max(1000, 1u)));
    vec2 uv = gl_FragCoord.xy / resolution;

    // Mouse position (normalized)
    vec2 mouse = vec2(float(pc.mouse_x), float(pc.mouse_y)) / resolution;

    // Distance from mouse
    float mouseDistance = distance(uv, mouse);

    // Base color - simple gradient
    vec3 color = vec3(0.1, 0.2, 0.4);

    // Show time as a slow color shift
    color.r += 0.3 * sin(pc.time * 0.5);
    color.g += 0.2 * cos(pc.time * 0.3);

    // Show CC1 control as blue intensity
    color.b += pc.cc1 * 0.5;

    // Show CC74 control as green shift
    color.g += pc.cc74 * 0.3;

    // Show pitch bend as horizontal color shift
    color.r += pc.pitch_bend * 0.4;

    // Show note velocity as brightness
    color *= (1.0 + pc.note_velocity * 0.8);

    // Mouse interaction - bright circle around mouse
    if (mouseDistance < 0.1) {
        color += vec3(0.5, 0.3, 0.1) * (1.0 - mouseDistance * 10.0);
    }

    // Mouse pressed - red tint when pressed
    if (pc.mouse_pressed != 0u) {
        color.r += 0.3;
    }

    // Show quadrants for easy reference
    if (abs(uv.x - 0.5) < 0.002 || abs(uv.y - 0.5) < 0.002) {
        color = vec3(0.8, 0.8, 0.8);
    }

    fragColor = vec4(color, 1.0);
}