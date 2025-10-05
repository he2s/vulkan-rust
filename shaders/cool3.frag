#version 450

// Push constants from application
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
    float osc_ch1;
    float osc_ch2;
    uint  render_w;
    uint  render_h;
} pc;

layout(location = 0) in vec2 fragUV;
layout(location = 1) in float vertexEnergy;
layout(location = 2) in vec3 worldPos;

layout(location = 0) out vec4 outColor;

// Psychedelic Feedback - Combines iterative transformation with Mandelbrot-style coloring

#define PI 3.141592653589793238
#define TAU (2.0 * PI)
#define sat(x) clamp(x, 0.0, 1.0)

mat2 rot(float a) {
    float c = cos(a), s = sin(a);
    return mat2(c, -s, s, c);
}

vec3 palette(float t) {
    vec3 a = vec3(0.5);
    vec3 b = vec3(0.5);
    vec3 c = vec3(1.0, 0.7, 0.4);
    vec3 d = vec3(0.0, 0.15, 0.20);
    d += vec3(pc.osc_ch1, pc.osc_ch2, pc.pitch_bend) * 0.3;
    return a + b * cos(TAU * (c * t + d));
}

vec3 complexFeedback(vec2 uv) {
    float energy = sat(pc.note_velocity);
    float modulation = sat(pc.cc1);
    float brightness = sat(pc.cc74);
    float t = pc.time;

    // Scale and rotate UV
    uv *= mix(0.8, 1.5, modulation);
    uv = rot(pc.pitch_bend * PI + t * 0.1) * uv;

    vec2 z = uv;
    vec2 c = uv * 0.5;
    float escape = 0.0;
    float minDist = 10.0;

    int maxIter = int(15.0 + brightness * 25.0);

    for(int i = 0; i < 40; i++) {
        if(i >= maxIter) break;

        // Complex squaring with audio twist
        float zx = z.x * z.x - z.y * z.y;
        float zy = 2.0 * z.x * z.y;

        // Add feedback
        z = vec2(zx, zy) + c;

        // Audio perturbation
        z += vec2(sin(t + float(i) * 0.2), cos(t * 1.3 + float(i) * 0.15)) * energy * 0.05;
        z = rot(t * 0.05 + float(i) * 0.1) * z;

        // Track minimum distance for orbit trap coloring
        minDist = min(minDist, length(z));

        if(length(z) > 4.0) {
            escape = float(i);
            break;
        }
    }

    vec3 col;

    if(escape > 0.0) {
        // Escaped - create bands
        float t1 = escape / float(maxIter);
        t1 = fract(t1 * 8.0 + pc.time * 0.3);
        col = palette(t1);

        // Add orbit trap coloring
        float t2 = sat(minDist * 0.5);
        col = mix(col, palette(t2 + 0.5), 0.5);
    } else {
        // Didn't escape - interior
        float interior = length(z) * 0.2;
        interior = fract(interior * 5.0 + pc.time * 0.2);
        col = palette(interior) * 0.5;
    }

    // Apply vertex energy
    col *= 1.0 + vertexEnergy * 0.4;

    return col;
}

void main() {
    vec2 resolution = vec2(float(pc.render_w), float(pc.render_h));
    vec2 uv = (fragUV - 0.5) * 2.0;
    uv.x *= resolution.x / resolution.y;

    float energy = sat(pc.note_velocity);
    float modulation = sat(pc.cc1);

    // Mouse/OSC interaction
    vec2 offset = vec2(0.0);
    if(pc.mouse_pressed > 0u) {
        offset = (vec2(float(pc.mouse_x), float(pc.mouse_y)) / resolution - 0.5) * 4.0;
    } else {
        offset = vec2(pc.osc_ch1, pc.osc_ch2) * 2.0;
        offset += vec2(sin(pc.time * 0.3), cos(pc.time * 0.25)) * 0.5;
    }

    uv += offset;

    // Audio zoom with pulsing
    float zoom = 1.0 + energy * 2.0;
    zoom *= 1.0 + sin(pc.time * 8.0) * energy * 0.3;
    uv /= zoom;

    // Get base color
    vec3 col = complexFeedback(uv);

    // Kaleidoscope effect at high energy
    if(energy > 0.7) {
        float angle = atan(uv.y, uv.x);
        float segments = 4.0 + floor(energy * 8.0);
        angle = mod(angle, TAU / segments) * segments;
        vec2 kaleidoUV = vec2(cos(angle), sin(angle)) * length(uv);
        kaleidoUV += offset;
        kaleidoUV /= zoom;
        vec3 kaleidoCol = complexFeedback(kaleidoUV);
        col = mix(col, kaleidoCol, (energy - 0.7) * 2.0);
    }

    // Chromatic aberration
    float aberration = energy * 0.02;
    vec3 colR = complexFeedback(uv + vec2(aberration, 0));
    vec3 colB = complexFeedback(uv - vec2(aberration, 0));
    col = vec3(colR.r, col.g, colB.b);

    // Hue shift based on note count
    if(pc.note_count > 0u) {
        float hueShift = float(pc.note_count % 12u) * 0.08;
        col = mix(col, col.gbr, hueShift);
    }

    // Contrast and saturation
    float contrast = 1.1 + modulation * 0.4;
    col = (col - 0.5) * contrast + 0.5;

    float gray = dot(col, vec3(0.299, 0.587, 0.114));
    col = mix(vec3(gray), col, 1.2 + energy * 0.5);

    // Bloom
    vec3 bloom = max(col - vec3(1.0), 0.0) * 1.5;
    col += bloom * energy;

    // Vignette
    float vig = 1.0 - pow(length((fragUV - 0.5) * 1.5), 2.0);
    col *= vig;

    outColor = vec4(sat(col), 1.0);
}
