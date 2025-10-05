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

// Plasma Energy - Combines iterative sinusoidal patterns with energy flows

#define PI 3.141592653589793238
#define TAU (2.0 * PI)
#define sat(x) clamp(x, 0.0, 1.0)

mat2 rot(float a) {
    float c = cos(a), s = sin(a);
    return mat2(c, -s, s, c);
}

vec3 palette(float t) {
    vec3 a = vec3(0.5, 0.5, 0.5);
    vec3 b = vec3(0.5, 0.5, 0.5);
    vec3 c = vec3(1.0, 1.0, 1.0);
    vec3 d = vec3(0.0, 0.10, 0.20);
    d += vec3(pc.osc_ch1 * 0.3, pc.osc_ch2 * 0.3, pc.pitch_bend * 0.2);
    return a + b * cos(TAU * (c * t + d));
}

// Multi-layer plasma
float plasma(vec2 uv, float t) {
    float energy = sat(pc.note_velocity);
    float modulation = sat(pc.cc1);

    // Base frequencies
    float freq1 = 3.0 + modulation * 5.0;
    float freq2 = 4.0 + energy * 6.0;
    float freq3 = 2.0 + vertexEnergy * 3.0;

    // Layer 1 - horizontal waves
    float p1 = sin(uv.x * freq1 + t);

    // Layer 2 - vertical waves
    float p2 = sin(uv.y * freq2 + t * 1.3);

    // Layer 3 - diagonal waves
    float p3 = sin((uv.x + uv.y) * freq3 + t * 0.8);

    // Layer 4 - circular waves
    float p4 = sin(length(uv) * (5.0 + energy * 3.0) - t * 2.0);

    // Layer 5 - spiral
    float angle = atan(uv.y, uv.x);
    float p5 = sin(angle * 3.0 + length(uv) * 8.0 - t * 1.5);

    // Combine layers
    float plasma = (p1 + p2 + p3 + p4 + p5) / 5.0;

    // Add turbulence
    plasma += sin(uv.x * 10.0 + sin(uv.y * 8.0 + t * 0.5)) * 0.1 * energy;

    return plasma;
}

// Voronoi-like energy cells
float voronoi(vec2 uv, float t) {
    vec2 i = floor(uv);
    vec2 f = fract(uv);

    float minDist = 1.0;

    for(int y = -1; y <= 1; y++) {
        for(int x = -1; x <= 1; x++) {
            vec2 neighbor = vec2(float(x), float(y));
            vec2 point = 0.5 + 0.5 * sin(t + TAU * fract(sin(dot(i + neighbor, vec2(127.1, 311.7))) * 43758.5453));
            vec2 diff = neighbor + point - f;
            float dist = length(diff);
            minDist = min(minDist, dist);
        }
    }

    return minDist;
}

void main() {
    vec2 resolution = vec2(float(pc.render_w), float(pc.render_h));
    vec2 uv = (fragUV - 0.5) * 2.0;
    uv.x *= resolution.x / resolution.y;

    float energy = sat(pc.note_velocity);
    float modulation = sat(pc.cc1);
    float brightness = sat(pc.cc74);
    float t = pc.time;

    // Mouse/OSC interaction
    vec2 offset = vec2(0.0);
    if(pc.mouse_pressed > 0u) {
        offset = (vec2(float(pc.mouse_x), float(pc.mouse_y)) / resolution - 0.5) * 2.0;
    } else {
        offset = vec2(pc.osc_ch1, pc.osc_ch2) * 1.0;
    }

    uv += offset;

    // Audio-reactive rotation and zoom
    uv = rot(pc.pitch_bend * PI + t * 0.1 * modulation) * uv;
    float zoom = 1.0 + energy * 0.5;
    zoom *= 1.0 + sin(t * 10.0) * energy * 0.1;
    uv *= zoom;

    // UV distortion
    float distortStr = energy * 0.3;
    uv += vec2(
        sin(uv.y * 3.0 + t) * distortStr,
        cos(uv.x * 3.0 + t * 1.2) * distortStr
    );

    // Get plasma value
    float p = plasma(uv, t);

    // Get voronoi cells
    float v = voronoi(uv * (2.0 + brightness * 3.0), t);

    // Combine patterns
    float combined = mix(p, v, 0.3 + modulation * 0.4);
    combined = fract(combined * 3.0 + t * 0.2);

    // Color from palette
    vec3 col = palette(combined);

    // Add secondary layer with different timing
    float p2 = plasma(uv * 1.3 + vec2(0.5), t * 0.7);
    p2 = fract(p2 * 2.0 + t * 0.3);
    vec3 col2 = palette(p2 + 0.5);
    col = mix(col, col2, 0.4);

    // Energy bursts
    if(energy > 0.5) {
        float burst = pow(energy - 0.5, 2.0) * 4.0;
        burst *= (1.0 + sin(t * 20.0) * 0.5);
        col += palette(t * 0.5) * burst;
    }

    // Brightness control
    col *= 0.8 + brightness * 0.7;

    // Add edge glow based on voronoi
    float edge = smoothstep(0.0, 0.1, v) - smoothstep(0.1, 0.2, v);
    col += edge * palette(t * 0.3) * (1.0 + energy);

    // Hue shift based on note count
    if(pc.note_count > 0u) {
        float shift = float(pc.note_count % 8u) * 0.125;
        col = mix(col, col.gbr, shift * 0.5);
    }

    // Contrast
    col = (col - 0.5) * (1.0 + modulation * 0.5) + 0.5;

    // Saturation
    float gray = dot(col, vec3(0.299, 0.587, 0.114));
    col = mix(vec3(gray), col, 1.3 + energy * 0.5);

    // Radial gradient for depth
    float radial = 1.0 - length(uv) * 0.2;
    radial = sat(radial);
    col *= radial;

    // Vignette
    float vig = 1.0 - pow(length((fragUV - 0.5) * 1.3), 1.5);
    col *= vig;

    // Bloom
    vec3 bloom = max(col - vec3(0.8), 0.0) * 2.0;
    col += bloom * energy * 0.5;

    outColor = vec4(sat(col), 1.0);
}
