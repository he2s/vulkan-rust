#version 450

// WILDBEAUTY6 - Heavy Glitch + Minimal 3D

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

#define PI 3.141592653589793238
#define TAU (2.0 * PI)
#define st(x) clamp(x, 0.0, 1.0)

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

float hash2(float n) {
    return fract(sin(n) * 43758.5453);
}

mat2 rot(float a) {
    float c = cos(a), s = sin(a);
    return mat2(c, -s, s, c);
}

// Simple band-based palette
vec3 bandPalette(float t, int band) {
    float energy = st(pc.note_velocity);
    float brightness = st(pc.cc74);

    vec3 a = vec3(0.5);
    vec3 b = vec3(0.5);
    vec3 c = vec3(1.0);
    vec3 d = vec3(0.0, 0.33, 0.67);

    if(band == 1) d = vec3(0.3, 0.2, 0.2);
    else if(band == 2) { a = vec3(0.8, 0.5, 0.4); b = vec3(0.3, 0.5, 0.3); }
    else if(band == 3) c = vec3(1.0, 0.7, 0.4);
    else if(band == 4) d = vec3(0.8, 0.9, 0.3);
    else if(band == 5) c = vec3(2.0, 1.0, 1.0);

    c *= 1.0 + brightness * 0.5;

    return a + b * cos(TAU * (c * t + d));
}

// Minimal 3D depth hint
float getDepth(vec2 uv) {
    vec3 p = vec3(uv * 2.0, 0.0);
    p.xy *= rot(pc.time * 0.5 + pc.osc_ch1);

    float d = length(vec2(length(p.xy) - 1.2, p.z)) - 0.2;
    return st(1.0 - d * 0.5);
}

void main() {
    vec2 resolution = vec2(float(pc.render_w), float(pc.render_h));
    vec2 uv = (fragUV - 0.5) * 2.0;
    uv.x *= resolution.x / resolution.y;

    float energy = st(pc.note_velocity);
    float modulation = st(pc.cc1);
    float brightness = st(pc.cc74);

    vec2 uvOriginal = uv;

    // EXTREME GLITCH EFFECTS
    float glitchTime = floor(pc.time * 10.0) / 10.0;
    float glitchFrame = hash2(glitchTime);

    // Data moshing
    float mosh = hash(vec2(floor(uv.y * 20.0), glitchTime));
    if(mosh > 0.7 - energy * 0.3) {
        uv.x += (mosh - 0.5) * 0.3 * energy;
        uv.x = fract(uv.x + 0.5) - 0.5;
    }

    // Vertical slice corruption
    float slice = floor(uv.x * 30.0);
    if(hash2(slice + glitchTime) > 0.8) {
        uv.y += (hash2(slice + glitchTime + 1.0) - 0.5) * 0.5 * modulation;
    }

    // Heavy pixelation
    float pixelSize = 2.0 + energy * 30.0 * step(0.85, glitchFrame);
    vec2 pixelated = floor(uv * pixelSize) / pixelSize;
    uv = mix(uv, pixelated, step(0.5, energy) * 0.7);

    // Stuttering time
    float stutterTime = pc.time;
    if(energy > 0.4) {
        stutterTime = floor(pc.time * 20.0) / 20.0;
    }

    // Band selection
    float bandPos = (uv.y + 1.0) * 0.5 * 6.0 + sin(stutterTime * 5.0) * modulation * 0.5;

    // Random band jumping
    if(glitchFrame > 0.9 && energy > 0.6) {
        bandPos += floor(hash2(glitchTime + uv.x) * 6.0) - 3.0;
    }

    int band = int(bandPos) % 6;

    // Get depth for slight 3D hint
    float depth = getDepth(uv);

    // Animate
    float t = uv.x + stutterTime * 0.02 + depth * 0.1;

    // Get color
    vec3 c = bandPalette(t, band);

    // RGB glitch
    if(hash2(glitchTime) > 0.6) {
        float shift = energy * 0.05;
        vec3 cr = bandPalette(t + shift, band);
        vec3 cb = bandPalette(t - shift, band);
        c.r = cr.r;
        c.b = cb.b;
    }

    // Heavy bit crushing
    float bitDepth = 64.0 - energy * 50.0;
    c = floor(c * bitDepth) / bitDepth;

    // Digital artifacts
    if(hash(uv * 100.0 + glitchTime) > 0.95 - modulation * 0.15) {
        c = vec3(hash(uv), hash(uv + 1.0), hash(uv + 2.0));
    }

    // Color channel swapping
    if(hash2(glitchTime + 2.0) > 0.8 && energy > 0.5) {
        c = c.gbr;
    }

    // Band borders
    float f = fract(bandPos);
    float border = smoothstep(0.48, 0.5, abs(f - 0.5));
    c *= border * 0.5 + 0.5;

    // Scanlines
    float scanline = sin(uv.y * resolution.y * 2.0 + stutterTime * 12.0);
    c *= 1.0 - step(0.85, scanline) * energy * 0.25;

    // Signal corruption
    float signalLoss = step(0.96, sin(stutterTime * 10.0 + uv.x * 60.0));
    c *= 1.0 - signalLoss * brightness * 0.6;

    // Strobe flash
    if(energy > 0.7 && hash2(floor(stutterTime * 30.0)) > 0.7) {
        c = vec3(1.0) - c;
    }

    // Static noise
    float staticNoise = hash(uv * 800.0 + vec2(pc.time * 80.0, 0.0));
    c = mix(c, vec3(staticNoise), energy * 0.15);

    // Vignette corruption
    float vignette = 1.0 - length(uvOriginal) * 0.5;
    if(hash2(glitchTime + 10.0) > 0.85) {
        vignette = 1.0 - vignette;
    }
    c *= st(vignette);

    outColor = vec4(st(c), 1.0);
}
