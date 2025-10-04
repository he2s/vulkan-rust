#version 450

// Push constants from application
layout(push_constant) uniform PushConstants {
    float time;
    uint  mouse_x;
    uint  mouse_y;
    uint  mouse_pressed;
    float note_velocity;
    float pitch_bend;
    float cc1;    // mid frequencies / mod wheel
    float cc74;   // high frequencies / cutoff
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

// Constants
#define PI 3.141592653589793238
#define TAU (2.0 * PI)
#define EPSILON 0.3
#define MAX_STEPS 8
#define MAX_DIST 8.0

// Helper functions
#define pos(x) (x * 0.5 + 0.5)
#define st(x) clamp(x, 0.0, 1.0)

// Hash functions for glitch effects
float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

float hash2(float n) {
    return fract(sin(n) * 43758.5453);
}

float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
}

// Rotation matrix
mat2 rot(float a) {
    float c = cos(a);
    float s = sin(a);
    return mat2(c, -s, s, c);
}

// IQ's palette function (like new10) but with glitch modifications
vec3 pal(in float t, in vec3 a, in vec3 b, in vec3 c, in vec3 d) {
    float glitchAmount = pc.note_velocity * 0.2;
    t += hash2(floor(t * 50.0 + pc.time * 30.0)) * glitchAmount;
    return a + b * cos(TAU * (c * t + d));
}

// Band-based palettes (from new10)
vec3 getPalette(int band, float t) {
    vec3 a, b, c, d;

    float energy = st(pc.note_velocity);
    float modulation = st(pc.cc1);
    float brightness = st(pc.cc74);

    if(band == 0) {
        a = vec3(0.5, 0.5, 0.5);
        b = vec3(0.5, 0.5, 0.5);
        c = vec3(1.0, 1.0, 1.0) * (1.0 + brightness * 0.5);
        d = vec3(0.00, 0.33, 0.67);
    }
    else if(band == 1) {
        a = vec3(0.5, 0.5, 0.5) + energy * 0.2;
        b = vec3(0.5, 0.5, 0.5) * (0.5 + modulation * 0.5);
        c = vec3(1.0, 1.0, 0.5);
        d = vec3(0.30, 0.20, 0.20);
    }
    else if(band == 2) {
        a = vec3(0.8, 0.5, 0.4);
        b = vec3(0.2, 0.4, 0.2);
        c = vec3(2.0, 1.0, 1.0) * (1.0 + brightness * 0.3);
        d = vec3(0.00, 0.25, 0.25) + vec3(pc.pitch_bend * 0.2);
    }
    else {
        a = vec3(0.5 + energy * 0.3, 0.5, 0.5);
        b = vec3(0.5, 0.5, 0.5);
        c = vec3(1.0, 0.7, 0.4);
        d = vec3(0.80, 0.90, 0.30);
    }

    return pal(t, a, b, c, d);
}

// Simplified SDF primitives
float sdf_torus(vec3 p, vec2 t) {
    vec2 q = vec2(length(p.xz) - t.x, p.y);
    return length(q) - t.y;
}

float sdf_sphere(vec3 p, float r) {
    return length(p) - r;
}

// Smooth min for blending
float smin(float a, float b, float k) {
    float h = st(0.5 + 0.5 * (b - a) / k);
    return mix(b, a, h) - k * h * (1.0 - h);
}

// Global glow accumulator
vec3 glow;

// Scene SDF
vec2 sdf(vec3 p) {
    float energy = st(pc.note_velocity);
    float modulation = st(pc.cc1);

    float t = pc.time * (1.0 + energy * 2.0);

    // Main torus - simplified
    vec3 p1 = p;
    p1.xy *= rot(t * 0.5);
    p1.yz *= rot(t * 0.3);
    float torus1 = sdf_torus(p1, vec2(1.2, 0.15 + energy * 0.1));

    // Secondary torus
    vec3 p2 = p;
    p2.xy *= rot(t * 0.7 + PI * 0.5);
    p2.xz *= rot(t * 0.4);
    float torus2 = sdf_torus(p2, vec2(1.0, 0.12));

    // Floating sphere
    vec3 ps = p + vec3(sin(t) * 1.5, cos(t * 1.3) * 1.0, 0.0);
    float sphere = sdf_sphere(ps, 0.3 + energy * 0.2);

    // Combine
    float smoothness = 0.1 + modulation * 0.3;
    float scene = smin(torus1, torus2, smoothness);
    scene = smin(scene, sphere, smoothness * 0.8);

    return vec2(scene, 1.0);
}

// Raymarching
vec2 trace(vec3 ro, vec3 rd) {
    vec3 p = ro;
    float td = 0.0;
    glow = vec3(0.0);

    float glowStrength = 0.05 + pc.note_velocity * 0.1;

    for(int i = 0; i < MAX_STEPS; i++) {
        if(td >= MAX_DIST) break;

        vec2 di = sdf(p);

        if(di.x < EPSILON) {
            return vec2(td, di.y);
        }

        p += di.x * rd;

        // Accumulate glow
        float glowFactor = (1.0 - st(di.x / 0.5)) * glowStrength;
        vec3 glowColor = pos(normalize(p)) * glowFactor;
        glowColor *= vec3(1.0 + pc.cc74, 1.0 + sin(pc.time * 3.0), 1.0 + pc.cc1);
        glow += glowColor;

        td = distance(ro, p);
    }

    return vec2(-1.0, -1.0);
}

// Normal calculation
vec3 get_normal(vec3 p) {
    vec2 e = vec2(EPSILON * 0.5, 0.0);
    return normalize(vec3(
        sdf(p + e.xyy).x - sdf(p - e.xyy).x,
        sdf(p + e.yxy).x - sdf(p - e.yxy).x,
        sdf(p + e.yyx).x - sdf(p - e.yyx).x
    ));
}

// Main rendering
vec3 render(vec2 uv) {
    float energy = st(pc.note_velocity);
    float modulation = st(pc.cc1);

    // Camera setup
    float camDist = 3.0 + sin(pc.time) * 0.5;
    vec3 ro = vec3(0.0, 0.0, -camDist);
    ro.xz *= rot(pc.time * 0.3 + pc.osc_ch1 * PI);

    vec3 rd = normalize(vec3(uv, 1.5));

    vec2 tdi = trace(ro, rd);

    if(tdi.x > 0.0) {
        vec3 p = ro + rd * tdi.x;
        vec3 n = get_normal(p);

        // Band-based coloring (blend 3D with band concept)
        float bandValue = (uv.y + 1.0) * 0.5 * 4.0;
        int band = int(bandValue) % 4;

        // Iridescence value
        float iridValue = dot(n, normalize(ro - p)) * 2.0 + length(p) * 0.3;

        vec3 color = getPalette(band, iridValue + pc.time * 0.3);

        // Basic lighting
        vec3 lightDir = normalize(vec3(1.0, 1.0, -1.0));
        float diff = st(dot(n, lightDir) * 0.5 + 0.5);
        color *= diff * 0.7 + 0.3;

        // Add glow
        color += glow * 1.5;

        return color;
    }

    // Background with glow
    vec3 bg = glow * 2.0;
    return bg;
}

void main() {
    vec2 resolution = vec2(float(pc.render_w), float(pc.render_h));
    vec2 uv = (fragUV - 0.5) * 2.0;
    uv.x *= resolution.x / resolution.y;

    float energy = st(pc.note_velocity);
    float modulation = st(pc.cc1);
    float brightness = st(pc.cc74);

    // Store original UV
    vec2 uvOriginal = uv;

    // GLITCH EFFECTS FROM NEW10

    // Glitch time quantization
    float glitchTime = floor(pc.time * 6.0) / 6.0;
    float glitchFrame = hash2(glitchTime);

    // Horizontal displacement glitches
    if(glitchFrame > 0.85 && energy > 0.4) {
        float displaceAmount = hash(vec2(floor(uv.y * 15.0), glitchTime));
        if(displaceAmount > 0.7) {
            uv.x += (displaceAmount - 0.5) * 0.15 * energy;
        }
    }

    // Vertical band shifting
    float bandGlitch = step(0.95, sin(pc.time * 80.0 + uv.y * 30.0));
    uv.x += bandGlitch * modulation * 0.15;

    // Pixelation effect
    float pixelSize = 1.0 + energy * 15.0 * step(0.92, hash2(glitchTime + 0.5));
    vec2 pixelated = floor(uv * vec2(resolution.x / resolution.y, 1.0) * pixelSize) / pixelSize;
    uv = mix(uv, pixelated, step(0.6, energy) * step(0.75, glitchFrame));

    // Chromatic aberration (simplified from wildbeauty3)
    float aberration = 0.005 + brightness * 0.015;
    vec3 c = vec3(0.0);
    c.r = render(uv + vec2(aberration, 0.0)).r;
    c.g = render(uv).g;
    c.b = render(uv - vec2(aberration, 0.0)).b;

    // RGB channel shifting (from new10)
    if(hash2(glitchTime + 1.0) > 0.7 && energy > 0.3) {
        float rgbShift = energy * 0.03;
        vec3 colR = render(uv + vec2(rgbShift, 0.0));
        vec3 colB = render(uv - vec2(rgbShift, 0.0));
        c.r = colR.r;
        c.b = colB.b;
    }

    // Color bit reduction (from new10)
    float colorDepth = 128.0 - energy * 100.0;
    c = floor(c * colorDepth) / colorDepth;

    // Random color corruption
    if(hash(uv * 80.0 + glitchTime) > 0.96 - modulation * 0.08) {
        c = vec3(hash(uv + glitchTime), hash(uv + glitchTime + 1.0), hash(uv + glitchTime + 2.0));
    }

    // Scanline effect (from new10)
    float scanline = sin(uv.y * resolution.y * 1.5 + pc.time * 8.0);
    scanline = step(0.75, scanline) * energy * 0.15;
    c *= 1.0 - scanline;

    // Band-based borders
    float bandPos = (uv.y + 1.0) * 0.5 * 4.0;
    float f = fract(bandPos);
    float border = smoothstep(0.45, 0.48, abs(f - 0.5));
    c *= border * 0.7 + 0.3;

    // Vignette (simplified from wildbeauty3)
    float vignette = 1.0 - pow(length(uvOriginal) * 0.6, 2.0);
    vignette = st(vignette);
    c *= vignette * 0.8 + 0.2;

    // Contrast and saturation
    float contrast = 1.1 + brightness * 0.3;
    c = (c - 0.5) * contrast + 0.5;

    // Note-triggered glitch bursts
    if(pc.note_count > 0u) {
        int flashBand = int(pc.note_count % 4u);
        int currentBand = int(bandPos) % 4;
        if(currentBand == flashBand && energy > 0.5) {
            c *= 1.0 + sin(pc.time * 30.0) * 0.3;
        }
    }

    // Static noise overlay
    float staticNoise = hash(uv * 500.0 + vec2(pc.time * 50.0, 0.0));
    c = mix(c, vec3(staticNoise), energy * 0.08);

    outColor = vec4(st(c), 1.0);
}
