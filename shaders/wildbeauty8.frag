#version 450

// WILDBEAUTY8 - Layered Bands + Floating Geometry

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
#define EPSILON 0.2
#define MAX_STEPS 10
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

// Band-specific palettes
vec3 getBandColor(int band, float t, float depth) {
    vec3 a, b, c, d;

    float energy = st(pc.note_velocity);
    float brightness = st(pc.cc74);

    if(band == 0) {
        a = vec3(0.2, 0.5, 0.8);
        b = vec3(0.6, 0.4, 0.5);
        c = vec3(1.2, 1.0, 1.5);
        d = vec3(0.0, 0.2, 0.4);
    }
    else if(band == 1) {
        a = vec3(0.5, 0.2, 0.6);
        b = vec3(0.5, 0.6, 0.4);
        c = vec3(1.5, 1.2, 1.0);
        d = vec3(0.6, 0.3, 0.1);
    }
    else if(band == 2) {
        a = vec3(0.8, 0.4, 0.2);
        b = vec3(0.4, 0.5, 0.6);
        c = vec3(1.0, 1.3, 1.8);
        d = vec3(0.3, 0.5, 0.7);
    }
    else if(band == 3) {
        a = vec3(0.3, 0.7, 0.4);
        b = vec3(0.6, 0.3, 0.5);
        c = vec3(1.8, 1.0, 1.2);
        d = vec3(0.8, 0.1, 0.3);
    }
    else {
        a = vec3(0.6, 0.3, 0.5);
        b = vec3(0.4, 0.6, 0.5);
        c = vec3(1.3, 1.5, 1.1);
        d = vec3(0.2, 0.6, 0.8);
    }

    // Modulate with depth and audio
    c *= 1.0 + brightness * 0.4;
    t += depth * 0.2 + energy * 0.1;

    return a + b * cos(TAU * (c * t + d));
}

float sdf_sphere(vec3 p, float r) {
    return length(p) - r;
}

float sdf_torus(vec3 p, vec2 t) {
    vec2 q = vec2(length(p.xz) - t.x, p.y);
    return length(q) - t.y;
}

vec3 glow;
int hitBand = 0;

vec2 sdf(vec3 p, vec2 uv) {
    float energy = st(pc.note_velocity);
    float modulation = st(pc.cc1);
    float t = pc.time * (1.2 + energy);

    // Determine band from UV Y position
    float bandPos = (uv.y + 1.0) * 0.5 * 5.0;
    int band = int(bandPos) % 5;
    hitBand = band;

    // Different geometry per band
    float d = 100.0;

    if(band == 0 || band == 2) {
        // Floating sphere
        vec3 ps = p + vec3(sin(t * 1.5) * 1.5, cos(t * 1.2) * 0.8, 0.0);
        d = sdf_sphere(ps, 0.4 + energy * 0.15);
    }
    else if(band == 1) {
        // Torus
        vec3 pt = p;
        pt.xy *= rot(t * 0.5);
        d = sdf_torus(pt, vec2(1.0, 0.2));
    }
    else if(band == 3) {
        // Multiple small spheres
        vec3 ps1 = p + vec3(sin(t) * 1.2, 0.0, cos(t) * 1.2);
        vec3 ps2 = p + vec3(sin(t + PI) * 1.2, 0.0, cos(t + PI) * 1.2);
        d = min(sdf_sphere(ps1, 0.25), sdf_sphere(ps2, 0.25));
    }
    else {
        // Torus
        vec3 pt = p;
        pt.xz *= rot(t * 0.7);
        d = sdf_torus(pt, vec2(0.8, 0.15 + modulation * 0.1));
    }

    return vec2(d, float(band));
}

vec2 trace(vec3 ro, vec3 rd, vec2 uv) {
    vec3 p = ro;
    float td = 0.0;
    glow = vec3(0.0);

    float glowStrength = 0.06 + pc.note_velocity * 0.1;

    for(int i = 0; i < MAX_STEPS; i++) {
        if(td >= 6.0) break;

        vec2 di = sdf(p, uv);

        if(di.x < EPSILON) {
            return vec2(td, di.y);
        }

        p += di.x * rd;

        float glowFactor = (1.0 - st(di.x / 0.5)) * glowStrength;
        int band = int(di.y);
        vec3 glowColor = getBandColor(band, length(p) + pc.time * 0.2, 0.5) * glowFactor;

        glow += glowColor;
        td = distance(ro, p);
    }

    return vec2(-1.0, -1.0);
}

vec3 get_normal(vec3 p, vec2 uv) {
    vec2 e = vec2(EPSILON * 0.5, 0.0);
    return normalize(vec3(
        sdf(p + e.xyy, uv).x - sdf(p - e.xyy, uv).x,
        sdf(p + e.yxy, uv).x - sdf(p - e.yxy, uv).x,
        sdf(p + e.yyx, uv).x - sdf(p - e.yyx, uv).x
    ));
}

vec3 render(vec2 uv) {
    float camDist = 3.2;
    vec3 ro = vec3(0.0, 0.0, -camDist);
    ro.xz *= rot(pc.time * 0.3 + pc.osc_ch1 * PI * 0.5);

    vec3 rd = normalize(vec3(uv, 1.5));

    vec2 tdi = trace(ro, rd, uv);

    if(tdi.x > 0.0) {
        vec3 p = ro + rd * tdi.x;
        vec3 n = get_normal(p, uv);

        int band = int(tdi.y);

        float depth = tdi.x / 6.0;
        float iridValue = dot(n, normalize(ro - p)) * 2.0;

        vec3 color = getBandColor(band, iridValue + pc.time * 0.3, depth);

        // Simple lighting
        float diff = st(dot(n, normalize(vec3(1.0, 1.0, -1.0))) * 0.5 + 0.5);
        color *= diff * 0.7 + 0.4;

        color += glow * 1.5;

        return color;
    }

    // Background bands
    float bandPos = (uv.y + 1.0) * 0.5 * 5.0;
    int band = int(bandPos) % 5;
    vec3 bg = getBandColor(band, uv.x + pc.time * 0.05, 0.0) * 0.15;
    bg += glow * 2.0;

    return bg;
}

void main() {
    vec2 resolution = vec2(float(pc.render_w), float(pc.render_h));
    vec2 uv = (fragUV - 0.5) * 2.0;
    uv.x *= resolution.x / resolution.y;

    float energy = st(pc.note_velocity);
    float modulation = st(pc.cc1);

    vec2 uvOriginal = uv;

    // MODERATE GLITCH EFFECTS
    float glitchTime = floor(pc.time * 8.0) / 8.0;
    float glitchFrame = hash2(glitchTime);

    // Band-based displacement
    float bandPos = (uv.y + 1.0) * 0.5 * 5.0;
    float bandF = fract(bandPos);

    if(hash2(floor(bandPos) + glitchTime) > 0.85 && energy > 0.5) {
        uv.x += (hash2(floor(bandPos) + glitchTime + 1.0) - 0.5) * 0.12 * energy;
    }

    // Pixelation on high energy
    if(energy > 0.7) {
        float pixelSize = 5.0 + energy * 10.0;
        uv = floor(uv * pixelSize) / pixelSize;
    }

    // Render with chromatic aberration
    float aberration = 0.006 + energy * 0.01;
    vec3 c = vec3(0.0);
    c.r = render(uv + vec2(aberration, 0.0)).r;
    c.g = render(uv).g;
    c.b = render(uv - vec2(aberration, 0.0)).b;

    // Band borders
    float border = smoothstep(0.46, 0.49, abs(bandF - 0.5));
    c *= border * 0.6 + 0.4;

    // Bit reduction
    float colorDepth = 150.0 - energy * 80.0;
    c = floor(c * colorDepth) / colorDepth;

    // Scanlines
    float scanline = sin(uv.y * resolution.y + pc.time * 5.0);
    c *= 1.0 - step(0.8, scanline) * energy * 0.12;

    // Random glitches
    if(hash(uv * 70.0 + glitchTime) > 0.97 - modulation * 0.05) {
        c *= vec3(hash(uv), hash(uv + 1.0), hash(uv + 2.0)) * 2.0;
    }

    // Vignette
    float vignette = 1.0 - pow(length(uvOriginal) * 0.6, 2.0);
    c *= st(vignette) * 0.8 + 0.2;

    outColor = vec4(st(c), 1.0);
}
