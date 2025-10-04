#version 450

// WILDBEAUTY10 - Kaleidoscope Chaos (symmetry + glitch + 3D)

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
#define EPSILON 0.25
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

// Psychedelic palette
vec3 kaleidoPalette(float t) {
    float energy = st(pc.note_velocity);
    float brightness = st(pc.cc74);

    vec3 a = vec3(0.6, 0.3, 0.8);
    vec3 b = vec3(0.9, 0.7, 0.8) * (1.0 + brightness);
    vec3 c = vec3(2.0, 1.5, 2.5) * (1.0 + energy * 0.5);
    vec3 d = vec3(0.0, 0.33, 0.67) + vec3(pc.osc_ch1, pc.osc_ch2, pc.pitch_bend) * 0.3;

    return a + b * cos(TAU * (c * t + d));
}

float sdf_torus(vec3 p, vec2 t) {
    vec2 q = vec2(length(p.xz) - t.x, p.y);
    return length(q) - t.y;
}

float sdf_sphere(vec3 p, float r) {
    return length(p) - r;
}

float sdf_box(vec3 p, vec3 b) {
    vec3 q = abs(p) - b;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}

float smin(float a, float b, float k) {
    float h = st(0.5 + 0.5 * (b - a) / k);
    return mix(b, a, h) - k * h * (1.0 - h);
}

vec3 glow;

vec2 sdf(vec3 p) {
    float energy = st(pc.note_velocity);
    float modulation = st(pc.cc1);

    float t = pc.time * (1.3 + energy * 1.5);

    // Rotating torus
    vec3 p1 = p;
    p1.xy *= rot(t * 0.6);
    p1.yz *= rot(t * 0.4);
    float torus = sdf_torus(p1, vec2(1.0, 0.15 + energy * 0.1));

    // Central sphere
    float sphere = sdf_sphere(p, 0.4 + sin(t * 2.0) * 0.1);

    // Orbiting box
    vec3 pb = p + vec3(sin(t * 1.5) * 1.5, 0.0, cos(t * 1.5) * 1.5);
    pb *= rot(t * 3.0);
    float box = sdf_box(pb, vec3(0.2 + modulation * 0.1));

    float scene = smin(torus, sphere, 0.2 + modulation * 0.3);
    scene = smin(scene, box, 0.15);

    return vec2(scene, 1.0);
}

vec2 trace(vec3 ro, vec3 rd) {
    vec3 p = ro;
    float td = 0.0;
    glow = vec3(0.0);

    float glowStrength = 0.06 + pc.note_velocity * 0.1;

    for(int i = 0; i < MAX_STEPS; i++) {
        if(td >= 8.0) break;

        vec2 di = sdf(p);

        if(di.x < EPSILON) {
            return vec2(td, di.y);
        }

        p += di.x * rd;

        float glowFactor = (1.0 - st(di.x / 0.5)) * glowStrength;
        vec3 glowColor = kaleidoPalette(length(p) * 0.5 + pc.time * 0.3) * glowFactor;

        glow += glowColor;
        td = distance(ro, p);
    }

    return vec2(-1.0, -1.0);
}

vec3 get_normal(vec3 p) {
    vec2 e = vec2(EPSILON * 0.5, 0.0);
    return normalize(vec3(
        sdf(p + e.xyy).x - sdf(p - e.xyy).x,
        sdf(p + e.yxy).x - sdf(p - e.yxy).x,
        sdf(p + e.yyx).x - sdf(p - e.yyx).x
    ));
}

vec3 render(vec2 uv) {
    float camDist = 3.0 + sin(pc.time * 0.5) * 0.5;
    vec3 ro = vec3(0.0, 0.0, -camDist);
    ro.xz *= rot(pc.time * 0.4 + pc.osc_ch1 * PI);

    vec3 rd = normalize(vec3(uv, 1.5));

    vec2 tdi = trace(ro, rd);

    if(tdi.x > 0.0) {
        vec3 p = ro + rd * tdi.x;
        vec3 n = get_normal(p);

        float iridValue = dot(n, normalize(ro - p)) * 2.5 + length(p) * 0.3;

        vec3 color = kaleidoPalette(iridValue + pc.time * 0.4);

        vec3 lightDir = normalize(vec3(1.0, 1.0, -1.0));
        float diff = st(dot(n, lightDir) * 0.6 + 0.5);

        float spec = pow(st(dot(reflect(rd, n), lightDir)), 16.0);

        color *= diff * 0.75 + 0.3;
        color += vec3(spec * 1.5);
        color += glow * 1.8;

        return color;
    }

    vec3 bg = glow * 2.2;
    return bg;
}

void main() {
    vec2 resolution = vec2(float(pc.render_w), float(pc.render_h));
    vec2 uv = (fragUV - 0.5) * 2.0;
    uv.x *= resolution.x / resolution.y;

    float energy = st(pc.note_velocity);
    float modulation = st(pc.cc1);
    float brightness = st(pc.cc74);

    vec2 uvOriginal = uv;

    // KALEIDOSCOPE EFFECT
    float angle = atan(uv.y, uv.x);
    float radius = length(uv);

    // Number of segments based on audio
    float segments = 6.0 + floor(modulation * 6.0);

    // Mirror across segments
    float segmentAngle = TAU / segments;
    angle = mod(angle, segmentAngle);

    // Mirror within segment
    if(mod(floor(atan(uv.y, uv.x) / segmentAngle), 2.0) > 0.5) {
        angle = segmentAngle - angle;
    }

    vec2 kaleidoUV = vec2(cos(angle), sin(angle)) * radius;

    // GLITCH EFFECTS
    float glitchTime = floor(pc.time * 7.0) / 7.0;
    float glitchFrame = hash2(glitchTime);

    // Segment jumping
    if(glitchFrame > 0.88 && energy > 0.5) {
        float jumpAngle = floor(hash2(glitchTime) * segments) * segmentAngle;
        kaleidoUV *= rot(jumpAngle);
    }

    // Radial glitch
    if(hash2(glitchTime + 1.0) > 0.85) {
        float radiusGlitch = hash2(floor(radius * 10.0) + glitchTime);
        if(radiusGlitch > 0.75) {
            kaleidoUV *= 1.0 + (radiusGlitch - 0.5) * 0.3 * energy;
        }
    }

    // Pixelation
    if(energy > 0.6 && glitchFrame > 0.8) {
        float pixelSize = 8.0 + energy * 12.0;
        kaleidoUV = floor(kaleidoUV * pixelSize) / pixelSize;
    }

    // Render with chromatic aberration
    float aberration = 0.008 + brightness * 0.015;
    vec3 c = vec3(0.0);
    c.r = render(kaleidoUV + vec2(aberration, 0.0)).r;
    c.g = render(kaleidoUV).g;
    c.b = render(kaleidoUV - vec2(aberration, 0.0)).b;

    // RGB shifting
    if(hash2(glitchTime + 2.0) > 0.75 && energy > 0.4) {
        float shift = energy * 0.04;
        c.rb = c.br;
    }

    // Bit reduction
    float colorDepth = 160.0 - energy * 80.0;
    c = floor(c * colorDepth) / colorDepth;

    // Random corruption
    if(hash(kaleidoUV * 60.0 + glitchTime) > 0.96 - modulation * 0.08) {
        c = vec3(hash(kaleidoUV + glitchTime), hash(kaleidoUV + glitchTime + 1.0), hash(kaleidoUV + glitchTime + 2.0));
    }

    // Circular scanlines
    float circularScan = sin(radius * 20.0 - pc.time * 8.0);
    c *= 1.0 - step(0.85, circularScan) * energy * 0.15;

    // Vignette
    float vignette = 1.0 - pow(radius * 0.6, 2.0);
    c *= st(vignette) * 0.8 + 0.2;

    // Contrast
    float contrast = 1.2 + brightness * 0.3;
    c = (c - 0.5) * contrast + 0.5;

    // Strobe on high energy
    if(energy > 0.8 && hash2(floor(pc.time * 25.0)) > 0.8) {
        c *= 1.5;
    }

    outColor = vec4(st(c), 1.0);
}
