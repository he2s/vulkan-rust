#version 450

// WILDBEAUTY9 - Liquid Glitch (flowing distortions + smooth 3D)

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
#define EPSILON 0.3
#define MAX_STEPS 10
#define st(x) clamp(x, 0.0, 1.0)

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

mat2 rot(float a) {
    float c = cos(a), s = sin(a);
    return mat2(c, -s, s, c);
}

// Smooth noise for flowing effects
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

// Flowing palette
vec3 fluidPalette(float t) {
    float energy = st(pc.note_velocity);
    float brightness = st(pc.cc74);

    vec3 a = vec3(0.5, 0.4, 0.7) * (1.0 + sin(pc.time) * 0.2);
    vec3 b = vec3(0.8, 0.6, 0.7) * (1.0 + brightness * 0.5);
    vec3 c = vec3(1.5, 1.2, 1.8) * (1.0 + energy * 0.3);
    vec3 d = vec3(pc.time * 0.05, pc.osc_ch1 * 0.5, pc.osc_ch2 * 0.5);

    return a + b * cos(TAU * (c * t + d));
}

float sdf_torus(vec3 p, vec2 t) {
    vec2 q = vec2(length(p.xz) - t.x, p.y);
    return length(q) - t.y;
}

float sdf_sphere(vec3 p, float r) {
    return length(p) - r;
}

float smin(float a, float b, float k) {
    float h = st(0.5 + 0.5 * (b - a) / k);
    return mix(b, a, h) - k * h * (1.0 - h);
}

vec3 glow;

vec2 sdf(vec3 p) {
    float energy = st(pc.note_velocity);
    float modulation = st(pc.cc1);

    float t = pc.time * 1.0;

    // Liquid distortion
    p += vec3(
        sin(p.y * 2.0 + t) * 0.1,
        cos(p.z * 2.0 + t * 1.3) * 0.1,
        sin(p.x * 2.0 + t * 0.8) * 0.1
    ) * (0.5 + energy * 0.5);

    // Flowing torus
    vec3 p1 = p;
    p1.xy *= rot(t * 0.4);
    float torus = sdf_torus(p1, vec2(1.2, 0.2 + energy * 0.15));

    // Pulsing sphere
    float pulseRadius = 0.5 + sin(t * 3.0) * 0.2 + energy * 0.3;
    float sphere = sdf_sphere(p, pulseRadius);

    // Combine smoothly
    float scene = smin(torus, sphere, 0.3 + modulation * 0.5);

    return vec2(scene, 1.0);
}

vec2 trace(vec3 ro, vec3 rd) {
    vec3 p = ro;
    float td = 0.0;
    glow = vec3(0.0);

    float glowStrength = 0.07 + pc.note_velocity * 0.12;

    for(int i = 0; i < MAX_STEPS; i++) {
        if(td >= 8.0) break;

        vec2 di = sdf(p);

        if(di.x < EPSILON) {
            return vec2(td, di.y);
        }

        p += di.x * rd;

        float glowFactor = (1.0 - st(di.x / 0.6)) * glowStrength;
        vec3 glowColor = fluidPalette(length(p) * 0.4 + pc.time * 0.2) * glowFactor;

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
    float camDist = 3.5;
    vec3 ro = vec3(sin(pc.time * 0.3) * 0.5, 0.0, -camDist);
    ro.xz *= rot(pc.osc_ch1 * PI * 0.3);

    vec3 rd = normalize(vec3(uv, 1.5));

    vec2 tdi = trace(ro, rd);

    if(tdi.x > 0.0) {
        vec3 p = ro + rd * tdi.x;
        vec3 n = get_normal(p);

        float iridValue = dot(n, normalize(ro - p)) * 2.0 + length(p) * 0.2;

        vec3 color = fluidPalette(iridValue + pc.time * 0.3);

        // Soft lighting
        vec3 lightDir = normalize(vec3(1.0, 1.0, -1.0));
        float diff = st(dot(n, lightDir) * 0.5 + 0.6);

        color *= diff;
        color += glow * 2.0;

        return color;
    }

    vec3 bg = glow * 2.5;
    bg += fluidPalette(length(uv) * 0.5 + pc.time * 0.1) * 0.1;

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

    // LIQUID DISTORTIONS

    // Flowing waves
    float wave1 = sin(uv.x * 5.0 + pc.time * 2.0) * cos(uv.y * 4.0 + pc.time * 1.5);
    float wave2 = noise(uv * 3.0 + vec2(pc.time * 0.5, pc.time * 0.7));

    uv += vec2(wave1, wave2) * 0.04 * (0.5 + energy * 0.5);

    // Ripple distortion
    float ripple = length(uv);
    float ripplePhase = ripple * 8.0 - pc.time * 4.0;
    uv += normalize(uv) * sin(ripplePhase) * 0.02 * modulation;

    // Turbulence
    vec2 turbulence = vec2(
        noise(uv * 4.0 + pc.time * 0.3),
        noise(uv * 4.0 + pc.time * 0.3 + 100.0)
    ) - 0.5;
    uv += turbulence * 0.05 * brightness;

    // Render with subtle chromatic aberration
    float aberration = 0.01 + energy * 0.015;
    vec3 c = vec3(0.0);
    c.r = render(uv + vec2(aberration, 0.0)).r;
    c.g = render(uv).g;
    c.b = render(uv - vec2(aberration, 0.0)).b;

    // Flowing color shifts
    vec3 shift = vec3(
        sin(pc.time * 2.0 + uv.x * 3.0),
        cos(pc.time * 1.7 + uv.y * 3.0),
        sin(pc.time * 2.3)
    ) * 0.05 * modulation;
    c += shift;

    // Smooth bit reduction
    float colorDepth = 180.0 - energy * 60.0;
    c = floor(c * colorDepth) / colorDepth;

    // Subtle scanlines
    float scanline = sin(uv.y * resolution.y * 0.5 + pc.time * 3.0);
    c *= 1.0 - step(0.9, scanline) * 0.08;

    // Vignette with flow
    float vignette = 1.0 - pow(length(uvOriginal) * 0.6, 2.0);
    vignette += sin(pc.time * 2.0 + length(uvOriginal) * 5.0) * 0.05;
    c *= st(vignette) * 0.85 + 0.15;

    // Color grading
    c = pow(c, vec3(0.95, 1.0, 1.05));

    outColor = vec4(st(c), 1.0);
}
