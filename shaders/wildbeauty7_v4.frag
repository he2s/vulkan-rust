#version 450

// WILDBEAUTY7 - Rich 3D + Subtle Glitch
// Variation 4: Blue Shift - Cool color palette

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
#define MAX_STEPS 12
#define MAX_DIST 10.0
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

mat3 rotY(float a) {
    float c = cos(a), s = sin(a);
    return mat3(c,0,s, 0,1,0, -s,0,c);
}

mat3 rotZ(float a) {
    float c = cos(a), s = sin(a);
    return mat3(c,-s,0, s,c,0, 0,0,1);
}

// Rich color palette
vec3 palette(float x) {
    float energy = st(pc.note_velocity);
    float brightness = st(pc.cc74);

    vec3 a = vec3(0.7, 0.3, 0.8) * (1.0 + sin(pc.time * 2.0) * 0.3);
    vec3 b = vec3(1.0 + brightness * 1.5, 0.9 + energy, 1.2);
    vec3 c = vec3(1.8, 1.3, 2.5) + vec3(pc.osc_ch1 * 3.0, pc.osc_ch2 * 4.0, 0.0);
    vec3 d = vec3(pc.time * 0.1, pc.pitch_bend, energy * 2.0);

    return a + b * cos(TAU * (c * x + d));
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

    float t = pc.time * (1.5 + energy * 2.0);

    vec3 op = p;
    p += sin(p * 2.0 + t * 0.5) * 0.08 * energy;

    // Main torus ring
    vec3 p1 = p;
    p1.xy *= rot(t * 0.6);
    p1.yz *= rot(t * 0.4);
    float torus1 = sdf_torus(p1, vec2(1.3, 0.18 + energy * 0.12));

    // Secondary intersecting torus
    vec3 p2 = p;
    p2.xy *= rot(t * 0.5 + PI * 0.5);
    p2.xz *= rot(t * 0.3);
    float torus2 = sdf_torus(p2, vec2(1.1, 0.15));

    // Third torus
    vec3 p3 = p;
    p3.yz *= rot(t * 0.7 + PI * 0.33);
    p3.xz *= rot(t * 0.35);
    float torus3 = sdf_torus(p3, vec2(0.9, 0.12));

    // Orbiting spheres
    vec3 ps1 = p + vec3(sin(t * 1.5) * 2.0, cos(t * 1.2) * 1.5, sin(t * 0.9) * 1.8);
    float sphere1 = sdf_sphere(ps1, 0.25 + energy * 0.15);

    vec3 ps2 = p + vec3(cos(t * 1.8) * 2.2, sin(t * 1.5) * 1.8, cos(t) * 2.0);
    float sphere2 = sdf_sphere(ps2, 0.2 + modulation * 0.1);

    // Central box
    vec3 pb = p;
    pb *= rotY(t * 2.0) * rotZ(t * 1.5);
    float box = sdf_box(pb, vec3(0.15 + modulation * 0.1));

    // Combine with smooth blending
    float smoothness = 0.15 + modulation * 0.4;
    float scene = smin(torus1, torus2, smoothness);
    scene = smin(scene, torus3, smoothness * 0.9);
    scene = smin(scene, sphere1, smoothness * 1.2);
    scene = smin(scene, sphere2, smoothness * 1.1);
    scene = smin(scene, box, smoothness * 0.7);

    float materialID = 1.0 + sin(length(op) * 4.0 + t) * 0.5;

    return vec2(scene, materialID);
}

vec2 trace(vec3 ro, vec3 rd) {
    vec3 p = ro;
    float td = 0.0;
    glow = vec3(0.0);

    float glowStrength = 0.08 + pc.note_velocity * 0.12;

    for(int i = 0; i < MAX_STEPS; i++) {
        if(td >= MAX_DIST) break;

        vec2 di = sdf(p);

        if(di.x < EPSILON) {
            return vec2(td, di.y);
        }

        p += di.x * rd;

        float glowFactor = (1.0 - st(di.x / 0.6)) * glowStrength;
        vec3 glowColor = palette(length(p) * 0.5 + pc.time * 0.3) * glowFactor;
        glowColor *= vec3(1.2 + pc.cc74 * 1.5, 1.0 + sin(pc.time * 4.0), 1.1 + pc.cc1);

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
    float energy = st(pc.note_velocity);

    float camDist = 3.5 + sin(pc.time * 0.8) * 0.8;
    vec3 ro = vec3(sin(pc.time * 0.4) * 0.5, cos(pc.time * 0.3) * 0.3, -camDist);
    ro.xz *= rot(pc.osc_ch1 * PI);

    vec3 rd = normalize(vec3(uv, 1.4));

    vec2 tdi = trace(ro, rd);

    if(tdi.x > 0.0) {
        vec3 p = ro + rd * tdi.x;
        vec3 n = get_normal(p);

        vec3 cd = normalize(ro - p);
        vec3 reflection = reflect(rd, n);

        // Multi-layer iridescence
        float iridValue1 = dot(n, cd) * 2.5;
        float iridValue2 = dot(n, reflection) * 1.5 + length(p) * 0.4;

        vec3 color1 = palette(iridValue1 + pc.time * 0.4);
        vec3 color2 = palette(iridValue2 + pc.time * 0.2);
        vec3 color = mix(color1, color2, 0.5);

        // Lighting
        vec3 lightDir = normalize(vec3(1.0, 1.0, -0.5));
        float diff = st(dot(n, lightDir) * 0.6 + 0.5);

        float spec = pow(st(dot(reflection, lightDir)), 12.0);

        color *= diff * 0.8 + 0.3;
        color += vec3(spec * 2.0, spec * 1.5, spec * 2.5);
        color += glow * 1.8;

        // Material variation
        if(tdi.y > 1.5) {
            color *= vec3(1.3, 0.9, 1.5);
        }

        return color;
    }

    vec3 bg = glow * 2.5;
    bg += palette(length(uv) * 0.5 + pc.time * 0.1) * 0.08;

    return bg;
}

void main() {
    vec2 resolution = vec2(float(pc.render_w), float(pc.render_h));
    vec2 uv = (fragUV - 0.5) * 2.0;
    uv.x *= resolution.x / resolution.y;

    float energy = st(pc.note_velocity);
    float brightness = st(pc.cc74);

    vec2 uvOriginal = uv;

    // SUBTLE GLITCH EFFECTS
    float glitchTime = floor(pc.time * 4.0) / 4.0;
    float glitchFrame = hash2(glitchTime);

    // Occasional displacement
    if(glitchFrame > 0.92 && energy > 0.6) {
        float displace = hash(vec2(floor(uv.y * 10.0), glitchTime));
        if(displace > 0.8) {
            uv.x += (displace - 0.5) * 0.08 * energy;
        }
    }

    // Light chromatic aberration
    float aberration = 0.008 + brightness * 0.012;
    vec3 c = vec3(0.0);
    c.r = render(uv + vec2(aberration, 0.0)).r;
    c.g = render(uv).g;
    c.b = render(uv - vec2(aberration, 0.0)).b;

    // Subtle bit reduction
    float colorDepth = 200.0 - energy * 50.0;
    c = floor(c * colorDepth) / colorDepth;

    // Rare color glitches
    if(hash(uv * 60.0 + glitchTime) > 0.98) {
        c.rg = c.gr;
    }

    // Vignette
    float vignette = 1.0 - pow(length(uvOriginal) * 0.55, 2.0);
    c *= st(vignette) * 0.85 + 0.15;

    // Color grading
    c = pow(c, vec3(0.9, 1.05, 0.95));
    float contrast = 1.15 + brightness * 0.25;
    c = (c - 0.5) * contrast + 0.5;

    outColor = vec4(st(c), 1.0);
}
