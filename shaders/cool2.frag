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

// Dancing Geometry - Combines multiple animated SDFs with smooth blending

#define PI 3.141592653589793238
#define TAU (2.0 * PI)
#define EPSILON 0.01
#define MAX_STEPS 100
#define sat(x) clamp(x, 0.0, 1.0)

mat2 rot(float a) {
    float c = cos(a), s = sin(a);
    return mat2(c, -s, s, c);
}

mat3 rotX(float a) {
    float c = cos(a), s = sin(a);
    return mat3(1,0,0, 0,c,-s, 0,s,c);
}

mat3 rotY(float a) {
    float c = cos(a), s = sin(a);
    return mat3(c,0,s, 0,1,0, -s,0,c);
}

mat3 rotZ(float a) {
    float c = cos(a), s = sin(a);
    return mat3(c,-s,0, s,c,0, 0,0,1);
}

vec3 palette(float t) {
    t += float(pc.note_count % 12u) * 0.08;
    vec3 a = vec3(0.5, 0.5, 0.5);
    vec3 b = vec3(0.5, 0.5, 0.5);
    vec3 c = vec3(1.0, 1.0, 0.5);
    vec3 d = vec3(0.8, 0.9, 0.3);
    d += vec3(pc.osc_ch1, pc.osc_ch2, pc.pitch_bend) * 0.3;
    return a + b * cos(TAU * (c * t + d));
}

float smoothUnion(float a, float b, float k) {
    float h = sat(0.5 + 0.5 * (b - a) / k);
    return mix(b, a, h) - k * h * (1.0 - h);
}

float sdfSphere(vec3 p, float r) {
    return length(p) - r;
}

float sdfBox(vec3 p, vec3 b) {
    vec3 q = abs(p) - b;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}

float sdfTorus(vec3 p, vec2 t) {
    vec2 q = vec2(length(p.xz) - t.x, p.y);
    return length(q) - t.y;
}

float sdfOctahedron(vec3 p, float s) {
    p = abs(p);
    return (p.x + p.y + p.z - s) * 0.57735027;
}

vec3 glow = vec3(0.0);

vec2 map(vec3 p) {
    float energy = sat(pc.note_velocity);
    float modulation = sat(pc.cc1);
    float brightness = sat(pc.cc74);
    float t = pc.time;

    // Dancing spheres
    vec3 p1 = p + vec3(sin(t * 1.2) * 1.5, cos(t * 0.8) * 1.2, sin(t * 0.6) * 1.0);
    float sphere1 = sdfSphere(p1, 0.5 + energy * 0.3);

    vec3 p2 = p + vec3(cos(t * 0.9) * 1.3, sin(t * 1.1) * 1.5, cos(t * 0.7) * 1.2);
    float sphere2 = sdfSphere(p2, 0.45 + modulation * 0.2);

    // Rotating boxes
    vec3 p3 = p + vec3(sin(t * 0.5 + PI) * 1.8, cos(t * 0.6 + PI) * 1.3, sin(t * 0.4) * 1.5);
    p3 = rotX(t * 0.7) * rotY(t * 0.5) * rotZ(t * 0.9) * p3;
    float box1 = sdfBox(p3, vec3(0.3 + brightness * 0.1));

    // Torus ring
    vec3 p4 = p;
    p4 = rotX(t * 0.3 + pc.pitch_bend) * rotY(t * 0.4) * p4;
    float torus1 = sdfTorus(p4, vec2(1.2 + vertexEnergy * 0.3, 0.2 + energy * 0.1));

    // Octahedrons
    vec3 p5 = p + vec3(cos(t * 0.8 + TAU/3.0) * 2.0, sin(t * 0.7) * 1.5, cos(t * 0.6 + PI/2.0) * 1.8);
    p5 = rotY(t * 0.9) * rotZ(t * 0.6) * p5;
    float octa1 = sdfOctahedron(p5, 0.6 + modulation * 0.2);

    vec3 p6 = p + vec3(sin(t * 0.7 - TAU/3.0) * 1.9, cos(t * 0.9) * 1.6, sin(t * 0.5 + PI) * 1.7);
    p6 = rotX(t * 0.5) * rotZ(t * 0.8) * p6;
    float octa2 = sdfOctahedron(p6, 0.55 + brightness * 0.15);

    // Smooth blend everything
    float smoothness = 0.3 + modulation * 0.4;
    float d = smoothUnion(sphere1, sphere2, smoothness);
    d = smoothUnion(d, box1, smoothness * 0.8);
    d = smoothUnion(d, torus1, smoothness * 1.2);
    d = smoothUnion(d, octa1, smoothness);
    d = smoothUnion(d, octa2, smoothness);

    return vec2(d, 1.0);
}

vec2 trace(vec3 ro, vec3 rd) {
    float t = 0.0;
    float glowStr = 0.08 + pc.note_velocity * 0.12;

    for(int i = 0; i < MAX_STEPS; i++) {
        vec3 p = ro + rd * t;
        vec2 h = map(p);

        if(h.x < EPSILON) return vec2(t, h.y);

        // Accumulate glow
        glow += palette(length(p) * 0.2 + pc.time * 0.15) * glowStr / (1.0 + h.x * h.x * 20.0);

        t += h.x * 0.7;
        if(t > 30.0) break;
    }

    return vec2(-1.0, -1.0);
}

vec3 getNormal(vec3 p) {
    vec2 e = vec2(EPSILON, 0.0);
    return normalize(vec3(
        map(p + e.xyy).x - map(p - e.xyy).x,
        map(p + e.yxy).x - map(p - e.yxy).x,
        map(p + e.yyx).x - map(p - e.yyx).x
    ));
}

void main() {
    vec2 resolution = vec2(float(pc.render_w), float(pc.render_h));
    vec2 uv = (fragUV - 0.5) * 2.0;
    uv.x *= resolution.x / resolution.y;

    // Camera setup with orbital motion
    float camDist = 5.0 - pc.cc1 * 1.5;
    vec3 ro = vec3(0.0, 0.0, -camDist);

    // Automatic camera orbit
    ro = rotY(pc.time * 0.2 + pc.osc_ch1 * PI) * ro;
    ro = rotX(sin(pc.time * 0.15) * 0.5 + pc.osc_ch2 * PI * 0.5) * ro;

    // Mouse override
    if(pc.mouse_pressed > 0u) {
        float mx = (float(pc.mouse_x) / float(pc.render_w) - 0.5) * TAU;
        float my = (float(pc.mouse_y) / float(pc.render_h) - 0.5) * PI;
        ro = rotY(mx) * ro;
        ro = rotX(my) * ro;
    }

    vec3 rd = normalize(vec3(uv, 1.0));
    vec2 hit = trace(ro, rd);

    vec3 col = vec3(0.0);

    if(hit.x > 0.0) {
        vec3 p = ro + rd * hit.x;
        vec3 n = getNormal(p);
        vec3 lightDir = normalize(vec3(1, 1, -1));

        // Rainbow iridescence
        float fresnel = pow(1.0 - sat(dot(-rd, n)), 2.0);
        vec3 irid = palette(fresnel + length(p) * 0.3 + pc.time * 0.2);

        // Lighting
        float diff = sat(dot(n, lightDir)) * 0.7 + 0.3;
        float spec = pow(sat(dot(reflect(rd, n), lightDir)), 32.0);

        col = irid * diff + vec3(spec) * 0.5 + glow;
    } else {
        col = glow;
    }

    // Energy pulse
    if(pc.note_velocity > 0.6) {
        float pulse = (pc.note_velocity - 0.6) * 2.5;
        pulse *= (1.0 + sin(pc.time * 30.0) * 0.5);
        col += palette(pc.time * 0.5) * pulse;
    }

    // Vignette
    float vig = 1.0 - length((fragUV - 0.5) * 1.2);
    col *= vig;

    outColor = vec4(sat(col), 1.0);
}
