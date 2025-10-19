#version 450

// Push constants from application
// Variation 8: Hyper - Extreme intense mode
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
#define EPSILON 0.01
#define MAX_STEPS 128
#define MAX_DIST 120.0

// Helper macros
#define min2(a, b) ((a.x < b.x) ? a : b)
#define pos(x) (x * 0.5 + 0.5)
#define sat(x) clamp(x, 0.0, 1.0)

// Rotation matrix
mat2 rot(float a) {
    float c = cos(a);
    float s = sin(a);
    return mat2(c, -s, s, c);
}

// Audio-reactive color palette
vec3 palette(float x) {
    // Shift palette based on note count
    x += float(pc.note_count % 8u) * 0.125;

    // Audio-reactive palette parameters
    vec3 a = vec3(0.5, 0.5, 0.0); // Base color (fire)
    vec3 b = vec3(0.5 + pc.cc74 * 0.3); // Amplitude
    vec3 c = vec3(0.1, 0.5, 0.0) + vec3(pc.osc_ch1, pc.osc_ch2, 0.0) * 0.3;
    vec3 d = vec3(0.0, pc.pitch_bend * 0.3, pc.note_velocity * 0.2);

    return a + b * cos(TAU * (c * x + d));
}

// Smooth union with audio-reactive smoothness
float smooth_union(float a, float b, float k) {
    float h = sat(pos((b - a) / k));
    return mix(b, a, h) - k * h * (1.0 - h);
}

// Torus SDF
float sdf_torus(vec3 p, vec2 t) {
    vec2 q = vec2(length(p.xz) - t.x, p.y);
    return length(q) - t.y;
}

// Global glow accumulator
vec3 glow;

// Main SDF scene
vec2 sdf(vec3 p) {
    vec2 di = vec2(120.0, -1.0);

    // Audio-reactive parameters
    float energy = clamp(pc.note_velocity, 0.0, 1.0);
    float modulation = clamp(pc.cc1, 0.0, 1.0);
    float brightness = clamp(pc.cc74, 0.0, 1.0);

    // Audio-reactive smoothness
    float smoothness = 0.2 + modulation * 0.4;

    // Time with audio modulation
    float t = pc.time * (0.5 + energy * 1.0);

    // Audio-reactive torus size
    float ringRadius = 1.0 + vertexEnergy * 0.2;
    float ringThickness = 0.15 + energy * 0.05;
    vec2 torusParams = vec2(ringRadius, ringThickness);

    // First ring with audio rotation
    vec3 p1 = p;
    p1.yz *= rot(t + pc.pitch_bend);
    p1.xy *= rot(t * 1.2);
    p1.xz *= rot(t * PI / 2.0 + PI / 3.0);
    float ring_1 = sdf_torus(p1, torusParams);

    // Second ring
    vec3 p2 = p;
    p2.yz *= rot(t + pc.pitch_bend);
    p2.xy *= rot(t * 1.2);
    p2.xz *= rot(t * PI / 2.0 + PI / 3.0);
    p2.yz *= rot(t * PI / 2.0 + PI / 5.0 + pc.osc_ch1 * PI);
    float ring_2 = sdf_torus(p2, torusParams * (1.0 + modulation * 0.2));

    // Third ring
    vec3 p3 = p;
    p3.yz *= rot(t + pc.pitch_bend);
    p3.xy *= rot(t * 1.2);
    p3.xz *= rot(t * PI / 2.0 + PI / 3.0);
    p3.xy *= rot(t * PI / 2.0 - PI / 7.0 + pc.osc_ch2 * PI);
    float ring_3 = sdf_torus(p3, torusParams * (1.0 - modulation * 0.1));

    // Combine rings with audio-reactive smoothness
    float combined = smooth_union(ring_1, smooth_union(ring_2, ring_3, smoothness), smoothness);

    di = min2(di, vec2(combined, 1.0));

    return di;
}

// Ray marching
vec2 trace(vec3 ro, vec3 rd) {
    vec3 p = ro;
    vec2 di;
    float td = 0.0;

    // Audio-reactive glow intensity
    float glowStrength = 0.05 + pc.note_velocity * 0.03;

    glow = vec3(0.0);

    for(int i = 0; i < MAX_STEPS; i++) {
        if(td >= MAX_DIST) break;

        di = sdf(p);

        if(di.x < EPSILON) {
            return vec2(td, di.y);
        }

        p += di.x * rd;

        // Accumulate glow with audio modulation
        float glowFactor = (1.0 - sat(di.x / 0.4)) * glowStrength;
        vec3 glowColor = pos(normalize(p)) * glowFactor;

        // Tint glow based on audio
        glowColor *= vec3(1.0 + pc.cc74 * 0.5, 1.0, 1.0 + pc.cc1 * 0.5);

        glow += glowColor;
        td = distance(ro, p);
    }

    return vec2(-1.0, -1.0);
}

// Normal calculation
vec3 get_normal(vec3 p) {
    vec2 e = EPSILON * vec2(1.0, -1.0);
    return normalize(
    e.xyy * sdf(p + e.xyy).x +
    e.yxy * sdf(p + e.yxy).x +
    e.yyx * sdf(p + e.yyx).x +
    e.xxx * sdf(p + e.xxx).x
    );
}

// Main rendering function
vec3 render(vec2 uv) {
    // Camera setup with audio influence
    float camDist = 3.0 - pc.cc1 * 0.5;
    vec3 ro = vec3(0.0, 0.0, -camDist);

    // Mouse/OSC camera rotation
    if(pc.mouse_pressed > 0u) {
        float mx = (float(pc.mouse_x) / float(pc.render_w) - 0.5) * TAU;
        float my = (float(pc.mouse_y) / float(pc.render_h) - 0.5) * PI;
        ro.xz *= rot(mx);
        ro.yz *= rot(my);
    }

    vec3 rd = normalize(vec3(uv, 1.0));
    vec3 lo = ro; // Light origin

    vec2 tdi = trace(ro, rd);

    if(tdi.x > 0.0) {
        vec3 p = ro + rd * tdi.x;
        vec3 n = get_normal(p);

        // Iridescence effect with audio modulation
        vec3 cd = normalize(ro - p);
        vec3 ld = normalize(lo - p);
        vec3 reflection = reflect(rd, n);

        // Audio-reactive perturbation
        float perturbStrength = 10.0 + pc.note_velocity * 15.0;
        vec3 perturbation = 0.05 * sin(p * perturbStrength);

        // Calculate iridescence
        float iridValue = dot(n + perturbation, cd) * 4.0 // Extreme;
        vec3 iridescence = palette(iridValue);

        // Specular with audio influence
        float specular = sat(dot(reflection, ld));
        float specIntensity = 0.1 + pc.cc74 * 0.2;
        specular *= specIntensity * pow(pos(sin(specular * 20.0 - 3.0)) + 0.1, 32.0);
specular += specIntensity * pow(sat(dot(reflection, ld)) + 0.3, 8.0);

// Shadow/ambient
float shadow = pow(sat(dot(n, vec3(0.0, 1.0, 0.0)) * 0.5 + 1.2), 3.0);

// Combine lighting
vec3 color = iridescence * shadow + specular + glow;

// Add energy flash
if(pc.note_velocity > 0.7) {
    color += vec3(0.2, 0.3, 0.5) * (pc.note_velocity - 0.7);
}

return color;
}

// Background with glow only
return vec3(0.0) + glow;
}

void main() {
    vec2 resolution = vec2(float(pc.render_w), float(pc.render_h));
    vec2 uv = (fragUV - 0.5) * 4.0 // Extreme;
    uv.x *= resolution.x / resolution.y;

    vec3 c = render(uv);

    // Subtle vignette
    float vignette = 1.0 - length(uv) * 0.3;
    c *= vignette;

    // Output with saturation
    outColor = vec4(sat(c), 1.0);
}