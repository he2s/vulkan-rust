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

// Fractal Crystal Kaleidoscope - Combines ray marching, fractals, and kaleidoscope effects

#define PI 3.141592653589793238
#define TAU (2.0 * PI)
#define EPSILON 0.01
#define MAX_STEPS 80
#define sat(x) clamp(x, 0.0, 1.0)

mat2 rot(float a) {
    float c = cos(a), s = sin(a);
    return mat2(c, -s, s, c);
}

vec3 palette(float t) {
    vec3 a = vec3(0.5, 0.5, 0.5);
    vec3 b = vec3(0.5, 0.5, 0.5);
    vec3 c = vec3(1.0, 1.0, 1.0);
    vec3 d = vec3(0.0 + pc.osc_ch1 * 0.3, 0.33 + pc.osc_ch2 * 0.3, 0.67);
    return a + b * cos(TAU * (c * t + d));
}

// Fractal SDF combining mandelbulb-like iteration with crystal geometry
float fractalCrystal(vec3 p) {
    vec3 z = p;
    float dr = 1.0;
    float r = 0.0;

    float power = 4.0 + pc.note_velocity * 4.0;
    int iterations = int(4.0 + pc.cc74 * 6.0);

    for(int i = 0; i < 10; i++) {
        if(i >= iterations) break;

        r = length(z);
        if(r > 2.0) break;

        // Convert to spherical coords
        float theta = acos(z.z / r);
        float phi = atan(z.y, z.x);
        dr = pow(r, power - 1.0) * power * dr + 1.0;

        // Scale and rotate
        float zr = pow(r, power);
        theta = theta * power + pc.time * 0.5;
        phi = phi * power + pc.pitch_bend * PI;

        // Convert back
        z = zr * vec3(sin(theta) * cos(phi), sin(phi) * sin(theta), cos(theta));
        z += p;
    }

    return 0.5 * log(r) * r / dr;
}

vec3 glow = vec3(0.0);

vec2 map(vec3 p) {
    float d = fractalCrystal(p);
    return vec2(d, 1.0);
}

vec2 trace(vec3 ro, vec3 rd) {
    float t = 0.0;

    for(int i = 0; i < MAX_STEPS; i++) {
        vec3 p = ro + rd * t;
        vec2 h = map(p);

        if(h.x < EPSILON || t > 20.0) break;

        // Accumulate glow
        float glowStr = 0.05 + pc.note_velocity * 0.1;
        glow += palette(length(p) * 0.3 + pc.time * 0.2) * glowStr / (1.0 + h.x * h.x * 10.0);

        t += h.x * 0.5;
    }

    return vec2(t, t < 20.0 ? 1.0 : -1.0);
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

    // Kaleidoscope effect
    float segments = 6.0 + floor(pc.cc1 * 6.0);
    float angle = atan(uv.y, uv.x);
    angle = mod(angle, TAU / segments) * segments;
    uv = vec2(cos(angle), sin(angle)) * length(uv);

    // Camera setup
    vec3 ro = vec3(0.0, 0.0, -3.0);
    vec3 rd = normalize(vec3(uv, 1.0));

    // Mouse/OSC control
    if(pc.mouse_pressed > 0u) {
        float mx = (float(pc.mouse_x) / float(pc.render_w) - 0.5) * TAU;
        float my = (float(pc.mouse_y) / float(pc.render_h) - 0.5) * PI;
        ro.xz = rot(mx) * ro.xz;
        ro.yz = rot(my) * ro.yz;
    } else {
        ro.xz = rot(pc.time * 0.2 + pc.osc_ch1 * TAU) * ro.xz;
        ro.yz = rot(pc.time * 0.15 + pc.osc_ch2 * PI) * ro.yz;
    }

    vec2 hit = trace(ro, rd);
    vec3 col = vec3(0.0);

    if(hit.y > 0.0) {
        vec3 p = ro + rd * hit.x;
        vec3 n = getNormal(p);

        // Iridescent lighting
        float fresnel = pow(1.0 - sat(dot(-rd, n)), 3.0);
        vec3 irid = palette(fresnel + length(p) * 0.2 + pc.time * 0.1);

        float diff = sat(dot(n, normalize(vec3(1, 1, -1)))) * 0.5 + 0.5;
        col = irid * diff + glow;
    } else {
        col = glow;
    }

    // Energy flash
    if(pc.note_velocity > 0.6) {
        col += vec3(0.5, 0.3, 0.8) * (pc.note_velocity - 0.6) * 2.0;
    }

    // Vignette
    float vig = 1.0 - length((fragUV - 0.5) * 1.5);
    col *= vig;

    outColor = vec4(sat(col), 1.0);
}
