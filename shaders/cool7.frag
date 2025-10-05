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

// Infinity Mirror - Combines recursive reflection with crystal geometry

#define PI 3.141592653589793238
#define TAU (2.0 * PI)
#define EPSILON 0.01
#define MAX_STEPS 50
#define sat(x) clamp(x, 0.0, 1.0)

mat2 rot(float a) {
    float c = cos(a), s = sin(a);
    return mat2(c, -s, s, c);
}

vec3 palette(float t) {
    vec3 a = vec3(0.5, 0.5, 0.5);
    vec3 b = vec3(0.5, 0.5, 0.5);
    vec3 c = vec3(1.0, 1.0, 1.0);
    vec3 d = vec3(0.3 + pc.osc_ch1 * 0.2, 0.2 + pc.osc_ch2 * 0.2, 0.5 + pc.pitch_bend * 0.1);
    return a + b * cos(TAU * (c * t + d));
}

// Mod operation that creates repeating space
vec3 modRepeat(vec3 p, float size) {
    return mod(p + size * 0.5, size) - size * 0.5;
}

// Crystal SDF with repetition
float crystalSDF(vec3 p) {
    float energy = sat(pc.note_velocity);
    float modulation = sat(pc.cc1);

    // Repeat space for infinite mirrors
    float repeatSize = 2.0 + modulation * 2.0;
    vec3 pr = modRepeat(p, repeatSize);

    // Rotating crystal
    pr.xy = rot(pc.time * 0.5 + energy * PI) * pr.xy;
    pr.yz = rot(pc.time * 0.3) * pr.yz;

    // Octahedron
    pr = abs(pr);
    float octa = (pr.x + pr.y + pr.z - (0.5 + energy * 0.3)) * 0.57735027;

    // Add inner detail
    vec3 pr2 = pr;
    pr2.xy = rot(pc.time * -0.7) * pr2.xy;
    float inner = length(pr2) - (0.3 + vertexEnergy * 0.2);

    return min(octa, inner);
}

vec3 glow = vec3(0.0);
int hitCount = 0;

vec2 trace(vec3 ro, vec3 rd) {
    float t = 0.0;
    float energy = sat(pc.note_velocity);
    float glowStr = 0.05 + energy * 0.1;

    for(int i = 0; i < MAX_STEPS; i++) {
        vec3 p = ro + rd * t;
        float d = crystalSDF(p);

        if(d < EPSILON) {
            hitCount++;
            return vec2(t, float(hitCount));
        }

        // Accumulate glow
        vec3 glowCol = palette(length(p) * 0.2 + pc.time * 0.2);
        glow += glowCol * glowStr / (1.0 + d * d * 50.0);

        t += d * 0.6;
        if(t > 20.0) break;
    }

    return vec2(-1.0, -1.0);
}

vec3 getNormal(vec3 p) {
    vec2 e = vec2(EPSILON, 0.0);
    return normalize(vec3(
        crystalSDF(p + e.xyy) - crystalSDF(p - e.xyy),
        crystalSDF(p + e.yxy) - crystalSDF(p - e.yxy),
        crystalSDF(p + e.yyx) - crystalSDF(p - e.yyx)
    ));
}

// Recursive reflection rendering
vec3 render(vec2 uv) {
    float energy = sat(pc.note_velocity);
    float modulation = sat(pc.cc1);
    float brightness = sat(pc.cc74);

    // Camera
    vec3 ro = vec3(0.0, 0.0, -5.0 + modulation);
    vec3 rd = normalize(vec3(uv, 1.0));

    // Mouse/OSC control
    if(pc.mouse_pressed > 0u) {
        float mx = (float(pc.mouse_x) / float(pc.render_w) - 0.5) * TAU;
        float my = (float(pc.mouse_y) / float(pc.render_h) - 0.5) * PI;
        ro.xz = rot(mx) * ro.xz;
        ro.yz = rot(my) * ro.yz;
    } else {
        ro.xz = rot(pc.time * 0.15 + pc.osc_ch1 * PI) * ro.xz;
        ro.yz = rot(sin(pc.time * 0.1) * 0.3 + pc.osc_ch2 * PI * 0.5) * ro.yz;
    }

    vec3 col = vec3(0.0);
    vec3 reflectionMask = vec3(1.0);

    // Multiple reflection bounces
    int maxBounces = int(2.0 + brightness * 3.0);

    for(int bounce = 0; bounce < 5; bounce++) {
        if(bounce >= maxBounces) break;

        hitCount = 0;
        glow = vec3(0.0);

        vec2 hit = trace(ro, rd);

        if(hit.x > 0.0) {
            vec3 p = ro + rd * hit.x;
            vec3 n = getNormal(p);

            // Surface color based on depth/position
            float depth = float(bounce) / float(maxBounces);
            vec3 surfaceCol = palette(depth + length(p) * 0.1 + pc.time * 0.15);

            // Fresnel
            float fresnel = pow(1.0 - sat(dot(-rd, n)), 2.0);

            // Add this bounce's contribution
            col += (surfaceCol * (1.0 - fresnel) + glow) * reflectionMask;

            // Reduce reflection for next bounce
            reflectionMask *= fresnel * (0.6 + energy * 0.3);

            // Set up next bounce
            ro = p + n * EPSILON * 2.0;
            rd = reflect(rd, n);

        } else {
            // Hit background
            col += glow * reflectionMask;
            break;
        }
    }

    return col;
}

void main() {
    vec2 resolution = vec2(float(pc.render_w), float(pc.render_h));
    vec2 uv = (fragUV - 0.5) * 2.0;
    uv.x *= resolution.x / resolution.y;

    float energy = sat(pc.note_velocity);

    // Add slight barrel distortion
    float dist = length(uv);
    uv *= 1.0 + dist * dist * (0.1 + energy * 0.1);

    // Get color with reflections
    vec3 col = render(uv);

    // Energy burst
    if(energy > 0.6) {
        float burst = pow(energy - 0.6, 2.0) * 6.0;
        burst *= (1.0 + sin(pc.time * 25.0) * 0.5);
        col += palette(pc.time * 0.4 + length(uv)) * burst;
    }

    // Note count affects color rotation
    if(pc.note_count > 0u) {
        float noteShift = float(pc.note_count % 6u) / 6.0;
        col = mix(col, col.gbr, noteShift * 0.5);
    }

    // Contrast
    col = (col - 0.5) * 1.2 + 0.5;

    // Saturation
    float gray = dot(col, vec3(0.299, 0.587, 0.114));
    col = mix(vec3(gray), col, 1.3);

    // Vignette
    float vig = 1.0 - pow(length((fragUV - 0.5) * 1.1), 1.8);
    col *= vig;

    // Bloom
    vec3 bloom = max(col - vec3(1.0), 0.0) * 1.2;
    col += bloom;

    outColor = vec4(sat(col), 1.0);
}
