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

// Liquid Metal - Combines metallic reflections with fluid simulation

#define PI 3.141592653589793238
#define TAU (2.0 * PI)
#define EPSILON 0.01
#define MAX_STEPS 64
#define sat(x) clamp(x, 0.0, 1.0)

mat2 rot(float a) {
    float c = cos(a), s = sin(a);
    return mat2(c, -s, s, c);
}

vec3 palette(float t) {
    // Metallic palette
    vec3 a = vec3(0.8, 0.8, 0.9);
    vec3 b = vec3(0.2, 0.2, 0.3);
    vec3 c = vec3(1.0, 1.0, 1.0);
    vec3 d = vec3(0.5 + pc.osc_ch1 * 0.3, 0.2 + pc.osc_ch2 * 0.3, 0.8 + pc.pitch_bend * 0.2);
    return a + b * cos(TAU * (c * t + d));
}

// Fluid displacement field
vec2 fluidField(vec2 p, float t) {
    float energy = sat(pc.note_velocity);
    float modulation = sat(pc.cc1);

    vec2 field = vec2(0.0);

    // Multiple vortices
    for(int i = 0; i < 3; i++) {
        float fi = float(i);
        vec2 center = vec2(
            sin(t * 0.5 + fi * TAU / 3.0) * 1.5,
            cos(t * 0.6 + fi * TAU / 3.0) * 1.2
        );

        vec2 diff = p - center;
        float dist = length(diff);
        float strength = (0.5 + energy * 0.5) / (dist + 0.1);

        // Vortex rotation
        vec2 vortex = vec2(-diff.y, diff.x) * strength;
        field += vortex;

        // Radial push
        field += diff * sin(dist * 5.0 - t * 2.0) * 0.1 * modulation;
    }

    return field * 0.3;
}

// Metallic surface using distance estimation
float metallicSurface(vec3 p) {
    float energy = sat(pc.note_velocity);
    float brightness = sat(pc.cc74);

    // Apply fluid distortion
    vec2 fluidOffset = fluidField(p.xy, pc.time);
    p.xy += fluidOffset;

    // Wavy metallic surface
    float surface = p.z;
    surface += sin(p.x * (3.0 + brightness * 2.0) + pc.time) * 0.3;
    surface += cos(p.y * (2.5 + energy * 2.0) + pc.time * 1.2) * 0.25;

    // Add smaller ripples
    surface += sin(p.x * 12.0 + p.y * 8.0 - pc.time * 3.0) * 0.08 * energy;

    // Vertex energy creates local deformation
    surface += sin(length(p.xy) * 8.0 - pc.time * 2.0) * vertexEnergy * 0.15;

    return surface;
}

vec3 glow = vec3(0.0);

vec2 trace(vec3 ro, vec3 rd) {
    float t = 0.0;
    float glowStr = 0.03 + pc.note_velocity * 0.05;

    for(int i = 0; i < MAX_STEPS; i++) {
        vec3 p = ro + rd * t;
        float d = metallicSurface(p);

        if(abs(d) < EPSILON || t > 10.0) break;

        // Glow accumulation
        glow += palette(length(p.xy) * 0.3 + pc.time * 0.1) * glowStr / (1.0 + abs(d) * abs(d) * 100.0);

        t += abs(d) * 0.8;
    }

    return vec2(t, t < 10.0 ? 1.0 : -1.0);
}

vec3 getNormal(vec3 p) {
    vec2 e = vec2(EPSILON, 0.0);
    return normalize(vec3(
        metallicSurface(p + e.xyy) - metallicSurface(p - e.xyy),
        metallicSurface(p + e.yxy) - metallicSurface(p - e.yxy),
        metallicSurface(p + e.yyx) - metallicSurface(p - e.yyx)
    ));
}

void main() {
    vec2 resolution = vec2(float(pc.render_w), float(pc.render_h));
    vec2 uv = (fragUV - 0.5) * 2.0;
    uv.x *= resolution.x / resolution.y;

    float energy = sat(pc.note_velocity);
    float modulation = sat(pc.cc1);

    // Camera looking down at liquid surface
    vec3 ro = vec3(0.0, 0.0, -2.5 + modulation * 0.5);

    // Mouse/OSC camera control
    if(pc.mouse_pressed > 0u) {
        float mx = (float(pc.mouse_x) / float(pc.render_w) - 0.5) * 4.0;
        float my = (float(pc.mouse_y) / float(pc.render_h) - 0.5) * 4.0;
        ro.xy += vec2(mx, my);
    } else {
        ro.xy += vec2(pc.osc_ch1, pc.osc_ch2) * 2.0;
    }

    vec3 rd = normalize(vec3(uv, 1.5));

    vec2 hit = trace(ro, rd);
    vec3 col = vec3(0.0);

    if(hit.y > 0.0) {
        vec3 p = ro + rd * hit.x;
        vec3 n = getNormal(p);

        // Metallic reflection
        vec3 refl = reflect(rd, n);

        // Environment reflection (procedural)
        vec3 envCol = palette(refl.x * 0.5 + refl.y * 0.3 + pc.time * 0.1);
        envCol += palette(length(refl.xy) * 0.5 + pc.time * 0.15 + 0.5) * 0.5;

        // Fresnel effect
        float fresnel = pow(1.0 - sat(dot(-rd, n)), 3.0);
        vec3 fresnelCol = palette(fresnel + pc.time * 0.2);

        // Metallic specular
        vec3 lightDir = normalize(vec3(1, 1, -1));
        float spec = pow(sat(dot(refl, lightDir)), 16.0);

        // Combine metallic look
        col = envCol * (0.7 + fresnel * 0.3);
        col += fresnelCol * fresnel * 0.4;
        col += vec3(spec) * (0.5 + energy * 0.5);
        col += glow * 0.8;

        // Add colored highlights based on normal
        vec3 normalCol = palette(dot(n, vec3(0.7, 0.3, 0.1)) + pc.time * 0.1);
        col = mix(col, normalCol, 0.2);

    } else {
        // Background gradient
        float bg = length(uv) * 0.5;
        col = palette(bg + pc.time * 0.05) * 0.3 + glow;
    }

    // Energy flash with metallic tint
    if(energy > 0.6) {
        float flash = (energy - 0.6) * 2.5;
        flash *= (1.0 + sin(pc.time * 25.0) * 0.3);
        col += palette(pc.time * 0.3) * flash * vec3(0.8, 0.9, 1.0);
    }

    // Color grading for metallic look
    float brightness = sat(pc.cc74);
    col = mix(col, col * vec3(0.9, 0.95, 1.1), 0.3); // Cool tint
    col *= 0.9 + brightness * 0.6;

    // Contrast
    col = (col - 0.5) * (1.1 + modulation * 0.3) + 0.5;

    // Subtle vignette
    float vig = 1.0 - length((fragUV - 0.5) * 0.8);
    col *= vig;

    outColor = vec4(sat(col), 1.0);
}
