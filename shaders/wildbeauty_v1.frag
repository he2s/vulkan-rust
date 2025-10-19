#version 450

// Push constants from application
// Variation 1: Faster - 1.5x speed
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

// Soothing Constants
#define PI 3.141592653589793238
#define TAU (2.0 * PI)
#define EPSILON 0.01
#define MAX_STEPS 64
#define MAX_DIST 10.0
#define GENTLENESS_FACTOR 0.3

// Gentle Helper macros
#define min2(a, b) ((a.x < b.x) ? a : b)
#define pos(x) (x * 0.5 + 0.5)
#define st(x) clamp(x, 0.0, 1.0)
#define gentle(x) (sin(x * GENTLENESS_FACTOR) * 0.1)
#define flow(x) (x + gentle(x + pc.time * 0.5) * 0.05)

// Gentle rotation matrix with smooth transitions
mat2 rot(float a) {
    a += gentle(a) * pc.note_velocity * 0.2;
    float c = cos(a);
    float s = sin(a);
    return mat2(c, -s, s, c);
}

// 3D rotation matrices for gentle movement
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

// Soothing peaceful color palette
vec3 palette(float x) {
    // Gentle shift palette for smooth transitions
    x += float(pc.note_count % 16u) * 0.01;
    x = flow(x); // Apply gentle flow

    // Calm Audio-reactive palette parameters
    vec3 a = vec3(0.5, 0.7, 0.9) * (1.0 + sin(pc.time * 0.5) * 0.2); // Gentle base
    vec3 b = vec3(0.6 + pc.cc74 * 0.3, 0.8 + pc.cc1 * 0.2, 0.9 + pc.note_velocity * 0.3);
    vec3 c = vec3(1.0, 1.2, 1.5) + vec3(pc.osc_ch1 * 0.5, pc.osc_ch2 * 0.5, pc.pitch_bend * 0.3);
    vec3 d = vec3(pc.time * 0.02, pc.pitch_bend * 0.1, pc.note_velocity * 0.2);

    // Soft layered colors
    vec3 color1 = a + b * cos(TAU * (c * x + d));
    vec3 color2 = vec3(0.6, 0.8, 0.9) + vec3(0.3) * sin(TAU * (x * 2.0 + pc.time * 0.3));
    vec3 color3 = vec3(0.7, 0.9, 0.8) * cos(x * 3.0 + pc.time * 0.5);

    // Blend all layers gently
    return mix(mix(color1, color2, 0.6), color3, gentle(x) * 0.1 + 0.05);
}

// Additional peaceful palettes
vec3 sky_palette(float x) {
    return 0.5 + 0.4 * cos(TAU * (x + vec3(0.0, 0.33, 0.67)) * 1.0 + pc.time * 0.3);
}

vec3 ocean_palette(float x) {
    vec3 ocean = vec3(0.2, 0.6, 0.9) * pow(x, 1.5) + vec3(0.4, 0.8, 1.0) * pow(1.0-x, 2.0);
    return ocean * (1.0 + gentle(x * 3.0 + pc.time * 0.5) * 0.2);
}

// Peaceful union operations
float smooth_union(float a, float b, float k) {
    k *= 1.0 + gentle(a + b + pc.time) * 0.02;
    float h = st(pos((b - a) / k));
    return mix(b, a, h) - k * h * (1.0 - h);
}

float gentle_union(float a, float b, float flow_factor) {
    float g = gentle(flow_factor + pc.time * 0.5);
    return min(a, b) + g * 0.01;
}

// Gentle SDF primitives
float sdf_torus(vec3 p, vec2 t) {
    p = flow(p); // Apply gentle flow to position
    vec2 q = vec2(length(p.xz) - t.x, p.y);
    return length(q) - t.y;
}

float sdf_sphere(vec3 p, float r) {
    return length(flow(p)) - r;
}

float sdf_box(vec3 p, vec3 b) {
    p = abs(flow(p)) - b;
    return length(max(p, 0.0)) + min(max(p.x, max(p.y, p.z)), 0.0);
}

float sdf_octahedron(vec3 p, float s) {
    p = abs(flow(p));
    float m = p.x + p.y + p.z - s;
    vec3 q;
    if (3.0 * p.x < m) q = p.xyz;
    else if (3.0 * p.y < m) q = p.yzx;
    else if (3.0 * p.z < m) q = p.zxy;
    else return m * 0.57735027;
    float k = clamp(0.5 * (q.z - q.y + s), 0.0, s);
    return length(vec3(q.x, q.y - s + k, q.z - k));
}

float sdf_pyramid(vec3 p, float h) {
    p = abs(flow(p));
    return (p.x + p.z < h) ? (h - p.y) : length(vec3(p.x, max(0.0, p.y - h), p.z));
}

// Global peaceful glow accumulator
vec3 glow;

// Peaceful flowing SDF scene
vec2 sdf(vec3 p) {
    vec2 di = vec2(200.0, -1.0);

    // Gentle Audio-reactive parameters
    float energy = clamp(pc.note_velocity, 0.0, 1.0);
    float modulation = clamp(pc.cc1, 0.0, 1.0);
    float brightness = clamp(pc.cc74, 0.0, 1.0);

    // Smooth gentle transitions
    float smoothness = 0.3 + modulation * 0.2 + gentle(pc.time) * 0.05;

    // Time with calm audio modulation
    float t = pc.time * (0.5 + energy * 0.5);
    float t2 = pc.time * 0.3 + pc.pitch_bend * 0.2;
    float t3 = pc.time * 0.4 + pc.osc_ch1 * 0.3;

    // Transform position with gentle movement
    vec3 op = p;
    p += sin(p * 1.0 + t) * 0.02 * energy;
    p *= rotX(t * 0.1 + pc.osc_ch1 * 0.2);
    p *= rotY(t * 0.15 + pc.osc_ch2 * 0.2);
    p *= rotZ(t * 0.08 + pc.pitch_bend * 0.2);

    // Gentle ring system with flowing movement
    float ringRadius = 1.5 + vertexEnergy * 0.2 + sin(t) * 0.1;
    float ringThickness = 0.15 + energy * 0.05 + cos(t * 0.5) * 0.02;
    vec2 torusParams = vec2(ringRadius, ringThickness);

    // Ring 1 - Main peaceful torus
    vec3 p1 = p;
    p1.yz *= rot(t * 0.5 + pc.pitch_bend * 0.3);
    p1.xy *= rot(t * 0.4 + gentle(t));
    p1.xz *= rot(t * 0.3 + PI / 6.0 + pc.note_velocity * 0.5);
    float ring_1 = sdf_torus(p1, torusParams);

    // Ring 2 - Gentle oscillation
    vec3 p2 = p;
    p2.yz *= rot(t2 * 0.3 + pc.osc_ch1 * 0.5);
    p2.xy *= rot(t2 * 0.4 + gentle(t2) * 0.5);
    p2.xz *= rot(t2 * 0.35 + PI / 8.0);
    float ring_2 = sdf_torus(p2, torusParams * (0.9 + modulation * 0.2));

    // Ring 3 - Flowing harmony
    vec3 p3 = p;
    p3.yz *= rot(t3 * 0.4 + pc.osc_ch2 * 0.3);
    p3.xy *= rot(t3 * 0.35 - PI / 12.0 + gentle(t3) * 0.3);
    p3.xz *= rot(t3 * 0.25 + pc.cc74 * 0.2);
    float ring_3 = sdf_torus(p3, torusParams * (1.1 - modulation * 0.1));

    // ADDITIONAL PEACEFUL GEOMETRY
    // Floating spheres
    vec3 ps1 = p + vec3(sin(t * 0.5) * 1.5, cos(t * 0.4) * 1.2, sin(t * 0.3) * 1.8);
    float sphere1 = sdf_sphere(ps1, 0.4 + energy * 0.1);

    vec3 ps2 = p + vec3(cos(t * 0.4) * 1.8, sin(t * 0.5) * 1.5, cos(t * 0.35) * 1.4);
    float sphere2 = sdf_sphere(ps2, 0.35 + brightness * 0.08);

    // Gentle boxes
    vec3 pb1 = p + vec3(sin(t * 0.3 + PI) * 1.2, cos(t * 0.4 + PI/2) * 1.5, sin(t * 0.25) * 1.3);
    pb1 *= rotX(t * 0.5) * rotY(t * 0.4) * rotZ(t * 0.6);
    float box1 = sdf_box(pb1, vec3(0.25 + modulation * 0.05));

    // Octahedrons of peace
    vec3 po1 = p + vec3(cos(t * 0.6) * 2.0, sin(t * 0.45) * 1.8, cos(t * 0.5) * 2.1);
    po1 *= rotY(t * 0.8) * rotZ(t * 0.7);
    float octa1 = sdf_octahedron(po1, 0.5 + energy * 0.1);

    // Pyramids
    vec3 pp1 = p + vec3(sin(t * 0.35) * 2.0, cos(t * 0.5) * 1.5, sin(t * 0.4) * 1.8);
    pp1 *= rotX(t * 0.7) * rotY(t * 0.6);
    float pyramid1 = sdf_pyramid(pp1, 0.7 + brightness * 0.1);

    // COMBINE EVERYTHING WITH GENTLE HARMONY
    float rings = smooth_union(ring_1, smooth_union(ring_2, ring_3, smoothness), smoothness * 1.2);
    float spheres = smooth_union(sphere1, sphere2, smoothness * 0.8);
    float geometry = gentle_union(box1, gentle_union(octa1, pyramid1, t), t * 0.5);

    // Final combination with smooth blending
    float scene1 = smooth_union(rings, spheres, smoothness);
    float scene2 = gentle_union(scene1, geometry, t + energy * 0.5);

    // Add gentle energy expansion
    if (pc.note_velocity > 0.7) {
        float energy_sphere = sdf_sphere(p, 0.8 + (pc.note_velocity - 0.7) * 0.5);
        scene2 = smooth_union(scene2, energy_sphere, smoothness * 2.0);
    }

    di = min2(di, vec2(scene2, 1.0 + sin(length(op) * 5.0 + t) * 0.5));

    return di;
}

// Peaceful Ray marching with gentle flow
vec2 trace(vec3 ro, vec3 rd) {
    vec3 p = ro;
    vec2 di;
    float td = 0.0;

    // Gentle glow intensity
    float glowStrength = 0.05 + pc.note_velocity * 0.08 + gentle(pc.time) * 0.02;

    glow = vec3(0.0);

    // Gentle ray direction perturbation
    rd += sin(rd * 5.0 + pc.time * 0.5) * 0.005 * pc.cc1;

    for(int i = 0; i < MAX_STEPS; i++) {
        if(td >= MAX_DIST) break;

        di = sdf(p);

        if(di.x < EPSILON) {
            return vec2(td, di.y);
        }

        // Gentle step size modulation
        float stepMod = 1.0 + gentle(td + pc.time * 0.5) * 0.02 * pc.note_velocity;
        p += di.x * rd * stepMod;

        // Peaceful glow accumulation
        float glowFactor = (1.0 - st(di.x / 1.0)) * glowStrength;
        vec3 glowColor = pos(normalize(p)) * glowFactor;

        // Multi-layer glow with harmony
        vec3 glowTint1 = vec3(0.8 + pc.cc74 * 0.4, 0.9 + sin(pc.time * 0.8) * 0.2, 1.0 + pc.cc1 * 0.3);
        vec3 glowTint2 = sky_palette(length(p) * 0.3 + pc.time * 0.1);
        vec3 glowTint3 = ocean_palette(di.x * 3.0 + pc.time * 0.2);

        // Blend glow colors gently
        glowColor *= mix(glowTint1, mix(glowTint2, glowTint3, 0.6), gentle(td + pc.time) * 0.2 + 0.3);

        // Add volumetric peace
        if (di.x < 2.0) {
            vec3 volumetric = palette(length(p) + pc.time * 0.2) * 0.01 * (1.0 - di.x * 0.5) * pc.note_velocity;
            glowColor += volumetric;
        }

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

// Peaceful flowing rendering function
vec3 render(vec2 uv) {
    // Gentle Camera setup with subtle audio influence
    float camDist = 3.5 + sin(pc.time * 0.5) * 0.3 - pc.cc1 * 0.3 + pc.note_velocity * 0.5;
    vec3 ro = vec3(
        sin(pc.time * 0.15 + pc.osc_ch1 * 0.3) * 0.2,
        cos(pc.time * 0.2 + pc.osc_ch2 * 0.3) * 0.15,
        -camDist
    );

    // Gentle automatic camera movement
    ro *= rotX(sin(pc.time * 0.08) * 0.1 + pc.pitch_bend * 0.1);
    ro *= rotY(cos(pc.time * 0.1) * 0.15 + pc.cc74 * 0.3);
    ro *= rotZ(sin(pc.time * 0.06) * 0.08 + pc.cc1 * 0.3);

    // Mouse/OSC camera override
    if(pc.mouse_pressed > 0u) {
        float mx = (float(pc.mouse_x) / float(pc.render_w) - 0.5) * TAU * 0.5;
        float my = (float(pc.mouse_y) / float(pc.render_h) - 0.5) * PI * 0.3;
        ro.xz *= rot(mx + gentle(pc.time) * 0.05);
        ro.yz *= rot(my + gentle(pc.time + 1.0) * 0.05);
    }

    // Gentle UV distortion
    uv += sin(uv * 3.0 + pc.time * 0.5) * 0.01 * pc.note_velocity;
    uv *= 1.0 + gentle(length(uv) + pc.time) * 0.02 * pc.cc1;

    vec3 rd = normalize(vec3(uv, 1.0));
    vec3 lo = ro + vec3(sin(pc.time * 0.4), cos(pc.time * 0.3), sin(pc.time * 0.25)) * 1.0; // Moving light

    vec2 tdi = trace(ro, rd);

    if(tdi.x > 0.0) {
        vec3 p = ro + rd * tdi.x;
        vec3 n = get_normal(p);

        // Gentle surface effects
        vec3 cd = normalize(ro - p);
        vec3 ld = normalize(lo - p);
        vec3 reflection = reflect(rd, n);

        // Subtle perturbation
        float perturbStrength = 3.0 + pc.note_velocity * 2.0 + sin(pc.time * 0.8) * 1.0;
        vec3 perturbation = 0.02 * sin(p * perturbStrength + pc.time * 0.3);
        perturbation += 0.01 * cos(p * perturbStrength * 1.2 + pc.time * 0.4);

        // Multi-layer iridescence harmony
        float iridValue1 = dot(n + perturbation, cd) * 1.5;
        float iridValue2 = dot(n, reflection) * 1.2 + length(p) * 0.2;
        float iridValue3 = gentle(length(p) + pc.time * 0.2) * 2.0;

        vec3 irid1 = palette(iridValue1 + pc.time * 0.1);
        vec3 irid2 = sky_palette(iridValue2 + pc.time * 0.08);
        vec3 irid3 = ocean_palette(iridValue3 + pc.time * 0.12);

        vec3 iridescence = mix(mix(irid1, irid2, 0.6), irid3, gentle(pc.time + length(p)) * 0.1 + 0.3);

        // Gentle specular highlights
        float spec1 = st(dot(reflection, ld));
        float spec2 = st(dot(reflection, normalize(lo + vec3(1, 0, 0) - p)));
        float spec3 = st(dot(reflection, normalize(lo + vec3(0, 1, 0) - p)));

        float specIntensity = 0.2 + pc.cc74 * 0.3 + sin(pc.time * 1.2) * 0.1;

        float specular = 0.0;
        specular += specIntensity * pow(pos(sin(spec1 * 8.0 - pc.time * 0.4)) + 0.2, 8.0);
        specular += specIntensity * 0.6 * pow(spec2 + 0.3, 6.0);
        specular += specIntensity * 0.4 * pow(spec3 + 0.4, 4.0);

        // Gentle animated specular
        specular *= 1.0 + sin(pc.time * 2.0 + length(p) * 1.5) * 0.1;

        // Gentle lighting
        float shadow1 = pow(st(dot(n, vec3(0.0, 1.0, 0.0)) * 0.5 + 0.8), 1.5);
        float shadow2 = pow(st(dot(n, normalize(vec3(1.0, 0.5, 0.2))) * 0.5 + 0.6), 1.8);
        float shadow3 = st(dot(n, normalize(vec3(-0.5, -1.0, 0.3))) * 0.3 + 0.7);

        float shadow = (shadow1 + shadow2 * 0.7 + shadow3 * 0.5) / 2.2;

        // Harmonious color mixing
        vec3 color = iridescence * shadow;
        color += vec3(specular * 0.8, specular * 0.6, specular * 1.0);
        color += glow * 0.8;

        // Gentle energy effects
        if(pc.note_velocity > 0.5) {
            vec3 energyGlow = vec3(
                0.8 + pc.note_velocity * 0.4,
                0.9 + pc.note_velocity * 0.3,
                1.0 + pc.note_velocity * 0.5
            ) * (pc.note_velocity - 0.5) * 0.3;
            color += energyGlow;
        }

        // Oscillator effects
        color += vec3(pc.osc_ch1 * 0.1, pc.osc_ch2 * 0.12, (pc.osc_ch1 + pc.osc_ch2) * 0.08);

        // Pitch bend chromatic effects
        color *= 1.0 + vec3(pc.pitch_bend * 0.15, 0.0, -pc.pitch_bend * 0.1);

        // CC effects
        color = mix(color, color * vec3(1.3, 0.9, 1.2), pc.cc1 * 0.1);
        color = mix(color, pow(color, vec3(0.9, 1.1, 0.95)), pc.cc74 * 0.15);

        // Romantic Material ID based effects 💕
        if (tdi.y > 1.5) {
            color *= vec3(1.4, 0.9, 1.8); // Passionate love material (hearts glow warmly)
        } else if (tdi.y > 1.0) {
            color *= vec3(1.2, 0.8, 1.5); // Gentle romance material (soft and dreamy)
        } else {
            color *= vec3(1.1, 1.0, 1.3); // Sparkle material (magical and ethereal)
        }

        return color;
    }

    // Peaceful Background with gentle glow layers
    vec3 bg = glow * 1.2;
    bg += sky_palette(length(uv) * 0.2 + pc.time * 0.05) * 0.05 * pc.note_velocity;
    bg += ocean_palette(gentle(length(uv) + pc.time * 0.3)) * 0.03 * pc.cc74;

    // Background animation
    bg += vec3(0.02, 0.03, 0.05) * (1.0 + sin(pc.time * 0.3 + length(uv) * 2.0) * 0.2);

    return bg;
}

void main() {
    vec2 resolution = vec2(float(pc.render_w), float(pc.render_h));
    vec2 uv = (fragUV - 0.5) * 2.0;
    uv.x *= resolution.x / resolution.y;

    // Gentle UV manipulation before rendering
    vec2 uvOriginal = uv;

    // Subtle barrel distortion with audio
    float distortion = 0.02 + pc.note_velocity * 0.05;
    float r = length(uv);
    uv *= 1.0 + distortion * r * r;

    // Minimal chromatic aberration
    float aberration = 0.002 + pc.cc74 * 0.005;
    vec3 c = vec3(0.0);
    c.r = render(uv + vec2(aberration, 0.0)).r;
    c.g = render(uv).g;
    c.b = render(uv - vec2(aberration, 0.0)).b;

    // Gentle post-processing effects

    // Soft reflection effect
    if (pc.note_velocity > 0.7) {
        float angle = atan(uvOriginal.y, uvOriginal.x);
        float segments = 4.0 + floor(pc.note_velocity * 2.0);
        angle = mod(angle, TAU / segments) * segments;
        vec2 reflectUV = vec2(cos(angle), sin(angle)) * length(uvOriginal);
        vec3 reflectColor = render(reflectUV * 0.9);
        c = mix(c, reflectColor, (pc.note_velocity - 0.7) * 0.2);
    }

    // Gentle feedback loop
    vec2 feedbackUV = uvOriginal * 0.98 + sin(pc.time * 0.4 + length(uvOriginal) * 2.0) * 0.005;
    vec3 feedback = render(feedbackUV);
    c = mix(c, feedback, 0.03 * pc.cc1);

    // Color grading harmony
    c = pow(c, vec3(0.95 + sin(pc.time * 0.2) * 0.05, 1.0 + cos(pc.time * 0.25) * 0.05, 0.98 + sin(pc.time * 0.18) * 0.05));

    // Gentle contrast and saturation
    float contrast = 1.05 + pc.cc74 * 0.15;
    c = (c - 0.5) * contrast + 0.5;

    // Saturation based on energy
    float gray = dot(c, vec3(0.299, 0.587, 0.114));
    c = mix(vec3(gray), c, 1.0 + pc.note_velocity * 0.3);

    // Romantic hue shift - like seeing the world through love 💖
    float hueShift = pc.osc_ch1 * 0.6 + pc.time * 0.03;
    c = mix(c, c.gbr, sin(hueShift) * 0.06); // Gentle warm shift
    c = mix(c, c.brg, cos(hueShift * 1.2) * 0.04); // Soft rose tint

    // Add extra romantic warmth during high energy
    if (pc.note_velocity > 0.6) {
        vec3 warmth = vec3(1.1, 0.95, 1.0) * (pc.note_velocity - 0.6) * 0.3;
        c *= warmth;
    }

    // Soft vignette with gentle animation
    float vignette = 1.0 - pow(length(uvOriginal) * 0.5, 1.5);
    vignette += sin(pc.time * 0.6 + length(uvOriginal) * 3.0) * 0.02 * pc.note_velocity;
    vignette = st(vignette);
    c *= vignette;

    // Romantic heart-beat movement effect 💓
    if (pc.note_velocity > 0.8) {
        float heartbeat = (pc.note_velocity - 0.8) * 0.012;
        // Heart-like pulsing movement
        vec2 heartOffset = vec2(
            sin(pc.time * 4.0) * heartbeat * 1.2, // Slower, more heart-like
            cos(pc.time * 4.0) * heartbeat * 0.8
        ) * (1.0 + sin(pc.time * 12.0) * 0.3); // Double beat effect
        vec3 heartColor = render(uvOriginal + heartOffset);
        c = mix(c, heartColor * vec3(1.1, 0.9, 1.0), 0.15); // Warm tint
    }

    // Final gentle modulation
    c *= 1.0 + gentle(length(uvOriginal) + pc.time * 0.05) * 0.02 * pc.cc1;

    // Romantic dreamy bloom effect - like love is glowing ✨
    vec3 bloom = max(c - vec3(0.7), 0.0) * 1.0;
    bloom *= vec3(1.1, 0.8, 0.9); // Warm romantic bloom
    c += bloom * pc.note_velocity * 0.4;

    // Add subtle heart-shaped highlights during peaks
    if (pc.note_velocity > 0.9) {
        float heartGlow = (pc.note_velocity - 0.9) * 2.0;
        vec2 heartUV = uvOriginal * 1.5;
        float heartShape = abs(heartUV.x) + abs(heartUV.y) - 0.5;
        float heartMask = smoothstep(0.2, 0.0, heartShape);
        c += vec3(0.8, 0.4, 0.6) * heartMask * heartGlow * 0.3;
    }

    // Final romantic color grading
    c = mix(c, c * vec3(1.05, 0.98, 1.02), 0.3); // Subtle warm/cool balance

    // Output with passionate saturation 💕
    outColor = vec4(st(c * (1.0 + pc.note_velocity * 0.25)), 1.0);
}