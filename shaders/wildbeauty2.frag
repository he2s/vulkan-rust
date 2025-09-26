#version 450

// Push constants from application
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

// CRAZY Constants
#define PI 3.141592653589793238
#define TAU (2.0 * PI)
#define EPSILON 0.5
#define MAX_STEPS 5
#define MAX_DIST 10.0
#define CHAOS_FACTOR 2.0


// INSANE Helper macros
#define min2(a, b) ((a.x < b.x) ? a : b)
#define pos(x) (x * 0.5 + 0.5)
#define st(x) clamp(x, 0.0, 1.0)
#define crazy(x) (sin(x * CHAOS_FACTOR) * cos(x * CHAOS_FACTOR * 0.7) * tan(x * CHAOS_FACTOR * 0.3))
#define warp(x) (x + crazy(x + pc.time) * 0.2)

// Chaos rotation matrix with time distortion
mat2 rot(float a) {
    a += crazy(a) * pc.note_velocity * 2.0;
    float c = cos(a);
    float s = sin(a);
    return mat2(c, -s, s, c);
}

// 3D rotation matrices for MAXIMUM CHAOS
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

// PSYCHEDELIC INSANE color palette
vec3 palette(float x) {
    // CHAOS shift palette based on EVERYTHING
    x += float(pc.note_count % 16u) * 0.0625;
    x = warp(x); // Apply chaos warp

    // EXTREME Audio-reactive palette parameters
    vec3 a = vec3(0.8, 0.2, 0.9) * (1.0 + sin(pc.time * 3.0) * 0.5); // Pulsing base
    vec3 b = vec3(1.2 + pc.cc74 * 2.0, 0.8 + pc.cc1 * 1.5, 1.5 + pc.note_velocity * 2.0);
    vec3 c = vec3(2.1, 1.5, 3.0) + vec3(pc.osc_ch1 * 5.0, pc.osc_ch2 * 7.0, pc.pitch_bend * 3.0);
    vec3 d = vec3(pc.time * 0.1, pc.pitch_bend * 2.0, pc.note_velocity * 3.0);

    // Multiple layers of chaos
    vec3 color1 = a + b * cos(TAU * (c * x + d));
    vec3 color2 = vec3(0.3, 0.7, 0.2) + vec3(0.7) * sin(TAU * (x * 7.0 + pc.time * 2.0));
    vec3 color3 = vec3(0.9, 0.1, 0.5) * cos(x * 13.0 + pc.time * 5.0);

    // Blend all layers with chaos
    return mix(mix(color1, color2, 0.3), color3, crazy(x) * 0.2 + 0.1);
}

// Additional crazy palettes
vec3 rainbow_palette(float x) {
    return 0.5 + 0.5 * cos(TAU * (x + vec3(0.0, 0.33, 0.67)) * 3.0 + pc.time * 2.0);
}

vec3 fire_palette(float x) {
    vec3 fire = vec3(1.0, 0.5, 0.1) * pow(x, 2.0) + vec3(0.9, 0.1, 0.0) * pow(1.0-x, 3.0);
    return fire * (1.0 + crazy(x * 10.0 + pc.time * 7.0) * 0.5);
}

// CHAOTIC union operations
float smooth_union(float a, float b, float k) {
    k *= 1.0 + crazy(a + b + pc.time) * 0.1;
    float h = st(pos((b - a) / k));
    return mix(b, a, h) - k * h * (1.0 - h);
}

float crazy_union(float a, float b, float chaos) {
    float c = crazy(chaos + pc.time * 3.0);
    return min(a, b) + c * 0.05;
}

// INSANE SDF primitives
float sdf_torus(vec3 p, vec2 t) {
    p = warp(p); // Apply chaos warp to position
    vec2 q = vec2(length(p.xz) - t.x, p.y);
    return length(q) - t.y;
}

float sdf_sphere(vec3 p, float r) {
    return length(warp(p)) - r;
}

float sdf_box(vec3 p, vec3 b) {
    p = abs(warp(p)) - b;
    return length(max(p, 0.0)) + min(max(p.x, max(p.y, p.z)), 0.0);
}

float sdf_octahedron(vec3 p, float s) {
    p = abs(warp(p));
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
    p = abs(warp(p));
    return (p.x + p.z < h) ? (h - p.y) : length(vec3(p.x, max(0.0, p.y - h), p.z));
}

// Global PSYCHEDELIC glow accumulator
vec3 glow;

// ULTIMATE CHAOS SDF scene
vec2 sdf(vec3 p) {
    vec2 di = vec2(200.0, -1.0);

    // EXTREME Audio-reactive parameters
    float energy = clamp(pc.note_velocity, 0.0, 1.0);
    float modulation = clamp(pc.cc1, 0.0, 1.0);
    float brightness = clamp(pc.cc74, 0.0, 1.0);

    // CHAOTIC smoothness
    float smoothness = 0.1 + modulation * 0.8 + crazy(pc.time) * 0.2;

    // Time with INSANE audio modulation
    float t = pc.time * (2.0 + energy * 5.0);
    float t2 = pc.time * 0.7 + pc.pitch_bend * 2.0;
    float t3 = pc.time * 1.3 + pc.osc_ch1 * 3.0;

    // Transform position with chaotic distortion
    vec3 op = p;
    p += sin(p * 3.0 + t) * 0.1 * energy;
    p *= rotX(t * 0.5 + pc.osc_ch1 * PI);
    p *= rotY(t * 0.7 + pc.osc_ch2 * PI);
    p *= rotZ(t * 0.3 + pc.pitch_bend * PI);

    // MASSIVE ring system with chaos
    float ringRadius = 1.5 + vertexEnergy * 0.5 + sin(t) * 0.3;
    float ringThickness = 0.1 + energy * 0.15 + cos(t * 2.0) * 0.05;
    vec2 torusParams = vec2(ringRadius, ringThickness);

    // Ring 1 - Main chaos torus
    vec3 p1 = p;
    p1.yz *= rot(t * 2.0 + pc.pitch_bend * 3.0);
    p1.xy *= rot(t * 1.8 + crazy(t));
    p1.xz *= rot(t * 1.4 + PI / 3.0 + pc.note_velocity * TAU);
    float ring_1 = sdf_torus(p1, torusParams);

    // Ring 2 - Oscillating madness
    vec3 p2 = p;
    p2.yz *= rot(t2 * 1.5 + pc.osc_ch1 * TAU);
    p2.xy *= rot(t2 * 2.2 + crazy(t2) * 2.0);
    p2.xz *= rot(t2 * 1.7 + PI / 5.0);
    float ring_2 = sdf_torus(p2, torusParams * (0.8 + modulation * 0.6));

    // Ring 3 - Complete chaos
    vec3 p3 = p;
    p3.yz *= rot(t3 * 3.0 + pc.osc_ch2 * TAU);
    p3.xy *= rot(t3 * 2.5 - PI / 7.0 + crazy(t3) * 3.0);
    p3.xz *= rot(t3 * 1.9 + pc.cc74 * PI);
    float ring_3 = sdf_torus(p3, torusParams * (1.2 - modulation * 0.3));

    // ADDITIONAL CHAOS GEOMETRY
    // Floating spheres
    vec3 ps1 = p + vec3(sin(t * 2.0) * 2.0, cos(t * 1.5) * 1.5, sin(t * 0.8) * 2.5);
    float sphere1 = sdf_sphere(ps1, 0.3 + energy * 0.2);

    vec3 ps2 = p + vec3(cos(t * 1.7) * 2.5, sin(t * 2.3) * 2.0, cos(t * 1.1) * 1.8);
    float sphere2 = sdf_sphere(ps2, 0.25 + brightness * 0.15);

    // Chaos boxes
    vec3 pb1 = p + vec3(sin(t * 1.1 + PI) * 1.8, cos(t * 1.9 + PI/2) * 2.2, sin(t * 0.6) * 2.0);
    pb1 *= rotX(t * 3.0) * rotY(t * 2.0) * rotZ(t * 4.0);
    float box1 = sdf_box(pb1, vec3(0.2 + modulation * 0.1));

    // Octahedrons of madness
    vec3 po1 = p + vec3(cos(t * 2.5) * 3.0, sin(t * 1.8) * 2.5, cos(t * 1.3) * 2.8);
    po1 *= rotY(t * 5.0) * rotZ(t * 3.5);
    float octa1 = sdf_octahedron(po1, 0.4 + energy * 0.3);

    // Pyramids
    vec3 pp1 = p + vec3(sin(t * 0.9) * 2.8, cos(t * 2.1) * 1.9, sin(t * 1.4) * 2.3);
    pp1 *= rotX(t * 4.0) * rotY(t * 2.5);
    float pyramid1 = sdf_pyramid(pp1, 0.6 + brightness * 0.2);

    // COMBINE EVERYTHING WITH MAXIMUM CHAOS
    float rings = smooth_union(ring_1, smooth_union(ring_2, ring_3, smoothness), smoothness * 0.8);
    float spheres = smooth_union(sphere1, sphere2, smoothness * 1.5);
    float geometry = crazy_union(box1, crazy_union(octa1, pyramid1, t), t * 2.0);

    // Final combination with multiple blend modes
    float scene1 = smooth_union(rings, spheres, smoothness);
    float scene2 = crazy_union(scene1, geometry, t + energy * 5.0);

    // Add periodic chaos bursts
    if (pc.note_velocity > 0.8) {
        float burst_sphere = sdf_sphere(p, 0.5 + (pc.note_velocity - 0.8) * 2.0);
        scene2 = min(scene2, burst_sphere);
    }

    di = min2(di, vec2(scene2, 1.0 + sin(length(op) * 5.0 + t) * 0.5));

    return di;
}

// INSANE Ray marching with CHAOS
vec2 trace(vec3 ro, vec3 rd) {
    vec3 p = ro;
    vec2 di;
    float td = 0.0;

    // EXTREME glow intensity
    float glowStrength = 0.1 + pc.note_velocity * 0.15 + crazy(pc.time) * 0.05;

    glow = vec3(0.0);

    // Chaotic ray direction perturbation
    rd += sin(rd * 20.0 + pc.time * 3.0) * 0.02 * pc.cc1;

    for(int i = 0; i < MAX_STEPS; i++) {
        if(td >= MAX_DIST) break;

        di = sdf(p);

        if(di.x < EPSILON) {
            return vec2(td, di.y);
        }

        // CHAOS step size modulation
        float stepMod = 1.0 + crazy(td + pc.time * 2.0) * 0.1 * pc.note_velocity;
        p += di.x * rd * stepMod;

        // PSYCHEDELIC glow accumulation
        float glowFactor = (1.0 - st(di.x / 0.6)) * glowStrength;
        vec3 glowColor = pos(normalize(p)) * glowFactor;

        // Multi-layer glow with chaos
        vec3 glowTint1 = vec3(1.0 + pc.cc74 * 2.0, 1.0 + sin(pc.time * 4.0), 1.0 + pc.cc1 * 1.5);
        vec3 glowTint2 = rainbow_palette(length(p) * 0.5 + pc.time);
        vec3 glowTint3 = fire_palette(di.x * 10.0 + pc.time * 0.5);

        // Blend glow colors with chaos
        glowColor *= mix(glowTint1, mix(glowTint2, glowTint3, 0.4), crazy(td + pc.time) * 0.3 + 0.2);

        // Add volumetric chaos
        if (di.x < 1.0) {
            vec3 volumetric = palette(length(p) + pc.time) * 0.02 * (1.0 - di.x) * pc.note_velocity;
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

// ULTIMATE CHAOS rendering function
vec3 render(vec2 uv) {
    // INSANE Camera setup with MAXIMUM audio influence
    float camDist = 2.0 + sin(pc.time * 2.0) * 1.0 - pc.cc1 * 1.5 + pc.note_velocity * 2.0;
    vec3 ro = vec3(
        sin(pc.time * 0.5 + pc.osc_ch1 * PI) * 0.5,
        cos(pc.time * 0.7 + pc.osc_ch2 * PI) * 0.3,
        -camDist
    );

    // CHAOTIC automatic camera movement
    ro *= rotX(sin(pc.time * 0.3) * 0.5 + pc.pitch_bend * 0.5);
    ro *= rotY(cos(pc.time * 0.4) * 0.7 + pc.cc74 * PI);
    ro *= rotZ(sin(pc.time * 0.2) * 0.3 + pc.cc1 * PI);

    // Mouse/OSC camera override
    if(pc.mouse_pressed > 0u) {
        float mx = (float(pc.mouse_x) / float(pc.render_w) - 0.5) * TAU * 2.0;
        float my = (float(pc.mouse_y) / float(pc.render_h) - 0.5) * PI * 1.5;
        ro.xz *= rot(mx + crazy(pc.time) * 0.2);
        ro.yz *= rot(my + crazy(pc.time + 1.0) * 0.2);
    }

    // CHAOS UV distortion
    uv += sin(uv * 10.0 + pc.time * 3.0) * 0.05 * pc.note_velocity;
    uv *= 1.0 + crazy(length(uv) + pc.time) * 0.1 * pc.cc1;

    vec3 rd = normalize(vec3(uv, 1.0));
    vec3 lo = ro + vec3(sin(pc.time * 2.0), cos(pc.time * 1.5), sin(pc.time * 0.8)) * 2.0; // Moving light

    vec2 tdi = trace(ro, rd);

    if(tdi.x > 0.0) {
        vec3 p = ro + rd * tdi.x;
        vec3 n = get_normal(p);

        // EXTREME surface effects
        vec3 cd = normalize(ro - p);
        vec3 ld = normalize(lo - p);
        vec3 reflection = reflect(rd, n);

        // INSANE perturbation
        float perturbStrength = 15.0 + pc.note_velocity * 30.0 + sin(pc.time * 5.0) * 10.0;
        vec3 perturbation = 0.1 * sin(p * perturbStrength + pc.time * 2.0);
        perturbation += 0.05 * cos(p * perturbStrength * 1.7 + pc.time * 3.0);

        // Multi-layer iridescence CHAOS
        float iridValue1 = dot(n + perturbation, cd) * 3.0;
        float iridValue2 = dot(n, reflection) * 2.0 + length(p) * 0.5;
        float iridValue3 = crazy(length(p) + pc.time * 0.5) * 5.0;

        vec3 irid1 = palette(iridValue1 + pc.time * 0.5);
        vec3 irid2 = rainbow_palette(iridValue2 + pc.time * 0.3);
        vec3 irid3 = fire_palette(iridValue3 + pc.time * 0.7);

        vec3 iridescence = mix(mix(irid1, irid2, 0.4), irid3, crazy(pc.time + length(p)) * 0.3 + 0.2);

        // MULTIPLE specular highlights
        float spec1 = st(dot(reflection, ld));
        float spec2 = st(dot(reflection, normalize(lo + vec3(1, 0, 0) - p)));
        float spec3 = st(dot(reflection, normalize(lo + vec3(0, 1, 0) - p)));

        float specIntensity = 0.3 + pc.cc74 * 0.8 + sin(pc.time * 7.0) * 0.2;

        float specular = 0.0;
        specular += specIntensity * pow(pos(sin(spec1 * 25.0 - pc.time * 2.0)) + 0.1, 16.0);
        specular += specIntensity * 0.7 * pow(spec2 + 0.2, 12.0);
        specular += specIntensity * 0.5 * pow(spec3 + 0.3, 8.0);

        // Crazy animated specular
        specular *= 1.0 + sin(pc.time * 10.0 + length(p) * 5.0) * 0.3;

        // CHAOTIC lighting
        float shadow1 = pow(st(dot(n, vec3(0.0, 1.0, 0.0)) * 0.5 + 1.2), 2.0);
        float shadow2 = pow(st(dot(n, normalize(vec3(1.0, 0.5, 0.2))) * 0.7 + 0.8), 2.5);
        float shadow3 = st(dot(n, normalize(vec3(-0.5, -1.0, 0.3))) * 0.3 + 0.9);

        float shadow = (shadow1 + shadow2 * 0.6 + shadow3 * 0.4) / 2.0;

        // INSANE color mixing
        vec3 color = iridescence * shadow;
        color += vec3(specular * 2.0, specular * 1.5, specular * 2.5);
        color += glow * 1.5;

        // EXTREME energy effects
        if(pc.note_velocity > 0.3) {
            vec3 energyFlash = vec3(
                1.0 + pc.note_velocity * 2.0,
                0.5 + pc.note_velocity * 1.5,
                0.8 + pc.note_velocity * 3.0
            ) * (pc.note_velocity - 0.3) * 0.8;
            color += energyFlash;
        }

        // Oscillator effects
        color += vec3(pc.osc_ch1 * 0.3, pc.osc_ch2 * 0.4, (pc.osc_ch1 + pc.osc_ch2) * 0.2);

        // Pitch bend chromatic effects
        color *= 1.0 + vec3(pc.pitch_bend * 0.5, 0.0, -pc.pitch_bend * 0.3);

        // CC effects
        color = mix(color, color * vec3(2.0, 0.5, 1.5), pc.cc1 * 0.3);
        color = mix(color, pow(color, vec3(0.7, 1.3, 0.8)), pc.cc74 * 0.4);

        // Material ID based effects
        if (tdi.y > 1.5) {
            color *= vec3(1.5, 0.8, 2.0); // Hot material
        } else if (tdi.y > 1.0) {
            color *= vec3(0.8, 1.5, 1.2); // Cool material
        }

        return color;
    }

    // PSYCHEDELIC Background with multiple glow layers
    vec3 bg = glow * 2.0;
    bg += rainbow_palette(length(uv) * 0.3 + pc.time * 0.2) * 0.1 * pc.note_velocity;
    bg += fire_palette(crazy(length(uv) + pc.time)) * 0.05 * pc.cc74;

    // Background animation
    bg += vec3(0.05, 0.02, 0.08) * (1.0 + sin(pc.time + length(uv) * 5.0) * 0.5);

    return bg;
}

void main() {
    vec2 resolution = vec2(float(pc.render_w), float(pc.render_h));
    vec2 uv = (fragUV - 0.5) * 2.0;
    uv.x *= resolution.x / resolution.y;

    // CHAOS UV manipulation before rendering
    vec2 uvOriginal = uv;

    // Barrel distortion with audio
    float distortion = 0.1 + pc.note_velocity * 0.3;
    float r = length(uv);
    uv *= 1.0 + distortion * r * r;

    // Simple anti-aliasing via smoothstep on edges
    vec2 pixelSize = 2.0 / resolution;
    vec3 c = render(uv);

    // Basic edge smoothing
    c = mix(c, render(uv + vec2(pixelSize.x, 0.0)), 0.1);
    c = mix(c, render(uv + vec2(0.0, pixelSize.y)), 0.1);

    // INSANE post-processing effects

    // Kaleidoscope effect
    if (pc.note_velocity > 0.5) {
        float angle = atan(uvOriginal.y, uvOriginal.x);
        float segments = 6.0 + floor(pc.note_velocity * 6.0);
        angle = mod(angle, TAU / segments) * segments;
        vec2 kaleidoUV = vec2(cos(angle), sin(angle)) * length(uvOriginal);
        vec3 kaleidoColor = render(kaleidoUV * 0.7);
        c = mix(c, kaleidoColor, (pc.note_velocity - 0.5) * 0.6);
    }

    // Feedback loop
    vec2 feedbackUV = uvOriginal * 0.95 + sin(pc.time * 2.0 + length(uvOriginal) * 5.0) * 0.02;
    vec3 feedback = render(feedbackUV);
    c = mix(c, feedback, 0.1 * pc.cc1);

    // Color grading CHAOS
    c = pow(c, vec3(0.8 + sin(pc.time) * 0.2, 1.0 + cos(pc.time * 1.2) * 0.2, 0.9 + sin(pc.time * 0.8) * 0.2));

    // Contrast and saturation
    float contrast = 1.2 + pc.cc74 * 0.5;
    c = (c - 0.5) * contrast + 0.5;

    // Saturation based on energy
    float gray = dot(c, vec3(0.299, 0.587, 0.114));
    c = mix(vec3(gray), c, 1.0 + pc.note_velocity * 1.5);

    // Hue shift
    float hueShift = pc.osc_ch1 * TAU + pc.time * 0.1;
    c = mix(c, c.gbr, sin(hueShift) * 0.3);
    c = mix(c, c.brg, cos(hueShift * 1.2) * 0.2);

    // EXTREME vignette with animation
    float vignette = 1.0 - pow(length(uvOriginal) * 0.7, 2.0);
    vignette += sin(pc.time * 3.0 + length(uvOriginal) * 10.0) * 0.1 * pc.note_velocity;
    vignette = st(vignette);
    c *= vignette;

    // Screen shake effect
    if (pc.note_velocity > 0.8) {
        float shake = (pc.note_velocity - 0.8) * 0.05;
        vec2 shakeOffset = vec2(
            sin(pc.time * 50.0) * shake,
            cos(pc.time * 37.0) * shake
        );
        vec3 shakeColor = render(uvOriginal + shakeOffset);
        c = mix(c, shakeColor, 0.3);
    }

    // Final chaos modulation
    c *= 1.0 + crazy(length(uvOriginal) + pc.time * 0.1) * 0.1 * pc.cc1;

    // Bloom effect
    vec3 bloom = max(c - vec3(1.0), 0.0) * 2.0;
    c += bloom * pc.note_velocity;

    // Simple blur at the end
    vec2 blurSize = pixelSize * (0.5 + pc.cc1 * 1.5);
    vec3 blur = c;
    blur += render(uv + vec2(blurSize.x, 0.0));
    blur += render(uv - vec2(blurSize.x, 0.0));
    blur += render(uv + vec2(0.0, blurSize.y));
    blur += render(uv - vec2(0.0, blurSize.y));
    blur *= 0.2;

    c = mix(c, blur, 0.3 + pc.note_velocity * 0.4);

    // Output with CHAOS saturation
    outColor = vec4(st(c * (1.0 + pc.note_velocity * 0.5)), 1.0);
}