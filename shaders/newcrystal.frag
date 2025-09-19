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

// Constants
#define PI 3.141592653589793238
#define TAU (2.0 * PI)
#define EPSILON 0.008
#define MAX_STEPS 20
#define MAX_DIST 100.0

// Helper macros
#define min2(a, b) ((a.x < b.x) ? a : b)
#define sat(x) clamp(x, 0.0, 1.0)

// Rotation matrix
mat2 rot(float a) {
    float c = cos(a);
    float s = sin(a);
    return mat2(c, -s, s, c);
}

// Neon color burst palette
vec3 neonBurst(float t, float intensity) {
    // Only activate on high energy
    if(intensity < 0.7) {
        return vec3(0.0);
    }

    // Pick neon color based on note count
    float colorIndex = float(pc.note_count % 6u);
    vec3 neonColor;

    if(colorIndex < 1.0) {
        neonColor = vec3(0.0, 1.0, 1.0); // Cyan
    } else if(colorIndex < 2.0) {
        neonColor = vec3(1.0, 0.0, 1.0); // Magenta
    } else if(colorIndex < 3.0) {
        neonColor = vec3(0.0, 1.0, 0.3); // Green
    } else if(colorIndex < 4.0) {
        neonColor = vec3(1.0, 0.3, 0.0); // Orange
    } else if(colorIndex < 5.0) {
        neonColor = vec3(0.3, 0.5, 1.0); // Blue
    } else {
        neonColor = vec3(1.0, 1.0, 0.0); // Yellow
    }

    // Pulse the neon with time
    float pulse = 0.5 + 0.5 * sin(pc.time * 10.0 + t * 5.0);
    return neonColor * (intensity - 0.7) * 3.0 * pulse;
}

// Smooth union
float smooth_union(float a, float b, float k) {
    float h = sat(0.5 + 0.5 * (b - a) / k);
    return mix(b, a, h) - k * h * (1.0 - h);
}

// Torus SDF
float sdf_torus(vec3 p, vec2 t) {
    vec2 q = vec2(length(p.xz) - t.x, p.y);
    return length(q) - t.y;
}

// Global glow accumulator
vec3 glow;

// Clean SDF scene with quirks
vec2 sdf(vec3 p) {
    vec2 di = vec2(120.0, -1.0);

    // Audio parameters
    float energy = clamp(pc.note_velocity, 0.0, 1.0);
    float modulation = clamp(pc.cc1, 0.0, 1.0);
    float brightness = clamp(pc.cc74, 0.0, 1.0);

    // QUIRK 1: Rings suddenly snap to different positions on beats
    vec3 snapOffset = vec3(0.0);
    if(energy > 0.6) {
        float snapTime = floor(pc.time * 4.0); // Snap 4 times per second
        snapOffset.x = sin(snapTime * 12.34) * 0.2;
        snapOffset.y = cos(snapTime * 23.45) * 0.2;
        snapOffset.z = sin(snapTime * 34.56) * 0.1;

        // Smooth transition to snap position
        float snapSmooth = smoothstep(0.6, 0.8, energy);
        p += snapOffset * snapSmooth;
    }

    // Smooth time with occasional hiccups
    float t = pc.time * (0.3 + energy * 0.5);

    // QUIRK 2: Time occasionally reverses
    if(sin(pc.time * 0.5) > 0.9) {
        t *= -1.0;
    }

    // Base torus parameters
    float ringRadius = 1.0;
    float ringThickness = 0.12 + energy * 0.03;

    // QUIRK 3: Rings pulse in size based on pitch bend
    ringRadius += sin(pc.time * 3.0 + pc.pitch_bend * PI) * 0.1;

    // QUIRK 4: Thickness varies per ring
    vec2 torusParams1 = vec2(ringRadius, ringThickness);
    vec2 torusParams2 = vec2(ringRadius * 0.95, ringThickness * 1.3);
    vec2 torusParams3 = vec2(ringRadius * 1.05, ringThickness * 0.7);

    // First ring with wobble
    vec3 p1 = p;
    float wobble1 = sin(pc.time * 7.0) * 0.1 * modulation;
    p1.yz *= rot(t + wobble1);
    p1.xy *= rot(t * 0.7);
    p1.xz *= rot(t * 0.5);
    float ring_1 = sdf_torus(p1, torusParams1);

    // Second ring - counter rotation
    vec3 p2 = p;
    p2.yz *= rot(-t * 0.8); // QUIRK: Counter rotation
    p2.xy *= rot(t * 0.6);
    p2.xz *= rot(t * 0.4 + PI * 0.5);
    float ring_2 = sdf_torus(p2, torusParams2);

    // Third ring - erratic
    vec3 p3 = p;
    // QUIRK 5: Third ring has jittery rotation on high brightness
    float jitter = brightness > 0.7 ? sin(pc.time * 50.0) * 0.05 : 0.0;
    p3.yz *= rot(t * 0.9 + jitter);
    p3.xy *= rot(t * 0.5);
    p3.xz *= rot(t * 0.6 + PI);

    // QUIRK 6: Third ring occasionally disappears
    float ring_3 = sdf_torus(p3, torusParams3);
    if(pc.osc_ch1 > 0.8 && sin(pc.time * 10.0) > 0.5) {
        ring_3 += 1.0; // Push it far away
    }

    // Variable smoothness based on OSC
    float smoothness = 0.3 + modulation * 0.2 + pc.osc_ch2 * 0.3;
    float combined = smooth_union(ring_1, smooth_union(ring_2, ring_3, smoothness), smoothness);

    // QUIRK 7: Random spikes/protrusions
    if(pc.note_count % 5u == 0u) {
        float spikePos = sin(p.x * 20.0 + p.y * 20.0 + pc.time * 5.0);
        if(spikePos > 0.95) {
            combined -= 0.05; // Create bumps
        }
    }

    di = min2(di, vec2(combined, 1.0));
    return di;
}

// Ray marching
vec2 trace(vec3 ro, vec3 rd) {
    vec3 p = ro;
    vec2 di;
    float td = 0.0;

    glow = vec3(0.0);

    for(int i = 0; i < MAX_STEPS; i++) {
        if(td >= MAX_DIST) break;

        di = sdf(p);

        if(di.x < EPSILON) {
            return vec2(td, di.y);
        }

        p += di.x * rd;

        // Subtle glow accumulation
        float glowFactor = (1.0 - sat(di.x / 0.5)) * 0.02;
        glow += vec3(glowFactor);

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

// Main rendering
vec3 render(vec2 uv) {
    // Camera setup - FIXED: camera stays centered
    float camDist = 3.5 - pc.cc1 * 0.5;
    vec3 ro = vec3(0.0, 0.0, -camDist);

    // Subtle orbit instead of drift
    float camRotSpeed = 0.1;
    ro.x = sin(pc.time * camRotSpeed) * 0.5;
    ro.y = cos(pc.time * camRotSpeed * 0.7) * 0.3;

    // Mouse control
    if(pc.mouse_pressed > 0u) {
        float mx = (float(pc.mouse_x) / float(pc.render_w) - 0.5) * TAU;
        float my = (float(pc.mouse_y) / float(pc.render_h) - 0.5) * PI;
        vec3 mouseRot = ro;
        mouseRot.xz = ro.xz * cos(mx) + ro.zy * sin(mx);
        mouseRot.y = ro.y * cos(my) - ro.z * sin(my);
        ro = mouseRot;
    }

    // Look at center
    vec3 target = vec3(0.0);
    vec3 forward = normalize(target - ro);
    vec3 right = normalize(cross(vec3(0, 1, 0), forward));
    vec3 up = cross(forward, right);

    vec3 rd = normalize(forward + uv.x * right + uv.y * up);
    vec3 lo = vec3(0.5, 1.0, -0.5); // Light position

    vec2 tdi = trace(ro, rd);

    if(tdi.x > 0.0) {
        vec3 p = ro + rd * tdi.x;
        vec3 n = get_normal(p);

        // Basic lighting
        vec3 ld = normalize(lo - p);
        vec3 reflection = reflect(rd, n);

        // Diffuse lighting
        float diff = max(dot(n, ld), 0.0);

        // Sharp specular
        float spec = pow(max(dot(reflection, ld), 0.0), 32.0);

        // QUIRK 8: Inverted lighting sometimes
        if(pc.osc_ch2 > 0.7 && sin(pc.time * 8.0) > 0.5) {
            diff = 1.0 - diff; // Invert diffuse
        }

        // Base monochrome value
        float mono = diff * 0.8 + spec * 0.5;

        // Edge detection for sharp lines
        float edge = 1.0 - abs(dot(n, -rd));
        edge = pow(edge, 3.0);
        mono += edge * 0.3;

        // QUIRK 9: Stripes pattern
        float stripes = sin(p.y * 30.0 + pc.time * 5.0) > 0.0 ? 1.0 : 0.8;
        mono *= stripes;

        // Start with black and white
        vec3 color = vec3(mono);

        // QUIRK 10: Double neon burst effect
        if(pc.note_velocity > 0.7) {
            vec3 neon = neonBurst(dot(n, rd), pc.note_velocity);

            // Neon affects edges more
            neon *= (1.0 + edge * 2.0);

            // QUIRK: Secondary neon flash offset in time
            vec3 neon2 = neonBurst(dot(n, rd) + 0.5, pc.note_velocity);
            neon2 *= sin(pc.time * 15.0) > 0.0 ? 1.0 : 0.0;

            // Mix both neons
            color = mix(color, color + neon + neon2 * 0.5, sat(pc.note_velocity - 0.7) * 3.0);
        }

        // QUIRK 11: Random bright flashes
        if(fract(pc.time * 0.3) < 0.02 && pc.note_velocity > 0.5) {
            color += vec3(1.0);
        }

        // High contrast
        color = smoothstep(0.1, 0.9, color);

        // Add subtle glow
        color += glow * 0.5;

        return color;
    }

    // Clean black background with subtle glow
    return glow * 0.3;
}

void main() {
    vec2 resolution = vec2(float(pc.render_w), float(pc.render_h));
    vec2 uv = (fragUV - 0.5) * 2.0;
    uv.x *= resolution.x / resolution.y;

    vec3 c = render(uv);

    // Final contrast adjustment
    c = pow(c, vec3(1.0 / 2.2));

    // Sharp vignette for focus
    float vignette = 1.0 - smoothstep(0.5, 1.5, length(uv));
    c *= vignette;

    // Threshold for pure black/white with neon
    if(pc.cc74 > 0.5) {
        float threshold = 0.5;
        c = step(threshold, c);
    }

    outColor = vec4(sat(c), 1.0);
}