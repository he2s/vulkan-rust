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

// Glitch random function
float glitchRand(vec2 co) {
    return fract(sin(dot(co.xy, vec2(12.9898, 78.233))) * 43758.5453);
}

// Audio-reactive color palette with glitch
vec3 palette(float x) {
    // Shift palette based on note count
    x += float(pc.note_count % 8u) * 0.125;

    // Glitch effect on high energy
    if(pc.note_velocity > 0.8) {
        float glitchAmount = glitchRand(vec2(floor(pc.time * 10.0), x));
        x += glitchAmount * 2.0 - 1.0;
    }

    // Audio-reactive palette parameters
    vec3 a = vec3(0.5, 0.5, 0.0); // Base color (fire)
    vec3 b = vec3(0.5 + pc.cc74 * 0.3); // Amplitude
    vec3 c = vec3(0.1, 0.5, 0.0) + vec3(pc.osc_ch1, pc.osc_ch2, 0.0) * 0.3;
    vec3 d = vec3(0.0, pc.pitch_bend * 0.3, pc.note_velocity * 0.2);

    vec3 color = a + b * cos(TAU * (c * x + d));

    // Digital color quantization glitch
    if(pc.cc1 > 0.7) {
        float levels = 4.0 + floor(pc.cc1 * 8.0);
        color = floor(color * levels) / levels;
    }

    return color;
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

// Main SDF scene with glitch distortions
vec2 sdf(vec3 p) {
    vec2 di = vec2(120.0, -1.0);

    // Audio-reactive parameters
    float energy = clamp(pc.note_velocity, 0.0, 1.0);
    float modulation = clamp(pc.cc1, 0.0, 1.0);
    float brightness = clamp(pc.cc74, 0.0, 1.0);

    // Glitch displacement
    if(energy > 0.6) {
        float glitchTime = floor(pc.time * 15.0) / 15.0;
        vec3 glitchOffset = vec3(
        glitchRand(vec2(glitchTime, 0.0)),
        glitchRand(vec2(glitchTime, 1.0)),
        glitchRand(vec2(glitchTime, 2.0))
        ) - 0.5;
        p += glitchOffset * energy * 0.1;

        // Digital stepping
        float gridSize = 0.05 / (1.0 + modulation);
        p = floor(p / gridSize) * gridSize;
    }

    // Audio-reactive smoothness
    float smoothness = 0.2 + modulation * 0.4;

    // Time with glitch stuttering
    float t = pc.time * (0.5 + energy * 1.0);
    if(pc.cc1 > 0.5) {
        // Time glitch - random time jumps
        float glitchJump = floor(glitchRand(vec2(floor(t * 5.0), 0.0)) * 3.0);
        t += glitchJump;
    }

    // Audio-reactive torus size with glitch scaling
    float ringRadius = 1.0 + vertexEnergy * 0.2;
    float ringThickness = 0.15 + energy * 0.05;

    // Random ring size glitches
    if(pc.note_velocity > 0.9) {
        ringRadius *= 0.5 + glitchRand(vec2(floor(pc.time * 20.0), 0.0));
        ringThickness *= 0.5 + glitchRand(vec2(floor(pc.time * 20.0), 1.0));
    }

    vec2 torusParams = vec2(ringRadius, ringThickness);

    // First ring with glitchy rotation
    vec3 p1 = p;
    float glitchRot1 = pc.pitch_bend + step(0.7, energy) * PI * glitchRand(vec2(floor(t * 10.0), 0.0));
    p1.yz *= rot(t + glitchRot1);
    p1.xy *= rot(t * 1.2);
    p1.xz *= rot(t * PI / 2.0 + PI / 3.0);
    float ring_1 = sdf_torus(p1, torusParams);

    // Second ring with random axis flips
    vec3 p2 = p;
    if(modulation > 0.6 && glitchRand(vec2(floor(t * 8.0), 1.0)) > 0.5) {
        p2.xy = p2.yx; // Axis swap glitch
    }
    p2.yz *= rot(t + pc.pitch_bend);
    p2.xy *= rot(t * 1.2);
    p2.xz *= rot(t * PI / 2.0 + PI / 3.0);
    p2.yz *= rot(t * PI / 2.0 + PI / 5.0 + pc.osc_ch1 * PI);
    float ring_2 = sdf_torus(p2, torusParams * (1.0 + modulation * 0.2));

    // Third ring with corruption
    vec3 p3 = p;
    // Bit-crush style position corruption
    if(brightness > 0.7) {
        float crushLevel = 16.0;
        p3 = floor(p3 * crushLevel) / crushLevel;
    }
    p3.yz *= rot(t + pc.pitch_bend);
    p3.xy *= rot(t * 1.2);
    p3.xz *= rot(t * PI / 2.0 + PI / 3.0);
    p3.xy *= rot(t * PI / 2.0 - PI / 7.0 + pc.osc_ch2 * PI);
    float ring_3 = sdf_torus(p3, torusParams * (1.0 - modulation * 0.1));

    // Combine with potential glitch artifacts
    float combined = smooth_union(ring_1, smooth_union(ring_2, ring_3, smoothness), smoothness);

    // Random geometry holes (data corruption effect)
    if(energy > 0.85) {
        float holeNoise = glitchRand(p.xy + vec2(floor(pc.time * 30.0)));
        if(holeNoise > 0.9) {
            combined += 0.5; // Create holes in geometry
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

// Main rendering function with glitch effects
vec3 render(vec2 uv) {
    // Camera setup with glitch jitter
    float camDist = 3.0 - pc.cc1 * 0.5;
    vec3 ro = vec3(0.0, 0.0, -camDist);

    // Camera glitch shake
    if(pc.note_velocity > 0.75) {
        float shakeAmount = (pc.note_velocity - 0.75) * 0.2;
        ro.x += (glitchRand(vec2(pc.time * 100.0, 0.0)) - 0.5) * shakeAmount;
        ro.y += (glitchRand(vec2(pc.time * 100.0, 1.0)) - 0.5) * shakeAmount;
    }

    // Mouse/OSC camera rotation
    if(pc.mouse_pressed > 0u) {
        float mx = (float(pc.mouse_x) / float(pc.render_w) - 0.5) * TAU;
        float my = (float(pc.mouse_y) / float(pc.render_h) - 0.5) * PI;
        ro.xz *= rot(mx);
        ro.yz *= rot(my);
    }

    vec3 rd = normalize(vec3(uv, 1.0));

    // Ray direction glitch distortion
    if(pc.cc74 > 0.8) {
        float distortTime = floor(pc.time * 20.0);
        rd.xy += (vec2(
        glitchRand(vec2(distortTime, uv.y)),
        glitchRand(vec2(distortTime + 1.0, uv.x))
        ) - 0.5) * 0.02 * pc.cc74;
    }

    vec3 lo = ro; // Light origin

    vec2 tdi = trace(ro, rd);

    if(tdi.x > 0.0) {
        vec3 p = ro + rd * tdi.x;
        vec3 n = get_normal(p);

        // Glitched normal for corrupted lighting
        if(pc.osc_ch1 > 0.7) {
            n.x += (glitchRand(vec2(floor(pc.time * 40.0), 0.0)) - 0.5) * 0.3;
            n = normalize(n);
        }

        // Iridescence effect with glitch modulation
        vec3 cd = normalize(ro - p);
        vec3 ld = normalize(lo - p);
        vec3 reflection = reflect(rd, n);

        // Audio-reactive perturbation with glitch spikes
        float perturbStrength = 10.0 + pc.note_velocity * 15.0;
        if(pc.note_velocity > 0.85) {
            perturbStrength *= 1.0 + glitchRand(vec2(floor(pc.time * 30.0), 0.0)) * 3.0;
        }
        vec3 perturbation = 0.05 * sin(p * perturbStrength);

        // Calculate iridescence with potential color corruption
        float iridValue = dot(n + perturbation, cd) * 2.0;
        vec3 iridescence = palette(iridValue);

        // Data corruption - random color channel swaps
        if(pc.cc1 > 0.8 && glitchRand(vec2(floor(pc.time * 15.0), 0.0)) > 0.7) {
            iridescence = iridescence.bgr; // Swap color channels
        }

        // Specular with glitch flashing
        float specular = sat(dot(reflection, ld));
        float specIntensity = 0.1 + pc.cc74 * 0.2;

        // Random specular spikes (electrical glitches)
        if(glitchRand(vec2(floor(pc.time * 60.0), dot(p, p))) > 0.95) {
            specIntensity *= 5.0;
        }

        specular *= specIntensity * pow(pos(sin(specular * 20.0 - 3.0)) + 0.1, 32.0);
specular += specIntensity * pow(sat(dot(reflection, ld)) + 0.3, 8.0);

// Shadow/ambient with random blackouts
float shadow = pow(sat(dot(n, vec3(0.0, 1.0, 0.0)) * 0.5 + 1.2), 3.0);
if(pc.osc_ch2 > 0.8 && glitchRand(vec2(floor(pc.time * 10.0), 0.0)) > 0.8) {
    shadow *= 0.1; // Random darkness
}

// Combine lighting
vec3 color = iridescence * shadow + specular + glow;

// Digital artifacts - scan lines
float scanline = step(0.5, fract(p.y * 50.0 + pc.time * 10.0));
if(pc.cc1 > 0.6) {
    color *= 0.8 + 0.2 * scanline;
}

// Chromatic aberration glitch
if(pc.note_velocity > 0.7) {
    float chromaShift = 0.01 * pc.note_velocity;
    color.r *= 1.0 + chromaShift;
    color.b *= 1.0 - chromaShift;
}

// Energy flash with corruption
if(pc.note_velocity > 0.7) {
    vec3 flashColor = vec3(0.2, 0.3, 0.5) * (pc.note_velocity - 0.7);
    // Random color inversions
    if(glitchRand(vec2(floor(pc.time * 25.0), 0.0)) > 0.9) {
        flashColor = vec3(1.0) - flashColor;
    }
    color += flashColor;
}

return color;
}

// Background with glitchy glow
vec3 bgColor = vec3(0.0) + glow;

// Static noise on background
if(pc.cc74 > 0.5) {
    float noise = glitchRand(uv + vec2(pc.time * 100.0));
    bgColor += vec3(noise) * 0.05 * pc.cc74;
}

return bgColor;
}

void main() {
    vec2 resolution = vec2(float(pc.render_w), float(pc.render_h));
    vec2 uv = (fragUV - 0.5) * 2.0;
    uv.x *= resolution.x / resolution.y;

    vec3 c = render(uv);

    // Subtle vignette
    float vignette = 1.0 - length(uv) * 0.3;
    c *= vignette;

    // Output with saturation
    outColor = vec4(sat(c), 1.0);
}