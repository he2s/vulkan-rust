#version 450

// Push constants matching your application
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

// Quality settings - define QUALITY_LEVEL before including this shader
// 0 = Low (fastest), 1 = Medium, 2 = High (original quality)
#ifndef QUALITY_LEVEL
#define QUALITY_LEVEL 2
#endif

// Quality-dependent constants
#if QUALITY_LEVEL == 0
const int MAX_RAYMARCH_STEPS = 50;
const int FRACTAL_ITERATIONS = 4;
#elif QUALITY_LEVEL == 1
const int MAX_RAYMARCH_STEPS = 80;
const int FRACTAL_ITERATIONS = 5;
#else
const int MAX_RAYMARCH_STEPS = 130;
const int FRACTAL_ITERATIONS = 6;
#endif

// Global variables for shader state
float prm1 = 0.;
vec2 bsMo = vec2(0);
float iTime;
vec2 iResolution;
vec2 iMouse;

// Optimized utility functions with inline hints
mat2 rot(in float a) {
    float c = cos(a), s = sin(a);
    return mat2(c, s, -s, c);
}

// Pre-computed matrix constant
const mat3 m3 = mat3(0.33338, 0.56034, -0.71817,
-0.87887, 0.32651, -0.15323,
0.15162, 0.69596, 0.61339) * 1.93;

// Inline dot product for 2D vectors
float mag2(vec2 p) { return dot(p, p); }

// Optimized linear step
float linstep(in float mn, in float mx, in float x) {
    return clamp((x - mn)/(mx - mn), 0., 1.);
}

// Optimized displacement with pre-computed constants
const float DISP_FREQ1 = 0.22;
const float DISP_FREQ2 = 0.175;

vec2 disp(float t) {
    // Pre-compute audio parameters once
    float energy = clamp(pc.note_velocity, 0.0, 1.0);
    float modulation = clamp(pc.cc1, 0.0, 1.0);

    float ampScale = 1.0 + energy * 0.5 + modulation * 0.3;
    // Optimized: single sin/cos calculation
    return vec2(sin(t * DISP_FREQ1), cos(t * DISP_FREQ2)) * 2.0 * ampScale;
}

vec2 map(vec3 p) {
    // Cache all audio parameters at once
    float energy = clamp(pc.note_velocity, 0.0, 1.0);
    float modulation = clamp(pc.cc1, 0.0, 1.0);
    float brightness = clamp(pc.cc74, 0.0, 1.0);
    float bend = clamp(pc.pitch_bend, -1.0, 1.0);

    vec3 p2 = p;
    p2.xy -= disp(p.z).xy;

    // Optimized rotation calculation
    float rotAmount = sin(p.z + iTime) * (0.1 + prm1 * 0.05) + iTime * 0.09 + bend * 0.3;
    p.xy *= rot(rotAmount);

    float cl = mag2(p2.xy);
    float d = 0.;

    // Single scaling operation
    p *= 0.61;

    float z = 1.;
    float trk = 1.;

    // Pre-compute displacement amplitude
    float dspAmp = 0.1 + prm1 * 0.2 + energy * 0.15;

    // Determine iteration count based on quality and brightness
    #if QUALITY_LEVEL == 0
    int iterations = 4;
    #else
    int iterations = min(FRACTAL_ITERATIONS, 4 + int(brightness * 2.0));
    #endif

    // Optimized fractal loop with pre-computed values
    for(int i = 0; i < FRACTAL_ITERATIONS; i++) {
        if(i >= iterations) break;

        // Combine calculations to reduce operations
        float vertInfluence = 1.0 + vertexEnergy * 0.1;
        vec3 sinOffset = sin(p.zxy * (0.75 * trk) + (iTime * trk * 0.8));
        p += sinOffset * dspAmp * vertInfluence;

        // Single trigonometric calculation
        d -= abs(dot(cos(p), sin(p.yzx)) * z);

        // Update iteration variables
        z *= 0.57;
        trk *= 1.4;
        p = p * m3;
    }

    // Optimized density calculation
    d = abs(d + prm1 * 3.) + prm1 * 0.3 - 2.5 + bsMo.y;
    d *= (1.0 - energy * 0.2);

    return vec2(d + cl * 0.2 + 0.25, cl);
}

vec4 render(in vec3 ro, in vec3 rd, float time) {
    vec4 rez = vec4(0);

    // Cache audio parameters once
    float energy = clamp(pc.note_velocity, 0.0, 1.0);
    float modulation = clamp(pc.cc1, 0.0, 1.0);
    float brightness = clamp(pc.cc74, 0.0, 1.0);

    const float ldst = 8.;
    vec3 lpos = vec3(disp(time + ldst) * 0.5, time + ldst);
    float t = 1.5;
    float fogT = 0.;

    // Adjust max steps based on quality
    #if QUALITY_LEVEL == 0
    int maxSteps = 50;
    #else
    int maxSteps = min(MAX_RAYMARCH_STEPS, 80 + int(brightness * 50.0));
    #endif

    for(int i = 0; i < MAX_RAYMARCH_STEPS; i++) {
        if(i >= maxSteps) break;
        if(rez.a > 0.99) break;

        vec3 pos = ro + t * rd;
        vec2 mpv = map(pos);
        float den = clamp(mpv.x - 0.3, 0., 1.) * 1.12;
        float dn = clamp((mpv.x + 2.), 0., 3.);

        vec4 col = vec4(0);
        if (mpv.x > 0.6) {
            // Pre-compute base color
            vec3 baseColor = vec3(5., 0.4, 0.2);

            // Optimized note influence calculation
            if(pc.note_count > 0u) {
                float noteInfluence = float(pc.note_count) * 0.125; // Multiplication instead of division
                baseColor = mix(baseColor, vec3(3., 0.8, 5.), clamp(noteInfluence, 0., 0.5));
            }

            // Add OSC color influence
            baseColor.xy += vec2(pc.osc_ch1, pc.osc_ch2) * 0.5;

            // Optimized color calculation - pre-compute common terms
            float posZ = sin(pos.z * 0.4) * 0.5 + 1.8;
            col = vec4(sin(baseColor + mpv.y * 0.1 + posZ) * 0.5 + 0.5, 0.08);

            // Combine density multiplications
            float denCubed = den * den * den;
            col *= denCubed;
            col.rgb *= linstep(4., -2.5, mpv.x) * 2.3;

            // Optimized lighting calculation
            #if QUALITY_LEVEL > 0
            // Full lighting for medium/high quality
            float dif = clamp((den - map(pos + .8).x) / 9., 0.001, 1.);
            dif += clamp((den - map(pos + .35).x) / 2.5, 0.001, 1.);
            #else
            // Single sample for low quality
            float dif = clamp((den - map(pos + 0.5).x) / 6., 0.001, 1.);
            #endif

            // Pre-compute and combine light colors
            vec3 lightColor = vec3(0.005, .045, .075) + 1.5 * vec3(0.033, 0.07, 0.03) * dif;
            lightColor *= (1.0 + brightness * 0.5);
            col.xyz *= den * lightColor;

            // Energy glow
            col.xyz += col.xyz * energy * 0.75;
        }

        // Optimized fog calculation
        float fogC = exp(t * 0.2 - 2.2);
        vec4 fogColor = vec4(0.06, 0.11, 0.11, 0.1);
        fogColor.rgb = mix(fogColor.rgb, vec3(0.08, 0.05, 0.12), modulation);
        col.rgba += fogColor * clamp(fogC - fogT, 0., 1.);
        fogT = fogC;

        // Accumulation
        rez = rez + col * (1. - rez.a);

        // Dynamic step size with pre-computed factor
        t += clamp(0.5 - dn * dn * 0.05, 0.09, 0.3);
    }

    return clamp(rez, 0.0, 1.0);
}

// Optimized saturation calculation
float getsat(vec3 c) {
    float mi = min(min(c.x, c.y), c.z);
    float ma = max(max(c.x, c.y), c.z);
    return (ma - mi) / (ma + 1e-7);
}

// Optimized color interpolation
vec3 iLerp(in vec3 a, in vec3 b, in float x) {
    vec3 ic = mix(a, b, x) + vec3(1e-6, 0., 0.);
    float sd = abs(getsat(ic) - mix(getsat(a), getsat(b), x));

    // Pre-compute common terms
    float icX2 = 2. * ic.x;
    float icY2 = 2. * ic.y;
    float icZ2 = 2. * ic.z;

    vec3 dir = normalize(vec3(icX2 - ic.y - ic.z,
    icY2 - ic.x - ic.z,
    icZ2 - ic.y - ic.x));

    float lgt = dot(vec3(1.0), ic);
    float ff = dot(dir, normalize(ic));
    ic += 1.5 * dir * sd * ff * lgt;

    return clamp(ic, 0., 1.);
}

void main() {
    // Setup resolution and time - optimized branching
    iResolution = (pc.render_w > 0u && pc.render_h > 0u)
    ? vec2(pc.render_w, pc.render_h)
    : vec2(800.0, 600.0);

    // Time with audio modulation
    float modulation = clamp(pc.cc1, 0.0, 1.0);
    float timeScale = mix(0.8, 1.5, modulation);
    iTime = pc.time * timeScale;

    // Mouse setup
    iMouse = vec2(float(pc.mouse_x), float(pc.mouse_y));

    vec2 q = fragUV;
    vec2 p = (fragUV * iResolution - 0.5 * iResolution.xy) / iResolution.y;

    // Interactive control - optimized conditional
    if (pc.mouse_pressed > 0u) {
        bsMo = (iMouse - 0.5 * iResolution.xy) / iResolution.y;
    } else {
        bsMo = vec2(pc.osc_ch1, pc.osc_ch2) * 0.5;
    }

    float time = iTime * 3.;
    vec3 ro = vec3(0, 0, time);

    // Audio-reactive camera movement
    float energy = clamp(pc.note_velocity, 0.0, 1.0);
    ro += vec3(sin(iTime) * 0.5 * (1.0 + energy),
    sin(iTime * 1.) * 0. + vertexEnergy * 0.2,
    0);

    float dspAmp = .85 + energy * 0.15;
    ro.xy += disp(ro.z) * dspAmp;
    float tgtDst = 3.5;

    vec3 target = normalize(ro - vec3(disp(time + tgtDst) * dspAmp, time + tgtDst));
    ro.x -= bsMo.x * 2.;

    // Optimized camera matrix construction
    vec3 rightdir = normalize(cross(target, vec3(0, 1, 0)));
    vec3 updir = normalize(cross(rightdir, target));
    rightdir = normalize(cross(updir, target));
    vec3 rd = normalize((p.x * rightdir + p.y * updir) * 1. - target);

    // Add pitch bend to camera rotation
    float bend = clamp(pc.pitch_bend, -1.0, 1.0);
    rd.xy *= rot(-disp(time + 3.5).x * 0.2 + bsMo.x + bend * 0.5);

    // Modulation parameter
    prm1 = smoothstep(-0.4, 0.4, sin(iTime * 0.3));
    prm1 = mix(prm1, 1.0, energy * 0.3);

    // Render the scene
    vec4 scn = render(ro, rd, time);
    vec3 col = scn.rgb;

    // Color interpolation with modulation control
    float lerpAmount = clamp(1. - prm1, 0.05, 1.);
    lerpAmount = mix(lerpAmount, 0.5, modulation);
    col = iLerp(col.bgr, col.rgb, lerpAmount);

    // Audio-reactive color grading
    float brightness = clamp(pc.cc74, 0.0, 1.0);
    vec3 powerCurve = vec3(.55, 0.65, 0.6);
    powerCurve = mix(powerCurve, vec3(0.45, 0.5, 0.55), brightness);
    col = pow(col, powerCurve) * vec3(1., .97, .9);

    // Note count affects tint - optimized modulo
    if(pc.note_count > 0u) {
        float noteInfluence = float(pc.note_count % 4u) * 0.25; // Pre-comp division
        vec3 tint = mix(vec3(1.0, 0.97, 0.9), vec3(0.9, 0.95, 1.1), noteInfluence);
        col *= tint;
    }

    // Vignette with energy influence - optimized power calculation
    float vignetteBase = 16.0 * q.x * q.y * (1.0 - q.x) * (1.0 - q.y);
    float vignettePower = 0.12 - energy * 0.05;
    float vignette = pow(vignetteBase, vignettePower);
    col *= vignette * 0.7 + 0.3;

    outColor = vec4(col, 1.0);
}