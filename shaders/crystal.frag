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

// Performance settings - can be adjusted via defines or push constants
#define QUALITY_LOW 0
#define QUALITY_MEDIUM 1
#define QUALITY_HIGH 2

// Set default quality (can be overridden by define)
#ifndef QUALITY_LEVEL
#define QUALITY_LEVEL QUALITY_HIGH
#endif

// Quality-based settings
#if QUALITY_LEVEL == QUALITY_HIGH
const int MAX_STEPS = 100;
const int FRACTAL_ITER = 5;
const float STEP_SIZE_MIN = 0.09;
const float STEP_SIZE_MAX = 0.3;
#elif QUALITY_LEVEL == QUALITY_MEDIUM
const int MAX_STEPS = 60;
const int FRACTAL_ITER = 4;
const float STEP_SIZE_MIN = 0.12;
const float STEP_SIZE_MAX = 0.4;
#else // QUALITY_LOW
const int MAX_STEPS = 40;
const int FRACTAL_ITER = 3;
const float STEP_SIZE_MIN = 0.15;
const float STEP_SIZE_MAX = 0.5;
#endif

// Global variables for shader state
float prm1 = 0.;
vec2 bsMo = vec2(0);
float iTime;
vec2 iResolution;

// Optimized rotation matrix - inline for better performance
mat2 rot(float a) {
    float c = cos(a), s = sin(a);
    return mat2(c, s, -s, c);
}

// Pre-computed constant matrix
const mat3 m3 = mat3(0.33338, 0.56034, -0.71817,
-0.87887, 0.32651, -0.15323,
0.15162, 0.69596, 0.61339) * 1.93;

// Optimized displacement function
vec2 disp(float t) {
    // Simplified audio-reactive displacement
    float energy = pc.note_velocity;
    float modulation = pc.cc1;

    // Use approximation for sin/cos
    float t1 = t * 0.22;
    float t2 = t * 0.175;

    // Combine amplitude scaling
    float ampScale = 2.0 + energy * 1.0 + modulation * 0.6;
    return vec2(sin(t1), cos(t2)) * ampScale;
}

// Optimized map function - the heart of the raymarcher
vec2 map(vec3 p) {
    // Cache audio parameters
    float energy = pc.note_velocity;
    float modulation = pc.cc1;
    float bend = pc.pitch_bend;

    vec3 p2 = p;
    p2.xy -= disp(p.z).xy;

    // Simplified rotation
    float rotAmount = sin(p.z + iTime) * 0.15 + iTime * 0.09 + bend * 0.3;
    p.xy *= rot(rotAmount);

    // Use squared length directly
    float cl = dot(p2.xy, p2.xy);
    float d = 0.;

    // Scale once
    p *= 0.61;

    // Simplified fractal calculation
    float z = 1.;
    float trk = 1.;
    float dspAmp = 0.1 + prm1 * 0.2 + energy * 0.15;

    // Unrolled and simplified loop for low quality
    #if QUALITY_LEVEL == QUALITY_LOW
    // Iteration 1
    vec3 sp = sin(p.zxy * 0.75 * trk + iTime * trk * 0.8) * dspAmp;
    p += sp * (1.0 + vertexEnergy * 0.1);
    d -= abs(dot(cos(p), sin(p.yzx))) * z;
    z *= 0.57;
    trk *= 1.4;
    p = p * m3;

    // Iteration 2
    sp = sin(p.zxy * 0.75 * trk + iTime * trk * 0.8) * dspAmp;
    p += sp;
    d -= abs(dot(cos(p), sin(p.yzx))) * z;
    z *= 0.57;
    trk *= 1.4;
    p = p * m3;

    // Iteration 3
    sp = sin(p.zxy * 0.75 * trk + iTime * trk * 0.8) * dspAmp;
    p += sp;
    d -= abs(dot(cos(p), sin(p.yzx))) * z;
    #else
    // Dynamic loop for medium/high quality
    for(int i = 0; i < FRACTAL_ITER; i++) {
        vec3 sp = sin(p.zxy * 0.75 * trk + iTime * trk * 0.8) * dspAmp;
        p += sp * (1.0 + vertexEnergy * 0.1 * float(i == 0));
        d -= abs(dot(cos(p), sin(p.yzx))) * z;
        z *= 0.57;
        trk *= 1.4;
        p = p * m3;
    }
    #endif

    // Simplified density calculation
    d = abs(d + prm1 * 3.) + prm1 * 0.3 - 2.5 + bsMo.y;
    d *= (1.0 - energy * 0.2);

    return vec2(d + cl * 0.2 + 0.25, cl);
}

// Optimized render function
vec4 render(vec3 ro, vec3 rd, float time) {
    vec4 rez = vec4(0);

    // Cache audio parameters
    float energy = pc.note_velocity;
    float modulation = pc.cc1;
    float brightness = pc.cc74;

    // Light position
    vec3 lpos = vec3(disp(time + 8.) * 0.5, time + 8.);

    float t = 1.5;
    float fogT = 0.;

    // Early termination threshold
    const float ALPHA_THRESHOLD = 0.95;

    // Simplified ray marching
    for(int i = 0; i < MAX_STEPS; i++) {
        if(rez.a > ALPHA_THRESHOLD) break;

        vec3 pos = ro + t * rd;
        vec2 mpv = map(pos);

        // Skip empty space quickly
        if(mpv.x > 0.6) {
            float den = clamp(mpv.x - 0.3, 0., 1.) * 1.12;

            // Simplified color calculation
            vec3 baseColor = vec3(5., 0.4, 0.2);

            // Audio color modulation (simplified)
            float noteInfluence = float(pc.note_count) * 0.125; // /8
            baseColor = mix(baseColor, vec3(3., 0.8, 5.), clamp(noteInfluence, 0., 0.5));
            baseColor.xy += vec2(pc.osc_ch1, pc.osc_ch2) * 0.5;

            // Simplified color formula
            vec4 col = vec4(sin(baseColor + mpv.y * 0.1 + sin(pos.z * 0.4) * 0.5 + 1.8) * 0.5 + 0.5, 0.08);
            col *= den * den * den;

            // Simplified lighting (only on medium/high quality)
            #if QUALITY_LEVEL > QUALITY_LOW
            col.rgb *= clamp(4. + 2.5 * mpv.x, 0., 1.) * 2.3;

            // Single light sample instead of two
            float dif = clamp((den - map(pos + 0.5).x) / 5., 0.001, 1.);
            vec3 lightColor = vec3(0.038, 0.115, 0.105) * dif * (1.0 + brightness * 0.5);
            col.xyz *= den * lightColor;
            #else
            // Very simplified lighting for low quality
            col.rgb *= 2.0;
            col.xyz *= den * vec3(0.05, 0.1, 0.1) * (1.0 + brightness * 0.5);
            #endif

            // Energy glow
            col.xyz *= (1.0 + energy * 0.75);

            // Accumulate
            rez = rez + col * (1. - rez.a);
        }

        // Simplified fog (only on medium/high quality)
        #if QUALITY_LEVEL > QUALITY_LOW
        float fogC = exp(t * 0.2 - 2.2);
        vec4 fogColor = vec4(0.06, 0.11, 0.11, 0.1);
        fogColor.rgb = mix(fogColor.rgb, vec3(0.08, 0.05, 0.12), modulation);
        rez += fogColor * clamp(fogC - fogT, 0., 1.) * (1. - rez.a);
        fogT = fogC;
        #endif

        // Adaptive step size
        float dn = clamp(mpv.x + 2., 0., 3.);
        t += clamp(0.5 - dn * dn * 0.05, STEP_SIZE_MIN, STEP_SIZE_MAX);
    }

    return clamp(rez, 0.0, 1.0);
}

// Simplified saturation calculation
float getsat(vec3 c) {
    float mi = min(min(c.x, c.y), c.z);
    float ma = max(max(c.x, c.y), c.z);
    return (ma - mi) / (ma + 1e-7);
}

// Simplified color interpolation
vec3 iLerp(vec3 a, vec3 b, float x) {
    #if QUALITY_LEVEL > QUALITY_LOW
    vec3 ic = mix(a, b, x) + vec3(1e-6, 0., 0.);
    float sd = abs(getsat(ic) - mix(getsat(a), getsat(b), x));
    vec3 dir = normalize(vec3(2. * ic.x - ic.y - ic.z,
    2. * ic.y - ic.x - ic.z,
    2. * ic.z - ic.y - ic.x));
    float lgt = dot(vec3(1.0), ic);
    float ff = dot(dir, normalize(ic));
    ic += 1.5 * dir * sd * ff * lgt;
    return clamp(ic, 0., 1.);
    #else
    // Simple linear interpolation for low quality
    return mix(a, b, x);
    #endif
}

void main() {
    // Setup resolution
    iResolution = (pc.render_w > 0u && pc.render_h > 0u)
    ? vec2(pc.render_w, pc.render_h)
    : vec2(800.0, 600.0);

    // Time with audio modulation (simplified)
    float timeScale = mix(0.8, 1.5, pc.cc1);
    iTime = pc.time * timeScale;

    // Setup coordinates
    vec2 q = fragUV;
    vec2 p = (fragUV * iResolution - 0.5 * iResolution.xy) / iResolution.y;

    // Interactive control
    if (pc.mouse_pressed > 0u) {
        bsMo = (vec2(float(pc.mouse_x), float(pc.mouse_y)) - 0.5 * iResolution.xy) / iResolution.y;
    } else {
        bsMo = vec2(pc.osc_ch1, pc.osc_ch2) * 0.5;
    }

    // Camera setup (simplified)
    float time = iTime * 3.;
    vec3 ro = vec3(0, 0, time);

    // Audio-reactive camera (simplified)
    float energy = pc.note_velocity;
    ro.x += sin(iTime) * 0.5 * (1.0 + energy);
    ro.y += vertexEnergy * 0.2;

    float dspAmp = 0.85 + energy * 0.15;
    ro.xy += disp(ro.z) * dspAmp;

    // Target and camera vectors (simplified)
    vec3 target = normalize(ro - vec3(disp(time + 3.5) * dspAmp, time + 3.5));
    ro.x -= bsMo.x * 2.;

    // Simplified camera matrix
    vec3 rightdir = normalize(vec3(target.y, -target.x, 0));
    vec3 updir = vec3(0, 1, 0);
    vec3 rd = normalize((p.x * rightdir + p.y * updir) - target);

    // Add pitch bend rotation
    float bend = pc.pitch_bend;
    rd.xy *= rot(-disp(time + 3.5).x * 0.2 + bsMo.x + bend * 0.5);

    // Modulation parameter (simplified)
    prm1 = smoothstep(-0.4, 0.4, sin(iTime * 0.3));
    prm1 = mix(prm1, 1.0, energy * 0.3);

    // Render the scene
    vec4 scn = render(ro, rd, time);
    vec3 col = scn.rgb;

    // Color grading (simplified for low quality)
    #if QUALITY_LEVEL > QUALITY_LOW
    float lerpAmount = clamp(1. - prm1, 0.05, 1.);
    lerpAmount = mix(lerpAmount, 0.5, pc.cc1);
    col = iLerp(col.bgr, col.rgb, lerpAmount);

    // Power curve adjustment
    float brightness = pc.cc74;
    vec3 powerCurve = mix(vec3(0.55, 0.65, 0.6), vec3(0.45, 0.5, 0.55), brightness);
    col = pow(col, powerCurve) * vec3(1., 0.97, 0.9);

    // Note count tint
    if(pc.note_count > 0u) {
        float noteInfluence = float(pc.note_count & 3u) * 0.25; // Bitwise AND instead of modulo
        vec3 tint = mix(vec3(1.0, 0.97, 0.9), vec3(0.9, 0.95, 1.1), noteInfluence);
        col *= tint;
    }
    #else
    // Simple color adjustment for low quality
    col = pow(col, vec3(0.55)) * vec3(1., 0.97, 0.9);
    #endif

    // Vignette (simplified)
    float vignette = 16.0 * q.x * q.y * (1.0 - q.x) * (1.0 - q.y);
    vignette = pow(vignette, 0.12 - energy * 0.05);
    col *= vignette * 0.7 + 0.3;

    outColor = vec4(col, 1.0);
}