#version 450

// ---------------- Push constants / IO ----------------
layout(push_constant) uniform PushConstants {
    float time; uint mouse_x; uint mouse_y; uint mouse_pressed;
    float note_velocity; float pitch_bend; float cc1; float cc74;
    uint  note_count; uint last_note; float osc_ch1; float osc_ch2;
    uint  render_w; uint render_h;
} pc;

layout(location=0) in vec2 frag_uv;          // 0..1
layout(location=1) in vec2 frag_screen_pos;  // optional
layout(location=0) out vec4 out_color;

// ---------------- Configuration ----------------
#define AUDIO_AFFECTS_CIRCUIT
#define AUDIO_AFFECTS_GLITCH
#define TV_GLITCHES

// Circuit raymarch settings
#define ITS 8
#define RAYMARCH_STEPS 50

// ---------------- Constants ----------------
#define PI 3.14159265359
#define TAU 6.283185307179586

// ---------------- Helpers ----------------
float saturate(float x){ return clamp(x, 0.0, 1.0); }
vec3  saturate(vec3  x){ return clamp(x, 0.0, 1.0); }

// Get normalized audio intensity
float getAudioIntensity() {
    return saturate((pc.note_velocity + pc.osc_ch1 + pc.osc_ch2) * 0.5);
}

// Hash functions for randomness
float hash11(float n){ return fract(sin(n*104729.0)*43758.5453123); }
float hash21(vec2 p){ return fract(sin(dot(p, vec2(127.1,311.7))) * 43758.5453123); }

// Noise functions from original
float nse(float x) {
    return fract(sin(x * 297.9712) * 90872.2961);
}

float nseI(float x) {
    float fl = floor(x);
    return mix(nse(fl), nse(fl + 1.0), smoothstep(0.0, 1.0, fract(x)));
}

float fbm(float x) {
    return nseI(x) * 0.5 + nseI(x * 2.0) * 0.25 + nseI(x * 4.0) * 0.125;
}

// 2D rotation
vec2 rotate(vec2 p, float a) {
    return vec2(p.x * cos(a) - p.y * sin(a), p.x * sin(a) + p.y * cos(a));
}

// ---------------- Circuit Generation ----------------
vec2 circuit(vec3 p) {
    p = mod(p, 2.0) - 1.0;
    float w = 1e38;
    vec3 cut = vec3(1.0, 0.0, 0.0);
    vec3 e1 = vec3(-1.0);
    vec3 e2 = vec3(1.0);
    float rnd = 0.23;
    float pos, plane;
    float fact = 0.9;
    float j = 0.0;

    #ifdef AUDIO_AFFECTS_CIRCUIT
    float audio = getAudioIntensity();
    // Audio affects the circuit complexity
    fact = mix(0.9, 0.7, audio * 0.5);
    #endif

    for(int i = 0; i < ITS; i++) {
        pos = mix(dot(e1, cut), dot(e2, cut), (rnd - 0.5) * fact + 0.5);
        plane = dot(p, cut) - pos;
        if(plane > 0.0) {
            e1 = mix(e1, vec3(pos), cut);
            rnd = fract(rnd * 9827.5719);
            cut = cut.yzx;
        }
        else {
            e2 = mix(e2, vec3(pos), cut);
            rnd = fract(rnd * 15827.5719);
            cut = cut.zxy;
        }
        j += step(rnd, 0.2);
        w = min(w, abs(plane));
    }
    return vec2(j / float(ITS - 1), w);
}

float scene(vec3 p) {
    vec2 cir = circuit(p);
    float audio = getAudioIntensity();

    // Base circuit visualization
    float base = exp(-100.0 * cir.y);

    // Animated pulsing based on circuit data
    float pulse = sin(p.z * 10.0 + pc.time * -5.0 + cir.x * 10.0) * 0.5 + 0.5;

    #ifdef AUDIO_AFFECTS_CIRCUIT
    // Audio modulates the pulse intensity
    pulse = mix(pulse, pulse * 1.5, audio);
    // Audio adds extra harmonics
    pulse += sin(p.z * 20.0 + pc.time * -8.0) * audio * 0.2;
    #endif

    float circuit_glow = pow(cir.x * 1.8 * pulse, 8.0);

    return base + circuit_glow;
}

// ---------------- TV Glitch Effects ----------------
#ifdef TV_GLITCHES

// Scanline effect
float scanline(vec2 uv, float time) {
    float line = sin(uv.y * 800.0 + time * 5.0) * 0.04;
    line *= sin(uv.y * 300.0 - time * 2.0) * 0.02 + 0.98;
    return line;
}

// Horizontal sync issues
vec2 horizontalSync(vec2 uv, float time, float audio) {
    float syncAmount = 0.0;

    // Random horizontal jumps
    float jumpTime = floor(time * 8.0);
    float jumpRand = hash11(jumpTime);
    if(jumpRand < 0.1 + audio * 0.2) {
        float jumpY = hash11(jumpTime + 100.0);
        if(abs(uv.y - jumpY) < 0.1) {
            syncAmount = (hash11(jumpTime + 200.0) - 0.5) * 0.2;
        }
    }

    // Rolling horizontal distortion
    float roll = sin(time * 2.0 + uv.y * 5.0) * 0.01;
    roll *= step(0.95, sin(time * 0.3)) * (1.0 + audio);

    uv.x += syncAmount + roll;
    return uv;
}

// Vertical hold problems
vec2 verticalHold(vec2 uv, float time, float audio) {
    float holdTime = time * 0.5;
    float holdAmount = 0.0;

    // Occasional vertical roll
    if(sin(holdTime) > 0.98 - audio * 0.1) {
        holdAmount = sin(holdTime * 20.0) * 0.05;
        uv.y = fract(uv.y + holdAmount);
    }

    return uv;
}

// Chromatic aberration (color separation)
vec3 chromaticAberration(vec2 uv, float amount) {
    return vec3(
    uv.x - amount,
    uv.x,
    uv.x + amount
    );
}

// Static/snow noise
float tvStatic(vec2 uv, float time, float intensity) {
    vec2 noise_uv = uv * 100.0 + time * 100.0;
    float static_noise = hash21(floor(noise_uv));
    return static_noise * intensity;
}

// Ghosting effect (image retention)
float ghosting(float current, float previous, float amount) {
    return mix(current, max(current, previous * 0.9), amount);
}

// VHS tracking lines
float trackingLines(vec2 uv, float time) {
    float lines = 0.0;
    float lineTime = time * 0.1;

    // Horizontal tracking lines
    for(int i = 0; i < 3; i++) {
        float linePos = hash11(floor(lineTime) + float(i)) * 2.0 - 1.0;
        float lineWidth = 0.01 + hash11(floor(lineTime) + float(i) + 100.0) * 0.02;
        lines += smoothstep(lineWidth, 0.0, abs(uv.y - linePos * 0.5 - 0.5)) * 0.5;
    }

    return lines;
}

// Signal interference bands
float interference(vec2 uv, float time, float audio) {
    float inter = 0.0;

    // Moving interference bands
    float bandY = sin(time * 0.7) * 0.5 + 0.5;
    float bandWidth = 0.05 + audio * 0.1;
    float bandIntensity = smoothstep(bandWidth, 0.0, abs(uv.y - bandY));

    // Add noise to the band
    if(bandIntensity > 0.0) {
        inter = sin(uv.x * 200.0 + time * 50.0) * bandIntensity;
        inter *= hash21(vec2(floor(uv.x * 50.0), floor(time * 10.0)));
    }

    return inter;
}

#endif // TV_GLITCHES

// ---------------- Main Raymarching ----------------
vec3 raymarchCircuit(vec2 uv) {
    float audio = getAudioIntensity();

    // Camera setup
    vec3 ro = vec3(0.0, pc.time * 0.2, 0.1);
    vec3 rd = normalize(vec3(uv, 0.9));

    // Add audio-reactive camera movement
    #ifdef AUDIO_AFFECTS_CIRCUIT
    float camSpeed = mix(0.2, 0.35, audio);
    ro.y = pc.time * camSpeed;

    // Extra rotation with audio
    float rotSpeed = mix(0.1, 0.2, audio);
    ro.xz = rotate(ro.xz, pc.time * rotSpeed + pc.pitch_bend);
    rd.xz = rotate(rd.xz, pc.time * 0.2 + sin(pc.time) * audio * 0.1);
    #else
    ro.xz = rotate(ro.xz, pc.time * 0.1);
    rd.xz = rotate(rd.xz, pc.time * 0.2);
    #endif

    ro.xy = rotate(ro.xy, 0.2);
    rd.xy = rotate(rd.xy, 0.2);

    // Raymarch accumulation
    float acc = 0.0;
    vec3 r = ro + rd * 0.5;

    for(int i = 0; i < RAYMARCH_STEPS; i++) {
        float noise_offset = nse(r.x) * 0.03;

        #ifdef AUDIO_AFFECTS_CIRCUIT
        // Audio adds more noise/variation
        noise_offset *= (1.0 + audio * 0.5);
        #endif

        acc += scene(r + noise_offset);
        r += rd * 0.015;
    }

    // Color grading
    vec3 col = pow(vec3(acc * 0.04), vec3(0.2, 0.6, 2.0) * 8.0) * 2.0;

    #ifdef AUDIO_AFFECTS_CIRCUIT
    // Audio affects color temperature
    vec3 audioColor = vec3(1.0, 0.7, 0.5);
    col = mix(col, col * audioColor, audio * 0.3);
    #endif

    // Clamp and apply flickering
    col = clamp(col, vec3(0.0), vec3(1.0));
    col *= fbm(pc.time * 6.0) * 2.0;

    return col;
}

// ---------------- Main ----------------
void main() {
    vec2 iResolution = vec2(float(pc.render_w), float(pc.render_h));
    vec2 fragCoord = frag_uv * iResolution;

    // Get base UV coordinates
    vec2 uv = fragCoord.xy / iResolution.xy;
    vec2 suv = uv; // Save original UV for effects

    // Convert to -1 to 1 range with aspect correction
    uv = 2.0 * uv - 1.0;
    uv.x *= iResolution.x / iResolution.y;

    float audio = getAudioIntensity();

    #ifdef TV_GLITCHES
    // Apply TV distortion effects to UV
    vec2 distorted_uv = uv;

    // Sync issues get worse with audio
    distorted_uv = horizontalSync(distorted_uv, pc.time, audio);
    distorted_uv = verticalHold(distorted_uv, pc.time, audio);

    // Use distorted UV for main rendering
    vec3 col = raymarchCircuit(distorted_uv);

    // Apply scanlines
    float scan = scanline(suv, pc.time);
    col *= 0.9 + scan;

    // Apply chromatic aberration (increases with audio)
    if(audio > 0.3) {
        vec3 chroma_uvs = chromaticAberration(distorted_uv, audio * 0.003);
        vec3 col_r = raymarchCircuit(vec2(chroma_uvs.r, distorted_uv.y));
        vec3 col_b = raymarchCircuit(vec2(chroma_uvs.z, distorted_uv.y));
        col.r = col_r.r;
        col.b = col_b.b;
    }

    //// Add TV static
    //float staticIntensity = 0.02 + audio * 0.03;
    //// Occasional static bursts
    //if(hash11(floor(pc.time * 2.0)) < 0.05 + audio * 0.1) {
    //    staticIntensity = 0.2 + audio * 0.3;
    //}
    //col += tvStatic(suv, pc.time, staticIntensity);

    // Add tracking lines
    col *= 1.0 - trackingLines(suv, pc.time) * 0.3;

    // Add interference
    float inter = interference(suv, pc.time, audio);
    col = mix(col, vec3(0.1, 0.15, 0.1), inter * 0.5);

    // Occasional complete signal loss
    float signalLoss = step(0.98 - audio * 0.05, hash11(floor(pc.time * 3.0)));
    if(signalLoss > 0.5) {
        col = vec3(tvStatic(suv, pc.time * 10.0, 1.0)) * 0.1;
    }

    // Vignette effect (CRT screen edges)
    float vignette = 1.0 - length(suv - 0.5) * 0.7;
    vignette = pow(saturate(vignette), 1.5);
    col *= vignette;

    // Phosphor glow simulation
    col = mix(col, smoothstep(0.0, 1.0, col), 0.5);

    #else
    // No TV glitches - clean rendering
    vec3 col = raymarchCircuit(uv);
    #endif

    // Gamma correction
    col = pow(col, vec3(1.0 / 2.2));

    // CC controls for brightness and gamma (from your setup)
    float bright = mix(0.85, 1.55, saturate(pc.cc74));
    float gammaC = 1.0 + 0.7 * saturate(pc.cc1);
    col = pow(max(col * bright, 0.0), vec3(gammaC));

    // MIDI note color tinting
    if(pc.note_count > 0u) {
        float noteTint = float(pc.last_note) / 127.0;
        vec3 tintColor = vec3(
        sin(noteTint * PI) * 0.2,
        sin(noteTint * PI + PI/3.0) * 0.2,
        sin(noteTint * PI + 2.0*PI/3.0) * 0.2
        );
        col += tintColor * pc.note_velocity;
    }

    out_color = vec4(clamp(col, 0.0, 3.0), 1.0);
}