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

// Quality settings
#ifndef QUALITY_LEVEL
#define QUALITY_LEVEL 2
#endif

// Global variables
float iTime;
vec2 iResolution;
vec2 iMouse;
vec2 bsMo = vec2(0);
float audioEnergy = 0.0;
float audioMod = 0.0;
float audioBright = 0.0;
float audioBend = 0.0;

// Constants
#define PI 3.14159265359
#define TAU 6.28318530718

// Smooth step functions
float smootherstep(float edge0, float edge1, float x) {
    x = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0);
}

// Create wave pattern
float wavePattern(vec2 p, float freq, float phase, float amp) {
    return sin(p.y * freq + phase) * amp;
}

// Create the interference pattern
float interferencePattern(vec2 uv) {
    vec2 p = uv;

    // Base frequency modulated by audio
    float baseFreq = 8.0 + audioEnergy * 4.0;

    // Multiple wave sources for interference
    float pattern = 0.0;

    // Primary horizontal waves
    float wave1 = sin(p.y * baseFreq + iTime * 2.0) * 0.5;
    float wave2 = sin(p.y * baseFreq * 1.1 + iTime * 2.2 + PI * 0.5) * 0.5;

    // Create interference in the center column
    float centerDist = abs(p.x);
    float centerInfluence = 1.0 - smoothstep(0.0, 0.3, centerDist);

    // Vertical modulation for the center distortion
    float vertMod = sin(p.y * baseFreq * 0.5 + iTime * 1.5 + audioMod * PI);
    vertMod *= centerInfluence;

    // Audio-reactive wave displacement
    float displacement = 0.0;
    displacement += sin(p.y * baseFreq * 2.0 + iTime * 3.0) * audioEnergy * 0.1;
    displacement += cos(p.y * baseFreq * 0.7 - iTime * 1.8) * audioMod * 0.1;

    // OSC inputs create additional modulation
    displacement += pc.osc_ch1 * sin(p.y * baseFreq * 1.5 + iTime) * 0.05;
    displacement += pc.osc_ch2 * cos(p.y * baseFreq * 0.8 - iTime * 1.2) * 0.05;

    // Combine waves with displacement
    p.x += displacement;
    p.x += vertMod * 0.2;

    // Main pattern with audio-reactive frequency
    pattern = sin(p.y * baseFreq + p.x * baseFreq * 0.3);

    // Add interference from pitch bend
    pattern += sin(p.y * baseFreq * 1.2 + audioBend * PI) * 0.3;

    // Vertex energy creates local distortions
    pattern += sin(p.y * baseFreq * 3.0 + vertexEnergy * TAU) * vertexEnergy * 0.2;

    // Note count creates harmonic variations
    float harmonic = float(pc.note_count % 8u + 1u);
    pattern += sin(p.y * baseFreq * harmonic * 0.25 + iTime) * 0.2;

    // Create sharp transitions
    float sharpness = 3.0 + audioBright * 2.0;
    pattern = sin(pattern * sharpness);

    return pattern;
}

// Create concentric wave effect from edges
float edgeWaves(vec2 uv) {
    vec2 p = uv;

    // Distance from edges
    float edgeDist = min(
    min(abs(p.x - 1.0), abs(p.x + 1.0)),
    min(abs(p.y - 1.0), abs(p.y + 1.0))
    );

    // Create waves emanating from edges
    float waveFreq = 15.0 + audioEnergy * 10.0;
    float waves = sin(edgeDist * waveFreq - iTime * 3.0);

    // Modulate with audio
    waves *= 1.0 + audioMod * 0.5;

    return waves;
}

// Main color calculation
vec3 getColor(vec2 uv) {
    // Get interference pattern
    float pattern = interferencePattern(uv);

    // Get edge waves
    float edges = edgeWaves(uv);

    // Combine patterns
    float combined = mix(pattern, edges, 0.3 * audioBright);

    // Create threshold for black/white/orange pattern
    float threshold1 = -0.3 + audioEnergy * 0.2;
    float threshold2 = 0.3 - audioEnergy * 0.2;

    vec3 col;

    // Three-color system based on the original image
    if (combined < threshold1) {
        // Black
        col = vec3(0.0);
    } else if (combined > threshold2) {
        // Orange (with audio modulation)
        col = vec3(1.0, 0.4, 0.1);
        col = mix(col, vec3(1.0, 0.3, 0.0), audioMod);

        // Note count shifts hue
        float hueShift = float(pc.note_count % 6u) / 6.0;
        col = mix(col, vec3(1.0, 0.5, 0.2), hueShift);
    } else {
        // White
        col = vec3(1.0);

        // Slight tint based on brightness
        col = mix(col, vec3(0.95, 0.95, 1.0), audioBright * 0.2);
    }

    // Add subtle gradient based on position
    float gradient = length(uv) * 0.1;
    col *= 1.0 - gradient * (1.0 - audioEnergy * 0.5);

    return col;
}

// Rotation matrix
mat2 rot(float a) {
    float c = cos(a);
    float s = sin(a);
    return mat2(c, -s, s, c);
}

void main() {
    // Setup resolution and time
    iResolution = (pc.render_w > 0u && pc.render_h > 0u)
    ? vec2(pc.render_w, pc.render_h)
    : vec2(800.0, 600.0);

    // Cache audio parameters
    audioEnergy = clamp(pc.note_velocity, 0.0, 1.0);
    audioMod = clamp(pc.cc1, 0.0, 1.0);
    audioBright = clamp(pc.cc74, 0.0, 1.0);
    audioBend = clamp(pc.pitch_bend, -1.0, 1.0);

    // Time with audio modulation
    float timeScale = mix(0.8, 2.0, audioMod);
    iTime = pc.time * timeScale;

    // Mouse setup
    iMouse = vec2(float(pc.mouse_x), float(pc.mouse_y));

    // Interactive control
    if (pc.mouse_pressed > 0u) {
        bsMo = (iMouse - 0.5 * iResolution.xy) / iResolution.y;
    } else {
        bsMo = vec2(pc.osc_ch1, pc.osc_ch2) * 0.5;
    }

    // UV coordinates
    vec2 q = fragUV;
    vec2 uv = (fragUV - 0.5) * 2.0;
    uv.x *= iResolution.x / iResolution.y;

    // Scale based on energy (zoom effect)
    float zoom = 1.0 + audioEnergy * 0.3;
    uv /= zoom;

    // Add rotation based on pitch bend
    uv *= rot(audioBend * 0.3);

    // Add mouse/OSC offset
    uv += bsMo * 0.5;

    // Get base color
    vec3 col = getColor(uv);

    // Add shimmer effect based on brightness
    if (audioBright > 0.5) {
        float shimmer = sin(iTime * 20.0 + uv.y * 50.0) * 0.05;
        shimmer *= (audioBright - 0.5) * 2.0;
        col += vec3(shimmer);
    }

    // Create pulsing effect with energy
    col *= 0.9 + audioEnergy * 0.1 * sin(iTime * 10.0);

    // Add strobe effect for high energy
    if (audioEnergy > 0.7) {
        float strobe = step(0.5, fract(iTime * 8.0));
        col = mix(col, 1.0 - col, strobe * (audioEnergy - 0.7) * 2.0);
    }

    // Vignette
    float vignette = 1.0 - length(uv * 0.5) * 0.3;
    vignette = pow(vignette, 0.5 - audioEnergy * 0.2);
    col *= vignette;

    // Add slight chromatic aberration for movement
    if (audioMod > 0.3) {
        vec2 caOffset = vec2(0.002, 0.0) * (audioMod - 0.3);
        vec3 colR = getColor(uv + caOffset);
        vec3 colB = getColor(uv - caOffset);
        col.r = colR.r;
        col.b = colB.b;
    }

    // Contrast adjustment based on brightness
    float contrast = 1.0 + audioBright * 0.5;
    col = pow(col, vec3(1.0 / contrast));

    // Final color output
    outColor = vec4(col, 1.0);
}