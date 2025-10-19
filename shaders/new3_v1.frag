#version 450

// Push constants matching your application (SAME AS ORIGINAL)
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

// Quality settings - same framework as original
#ifndef QUALITY_LEVEL
#define QUALITY_LEVEL 2
#endif

#if QUALITY_LEVEL == 0
const float POLE_LIMIT = 8.0;
const int COLOR_SAMPLES = 1;
#elif QUALITY_LEVEL == 1
const float POLE_LIMIT = 12.0;
const int COLOR_SAMPLES = 2;
#else
const float POLE_LIMIT = 15.0;
const int COLOR_SAMPLES = 3;
#endif

// Keep original shader's define
#define CORRECT_STREAMLINES
#define saturate(x) clamp(x, 0., 1.)

// Global variables matching original framework
float iTime;
vec2 iResolution;
vec2 iMouse;
vec2 bsMo = vec2(0);
float audioEnergy = 0.0;
float audioMod = 0.0;
float audioBright = 0.0;
float audioBend = 0.0;

// Constants
const float PI = 3.1419;

// Original shader functions with audio modifications
float rand(float n) {
    return fract(sin(n) * 43758.5453123);
}

vec2 force(vec2 p, vec2 pole) {
    // Original optimized version by Fabrice
    p -= pole;
    // Audio modulation affects force strength
    float forceScale = 1.0 + audioEnergy * 0.5;
    return p / (dot(p, p) * forceScale);
}

float calcVelocity(vec2 p) {
    vec2 velocity = vec2(0);
    vec2 pole;
    vec2 f;
    float o, r, m;
    float flip = 1.;
    float j = 0.;

    // Audio affects number of poles (complexity)
    float poleCount = mix(8.0, POLE_LIMIT, audioBright);

    for (float i = 0.; i < POLE_LIMIT; i++) {
        if(i >= poleCount) break;

        r = rand(i / POLE_LIMIT) - 0.5;
        m = rand(i + 1.) - 0.5;

        // Audio modulation affects pole movement speed
        float timeScale = (iTime + (23.78 * 1000.)) * 2.;
        timeScale *= (1.0 + audioMod * 2.0); // Mod wheel speeds up/slows down
        m *= timeScale;

        o = i + r + m;

        // Create pole position with audio influence
        float angleOffset = o / POLE_LIMIT * PI * 2.;
        angleOffset += audioBend * PI; // Pitch bend rotates field

        // Note count creates spiral patterns
        float spiralFactor = 1.0 + float(pc.note_count % 8u) * 0.1;
        float radius = 1.0 + sin(i * 0.5 + iTime) * audioEnergy * 0.3;

        pole = vec2(
        sin(angleOffset) * radius * spiralFactor,
        cos(angleOffset) * radius
        );

        // OSC inputs create additional pole offset
        pole += vec2(pc.osc_ch1, pc.osc_ch2) * 0.2;

        // Vertex energy creates local distortions
        pole += vec2(sin(vertexEnergy * PI), cos(vertexEnergy * PI)) * 0.1;

        f = force(p, pole);
        flip *= -1.;
        velocity -= f * flip;
        j += atan(f.x, f.y) * flip;
    }

    velocity = normalize(velocity);

    #ifdef CORRECT_STREAMLINES
    return j;
    #endif
    return atan(velocity.x, velocity.y);
}

vec2 dir(float a) {
    return vec2(sin(a), cos(a));
}

float calcDerivitive(float a, vec2 p) {
    vec2 v = dir(a);
    float n = 2. / iResolution.x;

    // Reduce samples for lower quality
    float d = 0.;
    d += length(v - dir(calcVelocity(p + vec2(0, n))));
    d += length(v - dir(calcVelocity(p + vec2(n, 0))));

    #if QUALITY_LEVEL > 0
    d += length(v - dir(calcVelocity(p + vec2(n, n))));
    d += length(v - dir(calcVelocity(p + vec2(n, -n))));
    d /= 4.;
    #else
    d /= 2.;
    #endif

    return d;
}

// Audio-reactive color generation
vec3 getStreamColor(float a, vec2 p, float lines) {
    vec3 baseColor;

    // Angle-based color with audio influence
    float colorAngle = a / PI + iTime * 0.1;

    // Note velocity affects color vibrancy
    float vibrancy = 0.5 + audioEnergy * 0.5;

    // Create HSV-like color
    vec3 hsv = vec3(
    colorAngle + audioBend * 0.2,  // Hue affected by pitch bend
    vibrancy,                       // Saturation from velocity
    lines                           // Value from line intensity
    );

    // Convert to RGB (simplified HSV to RGB)
    float c = hsv.y * hsv.z;
    float h = mod(hsv.x * 6.0, 6.0);
    float x = c * (1.0 - abs(mod(h, 2.0) - 1.0));

    if (h < 1.0) baseColor = vec3(c, x, 0);
    else if (h < 2.0) baseColor = vec3(x, c, 0);
    else if (h < 3.0) baseColor = vec3(0, c, x);
    else if (h < 4.0) baseColor = vec3(0, x, c);
    else if (h < 5.0) baseColor = vec3(x, 0, c);
    else baseColor = vec3(c, 0, x);

    // Note count creates color variations
    float noteHue = float(pc.note_count % 8u) * 0.125;
    vec3 noteColor = mix(
    vec3(0.2, 0.5, 1.0),  // Blue
    vec3(1.0, 0.3, 0.5),  // Pink
    noteHue
    );

    baseColor = mix(baseColor, noteColor, 0.3);

    // OSC channels add to color
    baseColor.rg += vec2(pc.osc_ch1, pc.osc_ch2) * 0.2;

    // CC74 (brightness) affects overall luminance
    baseColor *= (0.7 + audioBright * 0.6);

    return baseColor;
}

void main() {
    // Setup resolution and time (matching original framework)
    iResolution = (pc.render_w > 0u && pc.render_h > 0u)
    ? vec2(pc.render_w, pc.render_h)
    : vec2(800.0, 600.0);

    // Cache audio parameters (same as original)
    audioEnergy = clamp(pc.note_velocity, 0.0, 1.0);
    audioMod = clamp(pc.cc1, 0.0, 1.0);
    audioBright = clamp(pc.cc74, 0.0, 1.0);
    audioBend = clamp(pc.pitch_bend, -1.0, 1.0);

    // Time with audio modulation (same as original)
    float timeScale = mix(0.8, 1.5, audioMod);
    iTime = pc.time * timeScale;

    // Mouse setup (same as original)
    iMouse = vec2(float(pc.mouse_x), float(pc.mouse_y));

    vec2 q = fragUV;

    // Interactive control (same as original)
    if (pc.mouse_pressed > 0u) {
        bsMo = (iMouse - 0.5 * iResolution.xy) / iResolution.y;
    } else {
        bsMo = vec2(pc.osc_ch1, pc.osc_ch2) * 0.5;
    }

    // Transform coordinates (from original streamlines shader)
    vec2 p = (-iResolution.xy + 2.0 * fragUV * iResolution.xy) / iResolution.x;

    // Audio affects zoom level
    float zoomScale = 3.0 - audioEnergy * 1.0; // Zoom in with energy
    p *= zoomScale;

    // Add mouse/OSC position offset
    p += bsMo * 2.0;

    // Calculate streamlines
    float a = calcVelocity(p);
    float deriv = calcDerivitive(a, p);

    // Audio-reactive line spacing
    float spacing = 1.0 / (30.0 - audioEnergy * 15.0); // More lines with energy

    a /= PI * 2.;

    float lines = fract(a / spacing);

    // Create stripes
    lines = min(lines, 1. - lines) * 2.;

    // Thin stripes into lines
    lines /= deriv / spacing;

    // Maintain constant line width across different screen sizes
    lines -= iResolution.x * 0.0005;

    // Audio affects line thickness
    lines *= (1.0 + audioMod * 0.5);

    // Don't blow out contrast when blending
    lines = saturate(lines);

    // Create disc with audio-reactive size
    float discRadius = 1.0 + audioEnergy * 0.5;
    float disc = length(p) - discRadius;
    disc /= fwidth(disc);
    disc = saturate(disc);

    lines = mix(1. - lines, lines, disc);

    // Add glow effect based on energy
    float glow = 1.0 - length(p) / (3.0 * zoomScale);
    glow = saturate(glow);
    glow = pow(glow, 2.0 - audioEnergy);
    lines += glow * audioEnergy * 0.3;

    lines = pow(lines, 1./2.2);

    // Get color instead of grayscale
    vec3 col = getStreamColor(a, p, lines);

    // Add energy pulse
    col += vec3(0.1, 0.2, 0.3) * audioEnergy * 0.2 * (1.0 + sin(iTime * 10.0) * 0.5);

    // Post-processing matching original style

    // Audio-reactive color grading
    vec3 powerCurve = vec3(0.55, 0.65, 0.6);
    powerCurve = mix(powerCurve, vec3(0.45, 0.5, 0.55), audioBright);
    col = pow(col, powerCurve);

    // Note count affects tint (same as original)
    if(pc.note_count > 0u) {
        float noteInfluence = float(pc.note_count % 4u) * 0.25;
        vec3 tint = mix(vec3(1.0, 0.97, 0.9), vec3(0.9, 0.95, 1.1), noteInfluence);
        col *= tint;
    }

    // Vignette with energy influence (same calculation as original)
    float vignetteBase = 16.0 * q.x * q.y * (1.0 - q.x) * (1.0 - q.y);
    float vignettePower = 0.12 - audioEnergy * 0.05;
    float vignette = pow(vignetteBase, vignettePower);
    col *= vignette * 0.7 + 0.3;

    // Add vertex energy influence
    col *= 1.0 + vertexEnergy * 0.2;

    outColor = vec4(col, 1.0);
}