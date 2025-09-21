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

layout(location = 0) in vec2 frag_uv;
layout(location = 1) in flat uint frag_instance_id;
layout(location = 2) in vec2 frag_screen_pos;

layout(location = 0) out vec4 outColor;

#define PI 3.14159265359

// --------------------------------------------------------
// Simplex(ish) Noise
// --------------------------------------------------------

vec3 hash33(vec3 p) {
    float n = sin(dot(p, vec3(7, 157, 113)));
    return fract(vec3(2097152, 262144, 32768)*n)*2. - 1.;
}

float tetraNoise(in vec3 p) {
    vec3 i = floor(p + dot(p, vec3(0.333333)));
    p -= i - dot(i, vec3(0.166666));
    vec3 i1 = step(p.yzx, p), i2 = max(i1, 1.0-i1.zxy);
    i1 = min(i1, 1.0-i1.zxy);
    vec3 p1 = p - i1 + 0.166666, p2 = p - i2 + 0.333333, p3 = p - 0.5;
    vec4 v = max(0.5 - vec4(dot(p,p), dot(p1,p1), dot(p2,p2), dot(p3,p3)), 0.0);
    vec4 d = vec4(dot(p, hash33(i)), dot(p1, hash33(i + i1)), dot(p2, hash33(i + i2)), dot(p3, hash33(i + 1.)));
    return clamp(dot(d, v*v*v*8.)*1.732 + .5, 0., 1.);
}

// --------------------------------------------------------
// Rectangle distance function
// --------------------------------------------------------

float sRect(vec2 p, vec2 size) {
    vec2 d = abs(p) - size;
    return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}

// --------------------------------------------------------
// Smooth repeat functions
// --------------------------------------------------------

vec2 smoothRepeatStart(float x, float size) {
    return vec2(
    mod(x - size / 2., size),
    mod(x, size)
    );
}

float smoothRepeatEnd(float a, float b, float x, float size) {
    return mix(a, b,
    smoothstep(
    0., 1.,
    sin((x / size) * PI * 2. - PI * .5) * .5 + .5
    )
    );
}

void main() {
    // Get resolution
    vec2 iResolution = (pc.render_w > 0u && pc.render_h > 0u)
    ? vec2(pc.render_w, pc.render_h)
    : vec2(800.0, 600.0);

    vec2 fragCoord = frag_uv * iResolution;

    // Square uv centered and scaled to the screen height
    vec2 uv = (-iResolution.xy + 2. * fragCoord.xy) / iResolution.y;

    // Audio-reactive parameters
    float energyLevel = clamp(pc.note_velocity, 0.0, 1.0);
    float pitchFactor = clamp(pc.pitch_bend, -1.0, 1.0);
    float modulation = clamp(pc.cc1, 0.0, 1.0);
    float brightness = clamp(pc.cc74, 0.0, 1.0);

    // Zoom varies with energy level
    float zoom = mix(2.0, 1.2, energyLevel);
    uv /= zoom;

    // Audio-reactive time scaling
    float timeScale = mix(0.5, 2.0, modulation);
    float iTime = pc.time * timeScale;

    // Repeat size varies with pitch bend
    float repeatSize = mix(3.0, 6.0, abs(pitchFactor));
    float x = uv.x - mod(iTime, repeatSize / 2.);
    float y = uv.y;

    vec2 ab; // two sample points on one axis
    float noise;
    float noiseA, noiseB;

    // Audio-reactive noise scaling
    float noiseIntensity = mix(0.5, 2.0, energyLevel);

    // Blend noise at different frequencies, moving in different directions
    ab = smoothRepeatStart(x, repeatSize);
    noiseA = tetraNoise(16.+vec3(vec2(ab.x, uv.y) * 1.2, 0)) * 0.5 * noiseIntensity;
    noiseB = tetraNoise(16.+vec3(vec2(ab.y, uv.y) * 1.2, 0)) * 0.5 * noiseIntensity;
    noise = smoothRepeatEnd(noiseA, noiseB, x, repeatSize);

    ab = smoothRepeatStart(y, repeatSize / 2.);
    noiseA = tetraNoise(vec3(vec2(uv.x, ab.x) * 0.5, 0)) * 2.0;
    noiseB = tetraNoise(vec3(vec2(uv.x, ab.y) * 0.5, 0)) * 2.0;
    noise *= smoothRepeatEnd(noiseA, noiseB, y, repeatSize / 2.);

    // High frequency detail controlled by CC74
    float detailScale = mix(0.05, 0.15, brightness);
    ab = smoothRepeatStart(x, repeatSize);
    noiseA = tetraNoise(9.+vec3(vec2(ab.x, uv.y) * detailScale, 0)) * 5.;
    noiseB = tetraNoise(9.+vec3(vec2(ab.y, uv.y) * detailScale, 0)) * 5.;
    noise *= smoothRepeatEnd(noiseA, noiseB, x, repeatSize);

    noise *= 0.75;

    // Gradient direction changes with OSC or mouse input
    vec2 gradientDir = vec2(-0.66, 1.0) * 0.4;

    if (pc.osc_ch1 != 0.0 || pc.osc_ch2 != 0.0) {
        // Use OSC input to control gradient direction
        gradientDir.x += pc.osc_ch1 * 0.5;
        gradientDir.y += pc.osc_ch2 * 0.5;
    } else if (pc.mouse_pressed > 0u) {
        // Use mouse input as fallback
        vec2 mouseNorm = vec2(float(pc.mouse_x), float(pc.mouse_y)) / iResolution;
        mouseNorm = (mouseNorm - 0.5) * 2.0;
        gradientDir += mouseNorm * 0.3;
    }

    // Blend with gradient orientation
    noise = mix(noise, dot(uv, gradientDir), 0.6);

    // Audio-reactive line spacing
    float spacing = mix(1./30., 1./80., brightness);
    float lines = mod(noise, spacing) / spacing;

    // Convert sawtooth to triangle wave
    lines = min(lines * 2., 1.) - max(lines * 2. - 1., 0.);

    lines /= fwidth(noise / spacing);
    lines /= 2.;

    // Rectangle distance - size varies with note count and energy
    vec2 rectSize = vec2(0.4, 0.25);

    // Scale rectangle with note count
    if (pc.note_count > 0u) {
        float noteScale = 1.0 + float(pc.note_count) * 0.05;
        rectSize *= noteScale;
    }

    // Deform rectangle with pitch bend
    rectSize.x *= (1.0 + pitchFactor * 0.3);
    rectSize.y *= (1.0 - pitchFactor * 0.2);

    // Position offset with modulation
    vec2 rectPos = uv + vec2(0.0, modulation * 0.1);

    float d = sRect(rectPos, rectSize);

    // Create fuzzy border - sharpness varies with energy
    float borderSharpness = mix(0.02, 0.08, energyLevel);
    float weight = smoothstep(0.0, borderSharpness, d);

    // Audio-reactive line weight
    float innerWeight = mix(3.0, 6.0, energyLevel);
    float outerWeight = mix(0.8, 1.5, brightness);
    weight = mix(innerWeight, outerWeight, weight);

    // Scale weight with resolution
    weight *= iResolution.y / 287.;

    // Offset the line by the weight
    lines -= weight - 1.;

    // Invert for high energy sections
    if (energyLevel > 0.8) {
        lines = 1. - lines;
    }

    // Add some color tinting based on audio
    vec3 color = vec3(lines);

    // Subtle color shifts
    if (modulation > 0.1) {
        color.r *= (1.0 + modulation * 0.2);
        color.b *= (1.0 + brightness * 0.15);
    }

    // Contrast boost for high frequencies
    color = mix(color, color * color, brightness * 0.3);

    // Add per-instance color variation using optimized approach
    float instance_factor = float(frag_instance_id % 16u) / 16.0; // Back to modulo for correctness

    // Pre-computed phase offsets for better performance
    const float phase1 = 2.09439;
    const float phase2 = 4.18879;
    float base_angle = instance_factor * 6.28318;

    vec3 instance_color = vec3(
        0.5 + 0.5 * sin(base_angle),
        0.5 + 0.5 * sin(base_angle + phase1),
        0.5 + 0.5 * sin(base_angle + phase2)
    );
    color *= mix(vec3(1.0), instance_color, 0.3);

    outColor = vec4(color, 1.0);
}