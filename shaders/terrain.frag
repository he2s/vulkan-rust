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

#define PI 3.14159265359

// --------------------------------------------------------
// Noise functions
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
// Wave interference pattern
// --------------------------------------------------------

float waveSource(vec2 p, vec2 center, float freq, float phase, float amplitude) {
    float dist = length(p - center);
    return amplitude * sin(dist * freq + phase);
}

void main() {
    // Get resolution
    vec2 iResolution = (pc.render_w > 0u && pc.render_h > 0u)
    ? vec2(pc.render_w, pc.render_h)
    : vec2(800.0, 600.0);

    vec2 fragCoord = fragUV * iResolution;
    vec2 uv = (-iResolution.xy + 2.0 * fragCoord.xy) / iResolution.y;

    // Audio-reactive parameters
    float energyLevel = clamp(pc.note_velocity, 0.0, 1.0);
    float pitchFactor = clamp(pc.pitch_bend, -1.0, 1.0);
    float modulation = clamp(pc.cc1, 0.0, 1.0);
    float brightness = clamp(pc.cc74, 0.0, 1.0);

    // Time with audio reactivity
    float timeScale = mix(1.0, 3.0, modulation);
    float iTime = pc.time * timeScale;

    // Wave sources - positions and properties change with audio
    vec2 source1 = vec2(-0.5, 0.0);
    vec2 source2 = vec2(0.5, 0.0);
    vec2 source3 = vec2(0.0, 0.4);
    vec2 source4 = vec2(0.0, -0.4);

    // OSC/Mouse input moves wave sources
    if (pc.osc_ch1 != 0.0 || pc.osc_ch2 != 0.0) {
        source1 += vec2(pc.osc_ch1, pc.osc_ch2) * 0.3;
        source2 += vec2(-pc.osc_ch1, pc.osc_ch2) * 0.3;
    } else if (pc.mouse_pressed > 0u) {
        vec2 mouseNorm = vec2(float(pc.mouse_x), float(pc.mouse_y)) / iResolution;
        vec2 mouseOffset = (mouseNorm - 0.5) * 0.8;
        source1 += mouseOffset;
        source2 -= mouseOffset;
    }

    // Add orbital motion to sources
    float orbitalSpeed = mix(0.5, 2.0, energyLevel);
    float orbit1 = iTime * orbitalSpeed;
    float orbit2 = iTime * orbitalSpeed * 1.618; // Golden ratio

    source1 += vec2(cos(orbit1), sin(orbit1)) * 0.2;
    source2 += vec2(cos(orbit2), sin(orbit2)) * 0.15;
    source3 += vec2(cos(orbit1 * 0.7), sin(orbit1 * 0.7)) * 0.1;
    source4 += vec2(cos(orbit2 * 0.8), sin(orbit2 * 0.8)) * 0.12;

    // Wave frequencies based on audio
    float baseFreq = mix(15.0, 40.0, brightness);
    float freq1 = baseFreq;
    float freq2 = baseFreq * (1.0 + pitchFactor * 0.5);
    float freq3 = baseFreq * 0.7;
    float freq4 = baseFreq * 1.3;

    // Wave phases
    float phase1 = iTime * mix(2.0, 8.0, energyLevel);
    float phase2 = iTime * mix(1.5, 6.0, energyLevel) + PI * 0.25;
    float phase3 = iTime * mix(1.8, 7.0, energyLevel) + PI * 0.5;
    float phase4 = iTime * mix(2.2, 9.0, energyLevel) + PI * 0.75;

    // Amplitudes based on note count and energy
    float amp1 = mix(0.3, 1.0, energyLevel);
    float amp2 = mix(0.25, 0.8, energyLevel);
    float amp3 = (pc.note_count > 1u) ? mix(0.2, 0.6, modulation) : 0.0;
    float amp4 = (pc.note_count > 2u) ? mix(0.15, 0.5, brightness) : 0.0;

    // Calculate interference pattern
    float wave = 0.0;
    wave += waveSource(uv, source1, freq1, phase1, amp1);
    wave += waveSource(uv, source2, freq2, phase2, amp2);
    wave += waveSource(uv, source3, freq3, phase3, amp3);
    wave += waveSource(uv, source4, freq4, phase4, amp4);

    // Add some standing wave patterns
    float standingH = sin(uv.x * mix(8.0, 20.0, brightness) + iTime * 2.0) *
    cos(uv.y * mix(6.0, 15.0, modulation) + iTime * 1.5) * 0.3;
    float standingV = cos(uv.x * mix(10.0, 25.0, modulation) + iTime * 2.5) *
    sin(uv.y * mix(8.0, 18.0, brightness) + iTime * 1.8) * 0.3;

    wave += standingH + standingV;

    // Add noise for organic texture
    float noise1 = tetraNoise(vec3(uv * 4.0, iTime * 0.1)) * 0.2;
    float noise2 = tetraNoise(vec3(uv * 12.0, iTime * 0.05)) * 0.1;

    wave += noise1 + noise2;

    // Distance-based modulation (creates circular patterns)
    float centerDist = length(uv);
    float distMod = sin(centerDist * mix(8.0, 20.0, brightness) - iTime * 3.0) * 0.2;
    wave += distMod;

    // Pitch bend creates asymmetric distortion
    if (abs(pitchFactor) > 0.1) {
        float skew = pitchFactor * 0.5;
        wave += sin(uv.x * 15.0 + skew) * sin(uv.y * 12.0 - skew) * 0.3 * abs(pitchFactor);
    }

    // Convert to isolines
    float spacing = mix(1./20., 1./50., brightness);
    float lines = mod(wave * 0.5 + 0.5, spacing) / spacing;
    lines = min(lines * 2., 1.) - max(lines * 2. - 1., 0.);
    lines /= fwidth(wave / spacing);
    lines /= 2.;

    // Audio-reactive line weight with center bias
    float weight = mix(1.0, 4.0, energyLevel);
    weight *= mix(0.5, 1.5, smoothstep(1.0, 0.2, centerDist)); // Thicker in center
    weight *= iResolution.y / 287.;

    lines -= weight - 1.;

    // High modulation creates dramatic inversions
    if (modulation > 0.8) {
        float inversionAmount = (modulation - 0.8) * 5.0;
        lines = mix(lines, 1.0 - lines, inversionAmount);
    }

    // Edge fading for focus
    float edgeFade = smoothstep(1.8, 1.0, centerDist);
    lines *= edgeFade;

    // Brightness affects contrast
    lines = mix(lines * 0.7, lines, brightness);

    outColor = vec4(vec3(lines), 1.0);
}