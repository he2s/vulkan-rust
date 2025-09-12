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

const float TAU = 6.283185307179586;

// iquilez palette
vec3 palette(float t) {
    vec3 a = vec3(0.5);
    vec3 b = vec3(0.5);
    vec3 c = vec3(1.0);
    vec3 d = vec3(0.263, 0.416, 0.557);
    return a + b * cos(TAU * (c * t + d));
}

void main() {
    // Resolution fallback
    vec2 iResolution = (pc.render_w > 0u && pc.render_h > 0u)
    ? vec2(pc.render_w, pc.render_h)
    : vec2(800.0, 600.0);

    // Shadertoy-style fragCoord and time
    vec2 fragCoord = fragUV * iResolution;

    float energy     = clamp(pc.note_velocity, 0.0, 1.0);
    float modulation = clamp(pc.cc1,          0.0, 1.0); // speed/warp
    float brightness = clamp(pc.cc74,         0.0, 1.0); // glow/intensity
    float bend       = clamp(pc.pitch_bend,  -1.0, 1.0);

    // Time: slightly audio-reactive
    float timeScale = mix(0.6, 1.4, modulation);
    float iTime = pc.time * timeScale;

    // Base UV (preserve aspect like the original)
    vec2 uv  = (fragCoord * 2.0 - iResolution.xy) / iResolution.y;
    vec2 uv0 = uv;

    // Optional interactive offset (mouse or OSC)
    vec2 mouseNorm   = vec2(float(pc.mouse_x), float(pc.mouse_y)) / iResolution;
    vec2 mouseOffset = (mouseNorm - 0.5) * 2.0 * 0.25; // gentle push
    vec2 oscOffset   = vec2(pc.osc_ch1, pc.osc_ch2) * 0.25;

    if (pc.mouse_pressed > 0u) uv += mouseOffset;
    else if (abs(pc.osc_ch1) + abs(pc.osc_ch2) > 0.0) uv += oscOffset;

    // Tile scale reacts a touch to energy and CC74 (cutoff/brightness)
    float baseTile = 1.5;
    float tileBoost = 1.0 + energy * 0.3 + brightness * 0.2;
    float tile = baseTile * tileBoost;

    // Iterations (keep the original 4 look; allow small lift with energy)
    int iters = 4 + int(floor(energy * 1.0)); // 4..5

    vec3 finalColor = vec3(0.0);

    // Port of mainImage loop with minor safety tweaks & ints
    for (int i = 0; i < iters; ++i) {
        float fi = float(i);

        // iq tile fold
        uv = fract(uv * tile) - 0.5;

        float lenUV  = length(uv);
        float lenUV0 = length(uv0);

        float d = lenUV * exp(-lenUV0);

        // Original palette driving; add slight pitch-bend hue drift
        float t = lenUV0 + fi * 0.4 + iTime * 0.4 + bend * 0.2;
        vec3 col = palette(t);

        // Distortion & shaping (preserved)
        d = sin(d * 8.0 + iTime) * 0.125; // /8
        d = abs(d);

        // Power reacts to modulation; clamp to avoid extreme blowouts
        float powK = mix(1.1, 1.35, modulation); // ~1.2 default feel
        d = pow(max(0.0005, 0.01 / max(1e-5, d)), powK);

        finalColor += col * d;
    }

    // Vertex energy boosts brightness a bit
    finalColor *= (1.0 + vertexEnergy * 0.5);

    // CC74-driven glow & gentle clamp to 0..1
    float glow = dot(finalColor, vec3(1.0 / 3.0));
    finalColor += glow * brightness * 0.25;

    // Simple contrast curve
    finalColor = pow(clamp(finalColor, 0.0, 1.0), vec3(2.9));

    outColor = vec4(finalColor, 1.0);
}
