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

void main() {
    // Resolution setup
    vec2 iResolution = (pc.render_w > 0u && pc.render_h > 0u)
    ? vec2(pc.render_w, pc.render_h)
    : vec2(800.0, 600.0);

    // Convert UV to Shadertoy-style coordinates
    vec2 u = fragUV * iResolution;

    // Audio-reactive parameters
    float energy = clamp(pc.note_velocity, 0.0, 1.0);
    float modulation = clamp(pc.cc1, 0.0, 1.0);
    float brightness = clamp(pc.cc74, 0.0, 1.0);
    float bend = clamp(pc.pitch_bend, -1.0, 1.0);

    // Time with audio modulation
    float timeScale = mix(0.7, 1.3, modulation);
    float iTime = pc.time * timeScale;

    // Mouse/OSC interactivity
    vec2 mouseNorm = vec2(float(pc.mouse_x), float(pc.mouse_y)) / iResolution;
    vec2 interactiveOffset = vec2(0.0);

    if (pc.mouse_pressed > 0u) {
        interactiveOffset = (mouseNorm - 0.5) * 0.1;
    } else if (abs(pc.osc_ch1) + abs(pc.osc_ch2) > 0.01) {
        interactiveOffset = vec2(pc.osc_ch1, pc.osc_ch2) * 0.05;
    }

    // Apply interactive offset before normalization
    u += interactiveOffset * iResolution.y;

    // === Port of the compact shader ===
    vec2 v = iResolution.xy;
    u = 0.2 * (u + u - v) / v.y;

    // Scale UV based on energy for zoom effect
    u *= mix(1.0, 0.85, energy * 0.5);

    // Apply pitch bend as rotation
    if (abs(bend) > 0.01) {
        float angle = bend * 0.5;
        float c = cos(angle);
        float s = sin(angle);
        u = mat2(c, -s, s, c) * u;
    }

    vec4 z = vec4(1, 2, 3, 0);
    vec4 o = z;

    // Main iteration loop with audio modulation
    float a = 0.5;
    float t = iTime;

    // Adjust iteration count based on energy (19 base, up to 21 with high energy)
    int maxIters = 19 + int(energy * 2.0);

    for (int iter = 1; iter < maxIters; iter++) {
        float i = float(iter);

        // Core calculation from original
        o += (1.0 + cos(z + t))
        / length((1.0 + i * dot(v, v))
        * sin(1.5 * u / (0.5 - dot(u, u)) - 9.0 * u.yx + t));

        // Update v with modulation influence
        t += 1.0;
        a += 0.03 * mix(1.0, 1.2, modulation);
        v = cos(t - 7.0 * u * pow(a, i)) - 5.0 * u;

        // Transform u with rotation matrix
        vec4 angles = vec4(0, 11, 33, 0);
        float timemod = 0.02 * t * mix(1.0, 1.5, brightness);
        mat2 rot = mat2(cos(i + timemod - angles));
        u *= rot;

        // Complex feedback with audio reactivity
        float feedback = 40.0 * mix(1.0, 1.5, energy);
        u += tanh(feedback * dot(u, u) * cos(100.0 * u.yx + t)) / 200.0
        + 0.2 * a * u
        + cos(4.0 / exp(dot(o, o) / 100.0) + t) / 300.0;
    }

    // Final color calculation
    o = 25.6 / (min(o, 13.0) + 164.0 / o) - dot(u, u) / 250.0;

    // Enhance with vertex energy
    o *= (1.0 + vertexEnergy * 0.3);

    // Apply brightness control
    o = mix(o, o * 1.5, brightness * 0.7);

    // Color grading based on note count
    if (pc.note_count > 0u) {
        float noteInfluence = float(pc.note_count) / 10.0;
        o.rgb = mix(o.rgb, o.gbr, clamp(noteInfluence, 0.0, 0.3));
    }

    // Saturation boost with CC1
    vec3 color = o.rgb;
    float lum = dot(color, vec3(0.299, 0.587, 0.114));
    color = mix(vec3(lum), color, 1.0 + modulation * 0.5);

    // Final output with gamma correction
    color = pow(max(color, 0.0), vec3(2.75));
    outColor = vec4(color, 1.0);
}