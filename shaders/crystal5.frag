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

    // Time with audio modulation - faster base rate for more dynamism
    float timeScale = mix(0.5, 2.0, modulation);
    float iTime = pc.time * timeScale;

    // Mouse/OSC interactivity
    vec2 mouseNorm = vec2(float(pc.mouse_x), float(pc.mouse_y)) / iResolution;
    vec2 interactiveOffset = vec2(0.0);

    if (pc.mouse_pressed > 0u) {
        interactiveOffset = (mouseNorm - 0.5) * 0.15;
    } else if (abs(pc.osc_ch1) + abs(pc.osc_ch2) > 0.01) {
        interactiveOffset = vec2(pc.osc_ch1, -pc.osc_ch2) * 0.1;
    }

    // Apply interactive offset
    u += interactiveOffset * iResolution.y;

    // === Modified shader with new patterns ===
    vec2 v = iResolution.xy;
    vec2 originalU = 0.2 * (u + u - v) / v.y;
    u = originalU;

    // Stronger zoom with energy
    u *= mix(1.0, 0.6, energy * 0.7);

    // Pitch bend creates spiral distortion
    if (abs(bend) > 0.01) {
        float spiral = bend * 2.0 + energy * 0.5;
        float dist = length(u);
        float angle = atan(u.y, u.x) + spiral * dist;
        u = vec2(cos(angle), sin(angle)) * dist;
    }

    // Initialize with different values for variation
    vec4 z = vec4(0.8, 1.5, 2.2, 0);
    vec4 o = z;

    // Main iteration loop with enhanced audio modulation
    float a = 0.5;
    float t = iTime;

    // More aggressive iteration scaling
    int maxIters = 17 + int(energy * 5.0); // 17-22 iterations

    for (int iter = 1; iter < maxIters; iter++) {
        float i = float(iter);

        // Modified core with extra harmonics
        vec4 colorMod = 1.0 + cos(z * 1.5 + t * 1.2);
        float divisor = length((1.0 + i * dot(v, v))
        * sin(2.0 * u / (0.5 - dot(u, u)) - 11.0 * u.yx + t * 1.3));

        // Add some discontinuity for sharper edges
        divisor = max(divisor, 0.001);
        o += colorMod / divisor;

        // More chaotic v evolution
        t += 1.0;
        a += 0.025 * mix(1.0, 1.8, modulation);
        float chaos = mix(7.0, 12.0, brightness);
        v = cos(t - chaos * u * pow(a, i * 0.9)) - 6.0 * u;

        // Enhanced rotation matrix with phase shifts
        vec4 angles = vec4(0, 13, 37, 0); // Different angles for variation
        float timemod = 0.025 * t * mix(1.0, 2.0, brightness);
        mat2 rot = mat2(cos(i * 1.1 + timemod - angles));
        u *= rot;

        // More aggressive feedback with extra nonlinearity
        float feedback = mix(30.0, 60.0, energy);
        vec2 feedbackU = u;

        // Add wave distortion
        feedbackU += 0.1 * sin(5.0 * u.yx + t * 2.0) * modulation;

        u += tanh(feedback * dot(feedbackU, feedbackU) * cos(150.0 * feedbackU.yx + t * 1.5)) / 250.0
        + 0.18 * a * u
        + cos(5.0 / exp(dot(o, o) / 120.0) + t * 1.2) / 400.0;
    }

    // Different final transformation for sharper contrast
    o = 30.0 / (min(o, 15.0) + 180.0 / o) - dot(u, u) / 200.0;

    // Enhance with vertex energy more dramatically
    o *= (1.0 + vertexEnergy * 0.8);

    // === COLOR MODES ===
    vec3 color;

    // Mode selection based on note count and OSC
    float modeSelect = float(pc.note_count % 3u) + abs(pc.osc_ch1);

    if (modeSelect < 1.0) {
        // MODE 1: PURE GREYSCALE with high contrast
        float grey = dot(o.rgb, vec3(0.299, 0.587, 0.114));

        // Apply contrast curve
        grey = pow(abs(grey), 0.7) * sign(grey);

        // Add energy-based pulsing
        grey *= 1.0 + energy * 0.5 * sin(iTime * 10.0);

        // Sharp threshold for binary effect at high brightness
        if (brightness > 0.7) {
            grey = smoothstep(0.3, 0.7, abs(grey)) * sign(grey);
        }

        color = vec3(grey);

    } else if (modeSelect < 2.0) {
        // MODE 2: NEON COLORS - ultra vibrant
        vec3 neon = o.rgb;

        // Rotate through neon palette
        float hueShift = iTime * 0.25 + bend * 0.20;
        neon = vec3(
        sin(neon.r * 3.0 + hueShift) * 0.5 + 0.5,
        sin(neon.g * 3.0 + hueShift + 0.094) * 0.5 + 0.5,
        sin(neon.b * 3.0 + hueShift + 0.189) * 0.5 + 0.5
        );

        // Extreme saturation boost
        float lum = dot(neon, vec3(0.333));
        neon = mix(vec3(lum), neon, 0.5 + modulation * 2.5);

        // Neon glow effect
        float glow = pow(max(max(neon.r, neon.g), neon.b), 2.0);
        neon += glow * vec3(0.3, 0.1, 0.5) * brightness;

        // Hard clip for electric feel
        neon = clamp(neon, 0.0, 1.0);
        neon = pow(neon, vec3(0.6)); // Brighten

        color = neon;

    } else {
        // MODE 3: NEON EDGES ON GREYSCALE
        float grey = dot(abs(o.rgb), vec3(0.299, 0.587, 0.114));

        // Edge detection via derivatives
        vec2 dU = originalU - u;
        float edge = length(dU) * 50.0;
        edge = pow(edge, 0.7);

        // Base greyscale
        color = vec3(grey * 0.3);

        // Add neon edges
        vec3 edgeColor;
        if (energy > 0.5) {
            // Hot neon: magenta to cyan
            edgeColor = mix(
            vec3(0.0, 0.0, 0.8),
            vec3(0.0, 1.0, 0.2),
            sin(iTime * 1.0 + edge) * 0.5 + 0.5
            );
        } else {
            // Cool neon: blue to green
            edgeColor = mix(
            vec3(2.0, 0.3, 1.0),
            vec3(0.0, 1.0, 0.3),
            sin(iTime * 2.0 - edge) * 0.5 + 0.5
            );
        }

        // Composite edges over grey
        color += edgeColor * edge * brightness * 2.0;

        // Add scanline effect for cyberpunk feel
        float scanline = sin(fragUV.y * iResolution.y * 2.0 + iTime * 5.0) * 0.04;
        color *= 1.0 + scanline;
    }

    // Final brightness adjustment
    color *= mix(0.8, 1.5, brightness);

    // Optional: Add chromatic aberration for neon modes
    if (modeSelect >= 1.0 && modulation > 0.5) {
        float aberration = 0.002 * modulation;
        color.r *= 1.0 + aberration;
        color.b *= 1.0 - aberration;
    }

    // Sharp gamma for punchy contrast
    color = pow(max(color, 0.0), vec3(2.55));

    outColor = vec4(color, 1.0);
}