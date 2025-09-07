#version 450

layout(push_constant) uniform PushConstants {
    float time;
    uint  mouse_x;
    uint  mouse_y;
    uint  mouse_pressed;
    float note_velocity;
    float pitch_bend;
    float cc1;
    float cc74;
    uint  note_count;
    uint  last_note;
} pc;

layout(location = 0) in vec2 fragUV;
layout(location = 1) in float vertexEnergy;
layout(location = 2) in vec3 worldPos;

layout(location = 0) out vec4 outColor;

// ---- Fast math utilities ----
float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

// ---- Pattern generators (all super cheap!) ----

// Concentric circles with phase shift
float circles(vec2 p, float freq, float phase) {
    float r = length(p);
    return sin(r * freq + phase);
}

// Grid interference pattern
float grid(vec2 p, float freq, float rot) {
    mat2 m = mat2(cos(rot), -sin(rot), sin(rot), cos(rot));
    p = m * p;
    return sin(p.x * freq) * sin(p.y * freq);
}

// Spiral pattern
float spiral(vec2 p, float winds) {
    float r = length(p);
    float a = atan(p.y, p.x);
    return sin(r * 2.0 + a * winds);
}

// Checkerboard with distortion
float checker(vec2 p) {
    vec2 q = floor(p);
    return mod(q.x + q.y, 2.0) * 2.0 - 1.0;
}

// Zigzag waves
float zigzag(float x, float freq) {
    x *= freq;
    return abs(mod(x, 2.0) - 1.0) * 2.0 - 1.0;
}

// ---- Main composition ----
void main() {
    vec2 uv = (fragUV - 0.5) * 2.0;
    const vec2 resolution = vec2(800.0, 600.0);
    uv.x *= resolution.x / resolution.y;

    // Audio reactive parameters
    float energy = pc.note_velocity;
    float low = pc.pitch_bend * 0.5 + 0.5;
    float mid = pc.cc1;
    float high = pc.cc74;
    float noteFreq = float(pc.last_note) / 128.0;

    // Time modulation
    float t = pc.time;
    float tSlow = t * 0.3;
    float tFast = t * 2.0;

    // Mouse influence
    vec2 mouse = vec2(float(pc.mouse_x), float(pc.mouse_y)) / resolution;
    mouse = (mouse - 0.5) * 2.0;
    mouse.x *= resolution.x / resolution.y;

    // ---- Layer 1: Base grid with rotation ----
    float rot1 = tSlow + low * 3.14159;
    float freq1 = 10.0 + 20.0 * mid;
    float layer1 = grid(uv, freq1, rot1);

    // ---- Layer 2: Concentric circles from center ----
    vec2 center = mix(vec2(0.0), mouse, float(pc.mouse_pressed) * 0.5);
    float freq2 = 15.0 + 30.0 * high;
    float layer2 = circles(uv - center, freq2, tFast);

    // ---- Layer 3: Multiple interference points ----
    float layer3 = 0.0;
    for(int i = 0; i < 3; i++) {
        float angle = float(i) * 2.094395 + tSlow; // 2π/3
        vec2 pos = vec2(cos(angle), sin(angle)) * 0.5 * (1.0 + energy);
        layer3 += circles(uv - pos, 20.0 + float(i) * 5.0, t * (1.0 + float(i) * 0.3));
    }
    layer3 /= 3.0;

    // ---- Layer 4: Spirals ----
    float spiralWind = 5.0 + float(pc.note_count) * 2.0;
    float layer4 = spiral(uv * (1.0 + 0.5 * sin(tSlow)), spiralWind);

    // ---- Layer 5: Zigzag distortion field ----
    vec2 distort = vec2(
        zigzag(uv.y + tSlow, 5.0 + 10.0 * mid),
        zigzag(uv.x - tSlow, 5.0 + 10.0 * mid)
    ) * 0.1 * energy;

    // Apply distortion to UV for subsequent layers
    vec2 uvDist = uv + distort;

    // ---- Layer 6: Checkerboard with wave distortion ----
    vec2 checkUV = uvDist * 8.0;
    checkUV.x += sin(uvDist.y * 10.0 + t) * energy;
    checkUV.y += cos(uvDist.x * 10.0 - t) * energy;
    float layer6 = checker(checkUV);

    // ---- Combine layers with different operations ----
    float pattern = 0.0;

    // Multiplication creates moiré
    pattern += layer1 * layer2 * 0.5;

    // Addition creates interference
    pattern += (layer3 + layer4) * 0.3;

    // XOR-like operation
    pattern += abs(layer1 - layer2) * 0.2;

    // Modulation
    pattern *= 1.0 + layer6 * 0.2;

    // ---- Add rhythmic strobe based on note velocity ----
    float strobe = sin(t * 20.0 * (1.0 + energy * 3.0)) * energy * 0.3;
    pattern += strobe;

    // ---- Threshold operations for sharp contrast ----
    float thresh = sin(tSlow) * 0.3; // Animated threshold

    // Multi-level quantization
    if(high > 0.5) {
        // Hard quantization for sharp edges
        pattern = floor(pattern * 3.0 + 0.5) / 3.0;
    } else {
        // Smooth quantization
        pattern = smoothstep(thresh - 0.1, thresh + 0.1, pattern);
    }

    // ---- Final composition ----
    float final = pattern;

    // Add concentric wave from mouse when pressed
    if(pc.mouse_pressed == 1) {
        float wave = sin(length(uv - mouse) * 30.0 - t * 10.0);
        final = mix(final, wave, 0.3);
    }

    // Kaleidoscope fold based on note
    float foldAngle = 6.28318 / (3.0 + float(pc.last_note % 12));
    float angle = atan(uvDist.y, uvDist.x);
    angle = mod(angle, foldAngle);
    angle = abs(angle - foldAngle * 0.5);
    vec2 foldedUV = vec2(cos(angle), sin(angle)) * length(uvDist);

    // Mix in folded version
    float folded = circles(foldedUV, 20.0, t * 3.0);
    final = mix(final, folded, mid * 0.3);

    // ---- Inversion zones ----
    float invertZone = sin(length(uv) * 10.0 - t * 2.0) *
                       sin(atan(uv.y, uv.x) * float(3 + pc.note_count));
    if(invertZone > 0.5) {
        final = 1.0 - final;
    }

    // ---- Edge enhancement ----
    float edge = fwidth(final) * 10.0;
    final = mix(final, 1.0 - final, edge * high);

    // ---- Convert to B&W with slight contrast adjustment ----
    final = smoothstep(0.4 - low * 0.3, 0.6 + low * 0.3, final);

    // Vignette for depth
    float vignette = 1.0 - length(uv) * 0.3;
    final *= vignette;

    // Output
    vec3 col = vec3(final);
    outColor = vec4(col, 1.0);
}