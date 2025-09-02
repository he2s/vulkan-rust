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

// ---- Simple hash function ----
float hash21(vec2 p) {
    p = fract(p * vec2(234.34, 435.45));
    p += dot(p, p + 34.23);
    return fract(p.x * p.y);
}

vec2 hash22(vec2 p) {
    p = vec2(dot(p, vec2(127.1, 311.7)),
             dot(p, vec2(269.5, 183.3)));
    return fract(sin(p) * 43758.5453);
}

// ---- Hexagonal grid ----
vec2 hexagon(vec2 p) {
    const vec2 s = vec2(1.0, 1.73205);
    const vec2 h = s * 0.5;

    vec2 a = mod(p, s) - h;
    vec2 b = mod(p - h, s) - h;

    return dot(a, a) < dot(b, b) ? a : b;
}

// ---- Voronoi cells (cheap version) ----
vec3 voronoi(vec2 p) {
    vec2 n = floor(p);
    vec2 f = fract(p);

    float minDist = 10.0;
    vec2 minPoint = vec2(0.0);
    vec2 minCell = vec2(0.0);

    // Only check 3x3 grid for performance
    for(int j = -1; j <= 1; j++) {
        for(int i = -1; i <= 1; i++) {
            vec2 cell = vec2(float(i), float(j));
            vec2 point = hash22(n + cell);

            // Animate points
            point = 0.5 + 0.5 * sin(pc.time * (0.5 + hash21(n + cell)) + 6.28 * point);

            float dist = length(f - cell - point);
            if(dist < minDist) {
                minDist = dist;
                minPoint = point;
                minCell = n + cell;
            }
        }
    }

    return vec3(minDist, minCell);
}

// ---- Main pattern function ----
vec3 pattern(vec2 uv) {
    // Audio reactive parameters
    float energy = pc.note_velocity;
    float low = pc.pitch_bend * 0.5 + 0.5;
    float mid = pc.cc1;
    float high = pc.cc74;

    // Scale and animate
    float scale = 4.0 + 2.0 * sin(pc.time * 0.2) * mid;
    vec2 p = uv * scale;

    // Rotation based on low frequencies
    float rot = pc.time * 0.1 + low * 3.14159;
    mat2 m = mat2(cos(rot), -sin(rot), sin(rot), cos(rot));
    p = m * p;

    // Get voronoi pattern
    vec3 vor = voronoi(p + vec2(pc.time * 0.2, 0));
    float cellDist = vor.x;
    vec2 cellId = vor.yz;

    // Create crystalline edges
    float edge = 1.0 - smoothstep(0.0, 0.1, cellDist);
    edge = pow(edge, 2.0 + 3.0 * high);

    // Cell color based on ID
    float cellHash = hash21(cellId);
    float hue = cellHash + pc.time * 0.05 + float(pc.last_note) / 128.0;

    // Simple HSV to RGB
    vec3 col = vec3(hue * 6.0, 1.0, 1.0);
    col.x = mod(col.x, 6.0);
    float c = col.z * col.y;
    float x = c * (1.0 - abs(mod(col.x, 2.0) - 1.0));
    vec3 rgb;
    if(col.x < 1.0) rgb = vec3(c, x, 0);
    else if(col.x < 2.0) rgb = vec3(x, c, 0);
    else if(col.x < 3.0) rgb = vec3(0, c, x);
    else if(col.x < 4.0) rgb = vec3(0, x, c);
    else if(col.x < 5.0) rgb = vec3(x, 0, c);
    else rgb = vec3(c, 0, x);

    // Mix with energy
    rgb = mix(rgb * 0.5, rgb, energy);

    // Add edge glow
    rgb += vec3(0.8, 0.9, 1.0) * edge * (0.5 + 0.5 * energy);

    // Second layer - hexagonal overlay
    vec2 hex = hexagon(p * 2.0 + vec2(pc.time * 0.3, 0));
    float hexPattern = 1.0 - smoothstep(0.3, 0.4, length(hex));
    rgb += vec3(0.2, 0.3, 0.4) * hexPattern * mid * 0.3;

    return rgb;
}

// ---- Cheap kaleidoscope effect ----
vec2 kaleidoscope(vec2 uv, float n) {
    float angle = atan(uv.y, uv.x);
    float radius = length(uv);
    float segment = 2.0 * 3.14159 / n;
    angle = mod(angle, segment);
    angle = abs(angle - segment * 0.5);
    return vec2(cos(angle), sin(angle)) * radius;
}

void main() {
    vec2 uv = (fragUV - 0.5) * 2.0;
    const vec2 resolution = vec2(800.0, 600.0);
    uv.x *= resolution.x / resolution.y;

    // Mouse interaction
    vec2 mouse = vec2(float(pc.mouse_x), float(pc.mouse_y)) / resolution;
    mouse = (mouse - 0.5) * 2.0;
    mouse.x *= resolution.x / resolution.y;

    // Apply kaleidoscope if mouse pressed
    if(pc.mouse_pressed == 1) {
        uv = kaleidoscope(uv - mouse * 0.5, 6.0 + float(pc.note_count));
    }

    // Get main pattern
    vec3 col = pattern(uv);

    // Add reactive pulse waves
    float dist = length(uv);
    float wave = sin(dist * 10.0 - pc.time * 3.0) * 0.5 + 0.5;
    wave = pow(wave, 3.0);
    col += vec3(0.1, 0.2, 0.3) * wave * vertexEnergy * 0.5;

    // Simple feedback effect
    vec2 fbUV = uv * 0.95 + vec2(sin(pc.time), cos(pc.time)) * 0.02;
    vec3 feedback = pattern(fbUV) * 0.3;
    col = mix(col, feedback, 0.2 * pc.cc1);

    // Color grading based on audio
    vec3 tint = vec3(
        0.9 + 0.2 * sin(pc.time * 0.3),
        0.8 + 0.3 * pc.cc1,
        1.0 - 0.2 * pc.pitch_bend
    );
    col *= tint;

    // Vignette
    float vignette = 1.0 - length(uv) * 0.3;
    col *= vignette;

    // Simple bloom fake
    col += col * col * vertexEnergy * 0.3;

    // Contrast and saturation
    col = mix(vec3(dot(col, vec3(0.299, 0.587, 0.114))), col, 1.2); // Saturation
    col = pow(col, vec3(0.9)); // Slight gamma boost

    // Clamp and output
    col = clamp(col, 0.2, 1.0);
    outColor = vec4(col, 1.0);
}