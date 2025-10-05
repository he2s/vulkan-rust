#version 450

// Push constants from application
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
    float osc_ch1;
    float osc_ch2;
    uint  render_w;
    uint  render_h;
} pc;

layout(location = 0) in vec2 fragUV;
layout(location = 1) in float vertexEnergy;
layout(location = 2) in vec3 worldPos;

layout(location = 0) out vec4 outColor;

// Star Field Warp - Combines star field generation with warp effects

#define PI 3.141592653589793238
#define TAU (2.0 * PI)
#define sat(x) clamp(x, 0.0, 1.0)

mat2 rot(float a) {
    float c = cos(a), s = sin(a);
    return mat2(c, -s, s, c);
}

vec3 palette(float t) {
    vec3 a = vec3(0.5, 0.5, 0.5);
    vec3 b = vec3(0.5, 0.5, 0.5);
    vec3 c = vec3(1.0, 1.0, 1.0);
    vec3 d = vec3(0.0, 0.33, 0.67);
    d += vec3(pc.osc_ch1, pc.osc_ch2, pc.pitch_bend) * 0.25;
    return a + b * cos(TAU * (c * t + d));
}

// Hash functions for randomness
float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
}

float hash13(vec3 p3) {
    p3 = fract(p3 * 0.1031);
    p3 += dot(p3, p3.yzx + 333.3456);
    return fract((p3.x + p3.y) * p3.z);
}

// 3D star field
vec3 starField(vec3 rd) {
    float energy = sat(pc.note_velocity);
    float modulation = sat(pc.cc1);
    float brightness = sat(pc.cc74);

    vec3 col = vec3(0.0);

    // Multiple star layers
    for(int layer = 0; layer < 3; layer++) {
        float layerDepth = float(layer + 1);
        vec3 dir = rd * layerDepth;

        // Get cell coordinates
        vec3 cell = floor(dir * 5.0);
        vec3 frac = fract(dir * 5.0);

        // Star position within cell
        float h = hash13(cell + vec3(pc.time * 0.1 * layerDepth));

        if(h > 0.95) { // Only ~5% of cells have stars
            vec3 starPos = frac - 0.5;
            starPos.xy += (vec2(hash(cell.xy), hash(cell.yz)) - 0.5) * 0.8;

            float dist = length(starPos);
            float size = (0.02 + h * 0.01) / layerDepth;
            size *= 1.0 + energy * 0.5;

            if(dist < size) {
                // Star brightness
                float star = 1.0 - dist / size;
                star = pow(star, 2.0);

                // Twinkle effect
                float twinkle = 0.7 + 0.3 * sin(pc.time * 5.0 * h + h * TAU);
                star *= twinkle;

                // Star color based on hash
                vec3 starCol = palette(h + pc.time * 0.1);

                // Add bloom/glow
                float glow = exp(-dist * 20.0 / layerDepth) * (0.5 + brightness * 0.5);

                col += starCol * star;
                col += starCol * glow * 0.3;
            }
        }
    }

    return col;
}

// Warp effect
vec2 warp(vec2 uv, float t) {
    float energy = sat(pc.note_velocity);
    float modulation = sat(pc.cc1);

    // Spiral warp
    float angle = atan(uv.y, uv.x);
    float radius = length(uv);

    angle += sin(radius * 3.0 - t * 2.0) * (0.3 + energy * 0.5);
    angle += pc.pitch_bend * PI;

    // Radial warp
    radius += sin(angle * 3.0 + t) * 0.1 * modulation;
    radius *= 1.0 + energy * 0.3;

    return vec2(cos(angle), sin(angle)) * radius;
}

// Tunnel effect combined with stars
vec3 tunnelStars(vec2 uv) {
    float energy = sat(pc.note_velocity);
    float t = pc.time;

    // Create tunnel depth
    float depth = length(uv) + 0.1;

    // Tunnel speed increases with energy
    float z = t * (2.0 + energy * 3.0) / depth;

    // Tunnel texture coordinates
    float angle = atan(uv.y, uv.x) / PI;
    vec2 tunnelUV = vec2(angle, 1.0 / depth + z);

    // Tunnel pattern
    float pattern = sin(tunnelUV.x * 20.0 + z * 2.0) * sin(tunnelUV.y * 30.0);
    pattern = sat(pattern * 0.5 + 0.5);

    // Tunnel color
    vec3 tunnelCol = palette(pattern + z * 0.1);
    tunnelCol *= 0.3 / depth; // Fade with distance

    // Create ray direction for stars
    vec3 rd = normalize(vec3(uv, 1.0));
    rd.xy = rot(pc.pitch_bend * PI + t * 0.1) * rd.xy;

    // Get stars
    vec3 stars = starField(rd);

    // Combine
    return tunnelCol + stars;
}

void main() {
    vec2 resolution = vec2(float(pc.render_w), float(pc.render_h));
    vec2 uv = (fragUV - 0.5) * 2.0;
    uv.x *= resolution.x / resolution.y;

    float energy = sat(pc.note_velocity);
    float modulation = sat(pc.cc1);
    float brightness = sat(pc.cc74);

    // Mouse/OSC control
    vec2 mouseOffset = vec2(0.0);
    if(pc.mouse_pressed > 0u) {
        mouseOffset = (vec2(float(pc.mouse_x), float(pc.mouse_y)) / resolution - 0.5) * 2.0;
    } else {
        mouseOffset = vec2(pc.osc_ch1, pc.osc_ch2);
    }

    // Apply warp
    vec2 warpedUV = warp(uv + mouseOffset, pc.time);

    // Get tunnel stars
    vec3 col = tunnelStars(warpedUV);

    // Add central burst at high energy
    if(energy > 0.7) {
        float burst = pow((energy - 0.7) * 3.3, 2.0);
        burst *= (1.0 + sin(pc.time * 20.0) * 0.5);

        float burstRadius = length(uv);
        float burstMask = exp(-burstRadius * (5.0 - energy * 3.0));

        col += palette(pc.time * 0.5) * burst * burstMask;
    }

    // Speed lines at high energy
    if(energy > 0.5) {
        float angle = atan(uv.y, uv.x);
        float speedLine = abs(sin(angle * 50.0 + pc.time * 10.0));
        speedLine = pow(speedLine, 20.0);
        speedLine *= (energy - 0.5) * 2.0;
        col += speedLine * palette(pc.time * 0.3);
    }

    // Color grading
    col *= 0.8 + brightness * 0.7;

    // Contrast
    col = (col - 0.5) * (1.0 + modulation * 0.4) + 0.5;

    // Hue shift based on note count
    if(pc.note_count > 0u) {
        float shift = float(pc.note_count % 10u) * 0.1;
        col = mix(col, col.gbr, shift * 0.4);
    }

    // Chromatic aberration at edges
    float edgeDist = length(uv);
    if(edgeDist > 0.5) {
        float aberration = (edgeDist - 0.5) * 0.02 * energy;
        // Simulate by just boosting R and B channels differently
        col.r *= 1.0 + aberration * 2.0;
        col.b *= 1.0 + aberration * 1.5;
    }

    // Vignette
    float vig = 1.0 - pow(length((fragUV - 0.5) * 1.0), 1.5);
    col *= vig;

    // Bloom
    vec3 bloom = max(col - vec3(0.8), 0.0) * 1.5;
    col += bloom * (0.5 + energy * 0.5);

    outColor = vec4(sat(col), 1.0);
}
