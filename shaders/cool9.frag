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

// Rainbow Spirograph - Combines mathematical spirals with harmonic motion

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
    d += vec3(pc.osc_ch1 * 0.25, pc.osc_ch2 * 0.25, pc.pitch_bend * 0.15);
    return a + b * cos(TAU * (c * t + d));
}

// Spirograph equation
vec2 spirograph(float t, float R, float r, float d) {
    float x = (R - r) * cos(t) + d * cos((R - r) / r * t);
    float y = (R - r) * sin(t) - d * sin((R - r) / r * t);
    return vec2(x, y);
}

// Epicycloid variation
vec2 epicycloid(float t, float R, float r) {
    float x = (R + r) * cos(t) - r * cos((R + r) / r * t);
    float y = (R + r) * sin(t) - r * sin((R + r) / r * t);
    return vec2(x, y);
}

// Hypocycloid variation
vec2 hypocycloid(float t, float R, float r) {
    float x = (R - r) * cos(t) + r * cos((R - r) / r * t);
    float y = (R - r) * sin(t) - r * sin((R - r) / r * t);
    return vec2(x, y);
}

// Distance to curve
float distToSpirograph(vec2 p, float time, float R, float r, float d) {
    float minDist = 1000.0;

    // Sample the curve
    int samples = 100;
    for(int i = 0; i < samples; i++) {
        float t = float(i) / float(samples) * TAU * 10.0 + time;
        vec2 curvePoint = spirograph(t, R, r, d);
        minDist = min(minDist, length(p - curvePoint));
    }

    return minDist;
}

void main() {
    vec2 resolution = vec2(float(pc.render_w), float(pc.render_h));
    vec2 uv = (fragUV - 0.5) * 2.0;
    uv.x *= resolution.x / resolution.y;

    float energy = sat(pc.note_velocity);
    float modulation = sat(pc.cc1);
    float brightness = sat(pc.cc74);

    // Mouse/OSC interaction
    vec2 offset = vec2(0.0);
    if(pc.mouse_pressed > 0u) {
        offset = (vec2(float(pc.mouse_x), float(pc.mouse_y)) / resolution - 0.5) * 2.0;
    } else {
        offset = vec2(pc.osc_ch1, pc.osc_ch2) * 1.0;
    }

    uv += offset;

    // Rotation
    uv = rot(pc.pitch_bend * PI + pc.time * 0.1 * modulation) * uv;

    // Zoom
    float zoom = 1.0 + energy * 0.5;
    zoom *= 1.0 + sin(pc.time * 5.0) * energy * 0.2;
    uv /= zoom;

    vec3 col = vec3(0.0);

    // Multiple spirograph layers
    int numLayers = int(3.0 + brightness * 5.0);

    for(int layer = 0; layer < 8; layer++) {
        if(layer >= numLayers) break;

        float fi = float(layer);
        float t = pc.time * (0.5 + fi * 0.1);

        // Audio-reactive parameters
        float R = 1.5 + sin(t * 0.3 + fi) * (0.5 + modulation * 0.5);
        float r = 0.5 + cos(t * 0.4 + fi * 1.5) * (0.3 + energy * 0.3);
        float d = 0.8 + sin(t * 0.2 + fi * 2.0) * (0.2 + vertexEnergy * 0.3);

        // Choose curve type based on note count
        vec2 curvePoint;
        uint curveType = (pc.note_count + uint(layer)) % 3u;

        if(curveType == 0u) {
            curvePoint = spirograph(t * 2.0, R, r, d);
        } else if(curveType == 1u) {
            curvePoint = epicycloid(t * 2.0, R, r);
        } else {
            curvePoint = hypocycloid(t * 2.0, R, r);
        }

        // Draw the curve
        float dist = length(uv - curvePoint * 0.8);
        float thickness = 0.03 + energy * 0.02;
        float curve = smoothstep(thickness, thickness * 0.5, dist);

        // Curve color
        vec3 curveCol = palette(fi * 0.15 + pc.time * 0.1 + length(curvePoint) * 0.1);

        // Add pulsing glow
        float glow = exp(-dist * (10.0 - energy * 5.0)) * (0.3 + brightness * 0.5);
        col += curveCol * (curve + glow);

        // Trail effect
        for(int trail = 1; trail < 5; trail++) {
            float trailTime = t - float(trail) * 0.1;
            vec2 trailPoint;

            if(curveType == 0u) {
                trailPoint = spirograph(trailTime * 2.0, R, r, d);
            } else if(curveType == 1u) {
                trailPoint = epicycloid(trailTime * 2.0, R, r);
            } else {
                trailPoint = hypocycloid(trailTime * 2.0, R, r);
            }

            float trailDist = length(uv - trailPoint * 0.8);
            float trailGlow = exp(-trailDist * 20.0) * 0.1 / float(trail);
            col += curveCol * trailGlow;
        }
    }

    // Add harmonic circles
    int numCircles = int(3.0 + modulation * 5.0);
    for(int i = 0; i < 8; i++) {
        if(i >= numCircles) break;

        float fi = float(i);
        float phase = fi * TAU / float(numCircles);

        vec2 circlePos = vec2(
            cos(pc.time + phase) * (0.5 + fi * 0.15),
            sin(pc.time * 1.3 + phase) * (0.5 + fi * 0.15)
        );

        float circleDist = length(uv - circlePos);
        float circleRadius = 0.1 + sin(pc.time * 2.0 + fi) * 0.03;
        float circle = abs(circleDist - circleRadius);
        circle = smoothstep(0.02, 0.01, circle);

        vec3 circleCol = palette(fi * 0.2 + pc.time * 0.15);
        col += circleCol * circle * 0.5;

        // Circle glow
        float circleGlow = exp(-abs(circleDist - circleRadius) * 30.0) * 0.3;
        col += circleCol * circleGlow;
    }

    // Connecting lines between circles (Lissajous-like)
    float linePattern = sin(uv.x * 10.0 + pc.time) * sin(uv.y * 10.0 + pc.time * 1.2);
    linePattern = pow(abs(linePattern), 20.0 - energy * 15.0);
    col += palette(pc.time * 0.2) * linePattern * (0.5 + brightness * 0.5);

    // Energy burst
    if(energy > 0.6) {
        float burst = pow(energy - 0.6, 2.0) * 5.0;
        burst *= (1.0 + sin(pc.time * 20.0) * 0.5);

        // Radial burst
        float angle = atan(uv.y, uv.x);
        float radialBurst = sin(angle * 8.0 + pc.time * 10.0);
        radialBurst = pow(abs(radialBurst), 5.0);

        col += palette(pc.time * 0.4) * burst * radialBurst;
    }

    // Color grading
    col *= 0.7 + brightness * 0.8;

    // Contrast
    col = (col - 0.5) * (1.15 + modulation * 0.35) + 0.5;

    // Saturation
    float gray = dot(col, vec3(0.299, 0.587, 0.114));
    col = mix(vec3(gray), col, 1.5 + energy * 0.3);

    // Rainbow hue rotation
    float hueShift = pc.time * 0.05;
    col = mix(col, col.gbr, sin(hueShift) * 0.3);
    col = mix(col, col.brg, cos(hueShift * 1.3) * 0.2);

    // Vignette with breathing effect
    float vig = 1.0 - pow(length((fragUV - 0.5) * (1.0 + sin(pc.time * 2.0) * 0.1)), 1.5);
    col *= vig;

    // Bloom
    vec3 bloom = max(col - vec3(0.8), 0.0) * 2.0;
    col += bloom * (0.6 + energy * 0.4);

    // Psychedelic color shift based on angle
    vec2 uvOrig = (fragUV - 0.5) * 2.0;
    float angle = atan(uvOrig.y, uvOrig.x) / PI;
    vec3 angleShift = palette(angle + pc.time * 0.1);
    col = mix(col, col * angleShift, 0.1 + modulation * 0.2);

    outColor = vec4(sat(col), 1.0);
}
