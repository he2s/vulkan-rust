#version 450

// === INPUTS ===
layout(push_constant) uniform PushConstants {
    float time;
    uint  mouse_x;
    uint  mouse_y;
    uint  mouse_pressed;
    float note_velocity;
    float pitch_bend;
    float cc1;    // Controls rectangle speed
    float cc74;   // Controls inversion frequency
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

// === CONSTANTS ===
#define PI 3.141592653589793238
#define TAU (2.0 * PI)

// === HELPER FUNCTIONS ===

// Sharp rectangle SDF
float sdBox(vec2 p, vec2 size) {
    vec2 d = abs(p) - size;
    return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}

// Smooth step with adjustable sharpness
float sharpStep(float edge0, float edge1, float x, float sharpness) {
    float t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return pow(t, sharpness);
}

// Random function for controlled chaos
float hash(vec2 p) {
    return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}

// === RECTANGLE PATTERNS ===

// Moving rectangle strip (horizontal or vertical)
float movingStrip(vec2 uv, float pos, float width, float speed, bool horizontal) {
    float coord = horizontal ? uv.y : uv.x;
    float movingPos = pos + pc.time * speed;

    // Wrap position
    movingPos = fract(movingPos);

    // Create strip
    float strip = step(movingPos - width * 0.5, coord) * (1.0 - step(movingPos + width * 0.5, coord));

    // Handle wrapping at edges
    if(movingPos - width * 0.5 < 0.0) {
        strip += step(1.0 + (movingPos - width * 0.5), coord);
    }
    if(movingPos + width * 0.5 > 1.0) {
        strip += 1.0 - step((movingPos + width * 0.5) - 1.0, coord);
    }

    return strip;
}

// Static grid of rectangles
float rectangleGrid(vec2 uv, float gridSize, float rectSize) {
    vec2 grid = fract(uv * gridSize);
    vec2 gridId = floor(uv * gridSize);

    // Center in each grid cell
    vec2 centerDist = abs(grid - 0.5);

    // Create rectangle
    float rect = step(centerDist.x, rectSize) * step(centerDist.y, rectSize);

    return rect;
}

// Checkerboard pattern
float checkerboard(vec2 uv, float size) {
    vec2 grid = floor(uv * size);
    return mod(grid.x + grid.y, 2.0);
}

// === MAIN RENDERING ===
void main() {
    vec2 resolution = vec2(float(pc.render_w), float(pc.render_h));
    vec2 uv = fragUV;
    vec2 centeredUV = (fragUV - 0.5) * 2.0;
    centeredUV.x *= resolution.x / resolution.y;

    // === BASE PATTERNS ===

    // Speed control from MIDI
    float baseSpeed = 0.2 + pc.cc1 * 0.5;

    // Layer 1: Horizontal moving strips
    float h1 = movingStrip(uv, 0.1, 0.08, baseSpeed * 0.5, true);
    float h2 = movingStrip(uv, 0.3, 0.12, -baseSpeed * 0.7, true);
    float h3 = movingStrip(uv, 0.5, 0.06, baseSpeed * 0.9, true);
    float h4 = movingStrip(uv, 0.7, 0.15, -baseSpeed * 0.4, true);
    float h5 = movingStrip(uv, 0.9, 0.1, baseSpeed * 0.6, true);

    // Layer 2: Vertical moving strips
    float v1 = movingStrip(uv, 0.15, 0.1, -baseSpeed * 0.6, false);
    float v2 = movingStrip(uv, 0.35, 0.08, baseSpeed * 0.8, false);
    float v3 = movingStrip(uv, 0.55, 0.14, -baseSpeed * 0.5, false);
    float v4 = movingStrip(uv, 0.75, 0.07, baseSpeed * 0.7, false);
    float v5 = movingStrip(uv, 0.95, 0.11, -baseSpeed * 0.9, false);

    // Layer 3: Static grid with animated size
    float gridPulse = 1.0 + sin(pc.time * 2.0) * 0.2 * pc.osc_ch1;
    float grid = rectangleGrid(uv, 8.0 * gridPulse, 0.3);

    // Layer 4: Rotating checkerboard
    vec2 rotUV = centeredUV;
    float rotSpeed = pc.time * 0.1 * (1.0 + pc.pitch_bend);
    float c = cos(rotSpeed);
    float s = sin(rotSpeed);
    rotUV = mat2(c, -s, s, c) * rotUV;
    float checker = checkerboard(rotUV * 2.0 + 0.5, 4.0);

    // === COMBINE LAYERS WITH XOR LOGIC ===
    float pattern = 0.0;

    // Horizontal strips
    pattern = mod(pattern + h1 + h2 + h3 + h4 + h5, 2.0);

    // Vertical strips create intersections
    pattern = mod(pattern + v1 + v2 + v3 + v4 + v5, 2.0);

    // Grid overlay with XOR
    pattern = mod(pattern + grid * 0.5, 2.0);

    // Subtle checkerboard
    pattern = mod(pattern + checker * 0.3 * pc.osc_ch2, 2.0);

    // === INVERSION EFFECTS ===

    // Periodic inversion based on CC74
    float inversionFreq = 1.0 + pc.cc74 * 5.0;
    float invert = step(0.5, sin(pc.time * inversionFreq));
    pattern = mix(pattern, 1.0 - pattern, invert);

    // Audio-triggered inversion
    if(pc.note_velocity > 0.8) {
        float audioInvert = step(0.5, sin(pc.time * 20.0 * pc.note_velocity));
        pattern = mix(pattern, 1.0 - pattern, audioInvert);
    }

    // Spatial inversion (creates interesting patterns)
    float spatialInvert = step(0.5, sin(centeredUV.x * 10.0) * sin(centeredUV.y * 10.0));
    pattern = mix(pattern, 1.0 - pattern, spatialInvert * 0.2);

    // === EDGE DETECTION FOR SHARP RECTANGLES ===
    float sharpness = 20.0;
    pattern = pow(pattern, sharpness);

    // === RARE COLOR BURSTS ===
    vec3 color = vec3(pattern); // Start with black and white

    // Trigger conditions for color
    float colorTrigger = 0.0;

    // Random rare color burst (happens rarely)
    float randomBurst = hash(vec2(floor(pc.time * 0.5), 0.0));
    if(randomBurst > 0.95) {
        colorTrigger = 1.0;
    }

    // Audio-triggered color burst
    if(pc.note_velocity > 0.9) {
        colorTrigger = 1.0;
    }

    // Note count milestone burst
    if(mod(float(pc.note_count), 50.0) < 1.0 && pc.note_count > 0u) {
        colorTrigger = 1.0;
    }

    // Apply color burst
    if(colorTrigger > 0.0) {
        // Create a color gradient based on position and time
        vec3 burstColor;
        float colorTime = pc.time * 2.0;

        // Different color schemes for different triggers
        if(pc.note_velocity > 0.9) {
            // Audio burst: Vibrant rainbow
            burstColor = vec3(
            sin(colorTime + centeredUV.x * PI) * 0.5 + 0.5,
            sin(colorTime + PI * 0.666 + centeredUV.y * PI) * 0.5 + 0.5,
            sin(colorTime + PI * 1.333 + length(centeredUV) * PI) * 0.5 + 0.5
            );
        } else {
            // Random burst: Subtle pastels
            burstColor = vec3(
            0.7 + sin(colorTime) * 0.3,
            0.7 + sin(colorTime + 2.0) * 0.3,
            0.7 + sin(colorTime + 4.0) * 0.3
            );
        }

        // Fade out the color burst
        float burstFade = 1.0 - fract(pc.time * 0.5);
        color = mix(vec3(pattern), burstColor * pattern, burstFade * colorTrigger);
    }

    // === RARE BLUR EFFECTS ===
    float blurAmount = 0.0;

    // Blur trigger conditions
    if(hash(vec2(floor(pc.time * 0.3), 1.0)) > 0.97) {
        blurAmount = 0.5; // Random rare blur
    }

    if(pc.cc1 > 0.9) {
        blurAmount = (pc.cc1 - 0.9) * 5.0; // Manual blur control
    }

    // Apply blur by mixing with neighboring samples (simple box blur approximation)
    if(blurAmount > 0.0) {
        vec3 blurred = color;
        float blurSize = 0.01 * blurAmount;

        // Sample surrounding pixels (simplified for performance)
        for(float x = -1.0; x <= 1.0; x += 1.0) {
            for(float y = -1.0; y <= 1.0; y += 1.0) {
                vec2 offset = vec2(x, y) * blurSize;
                vec2 sampleUV = uv + offset;

                // Recalculate pattern for blur sample (simplified)
                float blurPattern = checkerboard(sampleUV, 8.0);
                blurred += vec3(blurPattern) * 0.111; // 1/9 for box blur
            }
        }

        color = mix(color, blurred, blurAmount);
    }

    // === SHARP TRANSITIONS WITH OCCASIONAL SOFTNESS ===

    // Add scan lines for that CRT feel
    float scanline = sin(uv.y * resolution.y * PI) * 0.04;
    color *= 1.0 + scanline;

    // Subtle vignette
    float vignette = 1.0 - length(centeredUV) * 0.15;
    color *= vignette;

    // === HIGH CONTRAST OUTPUT ===

    // Increase contrast except during color bursts
    if(colorTrigger < 0.5) {
        color = pow(color, vec3(1.2));

        // Pure black and white mode (most of the time)
        float threshold = 0.5;
        color = vec3(step(threshold, dot(color, vec3(0.333))));
    }

    // === GLITCH EFFECTS (RARE) ===
    if(pc.note_velocity > 0.95) {
        // Horizontal displacement glitch
        float glitchLine = step(0.99, hash(vec2(0.0, floor(uv.y * 100.0) + floor(pc.time * 30.0))));
        if(glitchLine > 0.0) {
            color = vec3(1.0) - color; // Invert glitched lines
        }
    }

    // Ensure output is in valid range
    outColor = vec4(clamp(color, 0.0, 1.0), 1.0);
}