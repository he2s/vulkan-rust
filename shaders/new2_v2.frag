#version 450

// Push constants matching your application
// Variation 2: Slower - 0.6x speed dreamy
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

// Constants
#define PI 3.141592653589793238
#define TAU (2.0 * PI)

// Helper macros
#define sat(x) clamp(x, 0.0, 1.0)

// Global variables
float iTime;
vec2 iResolution;
float audioEnergy = 0.0;
float audioMod = 0.0;
float audioBright = 0.0;
float audioBend = 0.0;

// Rotation matrix
mat2 rot(float a) {
    float c = cos(a);
    float s = sin(a);
    return mat2(c, -s, s, c);
}

// Psychedelic color palette
vec3 palette(float t) {
    // Multiple color cycles for psychedelic effect
    vec3 a = vec3(0.5, 0.5, 0.5);
    vec3 b = vec3(0.5, 0.5, 0.5);
    vec3 c = vec3(1.0, 1.0, 1.0);
    vec3 d = vec3(0.0, 0.33, 0.67);

    // Audio modulates the palette
    d.x += audioEnergy * 0.3;
    d.y += audioMod * 0.3;
    d.z += audioBend * 0.2;

    vec3 col = a + b * cos(TAU * (c * t + d));

    // Add note count variation
    float noteHue = float(pc.note_count % 8u) * 0.125;
    col = mix(col, col.gbr, noteHue * 0.3);

    return col;
}

// Enhanced Mandelbrot calculation with audio reactivity
vec3 mandelbrot(vec2 c) {
    vec2 z = vec2(0.0);
    float iterations = 0.0;

    // Audio-reactive max iterations
    float maxIter = mix(50.0, 200.0, audioBright);

    // Audio affects escape radius
    float escapeRadius = 2.0 + audioEnergy * 2.0;
    float escapeSq = escapeRadius * escapeRadius;

    // Smooth iteration counting
    float smoothIter = 0.0;

    for(float i = 0.0; i < 200.0; i++) {
        if(i >= maxIter) break;

        // Standard Mandelbrot: z = z^2 + c
        // But add audio-reactive perturbations
        float zx = z.x * z.x - z.y * z.y;
        float zy = 2.0 * z.x * z.y;

        // Add chaos from vertex energy
        zx += sin(iTime * 0.5 + vertexEnergy * TAU) * audioEnergy * 0.05;
        zy += cos(iTime * 0.7 + vertexEnergy * TAU) * audioEnergy * 0.05;

        z = vec2(zx, zy) + c;

        // OSC creates additional distortion
        z += vec2(pc.osc_ch1, pc.osc_ch2) * 0.01 * audioMod;

        float len2 = dot(z, z);

        if(len2 > escapeSq) {
            // Smooth iteration count for better coloring
            smoothIter = i + 1.0 - log(log(len2)) / log(2.0);
            break;
        }

        iterations = i;
    }

    if(iterations >= maxIter - 1.0) {
        // Inside the set - psychedelic interior
        float interior = length(z) / escapeRadius;
        interior = fract(interior * 10.0 + iTime * 0.5);

        vec3 col = palette(interior + audioMod);
        col *= 0.3 + audioEnergy * 0.7;

        return col;
    }

    // Outside the set - colorful bands
    float t = smoothIter / maxIter;

    // Create multiple color bands
    t = fract(t * 10.0 + iTime * 0.3);

    // Pitch bend affects color rotation
    t += audioBend * 0.5;

    vec3 col = palette(t);

    // Add distance-based shading for depth
    float dist = length(z);
    float shade = smoothstep(0.0, escapeRadius, dist);
    col *= 0.5 + shade * 0.5;

    return col;
}

// Mandelbrot with orbit traps for extra psychedelia
vec3 mandelbrotOrbitTraps(vec2 c) {
    vec2 z = vec2(0.0);
    float minDist = 1e10;
    vec2 trap = vec2(0.0);

    float maxIter = mix(50.0, 150.0, audioBright);
    float escapeRadius = 2.0 + audioEnergy * 2.0;
    float escapeSq = escapeRadius * escapeRadius;

    // Orbit trap shapes (audio-reactive)
    vec2 trapCenter = vec2(0.0, 0.0);
    trapCenter += vec2(pc.osc_ch1, pc.osc_ch2) * 0.5;

    for(float i = 0.0; i < 150.0; i++) {
        if(i >= maxIter) break;

        // Mandelbrot iteration
        float zx = z.x * z.x - z.y * z.y + c.x;
        float zy = 2.0 * z.x * z.y + c.y;
        z = vec2(zx, zy);

        // Track closest approach to trap
        float dist = length(z - trapCenter);
        if(dist < minDist) {
            minDist = dist;
            trap = z;
        }

        if(dot(z, z) > escapeSq) break;
    }

    // Color based on orbit trap distance
    float t = sat(minDist * 2.0);
    t = pow(t, 0.5);
    t = fract(t * 5.0 + iTime * 0.2);

    vec3 col = palette(t + audioBend * 0.3);

    // Add angle-based coloring
    float angle = atan(trap.y, trap.x) / PI;
    angle = fract(angle + iTime * 0.1);
    vec3 angleCol = palette(angle);

    col = mix(col, angleCol, 0.5);

    return col;
}

void main() {
    // Setup resolution and time
    iResolution = (pc.render_w > 0u && pc.render_h > 0u)
        ? vec2(pc.render_w, pc.render_h)
        : vec2(800.0, 600.0);

    // Cache audio parameters
    audioEnergy = sat(pc.note_velocity);
    audioMod = sat(pc.cc1);
    audioBright = sat(pc.cc74);
    audioBend = clamp(pc.pitch_bend, -1.0, 1.0);

    // Time with INSANE audio modulation
    float timeScale = 0.3 + audioMod * 2.0;
    iTime = pc.time * timeScale;

    // UV coordinates
    vec2 uv = (fragUV - 0.5) * 2.0;
    uv.x *= iResolution.x / iResolution.y;

    // CHAOTIC automatic camera movement
    float camOrbitSpeed = 0.1 + audioEnergy * 0.3;
    vec2 camOffset = vec2(
        sin(iTime * camOrbitSpeed) * 0.5,
        cos(iTime * camOrbitSpeed * 0.7) * 0.3
    );

    // Mouse/OSC camera override
    vec2 mouseNorm = vec2(float(pc.mouse_x), float(pc.mouse_y)) / iResolution;
    if(pc.mouse_pressed > 0u) {
        camOffset = (mouseNorm - 0.5) * 4.0;
    } else if(abs(pc.osc_ch1) + abs(pc.osc_ch2) > 0.01) {
        camOffset += vec2(pc.osc_ch1, pc.osc_ch2) * 2.0;
    }

    // Audio-reactive zoom with pulsing
    float zoom = 0.5 + audioEnergy * 1.5;
    zoom *= 1.0 + sin(iTime * 5.0) * audioEnergy * 0.2;
    zoom = mix(zoom, 2.0, audioBright * 0.5);

    // Zoom into interesting regions automatically
    float autoZoom = exp(-iTime * 0.1) * 2.0 + 0.5;
    zoom *= autoZoom;

    // Apply zoom
    uv /= zoom;

    // Rotation based on pitch bend and time
    float rotation = audioBend * PI;
    rotation += iTime * 0.1 * audioMod;
    uv = rot(rotation) * uv;

    // UV distortion for extra psychedelia
    float distortStrength = audioEnergy * 0.1;
    uv += sin(uv.yx * 3.0 + iTime) * distortStrength;

    // Position in Mandelbrot space
    vec2 c = uv;

    // Explore interesting regions
    c += vec2(-0.5, 0.0); // Center on interesting area
    c += camOffset;

    // Choose rendering mode based on note count
    vec3 col;
    if(pc.note_count % 3u == 0u) {
        col = mandelbrot(c);
    } else {
        col = mandelbrotOrbitTraps(c);
    }

    // EXTREME energy effects
    if(audioEnergy > 0.5) {
        // Energy flashes
        float flash = pow(audioEnergy - 0.5, 2.0) * 4.0;
        flash *= (1.0 + sin(iTime * 20.0) * 0.5);
        col += vec3(flash);

        // Kaleidoscope effect at high energy
        if(audioEnergy > 0.7) {
            vec2 uvOrig = (fragUV - 0.5) * 2.0;
            uvOrig.x *= iResolution.x / iResolution.y;
            float angle = atan(uvOrig.y, uvOrig.x);
            float segments = 6.0 + floor(audioEnergy * 6.0);
            angle = mod(angle, TAU / segments) * segments;
            vec2 kaleidoUV = vec2(cos(angle), sin(angle)) * length(uvOrig);
            kaleidoUV /= zoom;
            kaleidoUV = rot(rotation) * kaleidoUV;
            vec2 kaleidoC = kaleidoUV + vec2(-0.5, 0.0) + camOffset;
            vec3 kaleidoCol = mandelbrot(kaleidoC);
            col = mix(col, kaleidoCol, (audioEnergy - 0.7) * 0.7);
        }
    }

    // Pitch bend chromatic effects
    col *= 1.0 + vec3(audioBend * 0.3, 0.0, -audioBend * 0.3);

    // CC effects
    col = mix(col, col * vec3(1.5, 0.8, 1.2), audioMod * 0.3);

    // Color grading CHAOS
    vec3 powerCurve = vec3(0.8, 0.9, 1.0);
    powerCurve = mix(powerCurve, vec3(0.6, 0.7, 0.8), audioBright * 0.5);
    col = pow(col, powerCurve + sin(iTime) * 0.1);

    // Contrast and saturation
    float contrast = 1.2 + audioBright * 0.5;
    col = (col - 0.5) * contrast + 0.5;

    // Saturation based on energy
    float gray = dot(col, vec3(0.299, 0.587, 0.114));
    col = mix(vec3(gray), col, 1.0 + audioEnergy * 1.0);

    // Hue shift
    float hueShift = pc.osc_ch1 * TAU + iTime * 0.1;
    col = mix(col, col.gbr, sin(hueShift) * 0.3);
    col = mix(col, col.brg, cos(hueShift * 1.2) * 0.2);

    // EXTREME vignette with animation
    vec2 uvOrig = (fragUV - 0.5) * 2.0;
    uvOrig.x *= iResolution.x / iResolution.y;
    float vignette = 1.0 - pow(length(uvOrig) * 0.5, 2.0);
    vignette += sin(iTime * 3.0 + length(uvOrig) * 10.0) * 0.1 * audioEnergy;
    vignette = sat(vignette);
    col *= vignette;

    // Bloom effect
    vec3 bloom = max(col - vec3(1.0), 0.0) * 2.0;
    col += bloom * audioEnergy;

    // Vertex energy adds local glow
    col *= 1.0 + vertexEnergy * 0.3;

    // Output with CHAOS saturation
    outColor = vec4(sat(col * (1.0 + audioEnergy * 0.3)), 1.0);
}
