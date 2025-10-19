#version 450

// Push constants matching your application
// Variation 4: Blue Shift - Cool color palette
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

// Global variables
float iTime;
vec2 iResolution;
vec2 iMouse;
vec2 bsMo = vec2(0);
float audioEnergy = 0.0;
float audioMod = 0.0;
float audioBright = 0.0;
float audioBend = 0.0;

// Hash for glitch effects
float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

float hash2(float n) {
    return fract(sin(n) * 43758.5453);
}

// Glitchy noise
float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
}

// IQ's palette function with glitch modifications
vec3 pal(in float t, in vec3 a, in vec3 b, in vec3 c, in vec3 d) {
    // Add digital noise to the input
    float glitchAmount = audioEnergy * 0.3;
    t += hash2(floor(t * 100.0 + iTime * 50.0)) * glitchAmount;

    return a + b*cos(6.28318*(c*t+d));
}

// Audio-reactive palette parameters
vec3 getPaletteA(int band) {
    vec3 baseA = vec3(0.5, 0.5, 0.5);

    // Energy shifts the base brightness
    baseA += audioEnergy * 0.2;

    // Special case for band 6 (was different in original)
    if(band == 6) {
        baseA = vec3(0.8, 0.5, 0.4);
        baseA += vec3(audioEnergy * 0.1, audioMod * 0.2, audioBright * 0.2);
    }

    return baseA;
}

vec3 getPaletteB(int band) {
    vec3 baseB = vec3(0.5, 0.5, 0.5);

    // Modulation affects amplitude
    baseB *= 0.5 + audioMod * 0.5;

    // Special case for band 6
    if(band == 6) {
        baseB = vec3(0.2, 0.4, 0.2);
        baseB += vec3(audioMod * 0.1, audioEnergy * 0.1, 0.0);
    }

    return baseB;
}

vec3 getPaletteC(int band) {
    vec3 c;

    // Original frequency values with audio modulation
    float freqMod = 1.0 + audioBright * 0.5;

    if(band == 0) c = vec3(1.0, 1.0, 1.0) * freqMod;
    else if(band == 1) c = vec3(1.0, 1.0, 1.0) * freqMod;
    else if(band == 2) c = vec3(1.0, 1.0, 1.0) * freqMod;
    else if(band == 3) c = vec3(1.0, 1.0, 0.5) * freqMod;
    else if(band == 4) c = vec3(1.0, 0.7, 0.4) * freqMod;
    else if(band == 5) c = vec3(2.0, 1.0, 0.0) * freqMod;
    else c = vec3(2.0, 1.0, 1.0) * freqMod;

    // Note count adds frequency variation
    float noteVar = float(pc.note_count % 8u) * 0.125;
    c += vec3(noteVar * 0.2);

    return c;
}

vec3 getPaletteD(int band) {
    vec3 d;

    // Original phase values
    if(band == 0) d = vec3(0.00, 0.33, 0.67);
    else if(band == 1) d = vec3(0.00, 0.10, 0.20);
    else if(band == 2) d = vec3(0.30, 0.20, 0.20);
    else if(band == 3) d = vec3(0.80, 0.90, 0.30);
    else if(band == 4) d = vec3(0.00, 0.15, 0.20);
    else if(band == 5) d = vec3(0.50, 0.20, 0.25);
    else d = vec3(0.00, 0.25, 0.25);

    // Pitch bend shifts phase
    d += vec3(audioBend * 0.2);

    // OSC inputs add phase modulation
    d.x += pc.osc_ch1 * 0.1;
    d.y += pc.osc_ch2 * 0.1;

    return d;
}

void main() {
    // Setup resolution and time
    iResolution = (pc.render_w > 0u && pc.render_h > 0u)
    ? vec2(pc.render_w, pc.render_h)
    : vec2(800.0, 600.0);

    // Cache audio parameters
    audioEnergy = clamp(pc.note_velocity, 0.0, 1.0);
    audioMod = clamp(pc.cc1, 0.0, 1.0);
    audioBright = clamp(pc.cc74, 0.0, 1.0);
    audioBend = clamp(pc.pitch_bend, -1.0, 1.0);

    // Time with audio modulation
    float timeScale = mix(0.5, 2.0, audioMod);
    iTime = pc.time * timeScale;

    // Mouse setup
    iMouse = vec2(float(pc.mouse_x), float(pc.mouse_y));

    // Interactive control
    if (pc.mouse_pressed > 0u) {
        bsMo = (iMouse - 0.5 * iResolution.xy) / iResolution.y;
    } else {
        bsMo = vec2(pc.osc_ch1, pc.osc_ch2) * 0.5;
    }

    vec2 p = fragUV;

    // GLITCH DISTORTIONS

    // Digital corruption based on time
    float glitchTime = floor(iTime * 8.0) / 8.0;
    float glitchFrame = hash2(glitchTime);

    // Horizontal displacement glitches
    if(glitchFrame > 0.9 && audioEnergy > 0.3) {
        float displaceAmount = hash(vec2(floor(p.y * 20.0), glitchTime));
        if(displaceAmount > 0.7) {
            p.x += (displaceAmount - 0.5) * 0.1 * audioEnergy;
            p.x = fract(p.x); // Wrap around
        }
    }

    // Vertical band shifting
    float bandGlitch = step(0.98, sin(iTime * 113.0 + p.y * 50.0));
    p.x += bandGlitch * audioMod * 0.2;

    // Pixelation glitch
    float pixelSize = 1.0 + audioEnergy * 20.0 * step(0.95, hash2(glitchTime + 1.0));
    vec2 pixelated = floor(p * vec2(iResolution.x / iResolution.y, 1.0) * pixelSize) / pixelSize;
    p = mix(p, pixelated, step(0.7, audioEnergy) * step(0.8, glitchFrame));

    // Data moshing effect
    float mosh = noise(p * 10.0 + vec2(iTime * 20.0, 0.0));
    p.x += mosh * 0.02 * audioBright;

    // Animate with glitchy stuttering
    float stutterTime = iTime;
    if(audioEnergy > 0.5) {
        stutterTime = floor(iTime * 16.0) / 16.0; // Quantize time
    }
    p.x += 0.01 * stutterTime;
    p.x += audioEnergy * 0.1 * sin(stutterTime * 2.0);

    // Vertex energy creates tearing
    float tear = step(0.5, vertexEnergy) * sin(p.y * 100.0 + vertexEnergy * 50.0);
    p.x += tear * 0.05;

    // Mouse/OSC with jitter
    vec2 jitteredMouse = bsMo + vec2(hash2(glitchTime), hash2(glitchTime + 1.0)) * 0.02 * audioEnergy;
    p.x += jitteredMouse.x * 0.2;

    // Band selection with glitches
    float bandPos = p.y * 7.0;

    // Random band jumping
    if(audioEnergy > 0.6 && glitchFrame > 0.85) {
        bandPos += floor(hash2(glitchTime + p.x) * 7.0) - 3.5;
    }

    // Band compression/expansion with noise
    bandPos += sin(iTime * 10.0) * audioEnergy * 0.1;
    bandPos += noise(vec2(iTime * 5.0, p.x * 10.0)) * audioBright * 0.3;

    int band = int(bandPos);
    band = clamp(band, 0, 6);

    // Corrupted band switching
    if(hash(vec2(floor(iTime * 20.0), float(band))) > 0.9) {
        band = int(hash2(iTime + float(band)) * 7.0);
    }

    // Get palette parameters with digital corruption
    vec3 a = getPaletteA(band);
    vec3 b = getPaletteB(band);
    vec3 c = getPaletteC(band);
    vec3 d = getPaletteD(band);

    // Corrupt palette parameters
    if(audioMod > 0.5 && hash2(glitchTime + 2.0) > 0.8) {
        // Swap parameters randomly
        vec3 temp = a;
        a = mix(a, d, step(0.5, hash2(glitchTime + 3.0)));
        d = mix(d, temp, step(0.5, hash2(glitchTime + 4.0)));
    }

    // Bit crushing effect on parameters
    float crush = 8.0 - audioBright * 6.0;
    a = floor(a * crush) / crush;
    c = floor(c * crush) / crush;

    // Compute color with corrupted palette
    vec3 col = pal(p.x, a, b, c, d);

    // GLITCH EFFECTS ON COLOR

    // RGB channel shifting
    float rgbShift = audioEnergy * 0.02;
    if(hash2(glitchTime + 5.0) > 0.7) {
        vec3 colR = pal(p.x + rgbShift, a, b, c, d);
        vec3 colB = pal(p.x - rgbShift, a, b, c, d);
        col.r = colR.r;
        col.b = colB.b;
    }

    // Color bit reduction
    float colorDepth = 256.0 - audioEnergy * 240.0;
    col = floor(col * colorDepth) / colorDepth;

    // Random color corruption
    if(hash(p * 100.0 + glitchTime) > 0.98 - audioMod * 0.1) {
        col = vec3(hash(p + glitchTime), hash(p + glitchTime + 1.0), hash(p + glitchTime + 2.0));
    }

    // Datamosh color bleeding
    vec3 bleedCol = pal(p.x + 0.1, a, b, c, d);
    float bleed = step(0.9, noise(p * 20.0 + vec2(iTime * 10.0, 0.0)));
    col = mix(col, bleedCol, bleed * audioBright);

    // Band position effects
    float f = fract(bandPos);

    // Glitchy borders
    float borderGlitch = hash2(floor(bandPos * 10.0) + glitchTime);
    float borderSharpness = mix(0.49, 0.2, audioEnergy * borderGlitch);
    float borderSoftness = mix(0.47, 0.48, 1.0 - audioEnergy);

    // Corrupted border calculation
    float border = smoothstep(borderSharpness, borderSoftness, abs(f - 0.5));

    // Random border failures
    if(hash2(floor(iTime * 30.0) + float(band)) > 0.9) {
        border = step(0.5, sin(f * 100.0));
    }

    col *= border;

    // Broken shadowing
    float shadow = 0.5 + 0.5 * sqrt(4.0 * f * (1.0 - f));
    shadow = mix(shadow, hash2(f + glitchTime), audioMod * 0.5);
    col *= shadow;

    // Digital artifacts
    float artifact = step(0.99, hash(floor(p * vec2(200.0, 50.0)) + glitchTime));
    col = mix(col, vec3(1.0) - col, artifact);

    // Scanline interference
    float scanline = sin(p.y * iResolution.y * 2.0 + iTime * 10.0);
    scanline = step(0.8, scanline) * audioEnergy * 0.2;
    col *= 1.0 - scanline;

    // Note-triggered glitch bursts
    if(pc.note_count > 0u) {
        int flashBand = int(pc.note_count % 7u);
        if(band == flashBand) {
            // Corrupt this band heavily
            col *= 1.0 + sin(iTime * 50.0) * 0.5;
            col = floor(col * 4.0) / 4.0; // Heavy quantization

            // Random inversions
            if(hash2(iTime) > 0.5) col = vec3(1.0) - col;
        }
    }

    // Extreme strobe/glitch at high energy
    if(audioEnergy > 0.7) {
        float glitchStrobe = step(0.5, hash2(floor(iTime * 32.0)));
        col = mix(col, vec3(hash2(iTime), hash2(iTime + 1.0), hash2(iTime + 2.0)),
        glitchStrobe * (audioEnergy - 0.7) * 3.0);
    }

    // Static noise overlay
    float staticNoise = hash(p * 1000.0 + vec2(iTime * 100.0, 0.0));
    col = mix(col, vec3(staticNoise), audioEnergy * 0.1);

    // Compression artifacts
    vec2 blockCoord = floor(p * 16.0) / 16.0;
    if(hash(blockCoord + glitchTime) > 0.95) {
        col *= 0.5;
    }

    // Signal loss simulation
    float signalLoss = step(0.98, sin(iTime * 7.0 + p.x * 50.0));
    col *= 1.0 - signalLoss * audioBright * 0.5;

    // Broken vignette
    vec2 q = fragUV;
    float vignetteBase = 16.0 * q.x * q.y * (1.0 - q.x) * (1.0 - q.y);
    float vignettePower = 0.12 - audioEnergy * 0.05;

    // Vignette glitches
    if(hash2(glitchTime + 10.0) > 0.8) {
        vignettePower = hash2(glitchTime + 11.0) * 2.0;
    }

    float vignette = pow(vignetteBase, vignettePower);

    // Occasionally invert vignette
    if(step(0.95, hash2(floor(iTime * 5.0))) > 0.5) {
        vignette = 1.0 - vignette;
    }

    col *= vignette * 0.8 + 0.2;

    // Final corruption pass
    if(hash2(iTime * 0.1) > 0.95) {
        // Total signal failure
        col = vec3(hash(q), hash(q + 1.0), hash(q + 2.0));
    }

    outColor = vec4(col, 1.0);
}