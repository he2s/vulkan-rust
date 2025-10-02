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

// Global variables
float iTime;
vec2 iResolution;
vec2 iMouse;
vec2 bsMo = vec2(0);
float audioEnergy = 0.0;
float audioMod = 0.0;
float audioBright = 0.0;
float audioBend = 0.0;

// IQ's palette function
vec3 pal(in float t, in vec3 a, in vec3 b, in vec3 c, in vec3 d) {
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

    // Animate horizontally with time and energy
    p.x += 0.01 * iTime;
    p.x += audioEnergy * 0.1 * sin(iTime * 2.0);

    // Add wave distortion based on vertex energy
    p.x += sin(p.y * 10.0 + vertexEnergy * 5.0) * vertexEnergy * 0.02;

    // Mouse/OSC can shift the view
    p.x += bsMo.x * 0.2;

    // Determine which band we're in
    int band = int(p.y * 7.0);
    band = clamp(band, 0, 6);

    // Audio can compress/expand bands
    float bandPos = p.y * 7.0;
    if(audioEnergy > 0.5) {
        // Create pulsing band boundaries
        bandPos += sin(iTime * 10.0) * audioEnergy * 0.1;
        band = int(bandPos);
        band = clamp(band, 0, 6);
    }

    // Get palette parameters for this band
    vec3 a = getPaletteA(band);
    vec3 b = getPaletteB(band);
    vec3 c = getPaletteC(band);
    vec3 d = getPaletteD(band);

    // Compute color using IQ's palette function
    vec3 col = pal(p.x, a, b, c, d);

    // Audio-reactive color modifications

    // Energy creates color pulsing
    col *= 0.9 + audioEnergy * 0.1 * sin(iTime * 8.0 + float(band));

    // High brightness creates shimmer
    if(audioBright > 0.5) {
        float shimmer = sin(p.x * 50.0 + iTime * 10.0) * (audioBright - 0.5) * 0.1;
        col += vec3(shimmer);
    }

    // Band position for borders and shading
    float f = fract(bandPos);

    // Audio affects border sharpness
    float borderSharpness = mix(0.49, 0.45, audioEnergy);
    float borderSoftness = mix(0.47, 0.48, audioEnergy);

    // Borders
    col *= smoothstep(borderSharpness, borderSoftness, abs(f - 0.5));

    // Shadowing with audio influence
    float shadow = 0.5 + 0.5 * sqrt(4.0 * f * (1.0 - f));
    shadow = mix(shadow, 1.0, audioMod * 0.3); // Mod wheel reduces shadowing
    col *= shadow;

    // Special effects based on note count
    if(pc.note_count > 0u) {
        // Create rhythmic flashes on specific bands
        int flashBand = int(pc.note_count % 7u);
        if(band == flashBand) {
            col *= 1.0 + sin(iTime * 20.0) * 0.2;
        }
    }

    // Strobe effect at high energy
    if(audioEnergy > 0.8) {
        float strobe = step(0.5, fract(iTime * 16.0));
        col = mix(col, vec3(1.0) - col, strobe * (audioEnergy - 0.8) * 2.0);
    }

    // Vignette
    vec2 q = fragUV;
    float vignetteBase = 16.0 * q.x * q.y * (1.0 - q.x) * (1.0 - q.y);
    float vignettePower = 0.12 - audioEnergy * 0.05;
    float vignette = pow(vignetteBase, vignettePower);
    col *= vignette * 0.8 + 0.2;

    outColor = vec4(col, 1.0);
}