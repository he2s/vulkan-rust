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

// Optimized marching parameters
#define MAXSTEPS 32
#define HITTHRESHOLD 0.008
#define FAR 18.0

// IFS parameters
#define BASE_NIFS 5
#define BASE_SCALE 2.2
#define BASE_TRANSLATE 3.2

mat2 rot(float angle) {
    float c = cos(angle);
    float s = sin(angle);
    return mat2(c, -s, s, c);
}

vec4 sd2d(vec2 p, float o) {
    float energyLevel = clamp(pc.note_velocity, 0.0, 1.0);
    float modulation = clamp(pc.cc1, 0.0, 1.0);
    float brightness = clamp(pc.cc74, 0.0, 1.0);

    float timeScale = mix(0.6, 1.4, modulation);
    float iTime = pc.time * timeScale;
    float time = 0.15 * o + 0.5 * iTime;

    float s = mix(0.45, 0.65, energyLevel);
    p *= s;

    float RADIUS = 1.0 + 0.3 * sin(iTime) * mix(0.8, 1.2, brightness);

    int NIFS = BASE_NIFS;
    float SCALE = BASE_SCALE + clamp(pc.pitch_bend, -0.3, 0.3);
    float TRANSLATE = BASE_TRANSLATE + modulation * 1.5;

    vec3 col = vec3(0.0);
    p = p * rot(-0.3 * time + energyLevel);

    for (int i = 0; i < 5; i++) {
        p.x = abs(p.x); col.r += 1.0;
        p = p * rot(0.8 * sin(time));
        p.y = abs(p.y); col.g += 1.0;
        if (p.x - p.y < 0.0) { p.xy = p.yx; col.b += 1.0; }
        p = p * SCALE - TRANSLATE;
        p = p * rot(0.25 * iTime);
    }

    float d = 0.4 * (length(p) - RADIUS) * pow(SCALE, -5.0) / s;
    col *= 0.2;

    // Enhanced colors
    vec3 color1 = mix(vec3(0.9, 0.3, 0.1), vec3(1.0, 0.5, 0.0), energyLevel);
    vec3 color2 = mix(vec3(0.1, 0.4, 0.9), vec3(0.0, 0.6, 1.0), brightness);
    vec3 color3 = mix(vec3(0.8, 0.1, 0.8), vec3(1.0, 0.3, 1.0), modulation);

    vec3 oc = mix(mix(color1, color2, col.r), color3, col.g * col.b);

    // High saturation
    vec3 gray = vec3(dot(oc, vec3(0.299, 0.587, 0.114)));
    oc = mix(gray, oc, 2.5);

    return vec4(oc, d);
}

vec4 map(vec3 p) {
    return sd2d(p.xz, p.y);
}

float shadow(vec3 ro, vec3 rd) {
    float res = 1.0;
    float t = 0.3;

    for (int i = 0; i < 6; i++) {
        float h = map(ro + rd * t).w;
        res = min(res, 3.5 * h / t);
        if (h < HITTHRESHOLD) break;
        t += h * 1.2;
        if (t > 8.0) break;
    }
    return clamp(res, 0.15, 1.0);
}

void main() {
    vec2 iResolution = (pc.render_w > 0u && pc.render_h > 0u)
    ? vec2(pc.render_w, pc.render_h)
    : vec2(800.0, 600.0);

    vec2 fragCoord = fragUV * iResolution;

    // Audio-reactive parameters
    float energyLevel = clamp(pc.note_velocity, 0.0, 1.0);
    float pitchFactor = clamp(pc.pitch_bend, -1.0, 1.0);
    float modulation = clamp(pc.cc1, 0.0, 1.0);
    float brightness = clamp(pc.cc74, 0.0, 1.0);

    float timeScale = mix(0.4, 1.2, modulation);
    float iTime = pc.time * timeScale;

    // Camera setup
    float height = mix(-0.4, -0.2, energyLevel * 0.5);
    float rot = iTime * mix(0.03, 0.1, modulation);
    float dist = mix(8.0, 11.0, abs(pitchFactor) * 0.7) + 0.5 * sin(0.3 * iTime);

    vec2 cameraOffset = vec2(0.0);
    if (pc.osc_ch1 != 0.0 || pc.osc_ch2 != 0.0) {
        cameraOffset = vec2(pc.osc_ch1, pc.osc_ch2) * 1.5;
    } else if (pc.mouse_pressed > 0u) {
        vec2 mouseNorm = vec2(float(pc.mouse_x), float(pc.mouse_y)) / iResolution;
        cameraOffset = (mouseNorm - 0.5) * 3.0;
    }

    vec3 ro = dist * vec3(cos(rot), height, sin(rot)) + vec3(cameraOffset.x, 0.0, cameraOffset.y);
    vec3 lookAt = vec3(0.0, 0.0, 0.0);
    vec3 fw = normalize(lookAt - ro);

    vec3 right = normalize(cross(vec3(0.0, 1.0, 0.8), fw));
    vec3 up = normalize(cross(fw, right));

    float lightRot = rot + energyLevel * 1.5;
    vec3 lightPos = ro + vec3(cos(lightRot) * 3.0, 2.0, sin(lightRot) * 3.0);

    // Single sample rendering
    vec2 uv = fragCoord / iResolution.xy;
    uv -= 0.5;
    uv.x *= iResolution.x / iResolution.y;

    float focalLength = mix(0.5, 0.65, brightness);
    vec3 rd = normalize(fw * focalLength + right * uv.x + up * uv.y);

    // Raymarch
    float t = 0.0;
    vec3 pos;
    vec3 sdfCol = vec3(0.0);
    bool hit = false;

    for (int i = 0; i < MAXSTEPS; i++) {
        pos = ro + rd * t;
        vec4 mr = map(pos);
        float d = mr.w;

        if (abs(d) < HITTHRESHOLD) {
            sdfCol = mr.rgb;
            hit = true;
            break;
        }

        if (t > FAR) break;
        t += d * 0.8;
    }

    vec3 col = vec3(0.0);
    float depth = t / FAR;

    // DEBUG: Force blur parameters to visible ranges for testing
    float focusPoint = 0.5;        // Fixed focus point
    float focusRange = 0.3;        // Fixed focus range

    float focusDistance = abs(depth - focusPoint);
    float blurAmount = smoothstep(0.0, focusRange, focusDistance);

    if (hit) {
        col = sdfCol;

        // Lighting
        vec3 toLight = normalize(lightPos - pos);
        float s = shadow(pos, toLight);
        col *= mix(0.25, 1.1, s);
        col *= mix(1.3, 2.2, energyLevel);

        // Enhanced saturation
        vec3 gray = vec3(dot(col, vec3(0.299, 0.587, 0.114)));
        col = mix(gray, col, 1.9);

        // Bloom effect
        if (brightness > 0.3) {
            float bloomAmount = (brightness - 0.3) * 1.43;
            vec3 bloom = col * col * bloomAmount * 0.8;
            col += bloom;
        }

        // SIMPLIFIED AGGRESSIVE BLUR TEST
        if (blurAmount > 0.1) {
            // Very obvious blur effect for testing
            col *= 0.5;                    // Dim significantly
            col = mix(col, vec3(0.5), 0.7); // Mix with gray
            col.r += 0.3 * blurAmount;     // Add red tint to show blur
        }

    } else {
        // Background - also show depth info
        col = vec3(depth, depth, depth); // Show depth as grayscale
    }

    // DEBUG: Show blur amount as green channel in non-hit areas
    if (!hit) {
        col.g = blurAmount;
    }

    // Vignette
    float vignette = smoothstep(1.3, 0.3, length(uv));
    col *= mix(0.6, 1.0, vignette);

    outColor = vec4(col, 1.0);
}