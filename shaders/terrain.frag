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

// Marching parameters
#define MAXSTEPS 50
#define HITTHRESHOLD 0.009
#define FAR 25.0

// Audio-reactive IFS parameters
#define BASE_NIFS 6
#define BASE_SCALE 2.3
#define BASE_TRANSLATE 3.5

mat2 rot(float angle) {
    float c = cos(angle);
    float s = sin(angle);
    return mat2(c, -s, s, c);
}

vec4 sd2d(vec2 p, float o) {
    // Audio-reactive parameters
    float energyLevel = clamp(pc.note_velocity, 0.0, 1.0);
    float pitchFactor = clamp(pc.pitch_bend, -1.0, 1.0);
    float modulation = clamp(pc.cc1, 0.0, 1.0);
    float brightness = clamp(pc.cc74, 0.0, 1.0);

    // Time with audio reactivity
    float timeScale = mix(0.5, 2.0, modulation);
    float iTime = pc.time * timeScale;
    float time = 0.2 * o + 0.6 * iTime;

    float s = mix(0.4, 0.7, energyLevel);
    p *= s;

    // Audio-reactive radius
    float RADIUS = (1.0 + sin(iTime)) * mix(0.8, 1.5, brightness);

    // Dynamic IFS iterations based on note count
    int NIFS = BASE_NIFS + int(pc.note_count);
    NIFS = min(NIFS, 12); // Cap to prevent performance issues

    // Audio-reactive scale and translate
    float SCALE = BASE_SCALE + pitchFactor * 0.5;
    float TRANSLATE = BASE_TRANSLATE + modulation * 2.0;

    int i;
    vec3 col = vec3(0.0);

    // Initial twist with audio reactivity
    p = p * rot(-0.4 * time + energyLevel * 1.5);

    for (i = 0; i < NIFS; i++) {
        if (p.x < 0.0) {
            p.x = -p.x;
            col.r++;
        }

        // Rotation amount varies with audio
        float rotAmount = 0.9 * sin(time) + pitchFactor * 0.3;
        p = p * rot(rotAmount);

        if (p.y < 0.0) {
            p.y = -p.y;
            col.g++;
        }

        if (p.x - p.y < 0.0) {
            p.xy = p.yx;
            col.b++;
        }

        p = p * SCALE - TRANSLATE;

        // Additional rotation with brightness control
        p = p * rot(0.3 * iTime + brightness * 0.5);
    }

    float d = 0.425 * (length(p) - RADIUS) * pow(SCALE, float(-i)) / s;
    col /= float(NIFS);

    // Audio-reactive color mixing
    vec3 color1 = mix(vec3(0.7, col.g, 0.2), vec3(0.9, col.g * 1.2, 0.1), energyLevel);
    vec3 color2 = mix(vec3(0.2, col.r, 0.7), vec3(0.1, col.r * 1.3, 0.9), brightness);
    vec3 oc = mix(color1, color2, col.b + modulation * 0.3);

    // Boost saturation with high energy
    if (energyLevel > 0.7) {
        oc *= 1.0 + (energyLevel - 0.7) * 1.5;
    }

    return vec4(oc, d);
}

vec4 map(vec3 p) {
    return sd2d(p.xz, p.y);
}

float shadow(vec3 ro, vec3 rd) {
    float h = 0.0;
    float k = mix(2.0, 5.0, clamp(pc.cc74, 0.0, 1.0)); // Audio-reactive shadow softness
    float res = 1.0;
    float t = 0.2; // bias

    for (int i = 0; t < 15.0; i++) {
        h = map(ro + rd * t).w;
        res = min(res, k * h / t);
        if (h < HITTHRESHOLD) {
            break;
        }
        t = t + h;
    }
    return clamp(res + 0.05, 0.0, 1.0);
}

void main() {
    // Get resolution
    vec2 iResolution = (pc.render_w > 0u && pc.render_h > 0u)
    ? vec2(pc.render_w, pc.render_h)
    : vec2(800.0, 600.0);

    vec2 fragCoord = fragUV * iResolution;

    // Audio-reactive parameters
    float energyLevel = clamp(pc.note_velocity, 0.0, 1.0);
    float pitchFactor = clamp(pc.pitch_bend, -1.0, 1.0);
    float modulation = clamp(pc.cc1, 0.0, 1.0);

    // Time scaling
    float timeScale = mix(0.3, 1.5, modulation);
    float iTime = pc.time * timeScale;

    // Camera with audio reactivity
    float height = mix(-0.6, -0.2, energyLevel);
    float rot = iTime * mix(0.05, 0.2, modulation);
    float dist = mix(7.0, 12.0, abs(pitchFactor)) + 1.0 * sin(0.5 * iTime);

    // OSC/Mouse input affects camera position
    vec2 cameraOffset = vec2(0.0);
    if (pc.osc_ch1 != 0.0 || pc.osc_ch2 != 0.0) {
        cameraOffset = vec2(pc.osc_ch1, pc.osc_ch2) * 3.0;
    } else if (pc.mouse_pressed > 0u) {
        vec2 mouseNorm = vec2(float(pc.mouse_x), float(pc.mouse_y)) / iResolution;
        cameraOffset = (mouseNorm - 0.5) * 6.0;
    }

    vec3 ro = dist * vec3(cos(rot), height, sin(rot)) + vec3(cameraOffset.x, 0.0, cameraOffset.y);
    vec3 lookAt = vec3(0.0, 0.0, 0.0);
    vec3 fw = normalize(lookAt - ro);

    // Tilting camera for audio reactivity
    vec3 tiltAxis = mix(vec3(0.0, 1.0, 1.0), vec3(0.0, 1.0, 0.5), energyLevel);
    vec3 right = normalize(cross(tiltAxis, fw));
    vec3 up = normalize(cross(fw, right));
    right = normalize(cross(up, fw));

    // Light with audio reactivity
    rot += sin(iTime) * mix(0.1, 0.4, energyLevel);
    vec3 lightPos = dist * vec3(cos(rot), height, sin(rot));

    // Simplified single-sample raymarch for performance
    float t = 0.0;
    float smallest = 500.0;
    vec3 pos, closest;
    vec3 sdfCol;

    vec2 uv = fragCoord / iResolution.xy;
    uv -= 0.5;
    uv.x *= iResolution.x / iResolution.y;

    // Focal length varies with brightness
    float focalLength = mix(0.4, 0.7, clamp(pc.cc74, 0.0, 1.0));
    vec3 rd = normalize(fw * focalLength + right * uv.x + up * uv.y);

    for (int i = 0; i < MAXSTEPS; i++) {
        pos = ro + rd * t;
        vec4 mr = map(pos);
        float d = mr.w;

        if (d < smallest) {
            smallest = d;
            closest = pos;
            sdfCol = mr.rgb;
        }

        if (abs(d) < HITTHRESHOLD || t > FAR) {
            break;
        }
        t += d;
    }

    pos = closest;
    vec3 col;

    if (t < FAR) {
        col = sdfCol;
        vec3 toLight = normalize(lightPos - pos);
        float s = shadow(pos, toLight);
        col *= s;
        col = mix(col, 1.5 * col, 1.0 - s);

        // Audio-reactive brightness boost
        col *= mix(0.8, 1.8, energyLevel);
    } else {
        col = vec3(0.0);
    }

    // Add vertex energy contribution
    col *= (1.0 + 0.2 * clamp(vertexEnergy, 0.0, 1.0));

    // Output color with depth in alpha for post-processing
    outColor = vec4(col, t / FAR);
}