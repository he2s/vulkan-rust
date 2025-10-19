#version 450

// Audio-reactive animated lines - inspired by IQ's work
// Variation 6: Minimal - Subtle thin lines
// Adapted for interactive MIDI/audio control

layout(push_constant) uniform PushConstants {
    float time;
    uint  mouse_x;
    uint  mouse_y;
    uint  mouse_pressed;
    float note_velocity;
    float pitch_bend;
    float cc1;    // mod wheel / line speed
    float cc74;   // cutoff / line density
    uint  note_count;
    uint  last_note;
    float osc_ch1;
    float osc_ch2;
    uint  render_w;
    uint  render_h;
} pc;

layout(location = 0) in vec2 fragUV;
layout(location = 1) in float vertexEnergy;

layout(location = 0) out vec4 outColor;

// Global variables
float iTime;
vec2 iResolution;
vec2 iMouse;
float audioEnergy;
float audioMod;
float audioBright;
float audioBend;

float freqs[16];

// --------------------------------------------------------
// Noise functions
// --------------------------------------------------------

vec3 hash3(float n) {
    return fract(sin(vec3(n, n+1.0, n+2.0)) * vec3(43758.5453123, 22578.1459123, 19642.3490423));
}

vec3 snoise3(in float x) {
    float p = floor(x);
    float f = fract(x);

    f = f*f*(3.0-2.0*f);

    return -1.0 + 2.0*mix(hash3(p+0.0), hash3(p+1.0), f);
}

// --------------------------------------------------------
// Line segment utilities
// --------------------------------------------------------

float dot2(in vec3 v) {
    return dot(v,v);
}

vec2 usqdLineSegment(vec3 ro, vec3 rd, vec3 v0, vec3 v1) {
    vec3 oa = ro - v0;
    vec3 ob = ro - v1;
    vec3 va = rd*dot(oa,rd)-oa;
    vec3 vb = rd*dot(ob,rd)-ob;

    vec3 w1 = va;
    vec3 w2 = vb-w1;
    float h = clamp(-dot(w1,w2)/dot(w2,w2), 0.0, 1.0);

    float di = dot2(w1+w2*h);

    return vec2(di, h);
}

// --------------------------------------------------------
// Ray casting through animated lines
// --------------------------------------------------------

vec3 castRay(vec3 ro, vec3 rd, float linesSpeed) {
    vec3 col = vec3(0.0);

    vec3 p = vec3(0.2);
    float h = 0.0;

    // Number of line segments - controlled by cc74
    int segments = int(32.0 + 32.0 * audioBright); // Fewer lines

    // Precompute loop-invariant values
    const float h_step = 1.0 / 128.0;
    const float color_freq = 62.831; // 10.0 * 6.2831
    const vec3 color_phase = vec3(0.0, 0.6, 0.9);

    float noise_time_base = linesSpeed * 0.015 * iTime;
    float width_multiplier = 0.25 * (1.0 + audioEnergy * 0.5);

    for(int i=0; i<128; i++) {
        if (i >= segments) break;

        vec3 op = p;

        // Generate line position using noise
        p = 1.25 * normalize(snoise3(64.0*h + noise_time_base));

        // Distance from ray to line segment
        vec2 dis = usqdLineSegment(ro, rd, op, p);

        // Audio-reactive color
        vec3 lcol = 0.6 + 0.4*sin(color_freq*h + color_phase);

        // Modulate with audio frequency - optimize pow(m, 2.0) to m*m
        float m = freqs[int(h * 16.0)] * (1.0 + 2.0*h);
        m = m * m; // Faster than pow(m, 2.0)

        // Line tapering
        float f = 1.0 - 4.0*dis.y*(1.0-dis.y);
        float width = (1240.0 - 1000.0*f) * width_multiplier;

        // Falloff along line - cache dis.y*dis.y
        float dis_y_sq = dis.y * dis.y;
        float ff = exp(-0.06 * dis_y_sq * dis.y);
        ff *= m;

        // Glow effect - cache width*dis.x
        float width_dist = width * dis.x;
        col += 0.15*lcol*exp(-0.4*width_dist)*ff; // Subtle glow
        col += 0.25*lcol*exp(-10.0*width_dist)*ff; // Thin lines

        h += h_step;
    }

    return col;
}

// --------------------------------------------------------
// Main
// --------------------------------------------------------

void main() {
    // Setup resolution
    iResolution = (pc.render_w > 0u && pc.render_h > 0u)
        ? vec2(pc.render_w, pc.render_h)
        : vec2(800.0, 600.0);

    // Cache audio parameters
    audioEnergy = clamp(pc.note_velocity, 0.0, 1.0);
    audioMod = clamp(pc.cc1, 0.0, 1.0);
    audioBright = clamp(pc.cc74, 0.0, 1.0);
    audioBend = clamp(pc.pitch_bend, -1.0, 1.0);

    iTime = pc.time;
    iMouse = vec2(float(pc.mouse_x), float(pc.mouse_y));

    vec2 fragCoord = fragUV * iResolution;
    vec2 q = fragCoord.xy / iResolution.xy;
    vec2 p = -1.0 + 2.0*q;
    p.x *= iResolution.x / iResolution.y;

    // Mouse control
    vec2 mo = vec2(0.5);
    if (pc.mouse_pressed > 0u) {
        mo = iMouse.xy / iResolution.xy;
    } else {
        mo += vec2(pc.osc_ch1, pc.osc_ch2) * 0.5;
    }

    // Generate frequency data from audio parameters
    const float inv_16 = 1.0 / 16.0;
    const float two_pi = 6.28318;

    // Precompute note phase (used if note_count > 0)
    float notePhase = float(pc.note_count % 8u) * 0.125; // 1.0/8.0 = 0.125

    for(int i=0; i<16; i++) {
        float t = float(i) * inv_16;

        // Simulate frequency spectrum
        float freq = audioEnergy * exp(-t * 3.0)
                   + vertexEnergy * exp(-abs(t - 0.5) * 4.0)
                   + audioBright * exp(-(1.0 - t) * 3.0);

        // Add variation based on note count
        if (pc.note_count > 0u) {
            freq += 0.3 * sin(notePhase * two_pi + t * two_pi);
        }

        // Optimize pow(freq, 3.0) to freq*freq*freq
        freq = freq * freq * freq;
        freqs[i] = clamp(1.9 * freq, 0.0, 1.0);
    }

    float time = iTime;

    // Audio-reactive camera movement
    vec3 ta = vec3(0.0, 0.0, 0.0);

    // Camera speed controlled by mod wheel
    float camSpeed = 1.0 + 10.0 * audioMod;

    // Beat/rhythm from note count
    float beat = float(pc.note_count);
    time += beat * 0.5 * audioMod;

    // Line animation speed controlled by cc1
    float linesSpeed = audioMod;

    // Camera target with audio reactivity
    ta = 0.2*vec3(
        cos(0.1*time + audioBend),
        0.0*sin(0.1*time),
        sin(0.07*time + audioBend * 0.5)
    );

    // Camera position
    vec3 ro = vec3(
        1.0*cos(camSpeed*0.05*time + 6.28*mo.x),
        0.0,
        1.0*sin(camSpeed*0.05*time + 6.2831*mo.x)
    );

    // Camera roll with pitch bend
    float roll = 0.25*sin(camSpeed*0.01*time) + audioBend * 0.3;

    // Camera matrix
    vec3 cw = normalize(ta - ro);
    vec3 cp = vec3(sin(roll), cos(roll), 0.0);
    vec3 cu = normalize(cross(cw, cp));
    vec3 cv = normalize(cross(cu, cw));
    vec3 rd = normalize(p.x*cu + p.y*cv + 1.2*cw);

    // Distortion effect with energy
    float curve = audioEnergy * 0.5;
    rd.xy += curve*0.025*vec2(sin(34.0*q.y), cos(34.0*q.x));
    rd = normalize(rd);

    // Audio-reactive camera distance
    ro *= 1.0 - linesSpeed*0.5*freqs[1];

    // Cast ray and render
    vec3 col = castRay(ro, rd, 1.0 + 20.0*linesSpeed);

    // Optimize col*col*2.4
    col = col * col * 2.4;

    // Brightness boost with audio
    col *= 1.0 + audioEnergy * 0.5;

    // Color tint based on pitch bend
    if (abs(audioBend) > 0.1) {
        vec3 tint = mix(vec3(1.0, 0.9, 0.8), vec3(0.8, 0.9, 1.0), audioBend * 0.5 + 0.5);
        col *= tint;
    }

    // Note count affects color cycling
    if (pc.note_count > 0u) {
        float noteHue = float(pc.note_count % 6u) / 6.0;
        vec3 cycleColor = 0.5 + 0.5*cos(6.28318*(noteHue + vec3(0.0, 0.33, 0.67)));
        col = mix(col, col * cycleColor, 0.2);
    }

    // Vignette - cache vignette base calculation
    float vignette_base = 16.0 * q.x * q.y * (1.0 - q.x) * (1.0 - q.y);
    col *= 0.15 + 0.85 * pow(vignette_base, 0.15);

    outColor = vec4(col, 1.0);
}
