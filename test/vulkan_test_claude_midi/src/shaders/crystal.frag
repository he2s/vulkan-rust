#version 450

// Push constants
layout(push_constant) uniform PushConstants {
    float time;
    uint mouse_x;
    uint mouse_y;
    uint mouse_pressed;
    float note_velocity;
    float pitch_bend;
    float cc1;
    float cc74;
    uint note_count;
    uint last_note;
    float osc_ch1;
    float osc_ch2;
    uint render_w;
    uint render_h;
} pc;

layout(location = 0) in vec2 frag_uv;
layout(location = 1) in vec2 frag_screen_pos;

layout(location = 0) out vec4 out_color;

const float PI = 3.14159265359;
const float TAU = PI * 2.0;

// === NOISE FUNCTIONS ===

vec3 hash3(float n) {
    return fract(sin(vec3(n, n+1.0, n+2.0)) * vec3(43758.5453123, 22578.1459123, 19642.3490423));
}

vec3 snoise3(in float x) {
    float p = floor(x);
    float f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    return -1.0 + 2.0 * mix(hash3(p + 0.0), hash3(p + 1.0), f);
}

// === DISTANCE FUNCTIONS ===

float dot2(in vec3 v) {
    return dot(v, v);
}

vec2 usqdLineSegment(vec3 ro, vec3 rd, vec3 v0, vec3 v1) {
    vec3 oa = ro - v0;
    vec3 ob = ro - v1;
    vec3 va = rd * dot(oa, rd) - oa;
    vec3 vb = rd * dot(ob, rd) - ob;

    vec3 w1 = va;
    vec3 w2 = vb - w1;
    float h = clamp(-dot(w1, w2) / dot(w2, w2), 0.0, 1.0);

    float di = dot2(w1 + w2 * h);

    return vec2(di, h);
}

// === FREQUENCY SIMULATION ===

float getFrequency(int i) {
    // Simulate frequency bands using our available inputs
    float base_freq = 0.1;

    if (i < 4) {
        // Low frequencies - use note velocity and OSC1
        base_freq += pc.note_velocity * 0.4 + pc.osc_ch1 * 0.3;
    } else if (i < 8) {
        // Mid frequencies - use CC1 and pitch bend
        base_freq += pc.cc1 * 0.5 + abs(pc.pitch_bend) * 0.3;
    } else if (i < 12) {
        // High frequencies - use CC74 and OSC2
        base_freq += pc.cc74 * 0.6 + pc.osc_ch2 * 0.4;
    } else {
        // Ultra high frequencies - combination
        base_freq += (pc.cc1 + pc.cc74 + pc.note_velocity) / 3.0 * 0.5;
    }

    // Add some variation per frequency band
    base_freq += sin(pc.time * 2.0 + float(i) * 0.5) * 0.1;

    return clamp(1.9 * pow(base_freq, 3.0), 0.0, 1.0);
}

// === RAY CASTING ===

vec3 castRay(vec3 ro, vec3 rd, float linesSpeed) {
    vec3 col = vec3(0.0);

    float mindist = 10000.0;
    vec3 p = vec3(0.2);
    float h = 0.0;

    // Get primary frequency for tube radius
    float rad = 0.04 + 0.15 * getFrequency(0);
    float mint = 0.0;

    for (int i = 0; i < 128; i++) {
        vec3 op = p;

        // Generate organic movement using noise
        p = 1.25 * normalize(snoise3(64.0 * h + linesSpeed * 0.015 * pc.time));

        vec2 dis = usqdLineSegment(ro, rd, op, p);

        // Color variation along the line
        vec3 lcol = 0.6 + 0.4 * sin(10.0 * TAU * h + vec3(0.0, 0.6, 0.9));

        // Audio reactivity - simulate the texture lookup with our frequency data
        float freq_index = h * 8.0; // Map to our frequency range
        int freq_i = int(freq_index) % 16;
        float m = pow(getFrequency(freq_i), 2.0) * (1.0 + 2.0 * h);

        // MIDI note influence on colors
        if (pc.note_count > 0) {
            float note_hue = float(pc.last_note) / 127.0;
            lcol += 0.3 * vec3(
                sin(note_hue * TAU + h * 10.0),
                sin(note_hue * TAU + h * 10.0 + TAU/3.0),
                sin(note_hue * TAU + h * 10.0 + 2.0*TAU/3.0)
            ) * pc.note_velocity;
        }

        // OSC influence on colors
        lcol.r += pc.osc_ch1 * sin(h * 20.0 + pc.time * 3.0) * 0.2;
        lcol.g += pc.osc_ch2 * cos(h * 15.0 + pc.time * 2.0) * 0.2;

        // Pitch bend affects color intensity
        lcol *= 1.0 + abs(pc.pitch_bend) * 0.5;

        float f = 1.0 - 4.0 * dis.y * (1.0 - dis.y);
        float width = 1240.0 - 1000.0 * f;
        width *= 0.25;
        float ff = 1.0 * exp(-0.06 * dis.y * dis.y * dis.y);
        ff *= m;

        // Core glow
        col += 0.3 * lcol * exp(-0.3 * width * dis.x) * ff;
        // Outer glow
        col += 0.5 * lcol * exp(-8.0 * width * dis.x) * ff;

        h += 1.0 / 128.0;
    }

    return col;
}

void main() {
    vec2 q = frag_uv;
    vec2 p = -1.0 + 2.0 * q;
    p.x *= float(pc.render_w) / float(pc.render_h);

    // Mouse interaction
    vec2 mo = vec2(float(pc.mouse_x) / float(pc.render_w),
                   float(pc.mouse_y) / float(pc.render_h));

    float time = pc.time;

    // === AUDIO-REACTIVE CAMERA BEHAVIOR ===

    // Simulate the "isFast" behavior with our inputs
    float audio_intensity = (pc.note_velocity + pc.cc1 + pc.cc74) / 3.0;
    float isFast = smoothstep(0.3, 0.5, audio_intensity);

    // Camera speed based on audio intensity
    float camSpeed = 1.0 + 40.0 * isFast;

    // Beat-like behavior using note count and velocity
    float beat = floor(max((time - 10.0) / 0.8, 0.0)) * float(pc.note_count > 0 ? 1 : 0);
    time += beat * 10.0 * isFast * pc.note_velocity;
    camSpeed *= mix(1.0, sign(sin(beat * 1.0)), isFast);

    // Lines animation speed
    float linesSpeed = audio_intensity;

    // === CAMERA SETUP ===

    vec3 ta = vec3(0.0);
    ta = 0.2 * vec3(
        cos(0.1 * time) + pc.osc_ch1 * 0.5,
        sin(0.1 * time) * pc.pitch_bend,
        sin(0.07 * time) + pc.osc_ch2 * 0.3
    );

    // Camera position with mouse and audio control
    vec3 ro = vec3(
        1.0 * cos(camSpeed * 0.05 * time + TAU * mo.x),
        pc.pitch_bend * 0.5,
        1.0 * sin(camSpeed * 0.05 * time + TAU * mo.x)
    );

    float roll = 0.25 * sin(camSpeed * 0.01 * time) + pc.pitch_bend * 0.1;

    // Camera transformation
    vec3 cw = normalize(ta - ro);
    vec3 cp = vec3(sin(roll), cos(roll), 0.0);
    vec3 cu = normalize(cross(cw, cp));
    vec3 cv = normalize(cross(cu, cw));
    vec3 rd = normalize(p.x * cu + p.y * cv + 1.2 * cw);

    // Audio-reactive screen distortion
    float curve = audio_intensity * 0.5;
    rd.xy += curve * 0.025 * vec2(sin(34.0 * q.y), cos(34.0 * q.x));
    rd = normalize(rd);

    // Audio-reactive camera pull-back
    ro *= 1.0 - linesSpeed * 0.5 * getFrequency(1);

    // === RENDER ===

    vec3 col = castRay(ro, rd, 1.0 + 20.0 * linesSpeed);
    col = col * col * 2.4;

    // === AUDIO-REACTIVE POST-PROCESSING ===

    // Intensity boost based on note activity
    col *= 1.0 + pc.note_velocity * 0.5;

    // OSC channels affect color balance
    col.r *= 1.0 + pc.osc_ch1 * 0.2;
    col.b *= 1.0 + pc.osc_ch2 * 0.2;

    // Pitch bend creates color shifts
    if (abs(pc.pitch_bend) > 0.01) {
        col.rgb = mix(col.rgb, col.brg, abs(pc.pitch_bend) * 0.3);
    }

    // Mouse interaction brightens the scene
    if (pc.mouse_pressed != 0) {
        col *= 1.0 + 0.3;
    }

    // Vignette effect
    col *= 0.15 + 0.85 * pow(16.0 * q.x * q.y * (1.0 - q.x) * (1.0 - q.y), 0.15);

    // Ensure minimum visibility
    col = max(col, vec3(0.01, 0.005, 0.02));

    out_color = vec4(col, 1.0);
}