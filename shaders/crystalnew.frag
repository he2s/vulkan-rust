#version 450

// Push constants from application
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

// Beat detection values
    float bpm;
    float time_to_next_beat;      // 0.0 to 1.0
    float time_since_last_beat;   // 0.0 to 1.0
    uint  beats_per_bar;
} pc;

layout(location = 0) in vec2 fragUV;
layout(location = 1) in float vertexEnergy;
layout(location = 2) in vec3 worldPos;

layout(location = 0) out vec4 outColor;

// Constants
#define PI 3.141592653589793238
#define TAU (2.0 * PI)
#define EPSILON 0.008
#define MAX_STEPS 20
#define MAX_DIST 100.0

// Helper macros
#define min2(a, b) ((a.x < b.x) ? a : b)
#define sat(x) clamp(x, 0.0, 1.0)

// Rotation matrix
mat2 rot(float a) {
    float c = cos(a);
    float s = sin(a);
    return mat2(c, -s, s, c);
}

// Beat-synchronized neon color burst palette
vec3 neonBurst(float t, float intensity) {
    // Only activate on high energy OR on beat
    float beatPulse = 1.0 - pc.time_since_last_beat;
    beatPulse = pow(beatPulse, 2.0); // Exponential decay

    // Trigger on velocity OR beat
    if(intensity < 0.7 && beatPulse < 0.3) {
        return vec3(0.0);
    }

    // Pick neon color based on note count, but cycle with beats
    float beatCycle = float(uint(pc.time * pc.bpm / 60.0) % 6u);
    float colorIndex = mix(float(pc.note_count % 6u), beatCycle, 0.3);
    vec3 neonColor;

    if(colorIndex < 1.0) {
        neonColor = vec3(0.0, 1.0, 1.0); // Cyan
    } else if(colorIndex < 2.0) {
        neonColor = vec3(1.0, 0.0, 1.0); // Magenta
    } else if(colorIndex < 3.0) {
        neonColor = vec3(0.0, 1.0, 0.3); // Green
    } else if(colorIndex < 4.0) {
        neonColor = vec3(1.0, 0.3, 0.0); // Orange
    } else if(colorIndex < 5.0) {
        neonColor = vec3(0.3, 0.5, 1.0); // Blue
    } else {
        neonColor = vec3(1.0, 1.0, 0.0); // Yellow
    }

    // Pulse synchronized to BPM
    float bpmPulse = 0.5 + 0.5 * sin(pc.time * (pc.bpm / 60.0) * TAU + t * 5.0);

    // Combine beat pulse with BPM pulse
    float combinedPulse = max(beatPulse, bpmPulse * 0.5);

    return neonColor * max(intensity - 0.7, beatPulse) * 3.0 * combinedPulse;
}

// Smooth union
float smooth_union(float a, float b, float k) {
    float h = sat(0.5 + 0.5 * (b - a) / k);
    return mix(b, a, h) - k * h * (1.0 - h);
}

// Torus SDF
float sdf_torus(vec3 p, vec2 t) {
    vec2 q = vec2(length(p.xz) - t.x, p.y);
    return length(q) - t.y;
}

// Global glow accumulator
vec3 glow;

// Beat-aware SDF scene
vec2 sdf(vec3 p) {
    vec2 di = vec2(120.0, -1.0);

    // Audio parameters
    float energy = clamp(pc.note_velocity, 0.0, 1.0);
    float modulation = clamp(pc.cc1, 0.0, 1.0);
    float brightness = clamp(pc.cc74, 0.0, 1.0);

    // Beat pulse for size modulation
    float beatPulse = 1.0 - pc.time_since_last_beat;
    beatPulse = pow(beatPulse, 1.5);

    // BEAT QUIRK 1: Rings snap to different positions on downbeats
    vec3 snapOffset = vec3(0.0);
    float beatNumber = floor(pc.time * pc.bpm / 60.0);
    bool isDownbeat = (uint(beatNumber) % pc.beats_per_bar) == 0u;

    if(isDownbeat && pc.time_since_last_beat < 0.1) {
        // Strong snap on downbeat
        snapOffset.x = sin(beatNumber * 12.34) * 0.3;
        snapOffset.y = cos(beatNumber * 23.45) * 0.3;
        snapOffset.z = sin(beatNumber * 34.56) * 0.15;
        p += snapOffset * (1.0 - pc.time_since_last_beat * 10.0);
    } else if(beatPulse > 0.5) {
        // Subtle snap on regular beats
        snapOffset.x = sin(beatNumber * 12.34) * 0.1;
        snapOffset.y = cos(beatNumber * 23.45) * 0.1;
        p += snapOffset * beatPulse;
    }

    // Time synchronized to BPM
    float bpmTime = pc.time * (pc.bpm / 120.0); // Normalize to 120 BPM
    float t = bpmTime * (0.3 + energy * 0.5);

    // BEAT QUIRK 2: Time reverses on every 4th bar
    if(uint(beatNumber / float(pc.beats_per_bar)) % 4u == 3u) {
        t *= -1.0;
    }

    // Base torus parameters with beat modulation
    float ringRadius = 1.0 + beatPulse * 0.15;
    float ringThickness = 0.12 + energy * 0.03 + beatPulse * 0.02;

    // BEAT QUIRK 3: Rings pulse in size based on beat position in bar
    float barProgress = float(uint(beatNumber) % pc.beats_per_bar) / float(pc.beats_per_bar);
    ringRadius += sin(barProgress * TAU + pc.pitch_bend * PI) * 0.1;

    // Different thickness for each ring, modulated by beat
    vec2 torusParams1 = vec2(ringRadius, ringThickness * (1.0 + beatPulse * 0.2));
    vec2 torusParams2 = vec2(ringRadius * 0.95, ringThickness * 1.3);
    vec2 torusParams3 = vec2(ringRadius * 1.05, ringThickness * (0.7 + beatPulse * 0.3));

    // First ring with beat-synced wobble
    vec3 p1 = p;
    float wobble1 = sin(beatNumber * 7.0) * 0.1 * modulation;
    p1.yz *= rot(t + wobble1);
    p1.xy *= rot(t * 0.7);
    p1.xz *= rot(t * 0.5);
    float ring_1 = sdf_torus(p1, torusParams1);

    // Second ring - counter rotation, speeds up on beats
    vec3 p2 = p;
    float beatSpeed = 1.0 + beatPulse * 0.5;
    p2.yz *= rot(-t * 0.8 * beatSpeed);
    p2.xy *= rot(t * 0.6);
    p2.xz *= rot(t * 0.4 + PI * 0.5);
    float ring_2 = sdf_torus(p2, torusParams2);

    // Third ring - erratic, jumps on beat
    vec3 p3 = p;
    // BEAT QUIRK 4: Third ring has beat-synchronized jitter
    float beatJitter = beatPulse * sin(pc.time * 50.0) * 0.1;
    p3.yz *= rot(t * 0.9 + beatJitter);
    p3.xy *= rot(t * 0.5);
    p3.xz *= rot(t * 0.6 + PI);

    // BEAT QUIRK 5: Third ring disappears on off-beats in fast tempos
    float ring_3 = sdf_torus(p3, torusParams3);
    if(pc.bpm > 140.0 && uint(beatNumber) % 2u == 1u) {
        ring_3 += beatPulse * 0.5; // Fade out on off-beats
    }

    // Variable smoothness based on beat progress
    float smoothness = 0.3 + modulation * 0.2 + pc.time_to_next_beat * 0.2;
    float combined = smooth_union(ring_1, smooth_union(ring_2, ring_3, smoothness), smoothness);

    // BEAT QUIRK 6: Spikes appear on strong beats
    if(isDownbeat && beatPulse > 0.7) {
        float spikePattern = sin(p.x * 20.0) * sin(p.y * 20.0) * sin(p.z * 20.0);
        if(spikePattern > 0.8) {
            combined -= beatPulse * 0.1; // Create beat-synced bumps
        }
    }

    di = min2(di, vec2(combined, 1.0));
    return di;
}

// Ray marching
vec2 trace(vec3 ro, vec3 rd) {
    vec3 p = ro;
    vec2 di;
    float td = 0.0;

    glow = vec3(0.0);

    for(int i = 0; i < MAX_STEPS; i++) {
        if(td >= MAX_DIST) break;

        di = sdf(p);

        if(di.x < EPSILON) {
            return vec2(td, di.y);
        }

        p += di.x * rd;

        // Beat-aware glow accumulation
        float beatGlow = (1.0 - pc.time_since_last_beat) * 0.5;
        float glowFactor = (1.0 - sat(di.x / 0.5)) * (0.02 + beatGlow * 0.03);
        glow += vec3(glowFactor);

        td = distance(ro, p);
    }

    return vec2(-1.0, -1.0);
}

// Normal calculation
vec3 get_normal(vec3 p) {
    vec2 e = EPSILON * vec2(1.0, -1.0);
    return normalize(
    e.xyy * sdf(p + e.xyy).x +
    e.yxy * sdf(p + e.yxy).x +
    e.yyx * sdf(p + e.yyx).x +
    e.xxx * sdf(p + e.xxx).x
    );
}

// Main rendering with beat awareness
vec3 render(vec2 uv) {
    // Camera setup - pulses with beat
    float beatPulse = 1.0 - pc.time_since_last_beat;
    float camDist = 3.5 - pc.cc1 * 0.5 - beatPulse * 0.2;
    vec3 ro = vec3(0.0, 0.0, -camDist);

    // Camera orbit synchronized to tempo
    float camRotSpeed = pc.bpm / 600.0; // Scale rotation to BPM
    ro.x = sin(pc.time * camRotSpeed * TAU) * 0.5;
    ro.y = cos(pc.time * camRotSpeed * TAU * 0.7) * 0.3;

    // Beat-synced camera shake on downbeats
    float beatNumber = floor(pc.time * pc.bpm / 60.0);
    bool isDownbeat = (uint(beatNumber) % pc.beats_per_bar) == 0u;
    if(isDownbeat && pc.time_since_last_beat < 0.05) {
        ro += vec3(
        sin(beatNumber * 123.4) * 0.1,
        cos(beatNumber * 234.5) * 0.1,
        sin(beatNumber * 345.6) * 0.05
        ) * (0.05 - pc.time_since_last_beat) * 20.0;
    }

    // Mouse control
    if(pc.mouse_pressed > 0u) {
        float mx = (float(pc.mouse_x) / float(pc.render_w) - 0.5) * TAU;
        float my = (float(pc.mouse_y) / float(pc.render_h) - 0.5) * PI;
        vec3 mouseRot = ro;
        mouseRot.xz = ro.xz * cos(mx) + ro.zy * sin(mx);
        mouseRot.y = ro.y * cos(my) - ro.z * sin(my);
        ro = mouseRot;
    }

    // Look at center
    vec3 target = vec3(0.0);
    vec3 forward = normalize(target - ro);
    vec3 right = normalize(cross(vec3(0, 1, 0), forward));
    vec3 up = cross(forward, right);

    vec3 rd = normalize(forward + uv.x * right + uv.y * up);

    // Light position moves with beat
    vec3 lo = vec3(
    0.5 + sin(beatNumber * 2.0) * 0.5,
    1.0,
    -0.5 + cos(beatNumber * 3.0) * 0.5
    );

    vec2 tdi = trace(ro, rd);

    if(tdi.x > 0.0) {
        vec3 p = ro + rd * tdi.x;
        vec3 n = get_normal(p);

        // Basic lighting
        vec3 ld = normalize(lo - p);
        vec3 reflection = reflect(rd, n);

        // Diffuse lighting with beat modulation
        float diff = max(dot(n, ld), 0.0);
        diff = mix(diff, 1.0, beatPulse * 0.2); // Brighten on beat

        // Sharp specular, enhanced on beat
        float spec = pow(max(dot(reflection, ld), 0.0), 32.0 - beatPulse * 16.0);

        // BEAT QUIRK 7: Inverted lighting on every other beat at high BPM
        if(pc.bpm > 160.0 && uint(beatNumber) % 2u == 1u) {
            diff = 1.0 - diff;
        }

        // Base monochrome value
        float mono = diff * 0.8 + spec * (0.5 + beatPulse * 0.5);

        // Edge detection for sharp lines
        float edge = 1.0 - abs(dot(n, -rd));
        edge = pow(edge, 3.0 - beatPulse); // Sharper edges on beat
        mono += edge * (0.3 + beatPulse * 0.3);

        // BEAT QUIRK 8: Stripes pattern synchronized to beat grid
        float barProgress = float(uint(beatNumber) % pc.beats_per_bar) / float(pc.beats_per_bar);
        float beatGrid = beatNumber + barProgress;
        float stripes = sin(p.y * 30.0 + beatGrid * PI) > 0.0 ? 1.0 : 0.8;
        mono *= stripes;

        // Start with black and white
        vec3 color = vec3(mono);

        // BEAT-AWARE NEON: Triggers on beats even without MIDI
        if(pc.note_velocity > 0.7 || beatPulse > 0.7) {
            vec3 neon = neonBurst(dot(n, rd), max(pc.note_velocity, beatPulse));

            // Neon affects edges more, especially on downbeats
            float edgeMultiplier = isDownbeat ? 3.0 : 2.0;
            neon *= (1.0 + edge * edgeMultiplier);

            // Secondary neon flash on off-beats
            if(uint(beatNumber) % 2u == 1u) {
                vec3 neon2 = neonBurst(dot(n, rd) + 0.5, beatPulse);
                neon += neon2 * 0.5;
            }

            color = mix(color, color + neon, sat(max(pc.note_velocity - 0.7, beatPulse) * 3.0));
        }

        // BEAT QUIRK 9: Flash on downbeats
        if(isDownbeat && pc.time_since_last_beat < 0.02) {
            color += vec3(1.0);
        }

        // High contrast with beat modulation
        float contrastPower = 0.1 + beatPulse * 0.3;
        color = smoothstep(contrastPower, 0.9, color);

        // Add beat-pulsing glow
        color += glow * (0.5 + beatPulse * 0.5);

        return color;
    }

    // Background with beat-reactive glow
    return glow * (0.3 + beatPulse * 0.2);
}

void main() {
    vec2 resolution = vec2(float(pc.render_w), float(pc.render_h));
    vec2 uv = (fragUV - 0.5) * 2.0;
    uv.x *= resolution.x / resolution.y;

    // Beat-based UV distortion on strong beats
    float beatNumber = floor(pc.time * pc.bpm / 60.0);
    bool isDownbeat = (uint(beatNumber) % pc.beats_per_bar) == 0u;
    if(isDownbeat && pc.time_since_last_beat < 0.1) {
        float distortion = (0.1 - pc.time_since_last_beat) * 10.0;
        uv *= 1.0 + distortion * 0.1;
    }

    vec3 c = render(uv);

    // Final contrast adjustment
    c = pow(c, vec3(1.0 / 2.2));

    // Beat-aware vignette
    float vignette = 1.0 - smoothstep(0.5 - pc.time_since_last_beat * 0.2, 1.5, length(uv));
    c *= vignette;

    // Threshold for pure black/white with neon, pulses with beat
    if(pc.cc74 > 0.5) {
        float threshold = 0.5 - pc.time_since_last_beat * 0.2;
        c = step(threshold, c);
    }

    outColor = vec4(sat(c), 1.0);
}