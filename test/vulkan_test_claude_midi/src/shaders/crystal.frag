#version 450

layout(push_constant) uniform PushConstants {
    float time;
    uint  mouse_x;
    uint  mouse_y;
    uint  mouse_pressed;
    float note_velocity; // MIDI velocity or blended RMS
    float pitch_bend;    // [-1,1] or blended low band
    float cc1;           // mid band
    float cc74;          // high band
    uint  note_count;
    uint  last_note;

} pc;

layout(location = 0) in  vec2 fragCoord;   // OK if unused
layout(location = 0) out vec4 fragColor;

mat2 rot(float a) {
    float c = cos(a), s = sin(a);
    return mat2(c, -s, s, c);
}

// Distance/color helper (ported from the s(p) macro)
float wf_s(in vec3 p, inout vec4 accum, float T, float t) {
    vec3 q = p;

    // “Torus-ish” distance
    float d = length(vec2(length(q.xy + vec2(0.5)) - 0.5, q.z)) - 0.01;

    // Truchet: quantize angle and rotate tile frame
    float ang = atan(q.y, q.x);
    float k   = round((ang - T) * 3.8) / 3.8 + T;
    q.yx = rot(k) * q.yx;
    q.x -= 0.5;

    // Color/falloff (original form)
    vec4 col = (sin(t + T) * 0.1 + 0.1) * (1.0 + cos(t + T * 0.5 + vec4(0.0, 1.0, 2.0, 0.0)));
    float falloff = 0.5 + pow(length(q) * 50.0, 1.3);
    accum += col / falloff;

    return d;
}

void main() {
    // --- Interactivity mappings ---
    float speed    = 1.0 + pc.cc1 * 1.5 + pc.note_velocity * 0.5;  // mid/velocity -> speed
    float hueShift = pc.cc74;                                       // highs -> tint
    float wobble   = pc.pitch_bend;                                 // bend -> camera wobble
    float T        = pc.time * speed;

    // iResolution-style values
    vec2 Rxy = vec2(float(max(1000, 1u)), float(max(1000, 1u)));
    vec3 R   = vec3(Rxy, 1.0);

    // Shadertoy coordinate setup:
    // F starts as pixel coords, then center & scale: F += F - R.xy
    vec2 F = gl_FragCoord.xy;
    F = F + (F - Rxy);

    vec4 O = vec4(0.0);   // color accumulator
    float t = 0.0;

    // Raymarch — same structure as original (more iterations for quality)
    const int MAX_STEPS = 28;
    const float MAX_DIST = 100.0;

    for (int i = 0; i < MAX_STEPS; ++i) {
        // Direction from rotated screen coords
        vec2 Fr = rot(t * 0.10) * F;
        vec3 rd = normalize(vec3(Fr, R.y));
        vec3 p  = t * rd;

        // Camera motion (time + pitch wobble)
        p.xz *= rot(T / 4.0 + wobble * 0.5);
        p.yz *= rot(T / 3.0 + wobble * 0.2);
        p.x  += T;

        // Domain repetition in three orientations
        vec3 pf = fract(p) - 0.5;
        float d = wf_s(pf, O, T, t);
        d = min(d, wf_s(vec3(-pf.y, pf.z, pf.x), O, T, t));
        d = min(d, wf_s(-pf.zxy, O, T, t));

        t += d;
        if (t > MAX_DIST) break;
    }

    // Map accumulator to RGB and apply interactive tints
    vec3 base = O.rgb;

    // Highs add warm tint; velocity adds glow
    base *= mix(vec3(1.0), vec3(1.0, 0.8 + 0.2 * hueShift, 0.6 + 0.4 * hueShift), 0.35);
    base += pc.note_velocity * 0.25;

    // Mouse halo
    vec2 m  = vec2(float(pc.mouse_x), float(pc.mouse_y));
    float mg = exp(-length(gl_FragCoord.xy - m) / 200.0) * (pc.mouse_pressed != 0u ? 0.5 : 0.2);
    base += mg;

    // Simple gamma
    fragColor = vec4(pow(max(base, 0.0), vec3(1.7545)), 1.0);
}
