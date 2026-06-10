#version 450

layout(push_constant) uniform PushConstants {
    float time;
    uint  mouse_x;
    uint  mouse_y;
    uint  mouse_pressed;
    float note_velocity;
    float pitch_bend;
    float cc1;
    float cc74;
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

#define TAU 6.2831853

float hash(vec2 p){ return fract(sin(dot(p, vec2(91.3, 47.7))) * 43758.5453); }

void main(){
    vec2 res = vec2(pc.render_w, pc.render_h);
    vec2 uv = (fragUV - 0.5) * vec2(res.x / res.y, 1.0);

    float t = pc.time * (0.3 + pc.cc1 * 0.2);

    // Background — deep ocean / space
    vec3 col = mix(vec3(0.02, 0.0, 0.08), vec3(0.0, 0.02, 0.06), uv.y * 0.5 + 0.5);

    // Faint plankton stars
    vec2 sp = uv * 80.0;
    float star = pow(hash(floor(sp)), 50.0);
    col += vec3(0.6, 0.8, 1.0) * star * (0.5 + 0.5 * sin(t * 3.0 + hash(floor(sp)) * 20.0));

    // Multiple jellyfish floating
    for (int j = 0; j < 4; j++){
        float fj = float(j);
        float seed = fj * 17.3;

        // Position with gentle drift
        vec2 jc = vec2(
            sin(t * (0.15 + fj * 0.05) + seed) * 0.5,
            sin(t * 0.1 + seed * 1.7) * 0.3 + sin(t * 0.5 + seed) * 0.05
        );
        jc.x += (fj - 1.5) * 0.35;

        vec2 d = uv - jc;
        float dist = length(d);
        float ang = atan(d.y, d.x);

        // Bell shape — half-disc on top
        float bellR = 0.18 + 0.04 * sin(t * 2.0 + seed);
        float bell = smoothstep(bellR, bellR * 0.6, dist) * step(d.y, 0.0);

        // Bell ridges
        float ridges = 0.5 + 0.5 * cos(ang * 8.0);
        ridges *= smoothstep(0.0, 0.3, -d.y) * smoothstep(bellR, bellR * 0.6, dist);

        // Color per jelly — pastel
        vec3 jellyCol = vec3(
            0.4 + 0.6 * sin(seed),
            0.5 + 0.5 * sin(seed * 1.7),
            0.7 + 0.3 * sin(seed * 0.9 + 1.0)
        );
        jellyCol = abs(jellyCol);

        // Audio reactive pulse
        float pulse = 1.0 + 0.4 * sin(t * 3.0 + seed) * (0.4 + pc.note_velocity);

        col += jellyCol * bell * 0.5 * pulse;
        col += jellyCol * ridges * 0.3;

        // Tentacles below — wavy lines
        for (int k = 0; k < 5; k++){
            float fk = float(k);
            float tx = jc.x + (fk - 2.0) * 0.025;
            float wave = sin(uv.y * 14.0 + t * 1.5 + fk * 1.3 + seed) * 0.03;
            wave += sin(uv.y * 5.0 - t + seed) * 0.02;
            float tentX = tx + wave;
            float tentLen = step(jc.y - 0.35 - fk * 0.02, uv.y) * step(uv.y, jc.y - 0.04);
            float w = smoothstep(0.006, 0.0, abs(uv.x - tentX)) * tentLen;
            col += jellyCol * w * 0.6 * pulse;
        }

        // Soft glow
        col += jellyCol * exp(-dist * 6.0) * 0.15 * pulse;
    }

    // Soft caustic light from above
    float caust = 0.5 + 0.5 * sin(uv.x * 6.0 + t * 0.7);
    caust *= 0.5 + 0.5 * sin(uv.x * 11.0 - t * 0.5);
    col += vec3(0.1, 0.2, 0.3) * caust * smoothstep(0.5, 1.0, uv.y) * 0.4;

    // Vignette
    col *= 1.0 - 0.35 * dot(uv, uv);

    outColor = vec4(col, 1.0);
}
