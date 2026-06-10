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

float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }

float noise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    float a = hash(i), b = hash(i + vec2(1, 0));
    float c = hash(i + vec2(0, 1)), d = hash(i + vec2(1, 1));
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

float fbm(vec2 p) {
    float v = 0.0, a = 0.5;
    for (int i = 0; i < 6; i++) {
        v += a * noise(p);
        p *= 2.03;
        a *= 0.5;
    }
    return v;
}

void main() {
    vec2 res = vec2(pc.render_w, pc.render_h);
    vec2 uv = (fragUV - 0.5) * vec2(res.x / res.y, 1.0);
    float t = pc.time * (0.15 + pc.cc1 * 0.2);

    // Sky gradient — deep night to predawn
    vec3 sky = mix(vec3(0.01, 0.02, 0.06), vec3(0.03, 0.05, 0.12), smoothstep(-0.8, 0.6, uv.y));
    sky = mix(sky, vec3(0.08, 0.04, 0.13), smoothstep(0.2, -0.6, uv.y) * 0.5);

    // Stars
    vec2 sp = uv * 200.0;
    float star = pow(hash(floor(sp)), 60.0);
    star *= smoothstep(-0.1, 0.5, uv.y);
    vec3 col = sky + vec3(star) * (0.8 + sin(t * 4.0 + hash(floor(sp)) * 30.0) * 0.2);

    // Aurora ribbons — multiple stacked sheets
    for (int i = 0; i < 4; i++) {
        float fi = float(i);
        float y = 0.05 + fi * 0.18;
        float wave = sin(uv.x * (1.5 + fi * 0.4) + t * (0.3 + fi * 0.15)) * 0.08;
        wave += fbm(vec2(uv.x * 2.0 + t * 0.2, fi)) * 0.15;
        float band = uv.y - y - wave;

        float thickness = 0.18 + 0.08 * sin(t * 0.5 + fi);
        float density = exp(-band * band / (thickness * thickness));

        // Aurora curtain: vertical streaks
        float curtain = fbm(vec2(uv.x * 6.0 + fi * 7.0, t * 0.4 + fi));
        density *= 0.4 + 0.6 * curtain;

        // Audio-reactive shimmer
        density *= 1.0 + pc.note_velocity * 0.6 * sin(uv.x * 20.0 + t * 5.0 + fi);

        // Per-ribbon hue: green → cyan → violet → rose
        vec3 ribbon = mix(
            vec3(0.1, 0.95, 0.55),
            vec3(0.55, 0.4, 1.0),
            fi / 3.0 + pc.pitch_bend * 0.3
        );
        ribbon = mix(ribbon, vec3(1.0, 0.4, 0.7), smoothstep(0.5, 1.0, pc.cc74) * 0.4);

        col += ribbon * density * (0.7 + pc.osc_ch1 * 0.3);
    }

    // Distant horizon glow
    float horizon = smoothstep(-0.5, -0.2, uv.y) * smoothstep(-0.05, -0.2, uv.y);
    col += vec3(0.15, 0.08, 0.18) * horizon * 0.6;

    // Soft snow ground reflection
    if (uv.y < -0.15) {
        float ground = exp(-(uv.y + 0.15) * 4.0);
        col += vec3(0.06, 0.1, 0.13) * ground * (0.5 + pc.note_velocity * 0.3);
    }

    // Subtle vignette
    col *= 1.0 - 0.3 * dot(uv, uv);

    outColor = vec4(pow(col, vec3(0.95)), 1.0);
}
