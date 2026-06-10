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

float noise(vec2 p){
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1, 0)), f.x),
               mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), f.x), f.y);
}

float fbm(vec2 p){
    float v = 0.0, a = 0.5;
    mat2 r = mat2(0.8, 0.6, -0.6, 0.8);
    for (int i = 0; i < 5; i++){ v += a * noise(p); p = r * p * 2.0; a *= 0.5; }
    return v;
}

void main(){
    vec2 res = vec2(pc.render_w, pc.render_h);
    vec2 uv = (fragUV - 0.5) * vec2(res.x / res.y, 1.0);

    float t = pc.time * (0.08 + pc.cc1 * 0.08);

    // Pastel sky gradient
    vec3 top    = vec3(0.95, 0.75, 0.85);  // soft pink
    vec3 mid    = vec3(0.85, 0.85, 0.95);  // pale lilac
    vec3 bot    = vec3(0.7, 0.9, 0.95);    // pale teal

    float vert = uv.y * 0.5 + 0.5;
    vec3 col = mix(bot, mid, smoothstep(0.0, 0.6, vert));
    col = mix(col, top, smoothstep(0.5, 1.0, vert));

    // Slow drifting waves — gentle horizontal undulation
    float wave = sin(uv.x * 2.0 + fbm(uv * 1.2 + t * 0.2) * 3.0 + t * 0.15);
    wave += 0.5 * sin(uv.x * 5.0 - t * 0.1 + uv.y * 2.0);
    wave *= 0.04;
    float waveLine = smoothstep(0.004, 0.0, abs(uv.y - 0.0 - wave));
    col = mix(col, vec3(0.95, 0.85, 0.95), waveLine * 0.3);

    // Soft pastel clouds — large blurry blobs
    for (int i = 0; i < 5; i++){
        float fi = float(i);
        vec2 cp = vec2(sin(t * 0.05 + fi * 1.3) * 0.8, 0.3 + 0.4 * sin(fi));
        cp.x = mod(cp.x + 2.0, 2.6) - 1.3;
        float dist = length((uv - cp) * vec2(0.6, 1.2));
        float cloud = exp(-dist * 2.5) * (0.5 + 0.3 * fbm(uv * 3.0 + fi));
        vec3 cc = mix(vec3(1.0, 0.95, 0.95), vec3(0.95, 0.9, 1.0), hash(vec2(fi, 1.0)));
        col = mix(col, cc, cloud * 0.6);
    }

    // Floating orbs — softly pulsing
    for (int j = 0; j < 8; j++){
        float fj = float(j);
        float seed = fj * 11.7;
        vec2 op = vec2(
            sin(t * 0.2 + seed) * 0.7,
            cos(t * 0.15 + seed * 1.3) * 0.5 + sin(t * 0.05 + seed) * 0.1
        );
        float od = length(uv - op);
        float pulse = 0.5 + 0.5 * sin(t * 0.5 + seed) * (0.7 + pc.note_velocity * 0.4);
        vec3 oc = mix(vec3(1.0, 0.85, 0.9), vec3(0.85, 0.9, 1.0), 0.5 + 0.5 * sin(seed * 2.0));

        // Soft halo
        col = mix(col, oc, exp(-od * 12.0) * 0.45 * pulse);

        // Bright core
        col = mix(col, vec3(1.0), smoothstep(0.018, 0.0, od) * pulse * 0.8);

        // Subtle trailing line down
        float trail = smoothstep(0.003, 0.0, abs(uv.x - op.x)) * smoothstep(0.0, 0.3, op.y - uv.y);
        col = mix(col, oc, trail * 0.15 * pulse);
    }

    // Sparkles
    vec2 sp = uv * 80.0 + vec2(0.0, t * 0.5);
    float sparkle = pow(hash(floor(sp)), 100.0);
    sparkle *= 0.5 + 0.5 * sin(t * 4.0 + hash(floor(sp)) * 20.0);
    col += vec3(1.0, 0.95, 0.9) * sparkle * 0.5;

    // Soft warm vignette
    col *= 1.0 - 0.18 * dot(uv, uv);

    // Slight tone lift — dreamy
    col = pow(col, vec3(0.92));

    outColor = vec4(col, 1.0);
}
