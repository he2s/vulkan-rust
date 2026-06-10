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

float hash(vec2 p){ return fract(sin(dot(p, vec2(91.3, 47.7))) * 43758.5453); }

float noise(vec2 p){
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1, 0)), f.x),
               mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), f.x), f.y);
}

float fbm(vec2 p){
    float v = 0.0, a = 0.5;
    for (int i = 0; i < 4; i++){ v += a * noise(p); p *= 2.0; a *= 0.5; }
    return v;
}

void main(){
    vec2 res = vec2(pc.render_w, pc.render_h);
    vec2 uv = (fragUV - 0.5) * vec2(res.x / res.y, 1.0);

    float t = pc.time * (0.12 + pc.cc1 * 0.1);

    // Depth gradient — light at top, abyss at bottom
    float depth = 1.0 - (uv.y * 0.5 + 0.5);
    vec3 col = mix(vec3(0.05, 0.18, 0.28), vec3(0.0, 0.02, 0.06), depth);

    // Light shafts from above
    vec2 sp = uv;
    sp.y -= t * 0.05;
    float shafts = 0.0;
    for (int i = 0; i < 5; i++){
        float fi = float(i);
        float x = (hash(vec2(fi, 1.0)) - 0.5) * 2.0;
        float dx = abs(uv.x - x - sin(t * 0.2 + fi) * 0.05);
        float shaftWidth = 0.04 + 0.02 * sin(t * 0.4 + fi);
        float shaft = smoothstep(shaftWidth, 0.0, dx);
        // Fade with depth
        shaft *= smoothstep(-0.6, 1.0, uv.y);
        shafts += shaft * (0.5 + 0.5 * sin(t + fi));
    }
    col += vec3(0.4, 0.7, 0.85) * shafts * 0.4;

    // Drifting particles (marine snow)
    for (int k = 0; k < 30; k++){
        float fk = float(k);
        vec2 pp = vec2(hash(vec2(fk, 1.0)) * 2.0 - 1.0,
                       fract(hash(vec2(fk, 2.0)) - t * 0.05) * 2.0 - 1.0);
        pp.x += sin(t * 0.3 + fk) * 0.04;
        float pd = length(uv - pp);
        float dot_ = smoothstep(0.006, 0.002, pd);
        col += vec3(0.7, 0.85, 0.9) * dot_ * 0.5;
        col += vec3(0.3, 0.5, 0.6) * exp(-pd * 30.0) * 0.2;
    }

    // Bioluminescent creatures — slow blobs
    for (int j = 0; j < 6; j++){
        float fj = float(j);
        float seed = fj * 19.3;
        vec2 bp = vec2(sin(t * 0.2 + seed) * 0.6,
                       cos(t * 0.15 + seed * 1.7) * 0.4);
        float bd = length(uv - bp);
        vec3 bc = mix(vec3(0.2, 0.7, 0.95), vec3(0.6, 0.3, 0.9), 0.5 + 0.5 * sin(seed));
        float pulse = 0.5 + 0.5 * sin(t * 1.5 + seed * 3.0);
        pulse *= 0.5 + pc.note_velocity * 0.7;
        col += bc * exp(-bd * 6.0) * pulse * 0.15;

        // Bright core
        col += bc * smoothstep(0.025, 0.0, bd) * pulse * 0.6;
    }

    // Subtle water caustic flicker
    float caust = sin(uv.x * 8.0 + t * 0.4) * sin(uv.x * 12.0 - t * 0.6);
    col += vec3(0.05, 0.1, 0.12) * abs(caust) * smoothstep(0.0, 1.0, uv.y) * 0.5;

    // Fog / depth haze
    col = mix(col, vec3(0.0, 0.02, 0.06), depth * 0.5);

    // Vignette
    col *= 1.0 - 0.3 * dot(uv, uv);

    outColor = vec4(col, 1.0);
}
