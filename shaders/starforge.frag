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

float noise(vec3 p){
    vec3 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float n = i.x + i.y * 57.0 + 113.0 * i.z;
    float a = fract(sin(n) * 43758.5453);
    float b = fract(sin(n + 1.0) * 43758.5453);
    float c = fract(sin(n + 57.0) * 43758.5453);
    float d = fract(sin(n + 58.0) * 43758.5453);
    float e = fract(sin(n + 113.0) * 43758.5453);
    float g = fract(sin(n + 114.0) * 43758.5453);
    float h = fract(sin(n + 170.0) * 43758.5453);
    float k = fract(sin(n + 171.0) * 43758.5453);
    return mix(mix(mix(a, b, f.x), mix(c, d, f.x), f.y),
               mix(mix(e, g, f.x), mix(h, k, f.x), f.y), f.z);
}

float fbm(vec3 p){
    float v = 0.0, a = 0.5;
    for (int i = 0; i < 5; i++){ v += a * noise(p); p *= 2.07; a *= 0.5; }
    return v;
}

void main(){
    vec2 res = vec2(pc.render_w, pc.render_h);
    vec2 uv = (fragUV - 0.5) * vec2(res.x / res.y, 1.0);

    float t = pc.time * (0.4 + pc.cc1 * 0.8);

    float r = length(uv);
    float a = atan(uv.y, uv.x);

    // Core star — pulsing brightness
    float corePulse = 0.6 + 0.4 * sin(t * 1.5) * (0.5 + pc.note_velocity);
    float core = exp(-r * 8.0) * corePulse * 2.0;

    // Multi-layer corona — fbm in 3D for animated tendrils
    float corona = 0.0;
    for (int i = 0; i < 4; i++){
        float fi = float(i);
        float radial = exp(-r * (3.0 - fi * 0.4));
        vec3 sp = vec3(uv * (3.0 + fi), t * 0.3 + fi);
        float n = fbm(sp + vec3(cos(a * 2.0), sin(a * 2.0), 0.0) * r * 2.0);
        corona += radial * n * 0.4;
    }
    corona *= 1.5;

    // Plasma jets — two opposing
    float jet = 0.0;
    for (int j = 0; j < 2; j++){
        float fj = float(j);
        float dir = (fj * 2.0 - 1.0);
        float along = uv.y * dir; // top jet for j=0, bottom for j=1 (after sign flip in angle)
        // Project onto rotating axis
        float jetAng = t * 0.1 + pc.pitch_bend * 1.5;
        vec2 axis = vec2(sin(jetAng), cos(jetAng));
        float al = dot(uv, axis) * dir;
        float perp = abs(dot(uv, vec2(-axis.y, axis.x)));
        float width = 0.08 + perp * 0.3;
        float profile = exp(-perp * perp / (width * width));
        profile *= smoothstep(0.0, 0.3, al) * smoothstep(1.4, 0.4, al);
        jet += profile;
    }

    // Shockwave rings — expanding circles synced to notes
    float ringT = mod(t * 0.4, 2.0);
    float ring = exp(-pow((r - ringT) * 8.0, 2.0));
    ring *= (1.0 - smoothstep(1.2, 1.5, ringT));
    ring *= 0.4 + pc.note_velocity * 0.8;

    // Background — distant nebula + stars
    vec3 col = vec3(0.02, 0.0, 0.05) + vec3(0.1, 0.05, 0.2) * (1.0 - r) * 0.3;

    vec2 sp = uv * 200.0;
    float star = pow(hash(floor(sp)), 80.0);
    col += vec3(0.9, 0.9, 1.0) * star;

    // Compose star
    vec3 coreCol = mix(vec3(1.0, 1.0, 0.95), vec3(1.0, 0.7, 0.3), smoothstep(0.0, 0.4, r));
    vec3 coronaCol = mix(vec3(1.0, 0.5, 0.2), vec3(0.4, 0.1, 0.6), smoothstep(0.2, 1.0, r));
    vec3 jetCol = vec3(0.6, 0.85, 1.0);

    col += coreCol * core;
    col += coronaCol * corona;
    col += jetCol * jet * (0.6 + pc.cc74 * 0.6);
    col += vec3(1.0, 0.8, 0.5) * ring;

    // Lens flare streaks
    float flare = exp(-abs(uv.y) * 60.0) * exp(-r * 1.0) * 0.4;
    flare += exp(-abs(uv.x) * 60.0) * exp(-r * 1.0) * 0.4;
    col += vec3(1.0, 0.9, 0.7) * flare * (0.5 + pc.note_velocity * 0.7);

    outColor = vec4(col, 1.0);
}
