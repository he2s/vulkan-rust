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

mat2 rot(float a){ float c=cos(a),s=sin(a); return mat2(c,-s,s,c); }

vec3 pal(float t){
    return 0.5 + 0.5 * cos(TAU * (vec3(1.0, 1.0, 0.9) * t + vec3(0.0, 0.33, 0.67)));
}

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
    for (int i = 0; i < 5; i++) { v += a * noise(p); p = r * p * 2.1; a *= 0.5; }
    return v;
}

void main(){
    vec2 res = vec2(pc.render_w, pc.render_h);
    vec2 uv = (fragUV - 0.5) * vec2(res.x / res.y, 1.0) * 1.6;

    float t = pc.time * (0.5 + pc.cc1 * 0.6);

    // Distort space — molten flow
    vec2 q = uv;
    q += 0.4 * vec2(fbm(uv * 1.5 + t * 0.3), fbm(uv * 1.5 - t * 0.4));
    q += 0.15 * vec2(fbm(q * 3.0 + t * 0.7), fbm(q * 3.0 - t * 0.6));

    // Glass thickness via fbm — like a slab of warped molten material
    float thickness = fbm(q * 2.0 + t * 0.2);
    thickness = pow(thickness, 1.4);

    // Bands of iridescence
    float bands = sin(thickness * 18.0 + t * 1.5 + pc.pitch_bend * 3.0);
    bands = 0.5 + 0.5 * bands;

    // Color via iridescent palette
    vec3 col = pal(thickness * 1.2 + bands * 0.3 + t * 0.05);

    // Inner glow — where thickness is low, brighten with warm core
    float core = exp(-thickness * 3.0) * (0.6 + pc.note_velocity * 1.2);
    col += vec3(1.0, 0.5, 0.2) * core;

    // Surface highlights — animated specular streaks
    float spec = pow(max(0.0, sin(q.y * 6.0 + t * 2.0 + q.x * 3.0)), 30.0);
    col += vec3(1.0, 0.95, 0.85) * spec * (0.4 + pc.cc74 * 0.6);

    // OSC-driven rotating sheen
    vec2 sh = uv * rot(t * 0.2 + pc.osc_ch1 * 3.0);
    float sheen = exp(-abs(sh.x * 6.0)) * 0.5;
    col += sheen * pal(t * 0.3);

    // Edge darkening
    col *= 1.0 - 0.35 * length(uv) * length(uv);

    outColor = vec4(col, 1.0);
}
