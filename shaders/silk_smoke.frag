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
    mat2 r = mat2(0.8, 0.6, -0.6, 0.8);
    for (int i = 0; i < 6; i++){ v += a * noise(p); p = r * p * 2.0; a *= 0.5; }
    return v;
}

mat2 rot(float a){ float c=cos(a),s=sin(a); return mat2(c,-s,s,c); }

void main(){
    vec2 res = vec2(pc.render_w, pc.render_h);
    vec2 uv = (fragUV - 0.5) * vec2(res.x / res.y, 1.0);

    float t = pc.time * (0.08 + pc.cc1 * 0.07);

    // Slow domain warp
    vec2 p = uv * 1.5;
    p *= rot(t * 0.1 + sin(t * 0.2) * 0.2);
    vec2 w = vec2(fbm(p * 1.1 + t * 0.2), fbm(p * 1.1 - t * 0.15));
    p += w * 0.5;
    w = vec2(fbm(p * 2.0 + w + t * 0.05), fbm(p * 2.0 - w - t * 0.04));
    p += w * 0.25;

    // Silk strands — long thin sin-bands of noise
    float strands = 0.0;
    for (int i = 0; i < 5; i++){
        float fi = float(i);
        float scale = 1.5 + fi * 0.7;
        vec2 q = p * scale;
        q *= rot(fi * 0.4);
        float band = sin(q.y * 1.8 + fbm(q) * 3.0 + t * 0.3 + fi);
        band = pow(0.5 + 0.5 * band, 4.0);
        strands += band * (0.5 - fi * 0.05);
    }

    // Soft cream/pearl base
    vec3 base = mix(vec3(0.08, 0.07, 0.12), vec3(0.95, 0.9, 0.85), strands * 0.5);

    // Iridescent shimmer — varies with normal-like derivative
    float ir = fbm(p * 4.0 + strands);
    vec3 sheen = 0.5 + 0.5 * cos(6.28 * (vec3(0.7, 0.55, 0.85) * ir + vec3(0.0, 0.25, 0.55)) + t * 0.2);

    vec3 col = base * 0.7 + sheen * strands * 0.4;

    // Highlights at strand peaks
    float hi = smoothstep(1.4, 2.2, strands);
    col += vec3(1.0, 0.95, 0.92) * hi * 0.7;

    // Audio breathes the silk gently
    col *= 1.0 + 0.1 * sin(t * 0.5) * (0.3 + pc.note_velocity * 0.7);

    // Gentle vignette
    col *= 1.0 - 0.18 * dot(uv, uv);

    // Slightly cool shadows
    col = mix(vec3(0.04, 0.04, 0.06), col, smoothstep(0.0, 0.4, length(col)));

    outColor = vec4(col, 1.0);
}
