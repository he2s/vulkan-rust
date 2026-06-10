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

void main(){
    vec2 res = vec2(pc.render_w, pc.render_h);
    vec2 uv = (fragUV - 0.5) * vec2(res.x / res.y, 1.0);

    float t = pc.time * (0.1 + pc.cc1 * 0.1);

    // Slow drifting fbm — the nebula
    vec2 q = uv * 1.4;
    q += vec2(t * 0.05, -t * 0.03);

    float n1 = fbm(q * 1.0);
    float n2 = fbm(q * 2.3 + n1);
    float n3 = fbm(q * 4.5 + n2 * 0.8);

    float density = pow(n2, 1.5) * (0.6 + n3 * 0.4);

    // Deep purple → magenta → soft rose palette
    vec3 deep   = vec3(0.04, 0.0, 0.12);
    vec3 mid    = vec3(0.25, 0.05, 0.45);
    vec3 bright = vec3(0.85, 0.4, 0.7);
    vec3 hot    = vec3(1.0, 0.85, 0.95);

    vec3 col = mix(deep, mid, smoothstep(0.0, 0.5, density));
    col = mix(col, bright, smoothstep(0.4, 0.75, density));
    col = mix(col, hot, smoothstep(0.7, 0.9, density));

    // Distant pinprick stars — sparse
    vec2 sp = uv * 250.0;
    float star = pow(hash(floor(sp)), 80.0);
    star *= 0.7 + 0.3 * sin(t * 2.0 + hash(floor(sp)) * 30.0);
    col += vec3(0.95, 0.85, 1.0) * star * (0.5 + (1.0 - density) * 0.7);

    // Gentle breathing — audio reactivity
    col *= 1.0 + 0.15 * sin(t * 1.5) * (0.3 + pc.note_velocity * 0.7);

    // Soft inner glow / center pulse
    float r = length(uv);
    col += vec3(0.4, 0.2, 0.6) * exp(-r * 2.2) * 0.15 * (0.6 + pc.osc_ch1 * 0.4);

    // Soft vignette
    col *= 1.0 - 0.25 * dot(uv, uv);

    // Gamma — keep velvety not crushed
    col = pow(col, vec3(0.9));

    outColor = vec4(col, 1.0);
}
