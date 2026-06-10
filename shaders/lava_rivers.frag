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

    float t = pc.time * (0.15 + pc.cc1 * 0.2);

    // Domain warp — slow molten flow
    vec2 q = uv * 1.2;
    q.y += t * 0.3;

    vec2 warp = vec2(fbm(q * 1.0 + vec2(0.0, t * 0.5)),
                     fbm(q * 1.0 + vec2(3.7, -t * 0.4)));
    q += warp * 0.4;

    float core = fbm(q * 2.0);
    float detail = fbm(q * 6.0 + warp);

    // Heat map: rock is dark, cracks reveal hot lava
    // Cracks = where core is near 0.5 (boundary)
    float crack = 1.0 - smoothstep(0.0, 0.04, abs(core - 0.5));
    crack += 1.0 - smoothstep(0.0, 0.03, abs(core - 0.3));
    crack += 0.5 * (1.0 - smoothstep(0.0, 0.02, abs(core - 0.7)));
    crack = clamp(crack, 0.0, 2.0);

    // Heat — strongest in cracks, dimming with distance from crack
    float heat = crack * (0.7 + detail * 0.3);
    heat *= 0.6 + pc.note_velocity * 0.8;

    // Rock surface
    vec3 rock = mix(vec3(0.05, 0.03, 0.03), vec3(0.18, 0.08, 0.05), detail);
    rock *= 0.7 + 0.3 * core;

    // Lava color ramp — black → red → orange → yellow → white-hot
    vec3 lava = vec3(0.0);
    lava = mix(vec3(0.2, 0.0, 0.0), vec3(0.9, 0.2, 0.0), smoothstep(0.0, 0.4, heat));
    lava = mix(lava, vec3(1.0, 0.5, 0.05), smoothstep(0.3, 0.7, heat));
    lava = mix(lava, vec3(1.0, 0.85, 0.4), smoothstep(0.6, 1.0, heat));
    lava = mix(lava, vec3(1.0, 1.0, 0.85), smoothstep(0.9, 1.3, heat));

    vec3 col = mix(rock, lava, smoothstep(0.0, 0.4, heat));

    // Hot glow from cracks blooming over rock
    col += lava * smoothstep(0.0, 1.0, crack) * 0.4 * (0.5 + pc.cc74 * 0.5);

    // Embers — small bright spots floating up
    vec2 ep = uv * 25.0;
    ep.y -= t * 2.5;
    float ember = pow(hash(floor(ep)), 30.0);
    ember *= smoothstep(0.3, 0.0, fract(ep.y));
    col += vec3(1.0, 0.6, 0.2) * ember * (0.5 + pc.note_velocity * 0.5);

    // Heat shimmer at top of frame
    float shimmer = smoothstep(0.4, 0.9, uv.y);
    col *= 1.0 + shimmer * 0.1 * sin(uv.y * 80.0 + t * 4.0);

    // Vignette
    col *= 1.0 - 0.25 * dot(uv, uv);

    outColor = vec4(col, 1.0);
}
