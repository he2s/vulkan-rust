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
    for (int i = 0; i < 5; i++){ v += a * noise(p); p *= 2.05; a *= 0.5; }
    return v;
}

void main(){
    vec2 res = vec2(pc.render_w, pc.render_h);
    vec2 uv = (fragUV - 0.5) * vec2(res.x / res.y, 1.0);

    float t = pc.time * (0.1 + pc.cc1 * 0.15);

    float r = length(uv);
    float a = atan(uv.y, uv.x);

    // Spiral arm parameter
    float spiral = a + log(max(r, 0.001)) * (3.0 + pc.osc_ch1 * 2.0) - t * 0.3;

    // 4 arms
    float arms = 0.5 + 0.5 * sin(spiral * 2.0);
    arms = pow(arms, 3.0);

    // Density along arms — fbm for dust lanes
    float dust = fbm(vec2(spiral * 3.0, r * 6.0));
    float density = arms * (0.5 + dust * 0.5);

    // Falloff with radius
    density *= exp(-r * 1.5);

    // Galactic core — bright bulge
    float core = exp(-r * 6.0) * 1.2;
    core += exp(-r * 12.0) * 1.5 * (0.7 + pc.note_velocity * 0.6);

    // Color: warm core, cool arms, magenta dust
    vec3 coreCol = mix(vec3(1.0, 0.9, 0.6), vec3(1.0, 0.7, 0.3), smoothstep(0.0, 0.3, r));
    vec3 armCol = mix(vec3(0.4, 0.6, 1.0), vec3(0.7, 0.5, 1.0), smoothstep(0.2, 1.0, r));
    vec3 dustCol = vec3(0.6, 0.15, 0.4);

    vec3 col = vec3(0.0, 0.005, 0.02);
    col += armCol * density * 0.6;
    col += dustCol * density * dust * 0.4 * (1.0 - smoothstep(0.0, 0.2, r));
    col += coreCol * core;

    // Stars — many tiny + a few bright
    vec2 sp = uv * 200.0;
    float star = pow(hash(floor(sp)), 70.0);
    star *= 0.5 + 0.5 * sin(t * 3.0 + hash(floor(sp)) * 30.0);
    col += vec3(0.9, 0.95, 1.0) * star * (0.8 + density * 1.2);

    // Brighter accent stars in arms
    vec2 sp2 = uv * 80.0;
    float bigStar = pow(hash(floor(sp2)), 90.0);
    col += vec3(1.0, 0.85, 0.6) * bigStar * 2.0 * density * (0.7 + pc.note_velocity);

    // Subtle nebular bloom around core
    col += coreCol * exp(-r * 2.0) * 0.15 * (1.0 + sin(t * 0.5) * 0.3);

    // Outer fade
    col *= 1.0 - smoothstep(1.0, 1.6, r) * 0.6;

    // Slight chromatic spin
    col.r += density * sin(spiral * 2.0 + t) * 0.05;
    col.b += density * cos(spiral * 2.0 - t) * 0.05;

    outColor = vec4(col, 1.0);
}
