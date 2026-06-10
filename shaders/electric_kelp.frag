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

void main(){
    vec2 res = vec2(pc.render_w, pc.render_h);
    vec2 uv = (fragUV - 0.5) * vec2(res.x / res.y, 1.0);

    float t = pc.time * (0.6 + pc.cc1 * 0.8);

    // Deep water background
    vec3 col = mix(vec3(0.0, 0.04, 0.1), vec3(0.0, 0.01, 0.04), uv.y * 0.5 + 0.5);

    // Kelp stalks — many wavy vertical bands
    for (int k = 0; k < 14; k++){
        float fk = float(k);
        float seed = fk * 13.7;

        float baseX = (hash(vec2(fk, 1.0)) - 0.5) * 2.6;

        // Tip position controlled by sway
        float sway = sin(uv.y * 1.4 + t * (0.5 + hash(vec2(fk, 2.0))) + seed) * 0.12;
        sway += sin(uv.y * 3.0 + t * 1.2 + seed) * 0.05;
        sway *= 0.7 + 0.3 * (uv.y + 1.0);

        float kelpX = baseX + sway;
        float dx = abs(uv.x - kelpX);

        // Thickness
        float thickness = 0.006 + 0.003 * sin(uv.y * 5.0 + seed);
        // Distance falloff for the stalk
        float stalk = smoothstep(thickness * 5.0, thickness, dx);

        // Lightning bursts — random along the stalk
        float burstY = sin(uv.y * 25.0 + t * 5.0 + seed);
        burstY = pow(0.5 + 0.5 * burstY, 12.0);
        float bolt = burstY * stalk * 4.0;

        // Audio reactive — more lightning with note velocity
        bolt *= 1.0 + pc.note_velocity * 4.0;

        // Color: cyan core, magenta edges
        vec3 boltCol = mix(vec3(0.6, 0.9, 1.0), vec3(0.9, 0.4, 1.0), hash(vec2(fk, 4.0)));

        col += boltCol * stalk * 0.4;
        col += boltCol * bolt * 0.6;

        // Glow halo
        col += boltCol * exp(-dx * 80.0) * 0.15;

        // Branches — diagonal jaggies
        float branchY = mod(uv.y * 6.0 + seed, 1.0);
        float branchX = abs((uv.x - kelpX) * 6.0 - (branchY - 0.5) * 0.8);
        float branch = smoothstep(0.06, 0.0, branchX) * smoothstep(0.5, 0.45, branchY) * smoothstep(0.0, 0.1, branchY);
        col += boltCol * branch * 0.3 * (0.5 + pc.cc74);
    }

    // Floating sparks
    vec2 sp = uv * 30.0;
    sp.y += t * 1.5;
    float spark = pow(hash(floor(sp)), 50.0);
    spark *= smoothstep(0.3, 0.0, fract(sp.y));
    col += vec3(0.5, 0.9, 1.0) * spark * (0.4 + pc.note_velocity * 0.6);

    // Water caustics on the floor
    if (uv.y < -0.5){
        float caust = sin(uv.x * 8.0 + t) * sin(uv.x * 5.0 - t * 0.7);
        col += vec3(0.05, 0.15, 0.25) * abs(caust) * smoothstep(-0.5, -1.0, uv.y);
    }

    // Vignette
    col *= 1.0 - 0.3 * dot(uv, uv);

    outColor = vec4(col, 1.0);
}
