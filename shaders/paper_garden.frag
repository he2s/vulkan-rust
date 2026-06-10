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

float hash(vec2 p){ return fract(sin(dot(p, vec2(91.3, 47.7))) * 43758.5453); }

void main(){
    vec2 res = vec2(pc.render_w, pc.render_h);
    vec2 uv = (fragUV - 0.5) * vec2(res.x / res.y, 1.0);

    float t = pc.time * (0.2 + pc.cc1 * 0.2);

    // Paper background — warm off-white with subtle fibre noise
    float fibre = hash(floor(uv * 800.0));
    vec3 col = vec3(0.96, 0.93, 0.88) - fibre * 0.04;
    // Soft warm gradient
    col *= 0.9 + 0.1 * (1.0 - length(uv));

    // Draw multiple folded flowers
    for (int j = 0; j < 5; j++){
        float fj = float(j);
        float seed = fj * 11.7;
        vec2 fc = vec2(
            cos(seed) * 0.35 + sin(t * 0.2 + seed) * 0.05,
            sin(seed * 1.3) * 0.3 + cos(t * 0.15 + seed) * 0.05
        );

        vec2 d = uv - fc;
        d *= rot(t * 0.1 + seed);

        float r = length(d);
        float a = atan(d.y, d.x);

        // Petal count varies per flower
        float petals = 5.0 + mod(seed, 3.0);
        float petAng = mod(a, TAU / petals) - 0.5 * TAU / petals;

        // Petal shape — sin curve in polar
        float petalR = 0.12 + 0.06 * cos(petAng * 2.0);
        petalR *= 0.7 + 0.3 * sin(t * 0.5 + seed);
        // Bloom — open/close with audio
        petalR *= 0.6 + 0.4 * (0.5 + 0.5 * sin(t * 0.7 + seed)) * (0.8 + pc.note_velocity * 0.4);

        float petal = smoothstep(petalR, petalR * 0.85, r);

        // Origami creases — thin radial lines
        float creases = step(0.5, abs(cos(a * petals)));
        creases *= smoothstep(petalR * 0.9, 0.0, r);

        // Color: soft pastel per flower
        vec3 petalCol = mix(
            vec3(0.95, 0.55, 0.65),
            vec3(0.55, 0.75, 0.95),
            0.5 + 0.5 * sin(seed * 2.7)
        );
        petalCol = mix(petalCol, vec3(0.95, 0.85, 0.4), abs(sin(seed * 0.9)) * 0.5);

        col = mix(col, petalCol, petal * 0.85);
        col = mix(col, petalCol * 0.6, creases * petal * 0.3);

        // Center disc
        float cdisc = smoothstep(0.02, 0.015, r);
        col = mix(col, vec3(0.95, 0.85, 0.3), cdisc);

        // Soft drop shadow on paper
        float shadow = smoothstep(petalR + 0.04, petalR + 0.01, r);
        col *= 1.0 - shadow * 0.08;
    }

    // Floating petals drifting
    for (int k = 0; k < 8; k++){
        float fk = float(k);
        vec2 dp = vec2(
            hash(vec2(fk, 1.0)) * 2.0 - 1.0,
            mod(hash(vec2(fk, 2.0)) - t * 0.05, 1.0) * 2.0 - 1.0
        );
        dp.x += sin(t * 0.3 + fk) * 0.05;

        float dpd = length((uv - dp) * vec2(1.0, 1.5));
        float petal = smoothstep(0.012, 0.005, dpd);

        vec3 dpc = mix(vec3(0.9, 0.5, 0.7), vec3(0.6, 0.85, 0.95), hash(vec2(fk, 3.0)));
        col = mix(col, dpc, petal * 0.6);
    }

    // Subtle warm vignette
    col *= 1.0 - 0.15 * dot(uv, uv);

    outColor = vec4(col, 1.0);
}
