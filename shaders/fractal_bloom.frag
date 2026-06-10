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
    return 0.5 + 0.5 * cos(TAU * (vec3(0.9, 0.7, 0.5) * t + vec3(0.0, 0.15, 0.35)));
}

void main(){
    vec2 res = vec2(pc.render_w, pc.render_h);
    vec2 uv = (fragUV - 0.5) * vec2(res.x / res.y, 1.0) * 2.4;

    float t = pc.time * (0.2 + pc.cc1 * 0.3);

    // Polar transformation for floral symmetry
    float r = length(uv);
    float a = atan(uv.y, uv.x);

    // Petal count varies with note
    float petals = 6.0 + float(pc.note_count % 8u);
    a = mod(a, TAU / petals) - 0.5 * TAU / petals;

    // Build vec2 back, but spiral-twist it
    vec2 p = vec2(cos(a), sin(a)) * r;
    p *= rot(r * 0.6 + t * 0.5 + pc.pitch_bend * 2.0);

    // Julia-style iteration in petal space
    vec2 z = p * 1.4;
    vec2 c = vec2(0.7885 * cos(t * 0.3 + pc.osc_ch1 * 3.0),
                  0.7885 * sin(t * 0.3 + pc.osc_ch2 * 3.0));

    float iter = 0.0;
    float trap = 1e10;
    for (int i = 0; i < 48; i++) {
        z = vec2(z.x*z.x - z.y*z.y, 2.0*z.x*z.y) + c;
        trap = min(trap, dot(z, z));
        if (dot(z, z) > 4.0) break;
        iter += 1.0;
    }

    float m = iter / 48.0;
    float t_orbit = sqrt(max(0.0, trap));

    // Color from iteration count and orbit trap
    vec3 col = pal(m * 1.5 + t * 0.05);
    col = mix(col, pal(t_orbit * 0.8 + 0.5), smoothstep(0.0, 1.0, t_orbit));

    // Bloom highlight where escape was fast (outer petals)
    float bloom = pow(1.0 - m, 3.0);
    col += pal(t * 0.1 + 0.2) * bloom * (0.4 + pc.note_velocity * 0.8);

    // Dark center for unescaped points (inside the set)
    if (iter >= 47.5) {
        col = mix(col, vec3(0.02, 0.0, 0.06), 0.85);
    }

    // Audio-reactive sparkle on edges
    float edge = smoothstep(0.4, 0.7, m) * (1.0 - smoothstep(0.7, 1.0, m));
    col += pal(iter * 0.1 + t) * edge * pc.cc74 * 0.8;

    // Soft vignette
    col *= 1.0 - 0.3 * dot(uv * 0.4, uv * 0.4);

    outColor = vec4(col, 1.0);
}
