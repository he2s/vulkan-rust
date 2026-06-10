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
    return 0.5 + 0.5 * cos(TAU * (vec3(0.8, 0.5, 0.7) * t + vec3(0.0, 0.2, 0.5)));
}

void main(){
    vec2 res = vec2(pc.render_w, pc.render_h);
    vec2 uv = (fragUV - 0.5) * vec2(res.x / res.y, 1.0) * 1.2;

    float t = pc.time * (0.3 + pc.cc1 * 0.4);

    // Kaleidoscope fold
    float segs = 6.0 + 2.0 * floor(mod(t * 0.2, 4.0));
    float a = atan(uv.y, uv.x);
    a = mod(a, TAU / segs);
    a = abs(a - 0.5 * TAU / segs);
    float r = length(uv);
    vec2 p = vec2(cos(a), sin(a)) * r;

    // IFS — repeat fold-and-scale
    float acc = 0.0;
    float energy = 0.0;
    for (int i = 0; i < 7; i++){
        p = abs(p) - vec2(0.5 + 0.1 * sin(t * 0.5 + float(i)),
                          0.4 + 0.1 * cos(t * 0.4 + float(i) * 1.3));
        p *= rot(t * 0.1 + float(i) * 0.3 + pc.pitch_bend * 0.5);
        p *= 1.18 + 0.08 * sin(t * 0.2 + float(i));
        acc += exp(-length(p) * 3.0);
        energy += dot(p, p);
    }

    // Color via accumulated energy
    float ce = energy * 0.05 + t * 0.05 + pc.osc_ch1 * 0.5;
    vec3 col = pal(ce);
    col *= 0.3 + acc * 0.4;

    // Highlight bright cells
    col += pal(ce + 0.4) * smoothstep(2.0, 4.0, acc) * 0.4;

    // Audio-reactive sparkle
    col += pal(t + acc) * (acc * 0.05) * pc.note_velocity;

    // Slight glow
    col += pal(t * 0.1) * exp(-length(uv) * 3.0) * 0.15;

    // Gamma + vignette
    col = pow(col, vec3(0.95));
    col *= 1.0 - 0.25 * dot(uv * 0.5, uv * 0.5);

    outColor = vec4(col, 1.0);
}
