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

vec3 neon(float t){
    return 0.5 + 0.5 * cos(TAU * (vec3(0.9, 0.6, 0.95) * t + vec3(0.0, 0.25, 0.55)));
}

float sdSegment(vec2 p, vec2 a, vec2 b){
    vec2 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h);
}

void main(){
    vec2 res = vec2(pc.render_w, pc.render_h);
    vec2 uv = (fragUV - 0.5) * vec2(res.x / res.y, 1.0);

    float t = pc.time * (0.3 + pc.cc1 * 0.4);

    vec3 col = vec3(0.02, 0.0, 0.06);

    // Background gentle gradient
    col += vec3(0.06, 0.02, 0.15) * (1.0 - length(uv) * 0.7);

    // Multiple folded polygon layers, each rotating
    for (int layer = 0; layer < 4; layer++){
        float fl = float(layer);
        float scale = 0.4 + fl * 0.18;
        vec2 lp = uv;
        lp *= rot(t * (0.15 + fl * 0.07) + fl * 0.9 + pc.pitch_bend);

        // Fold along symmetry — origami crease
        float fold = 4.0 + fl;
        float a = atan(lp.y, lp.x);
        float ra = mod(a + TAU / (2.0 * fold), TAU / fold) - TAU / (2.0 * fold);
        vec2 fp = vec2(cos(ra), sin(ra)) * length(lp);

        // Polygon vertices in folded coords
        vec2 v0 = vec2(scale, 0.0);
        vec2 v1 = vec2(scale * cos(TAU / fold), scale * sin(TAU / fold));

        // Edge line
        float edge = sdSegment(fp, v0, v1);

        // Creases — additional folding lines from center
        float crease = abs(ra);
        crease = smoothstep(0.01, 0.0, crease);

        // Neon color per layer
        vec3 lc = neon(fl * 0.3 + t * 0.1 + pc.osc_ch1 * 0.5);

        // Edge glow + sharp line
        float w = 0.008 + 0.005 * sin(t * 2.0 + fl);
        float line = smoothstep(w, 0.0, edge);
        float halo = exp(-edge * 60.0);

        // Audio reactive intensity
        float amp = 1.0 + pc.note_velocity * 0.8;

        col += lc * line * 1.5 * amp;
        col += lc * halo * 0.35 * amp;
        col += lc * crease * 0.4 * amp;

        // Bright vertex points
        float vert = exp(-length(fp - v0) * 60.0);
        col += vec3(1.0) * vert * 0.5 * amp;
    }

    // Particles trailing
    for (int p = 0; p < 12; p++){
        float fp = float(p);
        float pt = fract(t * 0.2 + fp * 0.083);
        float a = fp * 0.5 + t * 0.05;
        vec2 pp = vec2(cos(a), sin(a)) * (0.2 + pt * 1.2);
        float pd = length(uv - pp);
        col += neon(fp * 0.1 + t * 0.2) * smoothstep(0.008, 0.0, pd) * (1.0 - pt);
        col += neon(fp * 0.1 + t * 0.2) * exp(-pd * 80.0) * 0.2 * (1.0 - pt);
    }

    // Vignette
    col *= 1.0 - 0.2 * dot(uv, uv);

    outColor = vec4(col, 1.0);
}
