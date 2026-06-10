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

vec3 prism(float t){
    return 0.5 + 0.5 * cos(TAU * (vec3(1.0) * t + vec3(0.0, 0.33, 0.67)));
}

// Distance to a thin triangle (shard) at origin pointing up
float sdTri(vec2 p, float h, float w){
    p.x = abs(p.x);
    vec2 e0 = vec2(-w, h);
    vec2 e1 = vec2(w * 2.0, 0.0);
    vec2 v0 = p - vec2(w, 0.0);
    vec2 v1 = p - vec2(-w, 0.0);
    float c0 = dot(v0, vec2(h, w)) / sqrt(h*h + w*w);
    float d = max(c0, -v1.y);
    return d;
}

void main(){
    vec2 res = vec2(pc.render_w, pc.render_h);
    vec2 uv = (fragUV - 0.5) * vec2(res.x / res.y, 1.0);

    float t = pc.time * (0.7 + pc.cc1 * 1.2);

    vec3 col = vec3(0.02, 0.0, 0.05);

    // Stormy backdrop — radial chromatic streaks
    float r = length(uv);
    float ang = atan(uv.y, uv.x);
    float streak = sin(ang * 40.0 + t * 2.0 + r * 12.0);
    streak = pow(0.5 + 0.5 * streak, 6.0);
    col += prism(ang / TAU + t * 0.1) * streak * 0.2 * (1.0 - r * 0.6);

    // Many prism shards exploding outward
    for (int i = 0; i < 24; i++){
        float fi = float(i);
        float seed = fi * 13.7 + 0.5;

        // Angle around center
        float a = fract(seed * 0.123) * TAU + t * 0.1;

        // Radial position with explosion timing
        float phase = fract(t * 0.35 + seed * 0.27);
        float rad = phase * 1.4;

        vec2 sc = vec2(cos(a), sin(a)) * rad;

        // Shard local space
        vec2 dp = uv - sc;
        float spin = t * 2.0 + seed * 5.0;
        dp *= rot(spin);

        // Triangular shard
        float h = 0.07 + 0.04 * sin(seed * 3.0);
        float w = 0.018 + 0.012 * cos(seed * 5.0);
        float d = sdTri(dp, h, w);
        // Build edge band: |d| close to 0
        float edge = smoothstep(0.012, 0.0, abs(d));
        float fill = smoothstep(0.0, -0.005, d);

        // Color cycles through spectrum
        vec3 shardCol = prism(seed + t * 0.4);

        // Audio brightens
        float amp = 1.0 + pc.note_velocity * 2.0;

        // Fade in/out
        float life = sin(phase * 3.14159);
        life = smoothstep(0.0, 0.3, life);

        col += shardCol * (edge * 1.5 + fill * 0.5) * life * amp;

        // Streaking motion blur tail
        vec2 tailDp = uv - sc + vec2(cos(a), sin(a)) * 0.06;
        float tailD = length(tailDp - sc * 0.0); // dummy use
        float tail = exp(-length(tailDp) * 30.0);
        col += shardCol * tail * life * 0.3;
    }

    // Flash burst — periodic
    float flash = pow(fract(t * 0.5), 30.0);
    col += vec3(1.0) * flash * pc.note_velocity * 0.6;

    // Chromatic aberration on the whole image
    float chrom = pc.cc74 * 0.02;
    col.r *= 1.0 + sin(uv.x * 10.0 + t) * chrom;
    col.b *= 1.0 + sin(uv.y * 10.0 - t) * chrom;

    // Edge darkening
    col *= 1.0 - 0.3 * dot(uv, uv);

    outColor = vec4(col, 1.0);
}
