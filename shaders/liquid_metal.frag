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

#define MAX_STEPS 80
#define MAX_DIST 30.0
#define EPS 0.001
#define TAU 6.2831853

mat2 rot(float a){ float c=cos(a),s=sin(a); return mat2(c,-s,s,c); }

float hash(vec3 p){ return fract(sin(dot(p, vec3(91.3, 47.7, 19.1))) * 43758.5453); }

float noise(vec3 p){
    vec3 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float n000 = hash(i);
    float n100 = hash(i + vec3(1, 0, 0));
    float n010 = hash(i + vec3(0, 1, 0));
    float n110 = hash(i + vec3(1, 1, 0));
    float n001 = hash(i + vec3(0, 0, 1));
    float n101 = hash(i + vec3(1, 0, 1));
    float n011 = hash(i + vec3(0, 1, 1));
    float n111 = hash(i + vec3(1, 1, 1));
    return mix(mix(mix(n000, n100, f.x), mix(n010, n110, f.x), f.y),
               mix(mix(n001, n101, f.x), mix(n011, n111, f.x), f.y), f.z);
}

float fbm(vec3 p){
    float v = 0.0, a = 0.5;
    for (int i = 0; i < 4; i++){ v += a * noise(p); p *= 2.05; a *= 0.5; }
    return v;
}

float scene(vec3 p, float t){
    // Blob — sphere displaced by noise
    float r = 1.2 + 0.4 * sin(t * 0.7);
    float d = length(p) - r;
    d += 0.35 * fbm(p * 1.5 + t * 0.4) * (1.0 + pc.note_velocity);
    d += 0.08 * sin(p.x * 6.0 + t * 2.0) * sin(p.y * 5.0 + t * 1.5) * sin(p.z * 7.0);
    return d * 0.7;
}

vec3 normal(vec3 p, float t){
    vec2 e = vec2(EPS, 0.0);
    return normalize(vec3(
        scene(p + e.xyy, t) - scene(p - e.xyy, t),
        scene(p + e.yxy, t) - scene(p - e.yxy, t),
        scene(p + e.yyx, t) - scene(p - e.yyx, t)));
}

void main(){
    vec2 res = vec2(pc.render_w, pc.render_h);
    vec2 uv = (fragUV - 0.5) * vec2(res.x / res.y, 1.0);

    float t = pc.time * (0.4 + pc.cc1 * 0.6);

    // Camera
    vec3 ro = vec3(0.0, 0.0, -3.5);
    vec3 rd = normalize(vec3(uv, 1.2));
    ro.xz *= rot(t * 0.25 + pc.osc_ch1 * 2.0);
    rd.xz *= rot(t * 0.25 + pc.osc_ch1 * 2.0);
    ro.yz *= rot(sin(t * 0.2) * 0.3 + pc.osc_ch2);
    rd.yz *= rot(sin(t * 0.2) * 0.3 + pc.osc_ch2);

    // Raymarch
    float td = 0.0;
    bool hit = false;
    for (int i = 0; i < MAX_STEPS; i++){
        vec3 p = ro + rd * td;
        float d = scene(p, t);
        if (abs(d) < EPS) { hit = true; break; }
        if (td > MAX_DIST) break;
        td += d * 0.85;
    }

    vec3 col = vec3(0.02, 0.01, 0.04);

    if (hit){
        vec3 p = ro + rd * td;
        vec3 n = normal(p, t);
        vec3 v = -rd;

        // Environment: animated colored bands seen via reflection
        vec3 refl = reflect(rd, n);
        float band = atan(refl.y, length(refl.xz));
        vec3 envA = 0.5 + 0.5 * cos(TAU * (vec3(0.95, 0.7, 0.4) * (band * 2.0 + t * 0.3) + vec3(0.0, 0.2, 0.4)));
        vec3 envB = 0.5 + 0.5 * cos(TAU * (vec3(0.5, 0.8, 1.0) * (refl.x * 1.5 - t * 0.2) + vec3(0.3, 0.5, 0.8)));
        vec3 env = mix(envA, envB, 0.5 + 0.5 * sin(refl.z * 4.0 + t));

        // Fresnel
        float fres = pow(1.0 - max(0.0, dot(n, v)), 4.0);

        // Diffuse with key light
        vec3 lightDir = normalize(vec3(0.4, 0.7, -0.6));
        float diff = max(0.0, dot(n, lightDir));

        col = env * (0.4 + fres * 0.9);
        col += vec3(1.0, 0.95, 0.9) * pow(diff, 12.0) * (0.4 + pc.cc74);

        // Subsurface warmth where blob is thick
        float thick = max(0.0, dot(n, v));
        col += vec3(1.0, 0.3, 0.5) * pow(thick, 8.0) * 0.3 * (0.5 + pc.note_velocity);
    } else {
        // Background — soft radial palette
        float r = length(uv);
        col = mix(vec3(0.05, 0.02, 0.1), vec3(0.0), smoothstep(0.0, 1.4, r));
        col += 0.15 * vec3(0.7, 0.4, 0.8) * exp(-r * 2.5);
    }

    col *= 1.0 - 0.25 * dot(uv, uv);
    outColor = vec4(col, 1.0);
}
