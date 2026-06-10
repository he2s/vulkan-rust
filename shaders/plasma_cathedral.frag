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
#define MAX_STEPS 100
#define MAX_DIST 40.0
#define EPS 0.0015

mat2 rot(float a){ float c=cos(a),s=sin(a); return mat2(c,-s,s,c); }

float sdBox(vec3 p, vec3 b){ vec3 q = abs(p) - b; return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0); }

// Build a cathedral interior with arches via domain repetition
float scene(vec3 p, float t){
    // Floor
    float d = p.y + 1.6;

    // Repeat columns along x and z
    vec3 q = p;
    q.x = mod(q.x + 2.0, 4.0) - 2.0;
    q.z = mod(q.z + 4.0, 8.0) - 4.0;

    // Column shaft
    float col = length(q.xz) - 0.18;
    col = max(col, -p.y - 2.0);

    // Arches
    vec3 a = q;
    a.y -= 1.2;
    float archR = length(a.xy) - 1.1;
    archR = max(archR, -(length(a.xy) - 0.9));
    archR = max(archR, abs(a.z) - 0.25);
    archR = max(archR, -a.y);

    d = min(d, col);
    d = min(d, archR);

    // Vaulted ceiling — wavy
    float ceiling = (1.6 + 0.5 * sin(p.x * 0.8) * sin(p.z * 0.6)) - p.y;
    d = min(d, ceiling);

    return d;
}

void main(){
    vec2 res = vec2(pc.render_w, pc.render_h);
    vec2 uv = (fragUV - 0.5) * vec2(res.x / res.y, 1.0);

    float t = pc.time * (0.4 + pc.cc1 * 0.4);

    // Camera — slowly moving down the nave
    vec3 ro = vec3(0.0, -0.3, -8.0 + mod(t * 1.2, 8.0));
    vec3 rd = normalize(vec3(uv, 1.2));
    rd.xz *= rot(sin(t * 0.2) * 0.2 + pc.osc_ch1 * 0.5);

    // Raymarch — accumulate volumetric plasma fog
    vec3 col = vec3(0.0);
    float td = 0.1;
    float fog = 0.0;
    bool hit = false;
    for (int i = 0; i < MAX_STEPS; i++){
        vec3 p = ro + rd * td;
        float d = scene(p, t);

        // Plasma volumetric — sample colored noise
        float plasma = sin(p.x * 1.5 + t * 1.2) * cos(p.y * 1.3 - t) * sin(p.z * 0.8 + t * 0.5);
        plasma = plasma * 0.5 + 0.5;

        vec3 plasmaCol = 0.5 + 0.5 * cos(TAU * (vec3(0.8, 0.5, 0.95) * plasma + vec3(0.0, 0.3, 0.6)) + t * 0.3 + pc.pitch_bend * 2.0);

        float density = exp(-abs(d) * 4.0) * 0.04;
        density *= 0.5 + pc.note_velocity * 1.5;
        col += plasmaCol * density * (1.0 - fog);
        fog += density;

        if (abs(d) < EPS){ hit = true; break; }
        if (td > MAX_DIST) break;
        if (fog > 0.95) break;
        td += max(EPS * 2.0, d * 0.7);
    }

    if (hit){
        // Architectural surface lighting
        vec2 e = vec2(EPS, 0.0);
        vec3 p = ro + rd * td;
        vec3 n = normalize(vec3(
            scene(p + e.xyy, t) - scene(p - e.xyy, t),
            scene(p + e.yxy, t) - scene(p - e.yxy, t),
            scene(p + e.yyx, t) - scene(p - e.yyx, t)));

        vec3 lp = vec3(0.0, 1.0, ro.z + 3.0);
        vec3 ld = normalize(lp - p);
        float diff = max(0.0, dot(n, ld));

        // Stone color tinted by plasma above
        vec3 stone = mix(vec3(0.18, 0.16, 0.22), vec3(0.4, 0.25, 0.5), pc.cc74);
        col += stone * diff * (1.0 - fog) * 0.7;

        // Floor reflections of plasma
        if (p.y < -1.5){
            col += vec3(0.5, 0.3, 0.6) * 0.2;
        }
    }

    // Distance attenuation
    col *= 1.0 / (1.0 + td * 0.02);

    // Soft vignette
    col *= 1.0 - 0.3 * dot(uv, uv);

    outColor = vec4(col, 1.0);
}
