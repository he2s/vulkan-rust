#version 450

// Push constants from application
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

// Volumetric Tunnel - Combines cloud-like fractals with tunnel effect and iterative feedback

#define PI 3.141592653589793238
#define TAU (2.0 * PI)
#define sat(x) clamp(x, 0.0, 1.0)

mat2 rot(float a) {
    float c = cos(a), s = sin(a);
    return mat2(c, s, -s, c);
}

const mat3 m3 = mat3(0.33338, 0.56034, -0.71817,
                    -0.87887, 0.32651, -0.15323,
                     0.15162, 0.69596, 0.61339) * 1.93;

vec3 palette(float t) {
    vec3 a = vec3(0.8, 0.5, 0.4);
    vec3 b = vec3(0.2, 0.4, 0.2);
    vec3 c = vec3(2.0, 1.0, 1.0);
    vec3 d = vec3(0.0, 0.25, 0.25);
    return a + b * cos(TAU * (c * t + d + vec3(pc.osc_ch1, pc.osc_ch2, 0.0) * 0.5));
}

// Volumetric noise from cloud shader
float map(vec3 p) {
    float energy = sat(pc.note_velocity);
    float modulation = sat(pc.cc1);

    // Tunnel displacement
    p.xy -= vec2(sin(p.z * 0.2 + pc.time), cos(p.z * 0.175 + pc.time)) * (1.0 + energy * 0.5);

    // Rotation
    p.xy *= rot(sin(p.z + pc.time * 0.5) * 0.3 + pc.pitch_bend * PI);

    // Fractal iteration
    float d = 0.0;
    p *= 0.6;
    float z = 1.0;
    float trk = 1.0;

    for(int i = 0; i < 5; i++) {
        p += sin(p.zxy * 0.75 * trk + pc.time * trk) * (0.2 + energy * 0.15);
        d -= abs(dot(cos(p), sin(p.yzx)) * z);
        z *= 0.57;
        trk *= 1.4;
        p = p * m3;
    }

    d = abs(d + 2.0) - 2.5 + modulation * 0.5;

    // Tunnel shape
    float tunnel = length(p.xy) - 0.5;
    return mix(d, tunnel, 0.3);
}

void main() {
    vec2 resolution = vec2(float(pc.render_w), float(pc.render_h));
    vec2 uv = (fragUV - 0.5) * 2.0;
    uv.x *= resolution.x / resolution.y;

    // Camera setup - tunnel fly-through
    vec3 ro = vec3(0.0, 0.0, pc.time * 2.0);
    vec3 rd = normalize(vec3(uv, 1.0));

    // Mouse/OSC rotation
    if(pc.mouse_pressed > 0u) {
        float mx = (float(pc.mouse_x) / float(pc.render_w) - 0.5) * TAU;
        rd.xy *= rot(mx);
    } else {
        rd.xy *= rot(pc.osc_ch1 * PI + pc.time * 0.1);
    }

    // Ray marching with volumetric rendering
    vec4 acc = vec4(0.0);
    float t = 0.5;
    float energy = sat(pc.note_velocity);
    float brightness = sat(pc.cc74);

    for(int i = 0; i < 80; i++) {
        if(acc.a > 0.99) break;

        vec3 p = ro + rd * t;
        float d = map(p);

        float density = sat(d * -0.5 + 0.5);
        if(d < 0.0) {
            // Inside volume
            vec3 col = palette(length(p) * 0.1 + pc.time * 0.2);
            col *= density * (0.5 + brightness * 0.5);

            // Add energy glow
            col += col * energy * 0.8;

            vec4 src = vec4(col, density * 0.15);
            acc = acc + src * (1.0 - acc.a);
        }

        // Fog
        vec4 fog = vec4(palette(t * 0.05) * 0.02, 0.01);
        acc = acc + fog * (1.0 - acc.a);

        t += max(0.1, abs(d) * 0.5);
        if(t > 20.0) break;
    }

    vec3 col = acc.rgb;

    // Chromatic aberration
    if(energy > 0.5) {
        float shift = (energy - 0.5) * 0.01;
        vec2 uvR = uv + rd.xy * shift;
        vec2 uvB = uv - rd.xy * shift;
        // Just shift colors based on energy for effect
        col.r *= 1.0 + shift * 10.0;
        col.b *= 1.0 + shift * 10.0;
    }

    // Vignette
    float vig = 1.0 - length(uv) * 0.4;
    col *= vig;

    outColor = vec4(sat(col), 1.0);
}
