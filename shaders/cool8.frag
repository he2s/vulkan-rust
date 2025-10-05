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

// Neon Grid - Combines cyberpunk grid with volumetric fog

#define PI 3.141592653589793238
#define TAU (2.0 * PI)
#define sat(x) clamp(x, 0.0, 1.0)

mat2 rot(float a) {
    float c = cos(a), s = sin(a);
    return mat2(c, -s, s, c);
}

vec3 palette(float t) {
    // Neon colors
    vec3 a = vec3(0.5, 0.5, 0.5);
    vec3 b = vec3(0.5, 0.5, 0.5);
    vec3 c = vec3(1.0, 1.0, 0.5);
    vec3 d = vec3(0.8 + pc.osc_ch1 * 0.2, 0.2 + pc.osc_ch2 * 0.3, 0.4 + pc.pitch_bend * 0.1);
    return a + b * cos(TAU * (c * t + d));
}

// Grid pattern
float grid(vec2 p, float size) {
    vec2 grid = abs(fract(p / size - 0.5) - 0.5) / fwidth(p / size);
    float line = min(grid.x, grid.y);
    return 1.0 - min(line, 1.0);
}

// 3D grid in space
vec3 neonGrid(vec3 ro, vec3 rd) {
    float energy = sat(pc.note_velocity);
    float modulation = sat(pc.cc1);
    float brightness = sat(pc.cc74);

    vec3 col = vec3(0.0);

    // March through grid space
    float t = 0.0;
    float fogDensity = 0.0;

    for(int i = 0; i < 60; i++) {
        vec3 p = ro + rd * t;

        // Perspective grid on XZ plane
        if(abs(p.y + 1.0) < 0.05) {
            float gridVal = grid(p.xz, 1.0 + modulation * 2.0);
            vec3 gridCol = palette(length(p.xz) * 0.1 + pc.time * 0.1);

            // Pulsing grid lines
            gridVal *= 0.5 + 0.5 * sin(pc.time * 5.0 + length(p.xz) * 2.0);
            gridVal *= 1.0 + energy * 2.0;

            col += gridCol * gridVal * 0.3;
        }

        // Vertical grid lines
        vec2 gridXY = abs(fract(p.xy * (0.5 + brightness * 0.5)) - 0.5);
        float vertGrid = smoothstep(0.05, 0.0, min(gridXY.x, gridXY.y));
        if(vertGrid > 0.01) {
            vec3 vertCol = palette(p.z * 0.05 + pc.time * 0.15 + 0.3);
            col += vertCol * vertGrid * 0.2 * (1.0 + energy);
        }

        // Volumetric fog
        float fog = 0.02 + energy * 0.03;
        fog *= exp(-t * 0.1);

        vec3 fogCol = palette(t * 0.02 + pc.time * 0.1);
        fogCol *= 1.0 + sin(t * 2.0 - pc.time * 3.0) * 0.3; // Animated fog

        col += fogCol * fog;
        fogDensity += fog;

        // Floating cubes
        vec3 cubeP = p;
        cubeP.y += sin(pc.time + cubeP.x * 2.0) * 0.5;
        cubeP.x += cos(pc.time * 0.8 + cubeP.z * 2.0) * 0.3;

        vec3 cubeCell = floor(cubeP / 3.0);
        vec3 cubeFrac = fract(cubeP / 3.0) - 0.5;

        // Hash for random cube placement
        float h = fract(sin(dot(cubeCell, vec3(127.1, 311.7, 74.7))) * 43758.5453);

        if(h > 0.7) { // Some cells have cubes
            vec3 cubeLocal = abs(cubeFrac) - vec3(0.2 + h * 0.1);
            float cubeDist = max(cubeLocal.x, max(cubeLocal.y, cubeLocal.z));

            if(cubeDist < 0.1) {
                float cubeEdge = 1.0 - smoothstep(-0.05, 0.05, cubeDist);
                vec3 cubeCol = palette(h + pc.time * 0.2);
                col += cubeCol * cubeEdge * (1.0 + energy * 2.0);
            }
        }

        t += 0.15;
        if(t > 15.0 || fogDensity > 1.0) break;
    }

    return col;
}

void main() {
    vec2 resolution = vec2(float(pc.render_w), float(pc.render_h));
    vec2 uv = (fragUV - 0.5) * 2.0;
    uv.x *= resolution.x / resolution.y;

    float energy = sat(pc.note_velocity);
    float modulation = sat(pc.cc1);

    // Camera setup - looking down at grid
    vec3 ro = vec3(0.0, 2.0 + modulation * 2.0, pc.time * 2.0);

    // Camera tilt
    float tilt = -0.3 + sin(pc.time * 0.2) * 0.1;

    // Mouse/OSC control
    vec2 mouseOffset = vec2(0.0);
    if(pc.mouse_pressed > 0u) {
        mouseOffset = (vec2(float(pc.mouse_x), float(pc.mouse_y)) / resolution - 0.5) * 4.0;
        ro.xy += mouseOffset;
    } else {
        ro.x += pc.osc_ch1 * 3.0;
        ro.y += pc.osc_ch2 * 2.0;
    }

    // Ray direction
    vec3 rd = normalize(vec3(uv.x, uv.y + tilt, 1.0));

    // Camera roll
    rd.xy = rot(pc.pitch_bend * PI * 0.5) * rd.xy;

    // Render
    vec3 col = neonGrid(ro, rd);

    // Sun/light source
    vec3 sunDir = normalize(vec3(0.5, -0.3, 1.0));
    float sun = pow(sat(dot(rd, sunDir)), 8.0);
    col += palette(pc.time * 0.1) * sun * (2.0 + energy * 3.0);

    // Scanlines effect
    float scanline = sin(fragUV.y * resolution.y * 2.0 + pc.time * 10.0);
    scanline = scanline * 0.05 + 0.95;
    col *= scanline;

    // Energy flash
    if(energy > 0.7) {
        float flash = pow(energy - 0.7, 2.0) * 10.0;
        flash *= (1.0 + sin(pc.time * 30.0) * 0.5);
        col += palette(pc.time * 0.3) * flash;
    }

    // Note count affects color shift
    if(pc.note_count > 0u) {
        float shift = float(pc.note_count % 7u) / 7.0;
        col = mix(col, col.gbr, shift * 0.6);
    }

    // Contrast and saturation
    col = (col - 0.5) * (1.1 + modulation * 0.3) + 0.5;

    float gray = dot(col, vec3(0.299, 0.587, 0.114));
    col = mix(vec3(gray), col, 1.4);

    // Vignette
    float vig = 1.0 - pow(length((fragUV - 0.5) * 0.9), 2.0);
    col *= vig;

    // Bloom
    vec3 bloom = max(col - vec3(0.9), 0.0) * 2.0;
    col += bloom * (0.5 + energy * 0.5);

    // CRT curve effect
    uv = (fragUV - 0.5) * 2.0;
    float curve = 1.0 - dot(uv * uv, vec2(0.1));
    col *= curve;

    outColor = vec4(sat(col), 1.0);
}
