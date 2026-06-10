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

void main(){
    vec2 res = vec2(pc.render_w, pc.render_h);
    vec2 uv = (fragUV - 0.5) * vec2(res.x / res.y, 1.0);

    float t = pc.time * (0.5 + pc.cc1 * 0.6);

    // Sky gradient — synthwave dusk
    float sky = uv.y * 0.5 + 0.5;
    vec3 col = mix(vec3(0.6, 0.05, 0.4), vec3(0.05, 0.0, 0.2), sky);
    col = mix(col, vec3(1.0, 0.4, 0.2), pow(1.0 - sky, 5.0));

    // Sun — segmented disc
    vec2 sunCenter = vec2(0.0, -0.05);
    float sunR = length(uv - sunCenter);
    float sun = smoothstep(0.4, 0.39, sunR);
    // Horizontal bands cut into sun
    float bandCut = smoothstep(0.5, 0.49, sin(uv.y * 30.0 + t * 0.5));
    sun *= (uv.y > sunCenter.y) ? 1.0 : mix(1.0, 0.0, smoothstep(0.0, 1.0, -(uv.y - sunCenter.y) * 4.0) * (0.5 + 0.5 * bandCut));
    vec3 sunCol = mix(vec3(1.0, 0.9, 0.2), vec3(1.0, 0.2, 0.6), smoothstep(-0.3, 0.3, uv.y));
    col = mix(col, sunCol, sun);

    // Sun glow halo
    col += sunCol * exp(-sunR * 4.5) * 0.4 * (0.7 + pc.note_velocity * 0.6);

    // Grid floor — only in lower half
    if (uv.y < -0.05){
        // Perspective: scale grid by 1/(-uv.y)
        float perspY = 1.0 / max(-uv.y + 0.02, 0.001);
        float gridZ = perspY + t * 4.0;
        float gridX = uv.x * perspY;

        // Animated grid lines
        float lineZ = abs(fract(gridZ) - 0.5);
        float lineX = abs(fract(gridX) - 0.5);

        // Audio-reactive line thickness
        float thickness = 0.02 + pc.cc74 * 0.05;
        float gZ = smoothstep(thickness, 0.0, lineZ);
        float gX = smoothstep(thickness, 0.0, lineX);
        float grid = max(gZ, gX);

        // Fade distant lines
        grid *= exp(-perspY * 0.02);

        vec3 gridCol = mix(vec3(1.0, 0.2, 0.6), vec3(0.4, 0.8, 1.0), pc.osc_ch1 * 0.5 + 0.5);
        col = mix(col, gridCol * 1.4, grid);

        // Faint reflection of sun on the ground
        float refl = smoothstep(0.4, 0.39, length(vec2(uv.x, -uv.y) - sunCenter * vec2(1, -1)));
        col += sunCol * refl * 0.15 * (1.0 + pc.note_velocity * 0.5) * smoothstep(0.0, -0.5, uv.y);
    }

    // Stars in upper sky
    if (uv.y > 0.0){
        vec2 sp = uv * 90.0;
        float star = pow(fract(sin(dot(floor(sp), vec2(91.3, 47.7))) * 43758.5453), 50.0);
        col += vec3(1.0, 0.95, 0.8) * star * (0.6 + sin(t * 4.0 + sp.x) * 0.4);
    }

    // Mountains / horizon silhouette
    float mtn = sin(uv.x * 4.0 + t * 0.05) * 0.04 + sin(uv.x * 7.3 - 0.5) * 0.025;
    mtn -= 0.06;
    if (uv.y < mtn && uv.y > -0.08){
        col *= 0.15;
        col += vec3(0.2, 0.04, 0.15);
    }

    // Scanline subtle CRT
    col *= 0.92 + 0.08 * sin(uv.y * res.y * 1.5);

    outColor = vec4(col, 1.0);
}
