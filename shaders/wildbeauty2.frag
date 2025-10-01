#version 450

// The MIT License
// https://www.youtube.com/c/InigoQuilez
// https://iquilezles.org/
// Copyright © 2015 Inigo Quilez
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

// Push constants from application
layout(push_constant) uniform PushConstants {
    float time;
    uint  mouse_x;
    uint  mouse_y;
    uint  mouse_pressed;
    float note_velocity;
    float pitch_bend;
    float cc1;    // mid frequencies / mod wheel
    float cc74;   // high frequencies / cutoff
    uint  note_count;
    uint  last_note;
    float osc_ch1;
    float osc_ch2;
    uint  render_w;
    uint  render_h;
} pc;

layout(location = 0) in vec2 fragUV;
layout(location = 1) in vec2 frag_screen_pos;

layout(location = 0) out vec4 outColor;

// Constants
#define PI 3.141592653589793238
#define TAU (2.0 * PI)

// Helper macros
#define st(x) clamp(x, 0.0, 1.0)
#define pos(x) (x * 0.5 + 0.5)

// Rotation matrix
mat2 rot(float a) {
    float c = cos(a);
    float s = sin(a);
    return mat2(c, -s, s, c);
}

// 3D rotation matrices
mat3 rotX(float a) {
    float c = cos(a), s = sin(a);
    return mat3(1,0,0, 0,c,-s, 0,s,c);
}

mat3 rotY(float a) {
    float c = cos(a), s = sin(a);
    return mat3(c,0,s, 0,1,0, -s,0,c);
}

mat3 rotZ(float a) {
    float c = cos(a), s = sin(a);
    return mat3(c,-s,0, s,c,0, 0,0,1);
}

vec3 pal( in float t, in vec3 a, in vec3 b, in vec3 c, in vec3 d )
{
    return a + b*cos( 6.28318*(c*t+d) );
}

void main() {
    // Resolution setup
    vec2 iResolution = vec2(float(pc.render_w), float(pc.render_h));

    // EXTREME Audio-reactive parameters (from wildbeauty2)
    float energy = clamp(pc.note_velocity, 0.0, 1.0);
    float modulation = clamp(pc.cc1, 0.0, 1.0);
    float brightness = clamp(pc.cc74, 0.0, 1.0);

    // Time with INSANE audio modulation (from wildbeauty2)
    float timeScale = 2.0 + energy * 5.0;
    float iTime = pc.time * timeScale;

    float t2 = pc.time * 0.7 + pc.pitch_bend * 2.0;
    float t3 = pc.time * 1.3 + pc.osc_ch1 * 3.0;

    // Setup UV with aspect ratio correction
    vec2 uv = (fragUV - 0.5) * 2.0;
    uv.x *= iResolution.x / iResolution.y;

    // INSANE Camera setup with MAXIMUM audio influence (from wildbeauty2)
    float camDist = 2.0 + sin(pc.time * 2.0) * 1.0 - modulation * 1.5 + energy * 2.0;

    // CHAOTIC automatic camera movement (from wildbeauty2)
    vec3 viewDir = vec3(
        sin(pc.time * 0.5 + pc.osc_ch1 * PI) * 0.5,
        cos(pc.time * 0.7 + pc.osc_ch2 * PI) * 0.3,
        -camDist
    );

    // Apply rotation transformations
    mat3 transform = rotX(sin(pc.time * 0.3) * 0.5 + pc.pitch_bend * 0.5)
                   * rotY(cos(pc.time * 0.4) * 0.7 + brightness * PI)
                   * rotZ(sin(pc.time * 0.2) * 0.3 + modulation * PI);

    viewDir = viewDir * transform;

    // Mouse/OSC camera override (from wildbeauty2)
    if(pc.mouse_pressed > 0u) {
        float mx = (float(pc.mouse_x) / float(pc.render_w) - 0.5) * TAU * 2.0;
        float my = (float(pc.mouse_y) / float(pc.render_h) - 0.5) * PI * 1.5;
        uv = rot(mx) * uv;
        uv = rot(my) * uv;
    }

    // CHAOS UV distortion (from wildbeauty2)
    uv += sin(uv * 10.0 + pc.time * 3.0) * 0.05 * energy;

    // Barrel distortion with audio (from wildbeauty2)
    float distortion = 0.1 + energy * 0.3;
    float r = length(uv);
    uv *= 1.0 + distortion * r * r;

    // Convert to normalized coordinates for palette calculation
    vec2 p = (uv / 2.0 + 0.5);

    // Apply rotation and distortion to pattern
    p.x += sin(pc.time * 3.0) * 0.1;
    p.y += cos(pc.time * 2.0) * 0.1;

    // animate
    p.x += 0.01*iTime;

    // Pitch bend affects horizontal movement
    p.x += pc.pitch_bend * 0.1;

    // Energy affects vertical offset
    p.y += energy * 0.05 * sin(iTime * 2.0);

    // compute colors with different palettes per band
    vec3 col = pal( p.x, vec3(0.5,0.5,0.5),vec3(0.5,0.5,0.5),vec3(1.0,1.0,1.0),vec3(0.0,0.33,0.67) );
    if( p.y>(1.0/7.0) ) col = pal( p.x, vec3(0.5,0.5,0.5),vec3(0.5,0.5,0.5),vec3(1.0,1.0,1.0),vec3(0.0,0.10,0.20) );
    if( p.y>(2.0/7.0) ) col = pal( p.x, vec3(0.5,0.5,0.5),vec3(0.5,0.5,0.5),vec3(1.0,1.0,1.0),vec3(0.3,0.20,0.20) );
    if( p.y>(3.0/7.0) ) col = pal( p.x, vec3(0.5,0.5,0.5),vec3(0.5,0.5,0.5),vec3(1.0,1.0,0.5),vec3(0.8,0.90,0.30) );
    if( p.y>(4.0/7.0) ) col = pal( p.x, vec3(0.5,0.5,0.5),vec3(0.5,0.5,0.5),vec3(1.0,0.7,0.4),vec3(0.0,0.15,0.20) );
    if( p.y>(5.0/7.0) ) col = pal( p.x, vec3(0.5,0.5,0.5),vec3(0.5,0.5,0.5),vec3(2.0,1.0,0.0),vec3(0.5,0.20,0.25) );
    if( p.y>(6.0/7.0) ) col = pal( p.x, vec3(0.8,0.5,0.4),vec3(0.2,0.4,0.2),vec3(2.0,1.0,1.0),vec3(0.0,0.25,0.25) );

    // band
    float f = fract(p.y*7.0);
    // borders
    col *= smoothstep( 0.49, 0.47, abs(f-0.5) );
    // shadowing
    col *= 0.5 + 0.5*sqrt(4.0*f*(1.0-f));

    // EXTREME energy effects (from wildbeauty2)
    if(energy > 0.3) {
        vec3 energyFlash = vec3(
            1.0 + energy * 2.0,
            0.5 + energy * 1.5,
            0.8 + energy * 3.0
        ) * (energy - 0.3) * 0.8;
        col += energyFlash;
    }

    // Oscillator effects (from wildbeauty2)
    col += vec3(pc.osc_ch1 * 0.3, pc.osc_ch2 * 0.4, (pc.osc_ch1 + pc.osc_ch2) * 0.2);

    // Pitch bend chromatic effects (from wildbeauty2)
    col *= 1.0 + vec3(pc.pitch_bend * 0.5, 0.0, -pc.pitch_bend * 0.3);

    // CC effects (from wildbeauty2)
    col = mix(col, col * vec3(2.0, 0.5, 1.5), modulation * 0.3);
    col = mix(col, pow(col, vec3(0.7, 1.3, 0.8)), brightness * 0.4);

    // Kaleidoscope effect (from wildbeauty2)
    if (energy > 0.5) {
        vec2 uvOriginal = (fragUV - 0.5) * 2.0;
        uvOriginal.x *= iResolution.x / iResolution.y;
        float angle = atan(uvOriginal.y, uvOriginal.x);
        float segments = 6.0 + floor(energy * 6.0);
        angle = mod(angle, TAU / segments) * segments;
        vec2 kaleidoP = vec2(cos(angle), sin(angle)) * length(uvOriginal);
        kaleidoP = (kaleidoP / 2.0 + 0.5);
        kaleidoP.x += 0.01*iTime + pc.pitch_bend * 0.1;
        vec3 kaleidoCol = pal( kaleidoP.x, vec3(0.5),vec3(0.5),vec3(1.0),vec3(0.0,0.33,0.67) );
        float kf = fract(kaleidoP.y*7.0);
        kaleidoCol *= smoothstep( 0.49, 0.47, abs(kf-0.5) );
        kaleidoCol *= 0.5 + 0.5*sqrt(4.0*kf*(1.0-kf));
        col = mix(col, kaleidoCol, (energy - 0.5) * 0.6);
    }

    // Color grading CHAOS (from wildbeauty2)
    col = pow(col, vec3(0.8 + sin(pc.time) * 0.2, 1.0 + cos(pc.time * 1.2) * 0.2, 0.9 + sin(pc.time * 0.8) * 0.2));

    // Contrast and saturation (from wildbeauty2)
    float contrast = 1.2 + brightness * 0.5;
    col = (col - 0.5) * contrast + 0.5;

    // Saturation based on energy (from wildbeauty2)
    float gray = dot(col, vec3(0.299, 0.587, 0.114));
    col = mix(vec3(gray), col, 1.0 + energy * 1.5);

    // Hue shift (from wildbeauty2)
    float hueShift = pc.osc_ch1 * TAU + pc.time * 0.1;
    col = mix(col, col.gbr, sin(hueShift) * 0.3);
    col = mix(col, col.brg, cos(hueShift * 1.2) * 0.2);

    // EXTREME vignette with animation (from wildbeauty2)
    vec2 uvOriginal = (fragUV - 0.5) * 2.0;
    uvOriginal.x *= iResolution.x / iResolution.y;
    float vignette = 1.0 - pow(length(uvOriginal) * 0.7, 2.0);
    vignette += sin(pc.time * 3.0 + length(uvOriginal) * 10.0) * 0.1 * energy;
    vignette = st(vignette);
    col *= vignette;

    // Bloom effect (from wildbeauty2)
    vec3 bloom = max(col - vec3(1.0), 0.0) * 2.0;
    col += bloom * energy;

    // Output with CHAOS saturation (from wildbeauty2)
    outColor = vec4(st(col * (1.0 + energy * 0.5)), 1.0);
}
