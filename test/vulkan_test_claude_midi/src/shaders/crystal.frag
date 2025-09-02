#version 450

layout(push_constant) uniform PushConstants {
    float time; uint mouse_x; uint mouse_y; uint mouse_pressed;
    float note_velocity; float pitch_bend; float cc1; float cc74;
    uint  note_count; uint last_note; float osc_ch1; float osc_ch2;
    uint  render_w; uint render_h;
} pc;

layout(location=0) in vec2 frag_uv;
layout(location=1) in vec2 frag_screen_pos;
layout(location=0) out vec4 out_color;

const float PI=3.14159265359;
const float TAU=6.28318530718;

// ---------- tiny helpers (cheap!) ----------
mat2 rot(float a){ float c=cos(a), s=sin(a); return mat2(c,-s,s,c); }
float sinesm(vec2 p){ return sin(p.x)*sin(p.y); }     // cheap noise kernel
float fbm3(vec2 p){                                     // 3 octaves only
    float f=0.0, a=0.5;
    f += a*sinesm(p);          p = rot(1.5708)*p*1.8 + 0.73; a*=0.5;
    f += a*sinesm(p);          p = rot(1.5708)*p*1.8 + 0.37; a*=0.5;
    f += a*sinesm(p);
    return f;
}

// sine palette (fast, vivid)
vec3 pal(float h, vec3 a, vec3 b, vec3 c, vec3 d){
    return a + b * cos(TAU*(c*h + d));
}

void main(){
    vec2 uv = (frag_uv - 0.5) * 2.0;
    float aspect = float(pc.render_w)/float(pc.render_h);
    uv.x *= aspect;

    float t = pc.time;
    float audio = (pc.note_velocity + pc.osc_ch1 + pc.osc_ch2) * 0.5;

    // Mouse drift (branch is fine; taken rarely)
    vec2 m = vec2(0.0);
    if(pc.mouse_pressed != 0u){
        m = vec2(float(pc.mouse_x)/float(pc.render_w), float(pc.mouse_y)/float(pc.render_h));
        m = (m - 0.5) * 2.0; m.x *= aspect;
    }

    // ---------- domain warp (2 passes) ----------
    // Less math than before; still swirly
    float swirl = 0.45 + 0.9*audio;                     // strength with audio
    vec2 p = uv - 0.22*m;

    // Warp #1
    float a1 = t*0.28 + pc.pitch_bend*0.7;
    vec2 q1 = p*rot(a1);
    vec2 g1 = vec2( fbm3(q1*1.25), fbm3(q1*1.65) );
    vec2 v1 = vec2(-g1.y, g1.x);                        // curl-ish
    p += normalize(v1 + 1e-4) * (0.24*swirl);

    // Warp #2 (lighter)
    float a2 = t*0.17 - pc.pitch_bend*0.5;
    vec2 q2 = p*rot(a2);
    float g2 = fbm3(q2*1.1);
    p += vec2(g2, -g2) * (0.10*swirl);

    // ---------- dye pattern (radial bands + diffusion soften) ----------
    float r = length(p);
    float bands = 0.5 + 0.5*sin((10.0 + 9.0*audio)*r - t*1.8 + 2.4*fbm3(p*0.9));
    float bleed = 0.5 + 0.5*fbm3(p*2.0 - t*0.12);
    float ink = mix(bands, smoothstep(0.0, 1.0, bands), 0.5*bleed);

    // ---------- palettes (with surprises) ----------
    // Base hue from last_note; drift with time & audio
    float baseHue = (pc.note_count>0u ? float(pc.last_note)/127.0 : 0.36);
    float hue = baseHue + 0.06*t + 0.09*audio + 0.06*fbm3(p*0.6);

    // Two fast palettes: soft neon vs. "acid flip" (poster shock)
    vec3 colSoft = pal(hue,
    vec3(0.44,0.42,0.46),          // a
    vec3(0.56,0.55,0.60),          // b
    vec3(1.00,0.97,0.93),          // c
    vec3(0.00,0.33,0.67)           // d
    );

    vec3 colAcid = pal(hue + 0.15,
    vec3(0.10,0.08,0.12),
    vec3(1.15,1.10,1.05),
    vec3(1.00,1.00,1.00),
    vec3(0.05,0.38,0.72)
    );

    // Surprise trigger: brief flips driven by audio pulses + a slow timer.
    // Cheap, branchless blending.
    float pulse = smoothstep(0.35, 0.95, audio);
    float timer = 0.5 + 0.5*sin(t*0.8 + float(pc.last_note));     // note-salted
    float surprise = smoothstep(0.78, 0.92, timer) * pulse;        // 0..1 bursts

    // CCs: overall brightness & contrast feel
    float bright = mix(0.85, 1.35, clamp(pc.cc74,0.0,1.0));
    float gamma  = 1.0 + 0.7*clamp(pc.cc1,0.0,1.0);

    // Tiny per-channel coordinate offsets for prismatic sparkle (very cheap)
    float sep = 0.010 + 0.012*audio;
    float sR = ink + 0.12*fbm3((p*(1.0+sep))*1.05 + t*0.07);
    float sG = ink;
    float sB = ink + 0.12*fbm3((p*(1.0-sep))*1.05 - t*0.07);

    vec3 palMix = mix(colSoft, colAcid, surprise);
    vec3 col = palMix * vec3(sR, sG, sB) * bright;

    // subtle “paper” grain in screen space
    float gr = fract(sin(dot(frag_screen_pos.xy + t*55.0, vec2(12.9898,78.233)))*43758.5453) - 0.5;
    col += gr * (0.06 + 0.18*clamp(pc.cc1,0.0,1.0));

    // vignette + musical flicker
    float vig = smoothstep(1.55, 0.25, length(uv));
    col *= mix(0.80, 1.50, vig);
    col *= 1.0 + 0.18*audio*sin(t*92.0);

    // finalize
    col = pow(max(col, 0.0), vec3(gamma));
    out_color = vec4(clamp(col, 0.0, 3.0), 1.0);
}
