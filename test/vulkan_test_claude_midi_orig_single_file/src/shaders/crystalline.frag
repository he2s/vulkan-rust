#version 450
/*
  Neon Palettes — adapted from Inigo Quilez "Palettes" (MIT)
  https://iquilezles.org/articles/palettes
*/

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
    uint  render_w;
    uint  render_h;
} pc;

layout(location = 0) in  vec2 fragCoord; // unused; keeps interface
layout(location = 0) out vec4 fragColor;

vec3 pal(float t, vec3 a, vec3 b, vec3 c, vec3 d){
    return a + b * cos(6.2831853 * (c * t + d));
}

void main() {
    vec2 R    = vec2(float(max(1000,1u)), float(max(1000,1u)));
    vec2 uv0  = gl_FragCoord.xy / R;   // STATIC normalized coords (for vignette/scan)
    vec2 p    = uv0;                   // WORKING coords (we'll scroll/warp these)

    // ---- interactivity ----
    float speed   = 1.0 + pc.cc1 * 1.2 + pc.note_velocity * 0.45;
    float bendX   = 0.22 * pc.pitch_bend;
    float hueCtl  = pc.cc74;
    float energy  = clamp(pc.note_velocity * 1.6, 0.0, 1.6);
    vec2  mN      = vec2(float(pc.mouse_x), float(pc.mouse_y)) / R;
    float pressed = (pc.mouse_pressed != 0u) ? 1.0 : 0.0;

    // horizontal scroll + bend
    p.x += 0.012 * pc.time * speed + bendX;

    // slight vertical warp near mouse (only when pressed)
    float dmy = p.y - mN.y;
    p.y += pressed * 0.028 * dmy * exp(-abs(dmy) * 40.0);

    // ---- neon palette bands ----
    vec3 a = vec3(0.120, 0.38, 0.45);
    vec3 b = vec3(0.290, 0.75, 0.85) * (1.0 + 0.35 * energy);

    vec3 c = vec3(1.0), d = vec3(0.20, 0.33, 0.57);
    if (p.y > (1.0/7.0)) { c = vec3(1.0);           d = vec3(0.00, 0.10, 0.20); }
    if (p.y > (2.0/7.0)) { c = vec3(1.0);           d = vec3(0.30, 0.20, 0.20); }
    if (p.y > (3.0/7.0)) { c = vec3(1.0,0.20,0.5);   d = vec3(0.80, 0.50, 0.30); }
    if (p.y > (4.0/7.0)) { c = vec3(1.0,2.7,0.4);   d = vec3(0.20, 0.15, 0.20); }
    if (p.y > (5.0/7.0)) { c = vec3(0.5,0.50,0.70);   d = vec3(0.20, 0.50, 0.25); }
    if (p.y > (6.0/7.0)) { c = vec3(0.3,0.5,0.80);   d = vec3(0.00, 0.25, 0.55); }

    // neon sweep through phase space
    d += hueCtl * vec3(0.00, 0.18, 0.28) + 0.05 * sin(vec3(0.9,1.1,1.3) * (pc.time * 0.6));

    vec3 baseCol = pal(p.x, a, b, c, d);

    // ---- band shaping ----
    float bands = p.y * 7.0;
    float f     = fract(bands);
    float dEdge = 0.5 - abs(f - 0.5);        // 0 at edges, 0.5 at band center

    // bright central tube (reacts to energy)
    float lineW = mix(0.045, 0.11, clamp(energy,0.0,1.0));
    float line  = smoothstep(0.0, lineW, dEdge);

    // soft interior shading (as IQ)
    float shade = 0.5 + 0.5 * sqrt(4.0 * f * (1.0 - f));

    // wide glow around edges
    float haloW = mix(2.50, 0.32, clamp(energy,0.0,1.0));
    float glow  = pow(smoothstep(0.0, 1.5, dEdge), 6.0) * (0.65 + 0.55 * energy);
    glow       *= smoothstep(0.0, haloW, dEdge);

    vec3 col = baseCol * (0.15 + 0.85 * shade);
    col += baseCol * line * 1.25;
    col += baseCol * glow;

    // mouse hotspot (use static coords so it never disappears)
    float spot = exp(-dot(gl_FragCoord.xy - mN * R, gl_FragCoord.xy - mN * R) / 1800.0);
    col += baseCol * spot * (0.20 + 0.45 * pressed);

    // vignette & scanlines — IMPORTANT: use uv0 (static), not p (scrolled)
    float aspect = R.x / R.y;
    float vignR  = (energy * 2.25) * length((uv0 - 0.35) * vec2(aspect, 1.0));
    float vig    = smoothstep(0.055, 1.25, vignR);  // edges darker, center bright

    float scan   = 0.96 + 0.04 * sin(gl_FragCoord.y * 3.14159);

    col *= vig * 0.1 * scan;

    // contrast & gamma
    col = clamp((col - 0.5) * (1.2 + 0.8 * pc.cc1) + 0.5, 0.0, 1.0);
    col = pow(col, vec3(0.4545));

    fragColor = vec4(col, 1.0);
}
