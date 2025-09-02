#version 450

// ---------------- Optional inputs (uncomment if you bind them) ----------------
// layout(set=0, binding=0) uniform sampler2D uBloom; // blurred bloom chain
// #define APPLY_BLOOM
// layout(set=0, binding=1) uniform sampler2D uScene; // downsampled scene buffer
// #define USE_SCENE_TEX
// Integer downsample factor used when writing uScene (matches your kScreenDownsample)
#define SCREEN_DOWNSAMPLE 1

// How strongly to overlay the bands on top of the composite (0..1)
#define BANDS_WEIGHT 2.85

// ---------------- Push constants / IO ----------------
layout(push_constant) uniform PushConstants {
    float time; uint mouse_x; uint mouse_y; uint mouse_pressed;
    float note_velocity; float pitch_bend; float cc1; float cc74;
    uint  note_count; uint last_note; float osc_ch1; float osc_ch2;
    uint  render_w; uint render_h;
} pc;

layout(location=0) in vec2 frag_uv;          // 0..1
layout(location=1) in vec2 frag_screen_pos;  // optional
layout(location=0) out vec4 out_color;

// ---------------- Helpers ----------------
#define PI 3.1415926
#define ROOT2 1.41421356237
float saturate(float x){ return clamp(x, 0.0, 1.0); }
vec3  saturate(vec3  x){ return clamp(x, 0.0, 1.0); }
float hash11(float n){ return fract(sin(n*104729.0)*43758.5453123); }
float hash21(vec2 p){ return fract(sin(dot(p, vec2(127.1,311.7))) * 43758.5453123); }

// ---------------- Palettes (original) ----------------
vec3 palette1(float x){ float v = pow(sin(PI*x), 2.0); return v * vec3(1.0, 0.5*v, 0.5); }
vec3 palette2(float x){ float v = pow(sin(PI*x), 2.0); return v * vec3(0.5*v, 1.0*v, 0.25); }
vec3 palette3(float x){ float v = pow(sin(PI*x), 2.0); return v * vec3(1.0, 0.5*v, 0.25*v); }
vec3 palette4(float x){ float v = pow(sin(PI*x), 2.0); return v * vec3(0.5, 0.75*v, 0.5); }
vec3 getPalette(int idx, float x){
    if(idx==0) return palette1(x);
    if(idx==1) return palette2(x);
    if(idx==2) return palette3(x);
    return palette4(x);
}

// ---------------- Vignette (same math) ----------------
float Vignette(vec2 fragCoord, vec2 iResolution)
{
    const float kVignetteStrength = 0.5;
    const float kVignetteScale    = 0.6;
    const float kVignetteExponent = 3.0;

    vec2 uv = fragCoord / iResolution;
    uv.x = (uv.x - 0.5) * (iResolution.x / iResolution.y) + 0.5;

    float x = 2.0 * (uv.x - 0.5);
    float y = 2.0 * (uv.y - 0.5);
    float dist = sqrt(x*x + y*y) / ROOT2;

    float ring = max(0.0, 1.0 - pow(dist * kVignetteScale, kVignetteExponent));
    return mix(1.0, ring, kVignetteStrength);
}

// ---------------- Overlapped/shrinking bands ----------------
//
// - Vertical overlap: Gaussian row masks (sigma grows with audio)
// - Horizontal shrink: each band gets a random width gate that tightens with audio
// - Scroll: pitch bend + mouse + audio wobble
//
vec3 renderBandsOverlapped(vec2 uv, vec2 iResolution)
{
    float audio = saturate((pc.note_velocity + pc.osc_ch1 + pc.osc_ch2) * 0.5);
    float t     = pc.time;
    bool  mdown = (pc.mouse_pressed != 0u);

    // Global horizontal scroll
    float xShift = 0.20*pc.pitch_bend + 0.08*audio*sin(t*2.1);
    if(mdown){
        xShift += (float(pc.mouse_x)/max(iResolution.x,1.0) - 0.5) * 0.6;
    }

    // Band layout
    const int NBANDS = 10;                           // number of rows
    float baseH   = 1.0/float(NBANDS);              // base row height
    float sigma   = 0.5 * baseH * mix(0.36, 0.80, audio); // vertical Gaussian width => more audio = more overlap
    float rate    = mix(0.5, 6.0, audio);           // how fast random widths change

    // Note salt to vary palettes per note
    float noteSalt   = (pc.note_count>0u ? float(pc.last_note)/127.0 : 0.37);

    vec3 accum = vec3(0.0);
    float aSum = 0.0;

    for(int i=0;i<NBANDS;i++){
        float fi = float(i);

        // Row center with tiny jitter that breathes with audio
        float jitter = (hash11(fi*17.0 + floor(t*1.3)) - 0.5) * baseH * 0.35 * audio;
        float center = (fi + 0.5)*baseH + jitter;

        // Vertical Gaussian mask (always overlapping)
        float dy = (uv.y - center)/sigma;
        float maskY = exp(-0.5*dy*dy);             // 1 at center, falls off with sigma

        // Random horizontal shrink gate:
        // width varies in [minW, 1], where minW gets smaller with audio
        float rnd  = hash11(fi*113.0 + floor(t*rate));
        float minW = mix(0.75, 0.25, audio);       // more audio => narrower possible bands
        float wX   = mix(1.0, mix(minW, 1.0, rnd), audio); // 1..minW
        // Optional random x-center shift per band (keeps motion lively)
        float cx   = fract(0.5 + 0.22*sin(t*1.7 + fi*1.23) + 0.18*pc.pitch_bend
        + (hash11(fi*31.0) - 0.5)*0.2);

        // Smooth horizontal rectangle gate
        float ax   = abs(fract(uv.x + xShift) - cx);
        float edge = 0.5*wX;                       // half-width in normalized coords
        float soft = 0.02 + 0.08*(1.0 - wX);       // softer edge for thinner gates
        float gateX= 1.0 - smoothstep(edge, edge+soft, ax); // 1 inside gate, 0 outside

        // Per-band palette cycling
        int pidx = int(mod(fi + floor(t*1.1) + floor(noteSalt*8.0), 4.0));
        // Palette sampling still uses the scrolled x; it’s okay if the gate masks it later.
        float px = fract(uv.x + xShift + 0.07*fi);
        vec3 col;
        if(pidx==0) col = palette1(px);
        else if(pidx==1) col = palette2(px);
        else if(pidx==2) col = palette3(px);
        else             col = palette4(px);

        // Approx to linear
        col = pow(col, vec3(0.45454545));

        // Row edge shaping (from your original)
        float n = 1.0 - fract(uv.y*4.0)*2.0;
        col *= sqrt(max(0.0, 1.0 - n*n));

        // Composite this band
        float a = maskY * gateX;
        accum += col * a;
        aSum  += a;
    }

    // Normalize to avoid blowout when many bands overlap
    accum /= max(aSum, 0.7);

    return accum;
}

// ---------------- Main ----------------
void main(){
    vec2 iResolution = vec2(float(pc.render_w), float(pc.render_h));
    vec2 fragCoord   = frag_uv * iResolution;

    vec3 rgb = vec3(0.0);

    // Optional bloom contribution
    #ifdef APPLY_BLOOM
    rgb = texture(uBloom, frag_uv).rgb;
    #endif

    // Optional downsampled scene composite
    #ifdef USE_SCENE_TEX
    ivec2 ip = ivec2(floor(fragCoord));
    rgb += texelFetch(uScene, ip / SCREEN_DOWNSAMPLE, 0).rgb * 0.6;
    #endif

    // Overlapped, horizontally shrinking bands (audio-reactive)
    vec3 bands = renderBandsOverlapped(frag_uv, iResolution);
    rgb = mix(rgb, rgb + bands, BANDS_WEIGHT); // additive-ish overlay

    // Same shaping + vignette as your post pass
    rgb  = saturate(rgb);
    rgb  = pow(rgb, vec3(0.8));
    rgb  = mix(vec3(0.1), vec3(0.9), rgb);
    rgb *= Vignette(fragCoord, iResolution);
    rgb  = saturate(rgb);

    // Live CCs: brightness & gamma
    float bright = mix(0.85, 1.55, saturate(pc.cc74));
    float gammaC = 1.0 + 0.7*saturate(pc.cc1);
    rgb = pow(max(rgb*bright, 0.0), vec3(gammaC));

    out_color = vec4(clamp(rgb, 0.0, 3.0), 1.0);
}
