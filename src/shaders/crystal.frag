#version 450

// ===================================================================================
// Push constants & varyings (fits your scaffold)
// ===================================================================================
layout(push_constant) uniform PushConstants {
    float time;
    uint  mouse_x;
    uint  mouse_y;
    uint  mouse_pressed;
    float note_velocity;
    float pitch_bend;
    float cc1;    // mids / mod wheel
    float cc74;   // highs / cutoff
    uint  note_count;
    uint  last_note;
    uint  render_w;
    uint  render_h;
} pc;

layout(location = 0) in vec2  fragUV;        // 0..1
layout(location = 1) in float vertexEnergy;
layout(location = 2) in vec3  worldPos;

layout(location = 0) out vec4 outColor;

// ===================================================================================
// Globals (mirrors style of your source)
// ===================================================================================
#define kScreenDownsample 1

vec2  gResolution;
vec2  gFragCoord;
float gTime;
float gDxyDuv;

void SetGlobals(vec2 fragCoord, vec2 resolution, float time){
    gFragCoord  = fragCoord;
    gResolution = resolution;
    gTime       = time;
    gDxyDuv     = 1.0 / gResolution.x; // for SDF widths, assumes square pixels
}

// ===================================================================================
// Math & utilities
// ===================================================================================
#define kPi     3.14159265359
#define kTwoPi  (2.0 * kPi)
#define kRoot2  1.41421356237

float sqr(float a){ return a*a; }
float saturate(float a){ return clamp(a,0.0,1.0); }
float sin01(float a){ return 0.5*sin(a)+0.5; }
float SmoothStep01(float x){ return x*x*(3.0-2.0*x); }
float PaddedSmoothStep(float x, float a, float b){ return SmoothStep01(saturate(x*(a+b+1.0)-a)); }
float toRad(float deg){ return kTwoPi*deg/360.0; }

// ===================================================================================
/* Hashing / RNG (FNV + combiners) */
// ===================================================================================
uint HashCombine(uint a, uint b){
    return (((a << (31u - (b & 31u))) | (a >> (b & 31u)))) ^
    ((b << (a & 31u)) | (b >> (31u - (a & 31u))));
}

uint HashOf(uint i){
    const uint kFNVPrime=0x01000193u, kFNVOffset=0x811c9dc5u;
    uint h=(kFNVOffset ^ (i & 0xffu)) * kFNVPrime;
    h=(h ^ ((i>>8u)&0xffu))*kFNVPrime;
    h=(h ^ ((i>>16u)&0xffu))*kFNVPrime;
    h=(h ^ ((i>>24u)&0xffu))*kFNVPrime;
    return h;
}
uint HashOf(uint a, uint b){ return HashCombine(HashOf(a), HashOf(b)); }
uint HashOf(uint a, uint b, uint c){ return HashCombine(HashCombine(HashOf(a), HashOf(b)), HashOf(c)); }
uint HashOf(uint a, uint b, uint c, uint d){
    return HashCombine(HashCombine(HashOf(a), HashOf(b)), HashCombine(HashOf(c), HashOf(d)));
}
uint HashOf(uvec2 v){ return HashCombine(HashOf(v.x), HashOf(v.y)); }

float HashToFloat(uint i){ return float(i) / float(0xffffffffu); }

// Simple value noise
float vhash(vec2 p){
    uvec2 up = uvec2(floatBitsToUint(p.x*1639.0), floatBitsToUint(p.y*1531.0));
    return HashToFloat(HashOf(up));
}
float n2(vec2 p){
    vec2 i=floor(p), f=fract(p);
    float a=vhash(i);
    float b=vhash(i+vec2(1,0));
    float c=vhash(i+vec2(0,1));
    float d=vhash(i+vec2(1,1));
    vec2 u=f*f*(3.0-2.0*f);
    return mix(mix(a,b,u.x), mix(c,d,u.x), u.y);
}

// ===================================================================================
// Color helpers (HSV) — used for color bursts
// ===================================================================================
vec3 Hue(float phi){
    float t = 6.0*phi; int i=int(t);
    vec3 c0 = vec3(((i+4)/3)&1, ((i+2)/3)&1, ((i+0)/3)&1);
    vec3 c1 = vec3(((i+5)/3)&1, ((i+3)/3)&1, ((i+1)/3)&1);
    return mix(c0, c1, t - float(i));
}
vec3 HSVToRGB(vec3 hsv){ return mix(vec3(0.0), mix(vec3(1.0), Hue(hsv.x), hsv.y), hsv.z); }
vec3 RGBToHSV(vec3 rgb){
    float v = max(max(rgb.r,rgb.g),rgb.b);
    float m = min(min(rgb.r,rgb.g),rgb.b);
    float c = v - m;
    float h = (c < 1e-10) ? 0.0 :
    (v==rgb.r ? (rgb.g-rgb.b)/c/6.0 :
    v==rgb.g ? (2.0 + (rgb.b-rgb.r)/c)/6.0 :
    (4.0 + (rgb.r-rgb.g)/c)/6.0);
    if(h<0.0) h+=1.0;
    float s = (v<1e-10)?0.0:c/max(1e-10,v);
    return vec3(h,s,v);
}

// Psychedelic color palette for bursts
vec3 BurstColor(float t, float energy){
    vec3 col1 = vec3(1.0, 0.1, 0.8);  // hot pink
    vec3 col2 = vec3(0.1, 1.0, 0.3);  // electric green
    vec3 col3 = vec3(0.9, 0.9, 0.1);  // electric yellow
    vec3 col4 = vec3(0.2, 0.3, 1.0);  // electric blue
    vec3 col5 = vec3(1.0, 0.4, 0.1);  // electric orange

    float phase = fract(t);
    float band = floor(t * 5.0);

    vec3 base;
    if(band < 1.0) base = mix(col1, col2, phase);
    else if(band < 2.0) base = mix(col2, col3, phase);
    else if(band < 3.0) base = mix(col3, col4, phase);
    else if(band < 4.0) base = mix(col4, col5, phase);
    else base = mix(col5, col1, phase);

    return base * (1.0 + 2.0*energy);
}

// ===================================================================================
// Ordered dithering (4x4 Bayer)
// ===================================================================================
const mat4 kOrderedDither = mat4(
vec4( 0.0,  8.0,  2.0, 10.0),
vec4(12.0,  4.0, 14.0,  6.0),
vec4( 3.0, 11.0,  1.0,  9.0),
vec4(15.0,  7.0, 13.0,  5.0)
);
float OrderedDither(ivec2 p){ return (kOrderedDither[p.x & 3][p.y & 3] + 1.0) / 17.0; }

// ===================================================================================
// Screen->world transforms (2D homography-ish)
// ===================================================================================
// Column-major; multiply as (M * vec3(p,1))
mat3 WorldToViewMatrix(float rot, vec2 trans, float sca){
    float c=cos(rot), s=sin(rot);
    // columns
    return mat3(
    c/sca,  s/sca, 0.0,
    -s/sca,  c/sca, 0.0,
    trans.x, trans.y, 1.0
    );
}
vec2 TransformScreenToWorld(vec2 p){
    return (p - 0.5*gResolution) / gResolution.y;
}

// ===================================================================================
// Hex tiling helpers (barycentric mapping, like your source)
// ===================================================================================
vec3 Cartesian2DToBarycentric(vec2 p){
    return vec3(p,0.0) * mat3(
    vec3(0.0, 1.0/0.8660254037844387, 0.0),
    vec3(1.0, 0.5773502691896257,     0.0),
    vec3(-1.0,0.5773502691896257,     0.0)
    );
}
vec2 Cartesian2DToHexagonalTiling(in vec2 uv, out vec3 bary, out ivec2 ij){
    const vec2 kHexRatio = vec2(1.5, 0.8660254037844387);
    vec2 uvClip = mod(uv + kHexRatio, 2.0*kHexRatio) - kHexRatio;

    ij = ivec2((uv + kHexRatio) / (2.0*kHexRatio)) * 2;
    if(uv.x + kHexRatio.x <= 0.0) ij.x -= 2;
    if(uv.y + kHexRatio.y <= 0.0) ij.y -= 2;

    bary = Cartesian2DToBarycentric(uvClip);
    if(bary.x > 0.0){
        if(bary.z > 1.0){ bary += vec3(-1.0, 1.0, -2.0); ij += ivec2(-1, 1); }
        else if(bary.y > 1.0){ bary += vec3(-1.0, -2.0, 1.0); ij += ivec2(1, 1); }
    } else {
        if(bary.y < -1.0){ bary += vec3(1.0, 2.0, -1.0); ij += ivec2(-1, -1); }
        else if(bary.z < -1.0){ bary += vec3(1.0, -1.0, 2.0); ij += ivec2(1, -1); }
    }
    return vec2(bary.y*0.5773502691896257 - bary.z*0.5773502691896257, bary.x);
}

// ===================================================================================
// Dark vignette (stronger than original)
// ===================================================================================
float DarkVignette(vec2 fragCoord){
    vec2 uv = fragCoord / gResolution;
    uv.x = (uv.x - 0.5) * (gResolution.x / gResolution.y) + 0.5;
    vec2 p = 2.0*(uv - 0.5);
    float dist = length(p) / kRoot2;
    // Much stronger darkening
    const float kStrength=0.85, kScale=0.4, kExp=2.5;
    return mix(1.0, max(0.0, 1.0 - pow(dist*kScale, kExp)), kStrength);
}

// ===================================================================================
// Interference / displacement (enhanced for darker feel)
// ===================================================================================
bool Interfere(inout vec2 xy, inout vec3 tint, in vec2 res, float mids, float highs){
    // More aggressive interference for darker mood
    float kStaticFrequency       = mix(0.12, 0.25, highs);
    float kStaticLowMagnitude    = 0.015;
    float kStaticHighMagnitude   = 0.04 + 0.08*highs;

    float kVDisplaceFrequency    = mix(0.08, 0.18, mids);
    float kHDisplaceFrequency    = mix(0.25, 0.40, highs);
    float kHDisplaceVMagnitude   = 0.12 + 0.12*mids;
    float kHDisplaceHMagnitude   = 0.45 + 0.35*highs;

    // Frame bucket from time
    float frameBucketF = floor(gTime * 60.0 / 8.0);  // Faster interference
    uint  frameBucket  = uint(max(0.0, frameBucketF));
    float frameHash    = HashToFloat(HashOf(frameBucket));

    bool  isDisplaced  = false;
    tint = vec3(0.4); // Darker base tint

    // Static row jitter (more aggressive)
    {
        float interP     = 0.02;
        float displacement = res.x * kStaticLowMagnitude;
        if(frameHash < kStaticFrequency){
            interP      = 0.6;
            displacement = kStaticHighMagnitude * res.x;
            tint        = vec3(0.2); // Even darker during static
        }
        float rowSeed = n2(vec2(0.0, xy.y*0.3 + gTime*3.0));
        if(rowSeed < interP){
            float mag = n2(vec2(gTime*4.5, xy.y*0.25))*2.0 - 1.0;
            xy.x -= displacement * sign(mag) * sqr(abs(mag));
            isDisplaced = true;
        }
    }

    // Vertical displacement gate
    if(frameHash > 1.0 - kVDisplaceFrequency){
        float dispX = HashToFloat(HashOf(8783u, frameBucket));
        float dispY = HashToFloat(HashOf(364719u, frameBucket));
        if(xy.y < dispX * res.y){
            xy.y -= mix(-1.0, 1.0, dispY) * res.y * 0.25;
            isDisplaced = true;
            tint = vec3(0.1); // Very dark
        }
    }
    // Horizontal displacement band gate
    else if(frameHash > 1.0 - kHDisplaceFrequency - kVDisplaceFrequency){
        float dispX = HashToFloat(HashOf(147251u, frameBucket));
        float dispY = HashToFloat(HashOf(287512u, frameBucket));
        float dispZ = HashToFloat(HashOf(8756123u, frameBucket));
        if(xy.y > dispX * res.y && xy.y < (dispX + mix(0.0, kHDisplaceVMagnitude, dispZ)) * res.y){
            xy.x -= mix(-1.0, 1.0, dispY) * res.x * kHDisplaceHMagnitude;
            isDisplaced = true;
            tint = vec3(0.15);
        }
    }

    return isDisplaced;
}

// ===================================================================================
// Core "Render" — darker with color bursts
// ===================================================================================
vec3 Render(vec2 xy, int idx, int maxSamples, bool isDisplaced, float jpegDamage,
float mids, float highs, float velocity, out float blendOut)
{
    // Temporal sampling
    float xi = (float(idx) + 0.6180339) / float(maxSamples);
    float baseSpeed = mix(0.08, 0.18, 0.5*mids + 0.5*highs); // Slower, more deliberate
    float time = gTime * baseSpeed + xi * (isDisplaced ? 2.0 : 0.4);

    float phase = fract(time);
    int interval = (int(floor(time)) & 1) << 1;
    float morph, warpedTime;

    const float kIntervalPartition = 0.85;
    if(phase < kIntervalPartition){
        float y = (interval==0) ? xy.y : (gResolution.y - xy.y);
        warpedTime = (phase / kIntervalPartition) - 0.3 * sqrt(y / gResolution.y) - 0.15;
        phase = fract(warpedTime);
        morph = 1.0 - PaddedSmoothStep(sin01(kTwoPi * phase), 0.0, 0.3);
        blendOut = float(interval/2) * 0.3; // Darker blend
        if(interval == 2) warpedTime *= 0.6;
    } else {
        time -= 0.9 * baseSpeed * xi * (isDisplaced?1.2:0.5);
        warpedTime = time;
        phase = (fract(time) - kIntervalPartition) / (1.0 - kIntervalPartition);
        // KickDrop-ish curve
        float kd = exp(-sqr((phase - 0.15)/0.25)) * -0.15 + smoothstep(0.25, 0.8, phase);
        blendOut = (kd + float(interval/2)) * 0.4;
        morph = 1.0;
        interval++;
    }
    float beta = abs(2.0*max(0.0, blendOut) - 1.0);

    // Screen->world with chroma warp
    vec2 uvView = TransformScreenToWorld(xy);
    float xi2 = n2(uvView*vec2(113.0,97.0) + gTime);
    uvView /= 1.0 + 0.08 * length(uvView) * xi2;

    // Rotation + zoom
    float expMorph = pow(morph, 0.4);
    const float kZoom = 0.4;
    float kScale = mix(2.8, 1.0, expMorph);
    mat3 M = WorldToViewMatrix(blendOut * kTwoPi, vec2(0.0), kZoom);
    uvView = (M * vec3(uvView,1.0)).xy;

    // Y squish warp and rotation
    vec2 uvWarp = uvView;
    uvWarp.y *= mix(1.0, 0.08, sqr(1.0 - morph) * xi * saturate(sqr(0.4*(1.0 + uvView.y))));
    float thetaR = toRad(45.0) * beta;
    mat2 R = mat2(cos(thetaR), -sin(thetaR), sin(thetaR), cos(thetaR));
    uvWarp = R * uvWarp;

    // Hex-based inversion + ripple thresholding
    int invert = 0;
    const int kMaxIterations = 2;
    const int kTurns = 9;
    const int kNumRipples = 7;
    float kRippleDelay = float(kNumRipples) / float(kTurns);
    float kThickness = mix(0.6, 0.3, morph);
    float kExponent  = mix(0.03, 0.65, morph);

    for(int iter=0; iter<0; ++iter){
        vec3 bary; ivec2 ij;
        Cartesian2DToHexagonalTiling(uvWarp, bary, ij);

        // Random hex gate
        uint hexHash = HashOf(uint(floor(phase*1.0)), uint(iter), uint(ij.x), uint(ij.y));
        if((hexHash & 1u) == 0u){
            float alpha = PaddedSmoothStep(sin01(phase*25.0), 0.1, 0.8);
            float dist = mix(max(max(abs(bary.x),abs(bary.y)),abs(bary.z)), length(uvView)*3.0, 1.0 - alpha);
            float hsum = bary[hexHash%3u] + bary[(hexHash+1u)%3u];
            if(dist > 1.0 - 0.015) invert ^= 1;
            else if(fract(25.0*hsum) < 0.4) invert ^= 1;
            if(iter==0) break;
        }

        // Ring ripples
        float sigma=0.0, wsum=0.0;
        for(int j=0;j<kTurns;++j){
            float delta=float(j)/float(kTurns);
            float theta=kTwoPi*delta;
            for(int i=0;i<kNumRipples;++i){
                float l = length(uvWarp - vec2(cos(theta), sin(theta))) * 0.6;
                float weight = log2(1.0 / (l + 1e-10));
                float ph = fract((float(j) + float(i)/kRippleDelay)/float(kTurns) + warpedTime);
                sigma += fract(l - pow(ph, kExponent)) * weight;
                wsum  += weight;
            }
        }
        invert ^= int((sigma / max(1e-5, wsum)) > kThickness);

        // Spiral out
        float theta2=kTwoPi*(floor(sin01(-kTwoPi*phase)*7.0*6.0)/6.0);
        uvWarp = R * (uvWarp + vec2(cos(theta2), sin(theta2)) * 0.6);
        uvWarp *= kScale;
    }

    // Color burst calculation
    float burstTrigger = 0.0;

    // Velocity-based bursts
    if(velocity > 0.7) {
        burstTrigger += (velocity - 0.7) * 3.0;
    }

    // High frequency bursts
    if(highs > 0.6) {
        burstTrigger += (highs - 0.6) * 2.5;
    }

    // Random burst gates
    float burstHash = HashToFloat(HashOf(uint(floor(gTime * 8.0)), uint(floor(xy.x/64.0)), uint(floor(xy.y/64.0))));
    if(burstHash > 0.95) {
        burstTrigger += 1.5;
    }

    // Time-based pulse bursts
    float pulse = sin(gTime * 12.0 + length(uvWarp) * 8.0);
    if(pulse > 0.8 && velocity > 0.5) {
        burstTrigger += (pulse - 0.8) * 2.0;
    }

    // Base dark color (much darker than original)
    vec3 baseA = vec3(0.05);  // Very dark
    vec3 baseB = vec3(0.02, 0.03, 0.08);  // Dark blue-ish
    vec3 sigma = vec3(float(invert!=0));
    vec3 col = mix(baseA - sigma * 0.03, sigma * mix(baseB, vec3(0.08), sqr(beta)), beta * 0.6);

    // Apply color burst
    if(burstTrigger > 0.1) {
        float burstIntensity = smoothstep(0.1, 2.0, burstTrigger);
        vec3 burstCol = BurstColor(gTime * 2.0 + length(uvWarp) * 3.0, velocity);

        // Burst mask - more likely at pattern edges
        float burstMask = 1.0;
        if(invert != 0) {
            burstMask *= 1.5; // Brighter bursts on white areas
        }

        // Distance-based burst falloff
        float distMask = 0.5 - smoothstep(0.0, 1.0, length(uvWarp * 0.5));
        burstMask *= distMask;

        col = mix(col, burstCol, burstIntensity * burstMask * 0.8);
    }

    // Subtle dark glow (reduced from original)
    float glow = 0.0;
    for(int k=0;k<10;k++){
        float a = float(k) * (kTwoPi/4.0);
        vec2  off = 0.002 * vec2(cos(a), sin(a));
        float samp = n2((uvWarp + off)*50.0 + gTime);
        glow += smoothstep(0.8, 1.0, samp);
    }
    glow /= 4.0;
    col += vec3(glow) * 0.01; // Much subtler glow

    return col;
}

// ===================================================================================
// Main
// ===================================================================================
void main(){
    // Resolution & coords
    vec2 iRes = (pc.render_w>0u && pc.render_h>0u) ? vec2(pc.render_w, pc.render_h) : vec2(800.0,600.0);
    vec2 fragCoord = fragUV * iRes;

    // Globals
    SetGlobals(fragCoord, iRes, pc.time);

    // Audio knobs
    float A = clamp(pc.note_velocity, 0.0, 1.0);
    float M = clamp(pc.cc1, 0.0, 1.0);
    float H = clamp(pc.cc74, 0.0, 1.0);

    // Screen-space interference
    vec2 xy = fragCoord * float(kScreenDownsample);
    vec3 tint = vec3(0.4); // Start darker
    bool isDisplaced = Interfere(xy, tint, iRes, M, H);

    // Ordered dither for "damage"
    uint tHashA = HashOf(uint(floor(gTime + sin(gTime)*1.8)));
    uint tHashB = HashOf(uint(floor(xy.x/96.0)));
    uint tHashC = HashOf(uint(floor(xy.y/96.0)));
    int denom  = int(HashOf(tHashA, tHashB, tHashC) & 127u);
    denom = max(1, denom);
    ivec2 xyDither = ivec2(xy) / denom;
    float jpegDamage = OrderedDither(xyDither);

    // Supersampling
    const int kAA = 1; // Reduced for performance since it's darker
    vec3 rgb = vec3(0.0);
    float blendSum = 0.0;
    int idx = 0;
    for(int i=0;i<kAA;i++){
        for(int j=0;j<kAA;j++,idx++){
            vec2 xyAA = xy + vec2(float(i), float(j))/float(kAA);
            float b;
            rgb += Render(xyAA, idx, kAA*kAA, isDisplaced, jpegDamage, M, H, A, b);
            blendSum += b;
        }
    }
    rgb /= float(kAA*kAA);
    float blend = blendSum / float(kAA*kAA);

    // Dark grading (crush blacks, enhance color separation)
    vec3 hsv = RGBToHSV(rgb);
    hsv.x += -sin((hsv.x + 0.08) * kTwoPi) * 0.12; // More color shift
    hsv.y = min(1.0, hsv.y * 1.7); // Boost saturation
    hsv.z = pow(hsv.z, 1.4); // Crush blacks more
    rgb = HSVToRGB(hsv);

    // Displacement-triggered quantization (more aggressive)
    if(isDisplaced){
        const float kColourQuantisation = 20.0; // Fewer levels for harsher look
        vec3 q = rgb * kColourQuantisation;
        if(fract(q.r) > jpegDamage) q.r += 1.0;
        if(fract(q.g) > jpegDamage) q.g += 1.0;
        if(fract(q.b) > jpegDamage) q.b += 1.0;
        rgb = floor(q) / kColourQuantisation;
    }

    // Strong dark vignette
    rgb *= DarkVignette(xy);

    // Multiply by tint for interference darkening
    rgb *= tint;

    // Reduce vertex energy effect
    rgb *= (1.0 + 0.05 * clamp(vertexEnergy, 0.0, 1.0));

    // Final dark clamp
    rgb = clamp(rgb, 0.0, 1.0);

    outColor = vec4(rgb, 1.0);
}