#version 450

layout(push_constant) uniform PushConstants {
    float time;
    uint  mouse_x;
    uint  mouse_y;
    uint  mouse_pressed;
    float note_velocity;
    float pitch_bend;   // [-1..1]
    float cc1;          // Modulation wheel (mid)
    float cc74;         // Filter cutoff (high)
    uint  note_count;
    uint  last_note;
} pc;

layout(location = 0) out vec2 fragUV;
layout(location = 1) out float vertexEnergy;
layout(location = 2) out vec3 worldPos;

// --------------------- original fullscreen tri ---------------------
const vec2 positions[3] = vec2[](
    vec2(-1.0, -1.0),
    vec2( 3.0, -1.0),
    vec2(-1.0,  3.0)
);

const vec2 uvs[3] = vec2[](
    vec2(0.0,  1.0),
    vec2(2.0,  1.0),
    vec2(0.0, -1.0)
);

// --------------------- helpers & noise ---------------------
float sat(float x){ return clamp(x,0.0,1.0); }
vec2  rot(vec2 p, float a){ float c=cos(a), s=sin(a); return mat2(c,-s,s,c)*p; }
float noteToFreq(uint note){ return 440.0 * pow(2.0, (float(note) - 69.0) / 12.0); }

float hash12(vec2 p){
    return fract(sin(dot(p, vec2(127.1,311.7))) * 43758.5453123);
}

// value noise
float vnoise(vec2 p){
    vec2 i=floor(p), f=fract(p);
    vec2 u=f*f*(3.0-2.0*f);
    float a=hash12(i);
    float b=hash12(i+vec2(1,0));
    float c=hash12(i+vec2(0,1));
    float d=hash12(i+vec2(1,1));
    return mix(mix(a,b,u.x), mix(c,d,u.x), u.y);
}

float fbm(vec2 p){
    float a=0.0, amp=0.5;
    for(int i=0;i<5;i++){
        a += amp * vnoise(p);
        p  = p*2.03 + 19.19;
        amp*= 0.5;
    }
    return a;
}

// central-difference gradient of fbm (for curl)
vec2 grad_fbm(vec2 p){
    // small epsilon in *different* axes for numerical stability
    const float e = 0.0015;
    float fx1 = fbm(p + vec2(e,0.0));
    float fx0 = fbm(p - vec2(e,0.0));
    float fy1 = fbm(p + vec2(0.0,e));
    float fy0 = fbm(p - vec2(0.0,e));
    return vec2((fx1 - fx0)/(2.0*e), (fy1 - fy0)/(2.0*e));
}

// kaleidoscope fold in *displacement field* (not geometry)
vec2 kalei(vec2 p, float N){
    float a = atan(p.y, p.x);
    float r = length(p);
    float seg = 6.28318530718 / max(N, 1.0);
    a = mod(a, seg);
    a = abs(a - seg*0.5);
    return vec2(cos(a), sin(a)) * r;
}

void main() {
    // base tri
    vec2 pos = positions[gl_VertexIndex];
    vec2 uv  = uvs[gl_VertexIndex];

    // --- musical parameters
    float freq       = noteToFreq(pc.last_note);
    float freq_norm  = freq / 2000.0;
    float A          = sat(pc.note_velocity);               // loudness
    float L          = sat(pc.pitch_bend*0.5 + 0.5);        // low
    float M          = sat(pc.cc1);                         // mids
    float H          = sat(pc.cc74);                        // highs
    float crowd      = float(pc.note_count);

    // approximate resolution to match your frag shader's mouse math
    vec2 res = vec2(800.0, 600.0);
    vec2 m   = vec2(float(pc.mouse_x)/res.x, float(pc.mouse_y)/res.y);  // 0..1
    vec2 mndc= m*2.0 - 1.0; // -1..1
    mndc.y   = mndc.y; // keep as-is (frag also uses unflipped y)

    // --- time scaffolding & pitch class
    float theta = mod(float(pc.last_note), 12.0)/12.0 * 6.28318530718;
    float t  = pc.time;
    float tA = t * (0.5 + 0.5*A);
    float tM = t * (0.3 + 0.7*M);

    // --- original-style harmonic waves, beefed up
    float tf = t * (freq_norm*0.25 + 0.03 + 0.04*crowd);
    vec2 disp = vec2(0.0);

    // fundamental
    disp += 0.050*A * vec2(
        sin(tf + pos.x*10.0 + 0.7*theta),
        cos(tf + pos.y* 8.0 - 0.3*theta)
    );

    // 2nd & 3rd
    disp += 0.035*A * vec2(
        sin(tf*2.0 + pos.x*15.0),
        sin(tf*3.0 + pos.y*12.0)
    );

    // 5th & 7th harmonics with mid/treble accents
    disp += 0.028*(0.7*A+0.3*M) * vec2(
        sin(tf*5.0 + pos.x*23.0 - 0.9*theta),
        cos(tf*5.0 + pos.y*19.0 + 0.6*theta)
    );
    disp += 0.020*(0.6*A+0.4*H) * vec2(
        sin(tf*7.0 + pos.x*29.0),
        sin(tf*7.0 + pos.y*27.0)
    );

    // --- curl-noise flow (swirly, controlled by bands)
    // build a warped coordinate for the noise field (safe, tiny scale)
    vec2 p  = pos;
    p = rot(p, 0.15*t + 0.25*theta);
    vec2 pw = p*1.8 + vec2( fbm(p*0.6 + 0.3*t), fbm(p.yx*0.7 - 0.25*t) );

    vec2 g  = grad_fbm(pw*2.2);
    vec2 curl = vec2( g.y, -g.x ); // 90° rotated gradient
    // mix in a kalei fold to the *field* (not position)
    curl = kalei(curl, floor(mix(5.0, 11.0, M)));

    float curlAmt = 0.035*(0.5 + 0.5*A) + 0.02*L + 0.015*H;
    disp += curl * curlAmt;

    // --- mouse gravity / anti-gravity (press to attract)
    vec2 d  = (pos - mndc);
    float r = max(length(d), 1e-3);
    vec2 dir= d / r;
    float mouseForce = 0.035 * (0.6*A + 0.4*M) * exp(-r*1.5);
    if(pc.mouse_pressed==1) disp -= dir * mouseForce; // attract
    else                    disp += dir * mouseForce; // repel

    // --- gentle pitch-bend space curvature
    float bend = pc.pitch_bend * 0.30;
    vec2  curved = pos + normalize(pos)*bend * sin(length(pos)*4.5 + t);

    // --- micro “glass” shimmer driven by highs (very small -> no cracks)
    vec2 shimmer = vec2(
        sin(pos.y*24.0 + tM*2.0),
        cos(pos.x*22.0 - tM*1.7)
    ) * (0.006 + 0.010*H);

    // combine: keep offsets small for safe coverage
    vec2 final_pos = mix(pos, curved, sat(abs(pc.pitch_bend))) + disp + shimmer;

    // output
    gl_Position = vec4(final_pos, 0.0, 1.0);
    fragUV      = uv;

    // energy & world pos (useful to frag)
    vertexEnergy = A*(0.6 + 0.4*M) + 0.05*crowd;
    worldPos     = vec3(final_pos, freq_norm);
}
