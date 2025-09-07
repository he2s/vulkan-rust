#version 450

// ---------- Push constants & varyings (match your scaffold) ----------
layout(push_constant) uniform PushConstants {
    float time;
    uint  mouse_x;
    uint  mouse_y;
    uint  mouse_pressed;
    float note_velocity;
    float pitch_bend;
    float cc1;    // mid band / mod wheel
    float cc74;   // high band / cutoff
    uint  note_count;
    uint  last_note;
    uint  render_w;
    uint  render_h;
} pc;

layout(location = 0) in vec2 fragUV;
layout(location = 1) in float vertexEnergy;
layout(location = 2) in vec3 worldPos;

layout(location = 0) out vec4 outColor;

// ---------- Constants & helpers ----------
const float PI = 3.14159265359;

// tiny hash/noise helpers (deterministic, fast)
float hash11(float n){ return fract(sin(n)*43758.5453123); }
float hash21(vec2 p){ return fract(sin(dot(p, vec2(41.0, 289.0))) * 12543.90321); }
float n1(float x){ float i=floor(x), f=fract(x); float a=hash11(i), b=hash11(i+1.0); return mix(a,b,smoothstep(0.0,1.0,f)); }
float n2(vec2 p){
    vec2 i=floor(p), f=fract(p);
    float a=hash21(i);
    float b=hash21(i+vec2(1,0));
    float c=hash21(i+vec2(0,1));
    float d=hash21(i+vec2(1,1));
    vec2 u = f*f*(3.0-2.0*f);
    return mix(mix(a,b,u.x), mix(c,d,u.x), u.y);
}

void pR(inout vec2 p, float a) {
    p = cos(a)*p + sin(a)*vec2(p.y, -p.x);
}

float smax(float a, float b, float r) {
    vec2 u = max(vec2(r + a, r + b), vec2(0.0));
    return min(-r, max(a, b)) + length(u);
}

// IQ palette (kept for internal weighting even though final is B/W)
vec3 pal(float t, vec3 a, vec3 b, vec3 c, vec3 d){ return a + b*cos(6.28318*(c*t + d)); }
vec3 spectrum(float n){ return pal(n, vec3(0.5), vec3(0.5), vec3(1.0), vec3(0.0, 0.33, 0.67)); }

// ---------- Model space utilities ----------
vec4 inverseStereographic(vec3 p, out float k) {
    k = 2.0 / (1.0 + dot(p, p));
    return vec4(k*p, k-1.0);
}

float fTorus(vec4 p4) {
    float d1 = length(p4.xy) / length(p4.zw) - 1.0;
    float d2 = length(p4.zw) / length(p4.xy) - 1.0;
    float d  = (d1 < 0.0) ? -d1 : d2;
    return d / PI;
}

float fixDistance(float d, float k) {
    float sn = sign(d);
    d = abs(d);
    d = d / k * 1.82;
    d += 1.0;
    d = pow(d, 0.5);
    d -= 1.0;
    d *= 5.0/3.0;
    return d * sn;
}

// We’ll set this each frame from pc.time
float uTime;

// Rotation helpers
mat3 rotateY(float a){
    float c=cos(a), s=sin(a);
    return mat3( c,0.0, s,
    0.0,1.0,0.0,
    -s,0.0, c );
}
mat3 rotateX(float a){
    float c=cos(a), s=sin(a);
    return mat3( 1.0,0.0,0.0,
    0.0, c,-s,
    0.0, s, c );
}

// Distance field
float map(vec3 p) {
    float k;
    vec4 p4 = inverseStereographic(p, k);

    // original rotation but time-driven by uTime
    pR(p4.zy, uTime * -PI / 2.0);
    pR(p4.xw, uTime * -PI / 2.0);

    // thick walled Clifford torus intersected with a sphere
    float d = fTorus(p4);
    d = abs(d) - 0.2;
    d = fixDistance(d, k);
    d = smax(d, length(p) - 1.85, 0.2);
    return d;
}

// Camera
mat3 calcLookAtMatrix(vec3 ro, vec3 ta, vec3 up) {
    vec3 ww = normalize(ta - ro);
    vec3 uu = normalize(cross(ww, up));
    vec3 vv = cross(uu, ww);
    return mat3(uu, vv, ww);
}

void main() {
    // --- Resolution & coordinates (prefer push constants if set)
    vec2 iResolution = (pc.render_w > 0u && pc.render_h > 0u)
    ? vec2(pc.render_w, pc.render_h)
    : vec2(800.0, 600.0);

    vec2 fragCoord = fragUV * iResolution;
    vec2 uv = fragCoord / iResolution;          // 0..1
    vec2 centered = (fragCoord - 0.5 * iResolution) / iResolution.y; // aspect-corrected

    // --- Musical hooks
    float A = clamp(pc.note_velocity, 0.0, 1.0);
    float M = clamp(pc.cc1, 0.0, 1.0);
    float H = clamp(pc.cc74, 0.0, 1.0);

    // gentle base speed, reacts to loudness & mids
    float speed = mix(0.45, 0.85, 0.5*A + 0.5*M);
    uTime = mod(pc.time * speed / 2.0, 1.0);

    // ---------- GLITCH DOMAIN WARP (screen-space) ----------
    // line tear amount (bigger with energy/highs)
    float lineGlitchAmt = mix(0.2, 0.008, 0.4*A + 0.6*H);
    // blocky jump probability (also reacts to mids)
    float blockJitter = step(0.97 - 0.5*M, hash21(vec2(floor(pc.time*7.0), 19.3)));

    // per-scanline horizontal jitter
    float y = uv.y * iResolution.y;
    float lineNoise = (n1(pc.time*3.0 + y*0.031) - 0.5) * 2.0;
    float tear = lineNoise * lineGlitchAmt;

    // occasional chunky offset blocks
    vec2 blockUV = floor(uv * vec2(24.0, 14.0)) / vec2(24.0, 14.0);
    float blockHash = hash21(blockUV + floor(pc.time*2.0));
    float blockKick = (blockHash > 0.85 ? 1.0 : 0.0) * blockJitter;
    float blockShift = (hash21(blockUV+2.31) - 0.5) * 0.05 * blockKick;

    // mild vertical jitter / rolling
    float roll = (n1(pc.time*0.3) - 0.5) * 0.02 * (0.2 + 0.8*M);

    // final warped coords for camera ray construction
    vec2 p = centered;
    p.x += tear + blockShift;
    p.y += roll;

    // ---------- Camera setup (mostly original) ----------
    vec3 camPos = vec3(1.8, 5.5, -5.5) * 1.75;
    vec3 camTar = vec3(0.0, 0.0, 0.0);
    vec3 camUp  = vec3(-1.0, 0.0, -1.5);
    mat3 camMat = calcLookAtMatrix(camPos, camTar, camUp);

    float focalLength = 5.0;
    vec3 rayDirection = normalize(camMat * vec3(p, focalLength));
    vec3 rayPosition  = camPos;
    float rayLength   = 0.0;

    float dist = 0.0;
    vec3 color = vec3(0.0);
    vec3 c;

    // March params (kept)
    const float ITER = 32.0;
    const float FUDGE_FACTOR = 0.8;
    const float INTERSECTION_PRECISION = 0.001;
    const float MAX_DIST = 20.0;

    // internal tint vector rotates with CCs/pitch (only used as luminance weight later)
    vec3 rotatedVectorY = rotateY(pc.cc1 * 10.0) * vec3(0.6, 0.25, 0.7);
    vec3 rotatedVector  = rotateY(pc.pitch_bend * 10.0) * rotatedVectorY;

    for (float i = 0.0; i < ITER; i++) {
        rayLength += max(INTERSECTION_PRECISION, abs(dist) * FUDGE_FACTOR);
        rayPosition = camPos + rayDirection * rayLength;
        dist = map(rayPosition);

        float proximity = max(0.0, 0.01 - abs(dist)) * 0.5;
        c  = vec3(proximity);
        c *= mix(vec3(1.4, 2.1, 1.7), vec3(1.2, 2.2, 1.8), H);

        c += rotatedVector * FUDGE_FACTOR / 160.0;
        c *= smoothstep(20.0, 7.0, length(rayPosition));

        float rl = smoothstep(MAX_DIST, 0.1, rayLength);
        c *= rl;

        c *= spectrum(rl * 6.0 - 0.6);

        color += c;

        if (rayLength > MAX_DIST) break;
    }

    // Tonemapping & gamma (as original-ish)
    color = pow(color, vec3(1.0 / 1.8)) * 2.0;
    color = pow(color, vec3(2.0)) * 3.0;
    color = pow(color, vec3(1.0 / 2.2));

    // Optional: add tiny energy lift from vertexEnergy
    color *= (1.0 + 0.15 * clamp(vertexEnergy, 0.0, 1.0));

    // ---------- BLACK & WHITE CONVERSION ----------
    float g = dot(color, vec3(0.2126, 0.7152, 0.0722)); // luminance

    // ---------- POST FX: glitchy B/W treatment ----------
    // film grain (audio-reactive)
    float grain = n2(uv * iResolution.xy + vec2(pc.time*120.0, -pc.time*77.0)) - 0.5;
    g += grain * mix(0.02, 0.12, 0.4*A + 0.6*H);

    // scanlines & vertical subline flicker
    float scan = 0.8 + 0.2 * sin( (uv.y*iResolution.y) * 3.14159 + pc.time*6.0 );
    float subScan = 1.0 + 0.06 * sin((uv.x*iResolution.x)*0.5 + pc.time*40.0);
    g *= scan * subScan;

    // posterize / bit-crush depending on energy
    float levels = mix(32.0, 6.0, clamp(0.2 + 0.8*(0.5*A + 0.5*M), 0.0, 1.0));
    g = floor(g * levels) / levels;

    // white spark hits (salt noise)
    float spark = step(0.996, hash21(uv * iResolution.xy + floor(pc.time*90.0)));
    g = mix(g, 1.0, spark * mix(0.0, 0.5, H));

    // vignetting for contrast
    float vign = smoothstep(1.25, 0.25, length(centered + vec2(tear, roll)*0.5));
    g *= vign;

    // occasional hard tear (brief horizontal clamp band)
    float tearGate = 0.5 * step(0.985, hash11(floor(pc.time*5.0)+3.7));
    if(tearGate > 0.5){
        float bandY = fract(pc.time*0.7) * 0.8 + 0.1;
        float band = smoothstep(bandY-0.01, bandY, uv.y) * (1.0 - smoothstep(bandY, bandY+0.01, uv.y));
        g = mix(g, g*0.35, band);
    }

    outColor = vec4(vec3(g), 1.0);
}
