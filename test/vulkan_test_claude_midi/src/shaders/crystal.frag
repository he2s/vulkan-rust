#version 450

// ─────────────────────────────────────────────────────
// Push constants / IO (your setup)
// ─────────────────────────────────────────────────────
layout(push_constant) uniform PushConstants {
    float time; uint mouse_x; uint mouse_y; uint mouse_pressed;
    float note_velocity; float pitch_bend; float cc1; float cc74;
    uint  note_count; uint last_note; float osc_ch1; float osc_ch2;
    uint  render_w; uint render_h;
} pc;

layout(location=0) in vec2 frag_uv;          // 0..1
layout(location=1) in vec2 frag_screen_pos;  // optional
layout(location=0) out vec4 out_color;

// ─────────────────────────────────────────────────────
// Shadertoy compatibility shims + tunables
// ─────────────────────────────────────────────────────
#define PI 3.14159265358979323846
#define TAU 6.28318530717958647693

// iTime / iResolution / iMouse like on Shadertoy
#define iTime        (pc.time)
#define iResolution  vec3(float(pc.render_w), float(pc.render_h), 1.0)
vec4 iMouse = vec4(float(pc.mouse_x),
float(pc.render_h) - float(pc.mouse_y),
float(pc.mouse_pressed), 0.0);

// The original shader referenced these from a common tab.
// You can swap them with your own values at the top here:
const float TIME_OFFSET   = 0.0;
const float LOOP_DURATION = 6.0; // seconds per move loop

// Reduced AA for better performance - use 1 for high performance, 2 for balanced
#ifndef AA
#define AA 1
#endif

// Dummy iFrame used by their normal estimator macro trick
const int iFrame = 0;

// Audio proxy (used if you want a touch of reactivity)
float audioLevel() {
    return clamp((pc.note_velocity + pc.osc_ch1 + pc.osc_ch2) * 0.5, 0.0, 1.0);
}

// ─────────────────────────────────────────────────────
// Optimized quaternion utilities with reduced precision
// ─────────────────────────────────────────────────────
vec4 qmul(vec4 a, vec4 b) {
    return vec4(
    a.w*b.xyz + b.w*a.xyz + cross(a.xyz, b.xyz),
    a.w*b.w - dot(a.xyz, b.xyz)
    );
}
vec4 q_conj(vec4 q){ return vec4(-q.xyz, q.w); }

vec4 rotate_angle_axis(float angle, vec3 axis) {
    float h = 0.5*angle;
    float s = sin(h);
    return vec4(axis*s, cos(h)); // Removed normalize for performance
}
vec3 rotate_vector(vec3 v, vec4 q) {
    vec4 t = qmul(qmul(q, vec4(v, 0.0)), q_conj(q));
    return t.xyz;
}
vec4 q_norm(vec4 q){
    float l = inversesqrt(max(dot(q,q), 1e-8));
    return q * l;
}
// Simplified slerp with early exit for similar quaternions
vec4 q_slerp(vec4 a, vec4 b, float t) {
    a = q_norm(a); b = q_norm(b);
    float cosom = dot(a, b);
    if (cosom < 0.0) { b = -b; cosom = -cosom; }

    // Use linear interpolation for similar quaternions (performance boost)
    if (cosom > 0.95) {
        return q_norm(mix(a, b, t));
    }

    float omega = acos(cosom);
    float sinom = sin(omega);
    float k0 = sin((1.0 - t) * omega) / sinom;
    float k1 = sin(t * omega) / sinom;
    return q_norm(a*k0 + b*k1);
}

// ─────────────────────────────────────────────────────
// Move program - precomputed for better performance
// ─────────────────────────────────────────────────────
const int MOVE_COUNT = 6;
const vec4 moves[MOVE_COUNT] = vec4[](
vec4( 1.0, 0.0, 0.0,  1.0),
vec4( 0.0, 1.0, 0.0, -1.0),
vec4( 0.0, 0.0, 1.0,  1.0),
vec4(-1.0, 0.0, 0.0,  1.0),
vec4( 0.0, 1.0, 0.0,  1.0),
vec4( 0.0, 0.0,-1.0,  1.0)
);

// Precomputed normalized axes for moves
const vec3 move_axes[MOVE_COUNT] = vec3[](
vec3( 1.0, 0.0, 0.0),
vec3( 0.0, 1.0, 0.0),
vec3( 0.0, 0.0, 1.0),
vec3(-1.0, 0.0, 0.0),
vec3( 0.0, 1.0, 0.0),
vec3( 0.0, 0.0,-1.0)
);

// ─────────────────────────────────────────────────────
// Optimized utilities with reduced precision
// ─────────────────────────────────────────────────────

// Helpers - use mediump where possible for RTX 2070
void pR(inout vec2 p, float a) {
    vec2 cs = vec2(cos(a), sin(a));
    p = vec2(dot(p, vec2(cs.x, -cs.y)), dot(p, cs.yx));
}
float vmin(vec3 v) { return min(min(v.x, v.y), v.z); }
float vmax(vec3 v) { return max(max(v.x, v.y), v.z); }
float fBox(vec3 p, vec3 b) {
    vec3 d = abs(p) - b;
    return length(max(d, vec3(0))) + vmax(min(d, vec3(0)));
}
// Faster smooth min/max
float smin(float a, float b, float k){
    float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}
float smax(float a, float b, float k) { return -smin(-a, -b, k); }
float range(float vmin_, float vmax_, float value) {
    return clamp((value - vmin_) / (vmax_ - vmin_), 0., 1.);
}
// Optimized functions
float almostIdentity(float x) { return x*x*(2.0-x); }
float circularOut(float t) { return sqrt(max((2.0 - t) * t, 0.0)); }
// Cached palette calculation
vec3 spectrum(float n) {
    float t = n + 0.1 + 0.5;
    return vec3(0.5) + vec3(0.5) * cos(TAU * (t + vec3(0.0, 0.33, 0.67)));
}
vec3 erot(vec3 p, vec3 ax, float ro) {
    float cr = cos(ro), sr = sin(ro);
    return mix(dot(ax,p)*ax, p, cr) + sr*cross(ax,p);
}

// Animation globals
bool lightingPass;
float time; // local (loop-normalized) time [0..1] with motion blur variation

void applyMomentum(inout vec4 q, float tNow, int i, vec4 move) {
    float turns = move.w;
    vec3 axis = move_axes[i]; // Use precomputed axis
    float duration = abs(turns);
    float rotation = PI*0.375 * turns; // Precompute PI/2 * 0.75
    float start = float(i + 1);
    float t = tNow * float(MOVE_COUNT);
    float ramp = range(start, start + duration, t);
    float angle = circularOut(ramp) * rotation;
    vec4 q2 = rotate_angle_axis(angle, axis);
    q = qmul(q, q2);
}
void applyMove(inout vec3 p, int i, vec4 move) {
    float turns = move.w;
    vec3 axis = move_axes[i]; // Use precomputed axis
    float rotation = PI*0.5 * turns;
    float start = float(i);
    float t = time * float(MOVE_COUNT);
    float ramp = range(start, start + 1., t);
    ramp = pow(almostIdentity(ramp), 2.5);
    float angle = ramp * rotation;
    bool animSide = vmax(p * -axis) > 0.;
    if (animSide) angle = 0.;
    p = erot(p, axis, angle);
}

// Cache momentum calculations
vec4 momentum(float tNow) {
    vec4 q = vec4(0,0,0,1);
    for (int i = MOVE_COUNT-1; i >= 0; --i) applyMomentum(q, tNow, i, moves[i]);
    return q;
}
vec4 momentumLoop(float tNow) {
    vec4 q = momentum(3.0);
    q = q_conj(q);
    q = q_slerp(vec4(0,0,0,1), q, tNow);
    q = qmul(momentum(tNow + 1.), q);
    q = qmul(momentum(tNow), q);
    return q_norm(q);
}

// Optimized modelling with early exits
vec4 mapBox(vec3 p) {
    // shuffle blocks - optimized rotations
    pR(p.xy, step(0., -p.z) * -PI*0.5);
    pR(p.xz, step(0.,  p.y) * PI);
    pR(p.yz, step(0., -p.x) * PI*1.5);

    // face colors - optimized calculation
    vec3 face = step(vec3(vmax(abs(p))), abs(p)) * sign(p);
    float faceIndex = max(dot(face, vec3(0,1,2)), dot(face, -vec3(3,4,5)));
    vec3 col = spectrum(faceIndex * 0.16666667); // 1/6 precomputed

    // offset sphere shell
    const float thick = 0.033;
    float d = length(p + vec3(.1,.02,.05)) - 0.4;
    d = max(d, -d - thick);

    // grooves - simplified calculation
    vec3 ap = abs(p);
    vec3 plane = cross(abs(face), vec3(0.57735027)); // normalize(vec3(1)) precomputed
    float groove = max(-dot(ap.yzx, plane), dot(ap.zxy, plane));
    d = smax(d, -abs(groove), 0.01);

    const float gap = 0.005;
    const float r = 0.05;

    // block edge
    float cut = -fBox(abs(p) - (1.05 + gap), vec3(1.)) + r; // 1+r precomputed
    d = smax(d, -cut, thick * 0.5);

    // adjacent block edge bounding
    float opp = vmin(abs(p)) + gap;
    opp = max(opp, length(p) - 1.);
    if (opp < d) {
        return vec4(opp, vec3(-1));
    }

    return vec4(d, col * 0.4);
}

vec4 map(vec3 p) {
    // Mouse interaction - only if mouse is active
    if (iMouse.z > 0.5) {
        pR(p.yz, (iMouse.y / iResolution.y * -2. + 1.) * 2.);
        pR(p.xz, (iMouse.x / iResolution.x * -2. + 1.) * 4.);
    }

    // Base spin with pitch bend
    pR(p.xz, iTime * TAU + pc.pitch_bend * 0.25);

    vec4 q = momentumLoop(time);
    p = rotate_vector(p, q);

    // Unrolled loop for better performance
    applyMove(p, 5, moves[5]);
    applyMove(p, 4, moves[4]);
    applyMove(p, 3, moves[3]);
    applyMove(p, 2, moves[2]);
    applyMove(p, 1, moves[1]);
    applyMove(p, 0, moves[0]);

    return mapBox(p);
}

// Camera & shading helpers
mat3 calcLookAtMatrix( in vec3 ro, in vec3 ta, in float roll )
{
    vec3 ww = normalize( ta - ro );
    vec3 uu = normalize( cross(ww,vec3(sin(roll),cos(roll),0.0) ) );
    vec3 vv = normalize( cross(uu,ww));
    return mat3( uu, vv, ww );
}

// Optimized normal calculation with fewer samples
vec3 calcNormal(vec3 p)
{
    const vec2 h = vec2(0.001, 0.0);
    return normalize( vec3(map(p+h.xyy).x - map(p-h.xyy).x,
    map(p+h.yxy).x - map(p-h.yxy).x,
    map(p+h.yyx).x - map(p-h.yyx).x ) );
}

// origin sphere intersection
vec2 iSphere( in vec3 ro, in vec3 rd, float r )
{
    vec3 oc = ro;
    float b = dot( oc, rd );
    float c = dot( oc, oc ) - r*r;
    float h = b*b - c;
    if( h<0.0 ) return vec2(-1.0);
    h = sqrt(h);
    return vec2(-b-h, -b+h );
}

// Simplified shadow calculation - fewer samples
float softshadow( in vec3 ro, in vec3 rd, in float mint, in float tmax )
{
    float res = 1.0;
    vec2 bound = iSphere(ro, rd, 0.55);
    tmax = min(tmax, bound.y);

    float t = mint;
    for( int i=0; i<50; i++ ) // Reduced from 100 to 50
    {
        vec4 hit = map( ro + rd*t );
        float h = hit.x;
        if (hit.y > 0.) {
            res = min( res, 10.0*h/t );
        }
        t += h;
        if( res<0.0001 || t>tmax ) break;
    }
    return clamp( res, 0.0, 1.0 );
}

vec3 render(vec2 p) {
    vec3 col = vec3(0.02, 0.01, 0.025);

    vec3 camPos = vec3(0,0,2.);
    mat3 camMat = calcLookAtMatrix( camPos, vec3(0,0,-1), 0.);
    vec3 rd = normalize( camMat * vec3(p.xy, 2.8) );
    vec3 pos = camPos;

    vec2 bound = iSphere(pos, rd, 0.55);
    if (bound.x < 0.) return col;

    lightingPass = false;
    float rayLength = bound.x;
    float dist = 0.;
    bool background = true;
    vec4 res;

    // Reduced marching steps from 200 to 120
    for (int i = 0; i < 120; i++) {
        rayLength += dist;
        pos = camPos + rd * rayLength;
        res = map(pos);
        dist = res.x;

        if (abs(dist) < 0.001) { background = false; break; }
        if (rayLength > bound.y) break;
    }

    lightingPass = true;

    if (!background) {
        col = res.yzw;
        vec3 nor = calcNormal(pos);

        // Precomputed light directions
        const vec3 lig = vec3(-0.33, 0.3, 0.25);
        const vec3 lba = vec3(0.4472136, -0.8944272, -0.4472136); // normalized
        vec3 hal = normalize( lig - rd );

        float amb = sqrt(clamp( 0.5+0.5*nor.y, 0.0, 1.0 ));
        float dif = clamp( dot( nor, lig ), 0.0, 1.0 );
        float bac = clamp( dot( nor, lba ), 0.0, 1.0 )*clamp( 1.0-pos.y,0.0,1.0);
        float fre = pow( clamp(1.0+dot(nor,rd),0.0,1.0), 2.0 );

        // Conditional shadow calculation
        if (dif > 0.001) dif *= softshadow( pos, lig, 0.001, 0.9 );

        float spe = pow( clamp( dot( nor, hal ), 0.0, 1.0 ),16.0)* dif *
        (0.04 + 0.96*pow( clamp(1.0+dot(hal,rd),0.0,1.0), 5.0 ));

        // Simplified lighting calculation
        vec3 lin = 2.8*dif*vec3(1.3,1.0,0.7) +
        0.55*amb*vec3(0.4,0.6,1.15) +
        1.55*bac*vec3(0.5,0.0,0.25) +
        0.25*fre*vec3(1.0);

        col = col*lin + 5.0*spe*vec3(1.1,0.9,0.7);
    }

    return col;
}

float vmul(vec2 v) { return v.x * v.y; }

// ─────────────────────────────────────────────────────
// Optimized mainImage with reduced AA and motion blur
// ─────────────────────────────────────────────────────
void mainImage(out vec4 fragColor, in vec2 fragCoord) {

    float mTime = (iTime + TIME_OFFSET) / LOOP_DURATION;
    time = mTime;

    vec2 o = vec2(0.0);
    vec3 col = vec3(0.0);

    // Optimized AA + motion blur
    #if AA > 1
    for( int m=0; m<AA; m++ )
    for( int n=0; n<AA; n++ )
    {
        o = vec2(float(m),float(n)) / float(AA) - 0.5;
        float d = 0.5*vmul(sin(fragCoord.xy * vec2(0.021, 0.019))); // Optimized noise
        time = mTime - 0.00416667*(float(m*AA+n)+d)/float(AA*AA-1); // 1.0/24.0 precomputed
        #endif

        time = mod(time, 1.0);
        vec2 p = (-iResolution.xy + 2.0 * (fragCoord + o)) / iResolution.y;
        col += render(p);

        #if AA > 1
    }
    col *= 1.0/float(AA*AA); // Use multiplication instead of division
    #endif

    // Optimized gamma correction
    col = pow( col, vec3(0.4545) );

    fragColor = vec4(col, 1.0);
}

// ─────────────────────────────────────────────────────
// Pipeline main
// ─────────────────────────────────────────────────────
void main() {
    vec2 fragCoord = frag_uv * iResolution.xy;

    vec4 col;
    mainImage(col, fragCoord);

    // Optimized brightness & gamma
    float bright = mix(0.85, 1.55, pc.cc74);
    float gammaC = 1.0 + 0.7 * pc.cc1;
    col.rgb = pow(col.rgb * bright, vec3(gammaC));

    // MIDI note tinting - optimized trig calculations
    if(pc.note_count > 0u) {
        float noteTint = float(pc.last_note) * 0.007874016; // 1/127 precomputed
        float phase = noteTint * PI;
        vec3 tintColor = vec3(
        sin(phase),
        sin(phase + 1.047197551), // PI/3 precomputed
        sin(phase + 2.094395102)  // 2*PI/3 precomputed
        ) * (0.2 * pc.note_velocity);
        col.rgb += tintColor;
    }

    out_color = vec4(clamp(col.rgb, 0.0, 3.0), 1.0);
}