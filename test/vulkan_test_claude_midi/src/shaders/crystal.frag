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

// If you want supersampling, define AA to 2 or 3 before compile.
#ifndef AA
// #define AA 2
#endif

// Dummy iFrame used by their normal estimator macro trick
const int iFrame = 0;

// Audio proxy (used if you want a touch of reactivity)
float audioLevel() {
    return clamp((pc.note_velocity + pc.osc_ch1 + pc.osc_ch2) * 0.5, 0.0, 1.0);
}

// ─────────────────────────────────────────────────────
// Minimal quaternion utilities (to replace common tab)
// Represent quats as vec4(q.xyz, q.w)
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
    return vec4(normalize(axis)*s, cos(h));
}
vec3 rotate_vector(vec3 v, vec4 q) {
    vec4 t = qmul(qmul(q, vec4(v, 0.0)), q_conj(q));
    return t.xyz;
}
vec4 q_norm(vec4 q){
    float l = inversesqrt(max(dot(q,q), 1e-8));
    return q * l;
}
vec4 q_slerp(vec4 a, vec4 b, float t) {
    a = q_norm(a); b = q_norm(b);
    float cosom = dot(a, b);
    // Shortest path
    if (cosom < 0.0) { b = -b; cosom = -cosom; }
    float k0, k1;
    if (1.0 - cosom > 1e-6) {
        float omega = acos(cosom);
        float sinom = sin(omega);
        k0 = sin((1.0 - t) * omega) / sinom;
        k1 = sin(t * omega) / sinom;
    } else {
        // Nearly linear
        k0 = 1.0 - t;
        k1 = t;
    }
    return q_norm(a*k0 + b*k1);
}

// ─────────────────────────────────────────────────────
// Move program (axis.xyz, turns in .w)
// Replace with your original move list as desired.
// MOVE_COUNT must match the length of moves[].
// ─────────────────────────────────────────────────────
const int MOVE_COUNT = 6;
const vec4 moves[MOVE_COUNT] = vec4[](
vec4( 1.0, 0.0, 0.0,  1.0),  // +X quarter
vec4( 0.0, 1.0, 0.0, -1.0),  // -Y quarter
vec4( 0.0, 0.0, 1.0,  1.0),  // +Z quarter
vec4(-1.0, 0.0, 0.0,  1.0),  // -X quarter
vec4( 0.0, 1.0, 0.0,  1.0),  // +Y quarter
vec4( 0.0, 0.0,-1.0,  1.0)   // -Z quarter
);

// ─────────────────────────────────────────────────────
// Original utilities
// ─────────────────────────────────────────────────────

// Helpers
void pR(inout vec2 p, float a) { p = cos(a)*p + sin(a)*vec2(p.y, -p.x); }
float vmin(vec3 v) { return min(min(v.x, v.y), v.z); }
float vmax(vec3 v) { return max(max(v.x, v.y), v.z); }
float fBox(vec3 p, vec3 b) {
    vec3 d = abs(p) - b;
    return length(max(d, vec3(0))) + vmax(min(d, vec3(0)));
}
float smin(float a, float b, float k){
    float f = clamp(0.5 + 0.5 * ((a - b) / k), 0., 1.);
    return (1. - f) * a + f  * b - f * (1. - f) * k;
}
float smax(float a, float b, float k) { return -smin(-a, -b, k); }
float range(float vmin_, float vmax_, float value) {
    return clamp((value - vmin_) / (vmax_ - vmin_), 0., 1.);
}
float almostIdentity(float x) { return x*x*(2.0-x); }
float circularOut(float t) { return sqrt(max((2.0 - t) * t, 0.0)); }
vec3 pal( in float t, in vec3 a, in vec3 b, in vec3 c, in vec3 d ) {
    return a + b*cos( TAU*(c*t+d) );
}
vec3 spectrum(float n) {
    return pal( n, vec3(0.5), vec3(0.5), vec3(1.0), vec3(0.0,0.33,0.67) );
}
vec3 erot(vec3 p, vec3 ax, float ro) {
    return mix(dot(ax,p)*ax, p, cos(ro)) + sin(ro)*cross(ax,p);
}

// Animation globals
bool lightingPass;
float time; // local (loop-normalized) time [0..1] with motion blur variation

void applyMomentum(inout vec4 q, float tNow, int i, vec4 move) {
    float turns = move.w;
    vec3 axis = normalize(move.xyz);
    float duration = abs(turns);
    float rotation = PI/2. * turns * .75;
    float start = float(i + 1);
    float t = tNow * float(MOVE_COUNT);
    float ramp = range(start, start + duration, t);
    float angle = circularOut(ramp) * rotation;
    vec4 q2 = rotate_angle_axis(angle, axis);
    q = qmul(q, q2);
}
void applyMove(inout vec3 p, int i, vec4 move) {
    float turns = move.w;
    vec3 axis = normalize(move.xyz);
    float rotation = PI/2. * turns;
    float start = float(i);
    float t = time * float(MOVE_COUNT);
    float ramp = range(start, start + 1., t);
    ramp = pow(almostIdentity(ramp), 2.5);
    float angle = ramp * rotation;
    bool animSide = vmax(p * -axis) > 0.;
    if (animSide) angle = 0.;
    p = erot(p, axis, angle);
}
vec4 momentum(float tNow) {
    vec4 q = vec4(0,0,0,1);
    for (int i = MOVE_COUNT-1; i >= 0; --i) applyMomentum(q, tNow, i, moves[i]);
    return q;
}
vec4 momentumLoop(float tNow) {
    vec4 q;
    // end state
    q = momentum(3.);
    q = q_conj(q);
    q = q_slerp(vec4(0,0,0,1), q, tNow);
    // next loop
    q = qmul(momentum(tNow + 1.), q);
    // current loop
    q = qmul(momentum(tNow), q);
    return q_norm(q);
}

// Modelling
vec4 mapBox(vec3 p) {
    // shuffle blocks
    pR(p.xy, step(0., -p.z) * PI / -2.);
    pR(p.xz, step(0.,  p.y) * PI);
    pR(p.yz, step(0., -p.x) * PI * 1.5);

    // face colors
    vec3 face = step(vec3(vmax(abs(p))), abs(p)) * sign(p);
    float faceIndex = max(vmax(face * vec3(0,1,2)), vmax(face * -vec3(3,4,5)));
    vec3 col = spectrum(faceIndex / 6. + .1 + .5);

    // offset sphere shell
    float thick = .033;
    float d = length(p + vec3(.1,.02,.05)) - .4;
    d = max(d, -d - thick);

    // grooves
    vec3 ap = abs(p);
    float l = sqrt(sqrt(1.) / 3.);
    vec3 plane = cross(abs(face), normalize(vec3(1)));
    float groove = max(-dot(ap.yzx, plane), dot(ap.zxy, plane));
    d = smax(d, -abs(groove), .01);

    float gap = .005;

    // block edge
    float r = .05;
    float cut = -fBox(abs(p) - (1. + r + gap), vec3(1.)) + r;
    d = smax(d, -cut, thick / 2.);

    // adjacent block edge bounding
    float opp = vmin(abs(p)) + gap;
    opp = max(opp, length(p) - 1.);
    if (opp < d) {
        return vec4(opp, vec3(-1));
    }

    return vec4(d, col * .4);
}

vec4 map(vec3 p) {
    if (iMouse.x > 0.) {
        pR(p.yz, ((iMouse.y / -iResolution.y) * 2. + 1.) * 2.);
        pR(p.xz, ((iMouse.x / -iResolution.x) * 2. + 1.) * 4.);
    }

    // Base spin (let pitch bend mod subtly)
    pR(p.xz, iTime * PI * 2. + pc.pitch_bend * 0.25);

    vec4 q = momentumLoop(time);
    p = rotate_vector(p, q);

    for (int i = MOVE_COUNT-1; i >= 0; --i) applyMove(p, i, moves[i]);

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

// https://iquilezles.org/articles/normalsSDF
vec3 calcNormal(vec3 p)
{
    const float h = 0.001;
    #define ZERO (min(iFrame,0)) // non-constant zero
    vec3 n = vec3(0.0);
    for( int i=ZERO; i<4; i++ )
    {
        vec3 e = 0.5773*(2.0*vec3((((i+3)>>1)&1),((i>>1)&1),(i&1))-1.0);
        n += e*map(p+e*h).x;
    }
    return normalize(n);
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

// https://www.shadertoy.com/view/lsKcDD
float softshadow( in vec3 ro, in vec3 rd, in float mint, in float tmax )
{
    float res = 1.0;
    // limit to bounding sphere
    vec2 bound = iSphere(ro, rd, .55);
    tmax = min(tmax, bound.y);

    float t = mint;
    for( int i=0; i<100; i++ )
    {
        vec4 hit = map( ro + rd*t );
        float h = hit.x;
        if (hit.y > 0.) { // don't shadow from bounding objects
            res = min( res, 10.0*h/t );
        }
        t += h;
        if( res<0.0001 || t>tmax ) break;
    }
    return clamp( res, 0.0, 1.0 );
}

vec3 render(vec2 p) {
    vec3 col = vec3(.02,.01,.025);

    vec3 camPos = vec3(0,0,2.);
    mat3 camMat = calcLookAtMatrix( camPos, vec3(0,0,-1), 0.);
    vec3 rd = normalize( camMat * vec3(p.xy, 2.8) );
    vec3 pos = camPos;

    vec2 bound = iSphere(pos, rd, .55);
    if (bound.x < 0.) return col;

    lightingPass = false;
    float rayLength = bound.x;
    float dist = 0.;
    bool background = true;
    vec4 res;

    for (int i = 0; i < 200; i++) {
        rayLength += dist;
        pos = camPos + rd * rayLength;
        res = map(pos);
        dist = res.x;

        if (abs(dist) < .001) { background = false; break; }
        if (rayLength > bound.y) break;
    }

    lightingPass = true;

    if (!background) {
        col = res.yzw;
        vec3 nor = calcNormal(pos);
        vec3 lig = normalize(vec3(-.33,.3,.25));
        vec3 lba = normalize( vec3(.5, -1., -.5) );
        vec3 hal = normalize( lig - rd );
        float amb = sqrt(clamp( 0.5+0.5*nor.y, 0.0, 1.0 ));
        float dif = clamp( dot( nor, lig ), 0.0, 1.0 );
        float bac = clamp( dot( nor, lba ), 0.0, 1.0 )*clamp( 1.0-pos.y,0.0,1.0);
        float fre = pow( clamp(1.0+dot(nor,rd),0.0,1.0), 2.0 );

        if (dif > .001) dif *= softshadow( pos, lig, 0.001, .9 );

        float occ = 1.0;
        float spe = pow( clamp( dot( nor, hal ), 0.0, 1.0 ),16.0)* dif *
        (0.04 + 0.96*pow( clamp(1.0+dot(hal,rd),0.0,1.0), 5.0 ));

        vec3 lin = vec3(0.0);
        lin += 2.80*dif*vec3(1.30,1.00,0.70);
        lin += 0.55*amb*vec3(0.40,0.60,1.15)*occ;
        lin += 1.55*bac*vec3(0.25,0.25,0.25)*occ*vec3(2,0,1);
        lin += 0.25*fre*vec3(1.00,1.00,1.00)*occ;

        col = col*lin;
        col += 5.00*spe*vec3(1.10,0.90,0.70);
    }

    return col;
}

float vmul(vec2 v) { return v.x * v.y; }

// ─────────────────────────────────────────────────────
// Shadertoy-style mainImage
// ─────────────────────────────────────────────────────
void mainImage(out vec4 fragColor, in vec2 fragCoord) {

    float mTime = (iTime + TIME_OFFSET) / LOOP_DURATION;
    time = mTime;

    vec2 o = vec2(0.0);
    vec3 col = vec3(0.0);

    // Optional AA + motion blur (shutter ~0.5)
    #ifdef AA
    for( int m=0; m<AA; m++ )
    for( int n=0; n<AA; n++ )
    {
        o = vec2(float(m),float(n)) / float(AA) - 0.5;
        float d = 0.5*vmul(sin(mod(fragCoord.xy * vec2(147,131), vec2(PI * 2.))));
        time = mTime - 0.1*(1.0/24.0)*(float(m*AA+n)+d)/float(AA*AA-1);
        #endif

        time = mod(time, 1.0);
        vec2 p = (-iResolution.xy + 2.0 * (fragCoord + o)) / iResolution.y;
        col += render(p);

        #ifdef AA
    }
    col /= float(AA*AA);
    #endif

    // Gamma from original
    col = pow( col, vec3(0.4545) );

    fragColor = vec4(col, 1.0);
}

// ─────────────────────────────────────────────────────
// Pipeline main (your setup’s post controls preserved)
// ─────────────────────────────────────────────────────
void main() {
    vec2 fragCoord = frag_uv * iResolution.xy;

    vec4 col;
    mainImage(col, fragCoord);

    // Brightness & gamma via CCs (your setup)
    float bright = mix(0.85, 1.55, clamp(pc.cc74, 0.0, 1.0));
    float gammaC = 1.0 + 0.7 * clamp(pc.cc1, 0.0, 1.0);
    col.rgb = pow(max(col.rgb * bright, 0.0), vec3(gammaC));

    // MIDI note tinting (your setup)
    if(pc.note_count > 0u) {
        float noteTint = float(pc.last_note) / 127.0;
        vec3 tintColor = vec3(
        sin(noteTint * PI) * 0.2,
        sin(noteTint * PI + PI/3.0) * 0.2,
        sin(noteTint * PI + 2.0*PI/3.0) * 0.2
        );
        col.rgb += tintColor * pc.note_velocity;
    }

    out_color = vec4(clamp(col.rgb, 0.0, 3.0), 1.0);
}
