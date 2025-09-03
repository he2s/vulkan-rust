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

// ---------- Constants & helpers (from the Shadertoy, adjusted) ----------
const float PI = 3.14159265359;

void pR(inout vec2 p, float a) {
    p = cos(a)*p + sin(a)*vec2(p.y, -p.x);
}

float smax(float a, float b, float r) {
    vec2 u = max(vec2(r + a, r + b), vec2(0.0));
    return min(-r, max(a, b)) + length(u);
}

// IQ palette
vec3 pal( float t, vec3 a, vec3 b, vec3 c, vec3 d ) {
    return a + b*cos( 6.28318*(c*t + d) );
}

vec3 spectrum(float n) {
    return pal(n,
               vec3(0.5),
               vec3(0.5),
               vec3(1.0),
               vec3(0.0, 0.33, 0.67));
}

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

// Rotation around Y-axis
mat3 rotateY(float angle) {
    float c = cos(angle);
    float s = sin(angle);
    return mat3(
    c, 0.0, s,
    0.0, 1.0, 0.0,
    -s, 0.0, c
    );
}

// Rotation around Y-axis
mat3 rotateX(float angle) {
    float c = cos(angle);
    float s = sin(angle);
    return mat3(
    c, 0.0, s,
    1.0, 0.0, 0.0,
    -s, 0.0, c
    );
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
    // --- Resolution & coordinates (adapt from your pipeline)
    // If you pass real size via push constants later, swap this:
    const vec2 iResolution = vec2(800.0, 600.0);
    vec2 fragCoord = fragUV * iResolution;

    // --- Musical hooks
    // slow the animation; breathe with loudness & mids
    float A = clamp(pc.note_velocity, 0.0, 1.0);
    float M = clamp(pc.cc1, 0.0, 1.0);
    float speed = mix(0.45, 0.85, 0.5*A + 0.5*M); // gentle reactive speed
    uTime = mod(pc.time * speed / 2.0, 1.0);

    // --- Camera setup (mostly original)
    vec3 camPos = vec3(1.8, 5.5, -5.5) * 1.75;
    vec3 camTar = vec3(0.0, 0.0, 0.0);
    vec3 camUp  = vec3(-1.0, 0.0, -1.5);
    mat3 camMat = calcLookAtMatrix(camPos, camTar, camUp);

    float focalLength = 5.0;
    // p like original: [-iResolution.xy + 2*fragCoord]/iResolution.y
    vec2 p = (-iResolution.xy + 2.0 * fragCoord) / iResolution.y;

    vec3 rayDirection = normalize(camMat * vec3(p, focalLength));
    vec3 rayPosition  = camPos;
    float rayLength   = 0.0;

    float dist = 0.0;
    vec3 color = vec3(0.0);
    vec3 c;

    // March params (kept as in the Shadertoy)
    const float ITER = 32.0;
    const float FUDGE_FACTOR = 0.8;
    const float INTERSECTION_PRECISION = 0.001;
    const float MAX_DIST = 20.0;


    vec3 rotatedVectorY = rotateY(pc.cc1 * 10.0) * vec3(0.6, 0.25, 0.7);
    vec3 rotatedVector = rotateY(pc.pitch_bend * 10.0) * rotatedVectorY;
    for (float i = 0.0; i < ITER; i++) {
        // step a little slower so we can accumulate glow
        rayLength += max(INTERSECTION_PRECISION, abs(dist) * FUDGE_FACTOR);
        rayPosition = camPos + rayDirection * rayLength;
        dist = map(rayPosition);

        // close-to-surface glow (blue-green tint) — modulate slightly with highs
        float proximity = max(0.0, 0.01 - abs(dist)) * 0.5;
        c  = vec3(proximity);
        c *= mix(vec3(1.4, 2.1, 1.7), vec3(1.2, 2.2, 1.8), clamp(pc.cc74, 0.0, 1.0));

        // base purple-ish glow each step
        c += rotatedVector * FUDGE_FACTOR / 160.0;
        c *= smoothstep(20.0, 7.0, length(rayPosition)); // distance fade from origin

        // fade further from camera
        float rl = smoothstep(MAX_DIST, 0.1, rayLength);
        c *= rl;

        // vary color with space progression
        c *= spectrum(rl * 6.0 - 0.6);

        color += c;

        if (rayLength > MAX_DIST) {
            break;
        }
    }

    // Tonemapping & gamma (as original, kept)
    color = pow(color, vec3(1.0 / 1.8)) * 2.0;
    color = pow(color, vec3(2.0)) * 3.0;
    color = pow(color, vec3(1.0 / 2.2));

    // Optional: add tiny energy lift from vertexEnergy
    color *= (1.0 + 0.15 * clamp(vertexEnergy, 0.0, 1.0));

    outColor = vec4(color, 1.0);

    // --- If you prefer **grayscale**, un-comment this line:
    // float g = dot(color, vec3(0.2126, 0.7152, 0.0722)); outColor = vec4(vec3(g), 1.0);
}
