#version 450

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
layout(location = 1) in float vertexEnergy;
layout(location = 2) in vec3 worldPos;

layout(location = 0) out vec4 outColor;

// Constants
#define PI 3.14159265359
#define PHI 1.618033988749895
#define TAU 6.283185307179586
#define MAX_TRACE_DISTANCE 30.0
#define INTERSECTION_PRECISION 0.001
#define NUM_OF_TRACE_STEPS 100
#define FUDGE_FACTOR 0.5
#define ENABLE_CHAMFER

// Audio-reactive chamfer amount
float getChamfer() {
    return 0.003 + pc.cc1 * 0.005;
}

// Plane with normal n at some distance from the origin
float fPlane(vec3 p, vec3 n, float distanceFromOrigin) {
    return dot(p, n) + distanceFromOrigin;
}

// Rotation
void pR(inout vec2 p, float a) {
    p = cos(a)*p + sin(a)*vec2(p.y, -p.x);
}

// Repeat around the origin by a fixed angle
float pModPolar(inout vec2 p, float repetitions) {
    float angle = 2.*PI/repetitions;
    float a = atan(p.y, p.x) + angle/2.;
    float r = length(p);
    float c = floor(a/angle);
    a = mod(a,angle) - angle/2.;
    p = vec2(cos(a), sin(a))*r;
    if (abs(c) >= (repetitions/2.)) c = abs(c);
    return c;
}

// Intersection with chamfer
float fOpIntersectionChamfer(float a, float b, float r) {
    float m = max(a, b);
    if (r <= 0.) return m;
    if (((-a < r) && (-b < r)) || (m < 0.)) {
        return max(m, (a + r + b)*sqrt(0.5));
    } else {
        return m;
    }
}

// Orient matrix
mat3 orientMatrix(vec3 A, vec3 B) {
    mat3 Fi = mat3(
    A,
    (B - dot(A, B) * A) / length(B - dot(A, B) * A),
    cross(B, A)
    );
    mat3 G = mat3(
    dot(A, B),              -length(cross(A, B)),   0,
    length(cross(A, B)),    dot(A, B),              0,
    0,                      0,                      1
    );
    return Fi * G * inverse(Fi);
}

// Audio-reactive color palette with inner glow
vec3 spectrum(float n) {
    // Base spectrum influenced by audio
    float energy = clamp(pc.note_velocity, 0.0, 1.0);
    float brightness = clamp(pc.cc74, 0.0, 1.0);

    // Shift hue based on note count
    n += float(pc.note_count % 12u) * 0.08;

    // More dramatic color shifts
    vec3 a = vec3(0.5 + energy * 0.2);
    vec3 b = vec3(0.5 + brightness * 0.3);
    vec3 c = vec3(1.0, 0.7, 0.4);
    vec3 d = vec3(0.0 + pc.time * 0.01, 0.33, 0.67);

    // Modify palette with OSC
    d.xy += vec2(pc.osc_ch1, pc.osc_ch2) * 0.3;

    vec3 color = a + b * cos(TAU * (c * n + d));

    // Energy affects saturation
    float sat = 0.5 + energy * 0.5;
    vec3 gray = vec3(dot(color, vec3(0.299, 0.587, 0.114)));
    color = mix(gray, color, sat);

    // Inner glow based on energy
    vec3 glowColor = vec3(0.3, 0.6, 1.0) * energy;
    color += glowColor * 0.3;

    // Brightness control
    color *= 0.7 + brightness * 0.5;

    return color;
}

// Model structure
struct Model {
    float dist;
    vec3 colour;
    float id;
};

float quadrant(float a, float b) {
    return ((sign(a) + sign(b) * 2.) + 3.) / 2.;
}

// Nearest icosahedron vertex
vec4 icosahedronVertex(vec3 p) {
    vec3 v1, v2, v3, result, plane;
    float id;
    v1 = vec3(
    sign(p.x) * PHI,
    sign(p.y) * 1.,
    0
    );
    v2 = vec3(
    sign(p.x) * 1.,
    0,
    sign(p.z) * PHI
    );
    v3 = vec3(
    0,
    sign(p.y) * PHI,
    sign(p.z) * 1.
    );
    plane = normalize(cross(
    mix(v1, v2, .5),
    cross(v1, v2)
    ));
    if (dot(p, plane) > 0.) {
        result = v1;
        id = quadrant(p.y, p.x);
    } else {
        result = v2;
        id = quadrant(p.x, p.z) + 4.;
    }
    plane = normalize(cross(
    mix(v3, result, .5),
    cross(v3, result)
    ));
    if (dot(p, plane) > 0.) {
        result = v3;
        id = quadrant(p.z, p.y) + 8.;
    }
    return vec4(normalize(result), id);
}

vec3 rand(vec3 seed){
    return fract(mod(seed, 1.) * 43758.5453);
}

vec3 jitterOffset(float seed) {
    return normalize(rand(vec3(seed, seed + .2, seed + .8)) - .5);
}

vec3 jitterVec(vec3 v, float seed, float magnitude) {
    return normalize(v + jitterOffset(seed) * magnitude);
}

float alias(float t, float resolution) {
    return floor(t * resolution) / resolution;
}

float fCrystalShard(vec3 p, float size) {
    float d;
    // Breathing effect on width
    float breathe = sin(pc.time * 3.0 + pc.note_velocity * TAU) * 0.02;
    float width = size * .04 + .07 + breathe;
    vec3 o = normalize(vec3(1,0,-.04));

    // Audio-reactive number of sides with smoother transition
    float sides = 5.0 + floor(pc.cc1 * 3.0);
    // Add twist based on height
    pR(p.xy, p.z * pc.pitch_bend * 0.3);
    pModPolar(p.xy, sides);

    float part1, part2;
    p.y = abs(p.y);
    part1 = fPlane(p, vec3(1,0,-.04), -width);

    pR(p.xy, TAU/sides);
    part2 = fPlane(p, vec3(1,0,-.04), -width);

    d = fOpIntersectionChamfer(part1, part2, getChamfer());

    return d;
}

float fCrystalCap(vec3 p, float id, float side) {
    float jitter = id + side * .1;
    vec3 o = normalize(vec3(1,0,.55));
    float angle = TAU / 3.;
    float d, part;

    // Audio affects jitter magnitude
    float jitterMag = 0.1 * (1.0 + pc.pitch_bend * 0.5);

    d = fPlane(p, jitterVec(o, jitter + .3, jitterMag), 0.);

    pR(p.xy, angle);
    part = fPlane(p, jitterVec(o, jitter + .5, jitterMag), 0.);
    d = fOpIntersectionChamfer(d, part, getChamfer());

    pR(p.xy, angle);
    part = fPlane(p, jitterVec(o, jitter + .9, jitterMag), 0.);
    d = fOpIntersectionChamfer(d, part, getChamfer());

    return d;
}

float fCrystal(vec3 p, float id, float focus) {
    // Audio-reactive sizing
    float energy = clamp(pc.note_velocity, 0.0, 1.0);
    float modulation = clamp(pc.cc1, 0.0, 1.0);

    float size = sin(pc.time * TAU * 4. + focus * 5. + id + energy * PI) * .5 + .5;
    float size2 = cos(pc.time * TAU * 2. + focus * 5. + id + modulation * PI) * .5 + .5;

    size = alias(size, 2.);
    size2 = alias(size2, 2.);

    float height = size2 * .1 + .35 + vertexEnergy * 0.1;
    float offset = .9;
    float d;

    float shard = fCrystalShard(p, size);

    p.z -= offset;
    float side = sign(p.z) * .5 + .5;
    p.z = abs(p.z);
    p.z -= height;

    float cap = fCrystalCap(p, id, side);
    d = fOpIntersectionChamfer(shard, cap, getChamfer());

    return d;
}

Model model(vec3 p) {
    float d = 1000.;
    vec3 col = vec3(0);
    vec3 dir = normalize(vec3(0, -PHI, -1));

    vec4 iv = icosahedronVertex(p);
    vec3 v = iv.xyz;
    float id = iv[3] / 12.;

    p *= orientMatrix(v, vec3(0,0,1));
    pR(p.xy, id);

    // Audio-reactive rotation
    float rotSpeed = 1.0 + pc.note_velocity * 2.0;
    pR(p.xy, pc.time * TAU * rotSpeed);

    float focus = dot(v, dir) * .5 + .5;

    d = fCrystal(p, id, focus);

    return Model(d, col, 1.);
}

// Ray marching
struct Hit {
    float len;
    vec3 colour;
    float id;
};

Model map(vec3 p) {
    Model res = Model(1000000., vec3(0), 0.);

    // Audio-reactive rotation
    float rx = pc.pitch_bend * PI * 0.5;
    float ry = pc.time + pc.osc_ch1 * TAU;

    pR(p.yz, rx);
    pR(p.xz, ry);

    res = model(p);
    return res;
}

Hit calcIntersection(vec3 ro, vec3 rd) {
    float h = INTERSECTION_PRECISION * 2.0;
    float t = 0.0;
    float res = -1.0;
    float id = -1.;
    vec3 colour;

    for(int i = 0; i < NUM_OF_TRACE_STEPS; i++) {
        if(h < INTERSECTION_PRECISION || t > MAX_TRACE_DISTANCE) break;
        Model m = map(ro + rd * t);
        h = m.dist;
        t += h * FUDGE_FACTOR;
        id = m.id;
        colour = m.colour;
    }

    if(t < MAX_TRACE_DISTANCE) res = t;
    if(t > MAX_TRACE_DISTANCE) id = -1.0;

    return Hit(res, colour, id);
}

vec3 calcNormal(vec3 pos) {
    vec3 eps = vec3(0.001, 0.0, 0.0);
    vec3 nor = vec3(
    map(pos+eps.xyy).dist - map(pos-eps.xyy).dist,
    map(pos+eps.yxy).dist - map(pos-eps.yxy).dist,
    map(pos+eps.yyx).dist - map(pos-eps.yyx).dist
    );
    return normalize(nor);
}

vec3 render(Hit hit, vec3 ro, vec3 rd) {
    // Darker, richer background
    vec3 bg = vec3(0.01, 0.005, 0.02);

    // Add subtle background gradient
    float bgGradient = dot(rd, vec3(0, 1, 0)) * 0.5 + 0.5;
    bg += vec3(0.02, 0.01, 0.03) * bgGradient * pc.cc74;

    vec3 color = bg;

    if(hit.id == 1.) {
        vec3 pos = ro + rd * hit.len;
        vec3 norm = calcNormal(pos);
        vec3 ref = reflect(rd, norm);

        // Multiple light sources
        vec3 lig1 = normalize(vec3(.5, 1, -.5));
        vec3 lig2 = normalize(vec3(-.5, .5, .5));
        vec3 lig3 = normalize(vec3(sin(pc.time * 2.), cos(pc.time * 1.5), -1));

        vec3 dome = vec3(0, 1, 0);
        vec3 eye = vec3(0, 0, -1);

        // Audio-reactive perturbation
        float perturbStrength = 10.0 + pc.note_velocity * 20.0;
        vec3 perturb = sin(pos * perturbStrength);

        color = spectrum(dot(norm + perturb * .05, eye) * 2.);

        // Multi-light specular
        float specular1 = clamp(dot(ref, lig1), 0., 1.);
        float specular2 = clamp(dot(ref, lig2), 0., 1.);
        float specular3 = clamp(dot(ref, lig3), 0., 1.);

        specular1 = pow((sin(specular1 * 20. - 3.) * .5 + .5) + .1, 32.) * specular1;
        specular2 = pow(specular2, 16.0) * 0.3;
        specular3 = pow(specular3, 8.0) * 0.2 * pc.note_velocity;

        float totalSpecular = (specular1 + specular2 + specular3) * (0.1 + pc.cc74 * 0.2);

        // Rim lighting effect
        float rimLight = 1.0 - abs(dot(norm, -rd));
        rimLight = pow(rimLight, 2.0) * 0.5 * pc.note_velocity;
        color += vec3(0.3, 0.5, 1.0) * rimLight;

        float shadow = pow(clamp(dot(norm, dome) * .5 + 1.2, 0., 1.), 3.);
        color = color * shadow + totalSpecular;

        // Audio-reactive fog with color tint
        float near = 2.8 - pc.osc_ch2 * 0.5;
        float far = 4. + pc.osc_ch2 * 2.0;
        float fog = (hit.len - near) / (far - near);
        fog = clamp(fog, 0., 1.);

        // Fog has slight color based on energy
        vec3 fogColor = bg + vec3(0.02, 0.01, 0.04) * pc.note_velocity;
        color = mix(color, fogColor, fog);
    }

    return color;
}

void main() {
    vec2 resolution = vec2(float(pc.render_w), float(pc.render_h));
    vec2 p = (fragUV - 0.5) * 2.0;
    p.x *= resolution.x / resolution.y;

    // Camera setup with audio influence
    vec3 camPos = vec3(0., 0.1, 3. - pc.cc1 * 0.5);
    vec3 camTar = vec3(0.);

    // Audio-reactive camera orbit
    float orbitSpeed = 1.0 - pc.cc74 * 0.7;
    pR(camPos.yx, pc.time * TAU * orbitSpeed);
    camPos += camTar;

    // Camera matrix
    vec3 ww = normalize(camTar - camPos);
    vec3 uu = normalize(cross(ww, vec3(0, 1, 0)));
    vec3 vv = normalize(cross(uu, ww));
    mat3 camMat = mat3(uu, vv, ww);

    // Create view ray
    vec3 rd = normalize(camMat * vec3(p.xy, 2.0));

    Hit hit = calcIntersection(camPos, rd);
    vec3 color = render(hit, camPos, rd);

    // Gamma correction
    color = pow(color, vec3(1.0 / 2.2));

    outColor = vec4(color, 1.0);
}