#version 450

// Geodesic tiling adapted for interactive audio/MIDI control

// Push constants for interactivity
layout(push_constant) uniform PushConstants {
    float time;
    uint  mouse_x;
    uint  mouse_y;
    uint  mouse_pressed;
    float note_velocity;
    float pitch_bend;
    float cc1;    // mod wheel / subdivision control
    float cc74;   // cutoff / color control
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

// Global variables
float iTime;
vec2 iResolution;
vec2 iMouse;
float audioEnergy;
float audioMod;
float audioBright;
float audioBend;

#define PI 3.14159265359
#define HEX_TILE

// --------------------------------------------------------
// Icosahedral domain mirroring
// --------------------------------------------------------

vec3 facePlane;
vec3 uPlane;
vec3 vPlane;

int Type=5;
vec3 nc;
vec3 pab;
vec3 pbc;
vec3 pca;

void init() {
    float cospin=cos(PI/float(Type)), scospin=sqrt(0.75-cospin*cospin);
    nc=vec3(-0.5,-cospin,scospin);
    pbc=vec3(scospin,0.,0.5);
    pca=vec3(0.,scospin,cospin);
    pbc=normalize(pbc); pca=normalize(pca);
    pab=vec3(0,0,1);

    facePlane = pca;
    uPlane = cross(vec3(1,0,0), facePlane);
    vPlane = vec3(1,0,0);
}

void fold(inout vec3 p) {
    for(int i=0;i<5;i++){
        p.xy = abs(p.xy);
        p -= 2. * min(0., dot(p,nc)) * nc;
    }
}

// --------------------------------------------------------
// Triangle tiling
// --------------------------------------------------------

const float sqrt3 = 1.7320508075688772;
const float i3 = 0.5773502691896258;

const mat2 cart2tri = mat2(1, 0, i3, 2. * i3);
const mat2 tri2cart = mat2(1, 0, -.5, .5 * sqrt3);

vec2 pick3(vec2 a, vec2 b, vec2 c, float u) {
    float v = fract(u * 0.3333333333333);
    return mix(mix(a, b, step(0.3, v)), c, step(0.6, v));
}

vec2 closestHex(vec2 p) {
    p = cart2tri * p;
    vec2 pi = floor(p);
    vec2 pf = fract(p);
    vec2 nn = pick3(
        vec2(0, 0),
        vec2(1, 1),
        vec2(1, 0),
        pi.x + pi.y
    );
    vec2 hex = mix(nn.xy, nn.yx, step(pf.x, pf.y)) + pi;
    hex = tri2cart * hex;
    return hex;
}

vec2 closestTri(vec2 p) {
    p = cart2tri * p;
    vec2 pf = fract(p);
    vec2 v = vec2(1./3., 2./3.);
    vec2 tri = mix(v, v.yx, step(pf.y, pf.x));
    tri += floor(p);
    tri = tri2cart * tri;
    return tri;
}

// --------------------------------------------------------
// Geodesic tiling
// --------------------------------------------------------

vec3 intersection(vec3 n, vec3 planeNormal, float planeOffset) {
    float denominator = dot(planeNormal, n);
    float t = (dot(vec3(0), planeNormal) + planeOffset) / -denominator;
    return n * t;
}

vec2 icosahedronFaceCoordinates(vec3 p) {
    vec3 i = intersection(normalize(p), facePlane, -1.);
    return vec2(dot(i, uPlane), dot(i, vPlane));
}

vec3 faceToSphere(vec2 facePoint) {
    return normalize(facePlane + (uPlane * facePoint.x) + (vPlane * facePoint.y));
}

const float edgeLength = 1. / ((sqrt(3.) / 12.) * (3. + sqrt(5.)));
const float faceRadius = (1./6.) * sqrt(3.) * edgeLength;

vec3 geodesicTri(vec3 p, float subdivisions) {
    float uvScale = subdivisions / faceRadius;

    vec2 uv = icosahedronFaceCoordinates(p);

    #ifdef HEX_TILE
        uvScale /= 1.3333;
        vec2 closest = closestHex(uv * uvScale);
    #else
        uvScale /= 2.;
        vec2 closest = closestTri(uv * uvScale);
    #endif

    return faceToSphere(closest / uvScale);
}

// --------------------------------------------------------
// Modelling
// --------------------------------------------------------

struct Model {
    float dist;
    vec3 color;
};

void spin(inout vec3 p) {
    // Rotation controlled by time and pitch bend
    float r = iTime / (6. - audioEnergy * 2.);
    r += audioBend * 0.5;

    mat2 rot = mat2(cos(r), -sin(r), sin(r), cos(r));
    p.xz *= rot;

    // Secondary rotation with mod wheel
    float r2 = r * 0.5 + audioMod * PI;
    mat2 rot2 = mat2(cos(r2), -sin(r2), sin(r2), cos(r2));
    p.zy *= rot2;
}

// Audio-reactive subdivision count
float getSubdivisions() {
    // cc1 (mod wheel) controls subdivision level
    float minSub = 1.0;
    float maxSub = 12.0;

    // Map cc1 to subdivision range
    float targetSub = mix(minSub, maxSub, audioMod);

    // Add note-based variations
    if (pc.note_count > 0u) {
        float noteOffset = float(pc.note_count % 5u) * 0.5;
        targetSub += noteOffset;
    }

    // Smooth animation
    float t = mod(iTime * (0.5 + audioEnergy), 3.0) - 1.5;
    t = clamp(t, 0., 1.);
    t = cos(t * PI + PI) * .5 + .5;

    return mix(targetSub, targetSub + 1.0, t);
}

Model map(vec3 p) {
    spin(p);
    fold(p);

    float subdivisions = getSubdivisions();
    vec3 point = geodesicTri(p, subdivisions);

    // Audio-reactive sphere size
    float sphereSize = 0.195 / subdivisions;
    sphereSize *= 1.0 + vertexEnergy * 0.2;

    float sphere = length(p - point) - sphereSize;

    // Audio-reactive coloring
    vec3 color = vec3(0);

    // Base color from position
    color.gb = point.yx * 2.5 + .5;

    // cc74 controls color brightness
    color.r = audioBright * 0.8;

    // Note velocity adds warmth
    color += audioEnergy * vec3(0.3, 0.2, 0.1);

    // Pitch bend affects hue
    color = mix(color, color.gbr, audioBend * 0.5 + 0.5);

    // OSC inputs create color variation
    color.rg += vec2(pc.osc_ch1, pc.osc_ch2) * 0.2;

    color = clamp(color, 0., 1.);

    return Model(sphere, color);
}

// --------------------------------------------------------
// Ray Marching
// --------------------------------------------------------

const float MAX_TRACE_DISTANCE = 8.;
const float INTERSECTION_PRECISION = .001;
const int NUM_OF_TRACE_STEPS = 100;

struct CastRay {
    vec3 origin;
    vec3 direction;
};

struct Ray {
    vec3 origin;
    vec3 direction;
    float len;
};

struct Hit {
    Ray ray;
    Model model;
    vec3 pos;
    bool isBackground;
    vec3 normal;
    vec3 color;
};

vec3 calcNormal(in vec3 pos) {
    vec3 eps = vec3(0.001, 0.0, 0.0);
    vec3 nor = vec3(
        map(pos+eps.xyy).dist - map(pos-eps.xyy).dist,
        map(pos+eps.yxy).dist - map(pos-eps.yxy).dist,
        map(pos+eps.yyx).dist - map(pos-eps.yyx).dist
    );
    return normalize(nor);
}

Hit raymarch(CastRay castRay) {
    float currentDist = INTERSECTION_PRECISION * 2.0;
    Model model;

    Ray ray = Ray(castRay.origin, castRay.direction, 0.);

    for(int i=0; i<NUM_OF_TRACE_STEPS; i++) {
        if (currentDist < INTERSECTION_PRECISION || ray.len > MAX_TRACE_DISTANCE) {
            break;
        }
        model = map(ray.origin + ray.direction * ray.len);
        currentDist = model.dist;
        ray.len += currentDist;
    }

    bool isBackground = false;
    vec3 pos = vec3(0);
    vec3 normal = vec3(0);
    vec3 color = vec3(0);

    if (ray.len > MAX_TRACE_DISTANCE) {
        isBackground = true;
    } else {
        pos = ray.origin + ray.direction * ray.len;
        normal = calcNormal(pos);
    }

    return Hit(ray, model, pos, isBackground, normal, color);
}

// --------------------------------------------------------
// Rendering
// --------------------------------------------------------

vec3 render(Hit hit) {
    if (hit.isBackground) {
        // Audio-reactive background
        vec3 bgColor = vec3(0.02, 0.03, 0.05);
        bgColor *= 1.0 + audioEnergy * 0.3;
        return bgColor;
    }

    vec3 color = hit.model.color;

    // Dynamic lighting
    float lighting = sin(dot(hit.normal, vec3(0,1,0))) * 0.3 + 0.7;
    lighting += audioEnergy * 0.2;
    color *= lighting;

    // Audio-reactive fog
    float fogAmount = 0.4 + audioBright * 0.3;
    color *= 1. - clamp(hit.ray.len * fogAmount - 0.8, 0., 1.);

    // Add rim lighting
    vec3 viewDir = normalize(hit.ray.origin - hit.pos);
    float rim = 1.0 - max(0.0, dot(viewDir, hit.normal));
    rim = pow(rim, 3.0);
    color += rim * vec3(0.2, 0.3, 0.4) * (0.5 + audioEnergy * 0.5);

    return color;
}

// --------------------------------------------------------
// Camera
// --------------------------------------------------------

mat3 calcLookAtMatrix(in vec3 ro, in vec3 ta, in float roll) {
    vec3 ww = normalize(ta - ro);
    vec3 uu = normalize(cross(ww, vec3(sin(roll), cos(roll), 0.0)));
    vec3 vv = normalize(cross(uu, ww));
    return mat3(uu, vv, ww);
}

void main() {
    init();

    // Setup resolution
    iResolution = (pc.render_w > 0u && pc.render_h > 0u)
        ? vec2(pc.render_w, pc.render_h)
        : vec2(800.0, 600.0);

    // Cache audio parameters
    audioEnergy = clamp(pc.note_velocity, 0.0, 1.0);
    audioMod = clamp(pc.cc1, 0.0, 1.0);
    audioBright = clamp(pc.cc74, 0.0, 1.0);
    audioBend = clamp(pc.pitch_bend, -1.0, 1.0);

    // Time
    iTime = pc.time;

    // Mouse
    iMouse = vec2(float(pc.mouse_x), float(pc.mouse_y));

    vec2 fragCoord = fragUV * iResolution;
    vec2 p = (-iResolution.xy + 2. * fragCoord.xy) / iResolution.y;

    // Audio-reactive camera
    float camDist = 2.5 - audioEnergy * 0.5;

    // Mouse or OSC control
    vec2 camOffset = vec2(0);
    if (pc.mouse_pressed > 0u) {
        camOffset = (iMouse - 0.5 * iResolution.xy) / iResolution.y * 2.0;
    } else {
        camOffset = vec2(pc.osc_ch1, pc.osc_ch2) * 1.5;
    }

    vec3 camPos = vec3(camOffset.x, camOffset.y, camDist);
    vec3 camTar = vec3(0);

    // Pitch bend affects camera roll
    float camRoll = audioBend * 0.3;

    mat3 camMat = calcLookAtMatrix(camPos, camTar, camRoll);

    // Audio affects FOV
    float fov = 2.0 - audioEnergy * 0.3;
    vec3 rd = normalize(camMat * vec3(p.xy, fov));

    Hit hit = raymarch(CastRay(camPos, rd));
    vec3 color = render(hit);

    // Post-processing
    color = pow(color, vec3(0.4545)); // Gamma correction

    outColor = vec4(color, 1.0);
}
