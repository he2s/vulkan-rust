#version 450

// Fractured Orb with dispersion - adapted for interactive audio/MIDI control
// Variation 1: Faster - 1.5x speed

layout(push_constant) uniform PushConstants {
    float time;
    uint  mouse_x;
    uint  mouse_y;
    uint  mouse_pressed;
    float note_velocity;
    float pitch_bend;
    float cc1;    // mod wheel / dispersion control
    float cc74;   // cutoff / shape control
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

// Quality settings
#ifndef QUALITY_LEVEL
#define QUALITY_LEVEL 2
#endif

#if QUALITY_LEVEL == 0
const float MAX_DISPERSE = 3.;
const float MAX_BOUNCE = 6.;
#elif QUALITY_LEVEL == 1
const float MAX_DISPERSE = 5.;
const float MAX_BOUNCE = 8.;
#else
const float MAX_DISPERSE = 7.;
const float MAX_BOUNCE = 10.;
#endif

#define PI 3.14159265359
#define TAU 6.28318530718

#define saturate(x) clamp(x, 0., 1.)

// Global variables
float iTime;
vec2 iResolution;
vec2 iMouse;
float audioEnergy;
float audioMod;
float audioBright;
float audioBend;
float time;

// --------------------------------------------------------
// HG_SDF utilities
// --------------------------------------------------------

void pR(inout vec2 p, float a) {
    p = cos(a)*p + sin(a)*vec2(p.y, -p.x);
}

float smax(float a, float b, float r) {
    vec2 u = max(vec2(r + a,r + b), vec2(0));
    return min(-r, max (a, b)) + length(u);
}

float vmax(vec2 v) {
    return max(v.x, v.y);
}

float vmax(vec3 v) {
    return max(max(v.x, v.y), v.z);
}

float fBox(vec2 p, vec2 b) {
    vec2 d = abs(p) - b;
    return length(max(d, vec2(0))) + vmax(min(d, vec2(0)));
}

float fBox(vec3 p, vec3 b) {
    vec3 d = abs(p) - b;
    return length(max(d, vec3(0))) + vmax(min(d, vec3(0)));
}

float range(float vmin, float vmax, float value) {
    return clamp((value - vmin) / (vmax - vmin), 0., 1.);
}

// --------------------------------------------------------
// Spectrum palette
// --------------------------------------------------------

vec3 pal(in float t, in vec3 a, in vec3 b, in vec3 c, in vec3 d) {
    return a + b*cos(6.28318*(c*t+d));
}

vec3 spectrum(float n) {
    // Audio-reactive palette
    vec3 shift = vec3(0.0, 0.33, 0.67);
    shift.x += audioBend * 0.2;
    shift.y += audioMod * 0.1;

    float brightness = 0.5 + audioBright * 0.3;
    return pal(n, vec3(brightness), vec3(0.5), vec3(1.0), shift);
}

// --------------------------------------------------------
// Modelling
// --------------------------------------------------------

vec2 map(vec3 p) {
    // Audio-reactive scale
    float scl = 0.9 - audioEnergy * 0.15;

    // Interactive rotation
    vec2 mouseControl = vec2(0);
    if (pc.mouse_pressed > 0u) {
        mouseControl = vec2(
            0.5 - float(pc.mouse_y) / iResolution.y,
            0.5 - float(pc.mouse_x) / iResolution.x
        );
    } else {
        // OSC control
        mouseControl = vec2(pc.osc_ch1 * 0.5, pc.osc_ch2 * 0.5);
        // Default rotation
        mouseControl += vec2(-0.25, -0.125);
    }

    pR(p.yz, mouseControl.x * PI / 2.);
    pR(p.xz, mouseControl.y * PI * 2.);

    p /= scl;

    // Audio-reactive distortions
    float waveSpeed = 1.0 + audioMod * 2.0;
    p += sin(sin(p * 5.) * 3. + time * PI * 2. * waveSpeed) * 0.1;

    // Fine detail with vertex energy
    float detailAmp = 0.03 + vertexEnergy * 0.02;
    p += (sin(p.x * 10. + time * PI * 2.) * sin(p.y * 20.) * sin(p.z * 30.)) * detailAmp;

    // Micro detail affected by brightness
    float sc = 3. + audioBright * 2.;
    p += (sin(p.x * 20. * sc + time * PI * 2.) * sin(p.y * 20. * sc) * sin(p.z * 20. * sc)) * 0.002;

    // Pitch bend affects orientation
    pR(p.xy, -PI/4. + audioBend * 0.3);
    pR(p.xz, -PI/4. + audioBend * 0.2);

    float d = length(p) - 1.;

    // Shape morphing with cc74
    float r = 0.3 + audioBright * 0.2;
    float boxiness = 2.5 + audioMod * 2.0;
    d = mix(d, fBox(p, vec3(0.8 - r)) - r, boxiness);

    // Hollow interior
    d = max(d, -(d + 0.01));

    d *= scl;

    return vec2(d, 1);
}

// --------------------------------------------------------
// Lighting
// --------------------------------------------------------

vec3 BGCOL = vec3(0.9, 0.83, 1.0);

float intersectPlane(vec3 rOrigin, vec3 rayDir, vec3 origin, vec3 normal, vec3 up, out vec2 uv) {
    float d = dot(normal, (origin - rOrigin)) / dot(rayDir, normal);
    vec3 point = rOrigin + d * rayDir;
    vec3 tangent = cross(normal, up);
    vec3 bitangent = cross(normal, tangent);
    point -= origin;
    uv = vec2(dot(tangent, point), dot(bitangent, point));
    return max(sign(d), 0.);
}

mat3 envOrientation;

vec3 light(vec3 origin, vec3 rayDir) {
    origin = -origin;
    rayDir = -rayDir;

    origin *= envOrientation;
    rayDir *= envOrientation;

    vec2 uv;
    vec3 pos = vec3(-6.0 + audioEnergy * 2.0);
    float hit = intersectPlane(origin, rayDir, pos, normalize(pos), normalize(vec3(-1,1,0)), uv);

    // Audio-reactive light shape
    vec2 lightSize = vec2(0.5 + audioMod * 0.5, 2.0 + audioBright);
    float l = smoothstep(0.75, 0.0, fBox(uv, lightSize) - 1.);
    l *= smoothstep(6., 0., length(uv));
    l *= 1.0 + audioEnergy * 0.5;

    return vec3(l) * hit;
}

vec3 env(vec3 origin, vec3 rayDir) {
    origin = -(vec4(origin, 1)).xyz;
    rayDir = -(vec4(rayDir, 0)).xyz;

    origin *= envOrientation;
    rayDir *= envOrientation;

    float l = smoothstep(0.0, 1.7, dot(rayDir, vec3(0.5, -0.3, 1.))) * 0.4;
    l *= 1.0 + audioBright * 0.3;

    // Audio-reactive background color
    vec3 bgColor = BGCOL;
    bgColor = mix(bgColor, vec3(0.83, 0.9, 1.0), audioMod * 0.3);

    return vec3(l) * bgColor;
}

// --------------------------------------------------------
// Marching
// --------------------------------------------------------

vec3 normal(in vec3 pos) {
    vec3 n = vec3(0.0);
    for(int i = 0; i < 4; i++) {
        vec3 e = 0.5773*(2.0*vec3((((i+3)>>1)&1),((i>>1)&1),(i&1))-1.0);
        n += e*map(pos+0.001*e).x;
    }
    return normalize(n);
}

struct Hit {
    vec2 res;
    vec3 p;
    float len;
    float steps;
};

Hit march(vec3 origin, vec3 rayDir, float invert, float maxDist, float understep) {
    vec3 p;
    float len = 0.;
    float dist = 0.;
    vec2 res = vec2(0.);
    vec2 candidate = vec2(0.);
    float steps = 0.;

    understep *= 0.2;

    for (float i = 0.; i < 300.; i++) {
        len += dist * understep;
        p = origin + len * rayDir;
        candidate = map(p);
        dist = candidate.x * invert;
        steps += 1.;
        res = candidate;
        if (dist < 0.001) {
            break;
        }
        if (len >= maxDist) {
            len = maxDist;
            res.y = 0.;
            break;
        }
    }

    return Hit(res, p, len, steps);
}

mat3 sphericalMatrix(vec2 tp) {
    float theta = tp.x;
    float phi = tp.y;
    float cx = cos(theta);
    float cy = cos(phi);
    float sx = sin(theta);
    float sy = sin(phi);
    return mat3(
        cy, -sy * -sx, -sy * cx,
        0, cx, sx,
        sy, cy * -sx, cy * cx
    );
}

// --------------------------------------------------------
// Tonemapping
// --------------------------------------------------------

vec3 tonemap2(vec3 texColor) {
    texColor /= 2.;
    texColor *= 16.;
    vec3 x = max(vec3(0), texColor - 0.004);
    return (x*(6.2*x+0.5))/(x*(6.2*x+1.7)+0.06);
}

// --------------------------------------------------------
// Pseudo-random for dispersion
// --------------------------------------------------------

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}

// --------------------------------------------------------
// Main
// --------------------------------------------------------

void main() {
    // Setup resolution
    iResolution = (pc.render_w > 0u && pc.render_h > 0u)
        ? vec2(pc.render_w, pc.render_h)
        : vec2(800.0, 600.0);

    // Cache audio parameters
    audioEnergy = clamp(pc.note_velocity, 0.0, 1.0);
    audioMod = clamp(pc.cc1, 0.0, 1.0);
    audioBright = clamp(pc.cc74, 0.0, 1.0);
    audioBend = clamp(pc.pitch_bend, -1.0, 1.0);

    // Time with audio modulation
    float duration = 4.0 - audioEnergy * 1.5;
    time = mod(pc.time / duration, 1.0);
    iTime = pc.time;

    // Environment orientation
    vec2 envAngles = ((vec2(81.5, 119) / vec2(187)) * 2. - 1.) * 2.;
    envAngles += vec2(audioBend * 0.5, audioMod * 0.3);
    envOrientation = sphericalMatrix(envAngles);

    vec2 fragCoord = fragUV * iResolution;
    vec2 uv = (2. * fragCoord - iResolution.xy) / iResolution.y;

    // Audio-reactive zoom
    float zoom = 1.0 + audioEnergy * 0.5;
    if (pc.mouse_pressed > 0u) {
        uv /= 1.75;
    } else {
        uv /= zoom;
    }

    Hit hit, firstHit;
    vec2 res;
    vec3 p, rayDir, origin, sam, ref, raf, nor, camOrigin, camDir;
    float invert, ior, offset, extinctionDist, maxDist, firstLen, bounceCount, wavelength;

    vec3 col = vec3(0);
    vec3 bgCol = BGCOL * 0.22;
    bgCol *= 1.0 + audioEnergy * 0.3;

    invert = 1.;
    maxDist = 15.;

    // Audio-reactive camera
    float camDist = 9.5 - audioEnergy * 1.5;
    camOrigin = vec3(0, 0, camDist);
    camDir = normalize(vec3(uv * 0.168, -1.));

    firstHit = march(camOrigin, camDir, invert, maxDist, 0.8);
    firstLen = firstHit.len;

    float steps = 0.;

    // Pseudo-random for dispersion
    float rand = hash(fragUV + floor(iTime * 60.) * 10.);

    // Audio controls dispersion amount
    float maxDisperse = MAX_DISPERSE * (0.5 + audioMod * 0.5);

    for (float disperse = 0.; disperse < MAX_DISPERSE; disperse++) {
        if (disperse >= maxDisperse) break;

        invert = 1.;
        sam = vec3(0);

        origin = camOrigin;
        rayDir = camDir;

        extinctionDist = 0.;
        wavelength = disperse / MAX_DISPERSE;
        wavelength += (rand * 2. - 1.) * (0.5 / MAX_DISPERSE);
        wavelength = mix(-0.5/5., 1. - 0.5/5., mod(wavelength, 1.));

        bounceCount = 0.;

        // Audio controls bounce count
        float maxBounce = MAX_BOUNCE * (0.6 + audioBright * 0.4);

        for (float bounce = 0.; bounce < MAX_BOUNCE; bounce++) {
            if (bounce >= maxBounce) break;

            if (bounce == 0.) {
                hit = firstHit;
            } else {
                hit = march(origin, rayDir, invert, maxDist / 2., 1.);
            }

            steps += hit.steps;

            res = hit.res;
            p = hit.p;

            if (invert < 0.) {
                extinctionDist += hit.len;
            }

            // Hit background
            if (res.y == 0.) {
                break;
            }

            vec3 nor = normal(p) * invert;
            ref = reflect(rayDir, nor);

            // Shade with audio enhancement
            sam += light(p, ref) * (0.5 + audioEnergy * 0.3);
            sam += pow(max(1. - abs(dot(rayDir, nor)), 0.), 5.) * 0.1;
            sam *= vec3(0.85, 0.85, 0.98);

            // Refract with audio-modulated IOR
            float iorBase = mix(0.1, 0.95, wavelength);
            iorBase += audioBend * 0.1;
            ior = invert < 0. ? iorBase : 1. / iorBase;

            raf = refract(rayDir, nor, ior);
            bool tif = raf == vec3(0);
            rayDir = tif ? ref : raf;
            offset = 0.01 / abs(dot(rayDir, nor));
            origin = p + offset * rayDir;
            invert *= -1.;

            bounceCount = bounce;
        }

        sam += bounceCount == 0. ? bgCol : env(p, rayDir);

        if (bounceCount == 0.) {
            col += sam * MAX_DISPERSE / 2.;
            break;
        } else {
            // Audio-reactive extinction
            vec3 extinction = vec3(0.5) * (0.0 + audioEnergy * 0.1);
            extinction = 1. / (1. + (extinction * extinctionDist));
            col += sam * extinction * spectrum(-wavelength + 0.25);
        }
    }

    col /= MAX_DISPERSE;

    // Audio-reactive exposure
    float exposure = 2.5 + audioEnergy * 1.0;
    col = pow(col, vec3(1.25)) * exposure;
    col = tonemap2(col);

    // Vignette
    vec2 q = fragUV;
    float vignette = 16.0 * q.x * q.y * (1.0 - q.x) * (1.0 - q.y);
    float vignettePower = 0.15 - audioEnergy * 0.05;
    vignette = pow(vignette, vignettePower);
    col *= vignette * 0.7 + 0.3;

    outColor = vec4(col, 1.0);
}
