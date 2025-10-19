#version 450

// Push constants matching your application (SAME AS ORIGINAL)
// Variation 1: Faster - 1.5x speed
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

// Quality settings - same framework as original
#ifndef QUALITY_LEVEL
#define QUALITY_LEVEL 2
#endif

#if QUALITY_LEVEL == 0
const float MAX_DISPERSE = 3.;
const float MAX_BOUNCE = 3.;
const float MAX_MARCH_STEPS = 400.;
#elif QUALITY_LEVEL == 1
const float MAX_DISPERSE = 5.;
const float MAX_BOUNCE = 4.;
const float MAX_MARCH_STEPS = 600.;
#else
const float MAX_DISPERSE = 10.;
const float MAX_BOUNCE = 5.;
const float MAX_MARCH_STEPS = 800.;
#endif

// Global variables matching original framework
float iTime;
vec2 iResolution;
vec2 iMouse;
vec2 bsMo = vec2(0);
float audioEnergy = 0.0;
float audioMod = 0.0;
float audioBright = 0.0;
float audioBend = 0.0;

// Constants
#define PI 3.14159265359
#define saturate(x) clamp(x, 0., 1.)

// Global shader state
float time;
float invertg;

// Rotation function
void pR(inout vec2 p, float a) {
    p = cos(a)*p + sin(a)*vec2(p.y, -p.x);
}

// Smooth max function
float smax(float a, float b, float r) {
    vec2 u = max(vec2(r + a,r + b), vec2(0));
    return min(-r, max (a, b)) + length(u);
}

// Vector utilities
float vmax(vec2 v) { return max(v.x, v.y); }
float vmin(vec2 v) { return min(v.x, v.y); }
float vmax(vec3 v) { return max(max(v.x, v.y), v.z); }
float vmin(vec3 v) { return min(min(v.x, v.y), v.z); }

// Box distance functions
float fBox(vec2 p, vec2 b) {
    vec2 d = abs(p) - b;
    return length(max(d, vec2(0))) + vmax(min(d, vec2(0)));
}

float fBox(vec3 p, vec3 b) {
    vec3 d = abs(p) - b;
    return length(max(d, vec3(0))) + vmax(min(d, vec3(0)));
}

// Line distance function
float sdLine(vec3 p, float h, float r) {
    p.y -= clamp(p.y, 0.0, h);
    return length(p) - r;
}

// Spectrum palette with audio modulation
vec3 pal(in float t, in vec3 a, in vec3 b, in vec3 c, in vec3 d) {
    return a + b*cos(6.28318*(c*t+d));
}

vec3 spectrum(float n) {
    // Audio affects spectrum shift
    vec3 shift = vec3(0.0, 0.33, 0.67);
    shift.x += audioMod * 0.2;
    shift.y += audioEnergy * 0.1;

    // Note count creates color variations
    float noteShift = float(pc.note_count % 8u) * 0.125;
    shift += vec3(noteShift * 0.1);

    return pal(n, vec3(0.5), vec3(0.5), vec3(1.0), shift);
}

// Triple sine function
float sin3(vec3 v) {
    return sin(v.x) * sin(v.y) * sin(v.z);
}

// Main distance field with audio reactivity
vec2 map(vec3 p) {
    // Audio affects scale
    float scl = 1.3 - audioEnergy * 0.2;
    p /= scl;

    // Audio-reactive rotation
    float rotAmount = 0.2 + audioBend * 0.3;
    pR(p.yz, rotAmount * PI);
    pR(p.xz, (-0.25 + audioMod * 0.2) * PI);

    float flr = p.y + 0.5;

    // Primary wave distortion with audio
    float waveSpeed = mix(1.0, 3.0, audioMod);
    vec3 waveOffset = time * vec3(1, 3, 2) * PI * 2. * waveSpeed + vec3(0, .5, 3);

    // Vertex energy adds local distortion
    p += sin(p * 8. + waveOffset) * (0.1 + vertexEnergy * 0.05);

    // OSC inputs create additional displacement
    p.xy += vec2(pc.osc_ch1, pc.osc_ch2) * 0.05;

    vec3 p2 = p;

    // Fine detail with energy modulation
    p += sin3(p * 80.) * (0.0015 + audioEnergy * 0.001);

    // Main prism shape
    float boxSize = 0.5 - 0.01 - audioEnergy * 0.05;
    float b = fBox(p, vec3(boxSize)) - 0.01;

    // Edge details
    float d3 = 1e12;
    float rr = 0.0025 + audioBright * 0.002;
    p2 = abs(p2);
    p2 = vec3(vmin(p2.xz), p2.y, vmax(p2.xz));
    d3 = min(d3, sdLine(p2.xzy - vec3(.5, .7, .5), 0.2, rr));
    d3 = max(d3, -vmax(p * vec3(1, -1, -1)));

    // Secondary wave with audio modulation
    float secondWaveAmp = 0.1 + audioEnergy * 0.05;
    p += sin(p * 6. + time * vec3(-3, 2, 1) * PI * 2. + vec3(.1, .5, .6)) * secondWaveAmp;

    // Cut-out pattern affected by brightness
    float cutoutSize = 0.0125 + audioBright * 0.01;
    b = smax(b, -vmin(abs(p)) + cutoutSize, 0.01);

    float d2 = b + 0.1;
    float d = max(b, -d2 + 0.01);

    d *= invertg;
    d2 = max(d2 + 0.001, b);

    float id = 1.;

    if (d2 < d) {
        id = 3.;
        d = d2;
    }

    // Include edge details based on quality
    #if QUALITY_LEVEL > 0
    if (d3 < d) {
        d = d3;
        id = 4.;
    }
    #endif

    d *= scl;
    return vec2(d, id);
}

// Background color with audio modulation
vec3 getBGColor() {
    vec3 baseCol = vec3(0.86, 0.8, 1.0);

    // Energy darkens background
    baseCol *= 1.0 - audioEnergy * 0.3;

    // Mod wheel shifts hue
    baseCol = mix(baseCol, vec3(0.7, 0.85, 1.0), audioMod);

    // Note count adds color variation
    float noteHue = float(pc.note_count % 6u) / 6.0;
    baseCol = mix(baseCol, vec3(1.0, 0.8, 0.9), noteHue * 0.2);

    return baseCol;
}

// Environment orientation
mat3 envOrientation;

// Lighting with audio reactivity
vec3 light(vec3 origin, vec3 rayDir) {
    origin = -origin;
    rayDir = -rayDir;

    origin *= envOrientation;
    rayDir *= envOrientation;

    // Audio affects light position
    vec3 lightPos = vec3(-6.0 + bsMo.x * 2.0, bsMo.y * 2.0, 0.0);

    vec2 uv;
    float hit = 1.0; // Simplified for this version

    // Light intensity modulated by energy
    float l = smoothstep(0.75, 0.0, length(origin - lightPos) - 3.0);
    l *= 1.0 + audioEnergy;

    return vec3(l) * hit;
}

// Environment lighting
vec3 env(vec3 origin, vec3 rayDir) {
    origin = -(vec4(origin, 1)).xyz;
    rayDir = -(vec4(rayDir, 0)).xyz;

    origin *= envOrientation;
    rayDir *= envOrientation;

    float l = smoothstep(0.0, 1.7, dot(rayDir, vec3(0.5, -0.3, 1))) * 0.4;
    l *= 1.0 + audioBright * 0.5;

    return vec3(l) * getBGColor();
}

// Normal calculation
vec3 normal(in vec3 pos) {
    vec3 n = vec3(0.0);
    for(int i = 0; i < 4; i++) {
        vec3 e = 0.5773 * (2.0 * vec3((((i+3)>>1)&1), ((i>>1)&1), (i&1)) - 1.0);
        n += e * map(pos + 0.001 * e).x;
    }
    return normalize(n);
}

// Hit structure
struct Hit {
    vec2 res;
    vec3 p;
    float len;
    float steps;
};

// Ray marching
Hit march(vec3 origin, vec3 rayDir, float invert, float maxDist, float understep) {
    vec3 p;
    float len = 0.;
    float dist = 0.;
    vec2 res = vec2(0.);
    vec2 candidate = vec2(0.);
    float steps = 0.;
    invertg = invert;

    // Adjust max steps based on quality and brightness
    float maxSteps = MAX_MARCH_STEPS * (0.5 + audioBright * 0.5);

    for(float i = 0.; i < MAX_MARCH_STEPS; i++) {
        if(i >= maxSteps) break;

        len += dist * understep;
        p = origin + len * rayDir;
        candidate = map(p);
        dist = candidate.x;
        steps += 1.;
        res = candidate;

        if(dist < 0.00005) {
            break;
        }
        if(len >= maxDist) {
            len = maxDist;
            res.y = 0.;
            break;
        }
    }

    return Hit(res, p, len, steps);
}

// Matrix utilities
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

// Hex pattern for background
float hex(vec2 U) {
    U *= mat2(1, -1./1.73, 0, 2./1.73) * 5.;
    vec3 g = vec3(U, 1. - U.x - U.y), g2;
    vec3 id = floor(g);
    g = fract(g);
    if(length(g) > 1.) g = 1. - g;
    g2 = abs(2. * fract(g) - 1.);
    return length(1. - g2);
}

// Tonemapping
vec3 tonemap2(vec3 texColor) {
    texColor /= 2.;
    texColor *= 16.;
    vec3 x = max(vec3(0), texColor - 0.004);
    return (x * (6.2 * x + 0.5)) / (x * (6.2 * x + 1.7) + 0.06);
}

// Pseudo-random for dispersion
float hash(vec2 p) {
    return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}

void main() {
    // Setup resolution and time (matching original framework)
    iResolution = (pc.render_w > 0u && pc.render_h > 0u)
    ? vec2(pc.render_w, pc.render_h)
    : vec2(800.0, 600.0);

    // Cache audio parameters
    audioEnergy = clamp(pc.note_velocity, 0.0, 1.0);
    audioMod = clamp(pc.cc1, 0.0, 1.0);
    audioBright = clamp(pc.cc74, 0.0, 1.0);
    audioBend = clamp(pc.pitch_bend, -1.0, 1.0);

    // Time with audio modulation
    float timeScale = mix(0.8, 1.5, audioMod);
    float duration = 8.0 - audioEnergy * 3.0; // Faster with energy
    time = mod(pc.time * timeScale / duration + 0.1, 1.0);
    iTime = pc.time * timeScale;

    // Mouse setup
    iMouse = vec2(float(pc.mouse_x), float(pc.mouse_y));

    // Interactive control
    if(pc.mouse_pressed > 0u) {
        bsMo = (iMouse - 0.5 * iResolution.xy) / iResolution.y;
    } else {
        bsMo = vec2(pc.osc_ch1, pc.osc_ch2) * 0.5;
    }

    // Environment orientation with audio
    vec2 envAngles = vec2(0.87, 1.27) + bsMo * 0.5 + vec2(audioBend * 0.3, 0.0);
    envOrientation = sphericalMatrix(envAngles);

    vec2 q = fragUV;
    vec2 uv = (2. * fragUV - 1.) * vec2(iResolution.x / iResolution.y, 1.0);

    // Hex pattern with audio modulation
    vec2 hexOffset = time * vec2(0.1, 0.172) * 2.;
    hexOffset *= 1.0 + audioEnergy; // Speed up with energy
    float h = hex(uv.yx * 1.08 + hexOffset);
    h -= 0.03;
    h /= length(fwidth(uv * 10.));
    h = 1. - saturate(h);

    // Background colors
    vec3 BGCOL = getBGColor();
    vec3 bgCol = BGCOL * 0.08 * 0.5;
    vec3 bgCol2 = bgCol * 0.3;
    bgCol = mix(bgCol, bgCol2, h);

    Hit hit, firstHit;
    vec2 res;
    vec3 p, rayDir, origin, sam, ref, raf, nor, camOrigin, camDir;
    float invert, ior, offset, extinctionDist, maxDist, firstLen, bounceCount, wavelength;

    vec3 col = vec3(0);

    invert = 1.;
    maxDist = 15.;

    // Camera with audio reactivity
    float fl = 20. - audioEnergy * 5.; // Zoom with energy
    float camDist = 9.5 - audioEnergy * 2.; // Move closer with energy

    camOrigin = vec3(
    sin(iTime * 0.2) * bsMo.x * 2.,
    cos(iTime * 0.15) * bsMo.y * 2.,
    camDist * fl
    );

    // Add vertex energy to camera shake
    camOrigin.xy += vec2(sin(vertexEnergy * 10.), cos(vertexEnergy * 10.)) * 0.1;

    camDir = normalize(vec3(uv * 0.168, -fl));

    // First hit
    firstHit = march(camOrigin, camDir, invert, maxDist * fl, 0.6);
    firstLen = firstHit.len;

    float steps = 0.;

    // Adjust dispersion samples based on quality and brightness
    float maxDisperse = MAX_DISPERSE * (0.5 + audioBright * 0.5);

    for(float disperse = 0.; disperse < MAX_DISPERSE; disperse++) {
        if(disperse >= maxDisperse) break;

        invert = 1.;
        sam = vec3(0);

        origin = camOrigin;
        rayDir = camDir;

        extinctionDist = 0.;
        wavelength = disperse / MAX_DISPERSE;

        // Add randomness based on frame
        float rand = hash(fragUV + floor(iTime * 60.) * 10.);
        wavelength += (rand * 2. - 1.) * (0.5 / MAX_DISPERSE);

        bounceCount = 0.;
        vec3 nor;

        // Bounces affected by note count
        float maxBounce = min(MAX_BOUNCE, 2. + float(pc.note_count % 4u));

        for(float bounce = 0.; bounce < MAX_BOUNCE; bounce++) {
            if(bounce >= maxBounce) break;

            if(bounce == 0.) {
                hit = firstHit;
            } else {
                hit = march(origin, rayDir, invert, 1.2, 0.6);
            }

            steps += hit.steps;
            res = hit.res;
            p = hit.p;

            if(invert < 0.) {
                extinctionDist += hit.len;
            }

            // Hit background
            if(res.y == 0.) {
                break;
            }

            if(res.y == 4.) {
                break;
            }

            nor = normal(p);
            ref = reflect(rayDir, nor);

            if(res.y > 1.) {
                break;
            }

            // Shade with audio enhancement
            sam += light(p, ref) * (0.5 + audioEnergy * 0.3);
            sam += pow(max(1. - abs(dot(rayDir, nor)), 0.), 5.) * (0.1 + audioBright * 0.1);
            sam *= vec3(0.85, 0.85, 0.98);

            // Refract with audio-modulated IOR
            float iorBase = mix(1.2, 1.8, wavelength);
            iorBase += audioMod * 0.2; // Mod wheel affects refraction
            ior = invert < 0. ? iorBase : 1. / iorBase;

            raf = refract(rayDir, nor, ior);
            bool tif = raf == vec3(0); // Total internal reflection
            rayDir = tif ? ref : raf;
            offset = 0.01 / abs(dot(rayDir, nor));
            origin = p + offset * rayDir;
            invert *= -1.;

            bounceCount = bounce;
        }

        if(res.y > 1.) {
            sam = vec3(0);
            sam += light(p, ref) * 0.5;
            sam += pow(max(1. - abs(dot(rayDir, nor)), 0.), 5.) * 0.1;
            sam *= vec3(0.85, 0.85, 0.98);
            vec3 cc = res.y == 2. ? vec3(1) : vec3(0.033);
            rayDir = refract(rayDir, nor, 1./1.3);
            sam += env(p, rayDir) * cc;
        }

        if(bounceCount == 0.) {
            col += sam * MAX_DISPERSE;
            break;
        }

        if(res.y < 2.) {
            sam += env(p, rayDir);
        }

        // Extinction with audio modulation
        vec3 extinction = vec3(0.5) * (0.0 + audioEnergy * 0.05);
        extinction = 1. / (1. + (extinction * extinctionDist));
        col += sam * extinction * spectrum(-wavelength + 0.25);
    }

    col /= MAX_DISPERSE;

    if(bounceCount == 0. && res.y == 0.) {
        col = bgCol;
    }

    if(res.y == 4.) {
        col = bgCol2;
    }

    // Audio-reactive exposure
    float exposure = 2.5 + audioEnergy * 1.0;
    col = pow(col, vec3(1.19)) * exposure;

    col = tonemap2(col);

    // Post-processing matching original framework

    // Note count affects tint
    if(pc.note_count > 0u) {
        float noteInfluence = float(pc.note_count % 4u) * 0.25;
        vec3 tint = mix(vec3(1.0, 0.97, 0.9), vec3(0.9, 0.95, 1.1), noteInfluence);
        col *= tint;
    }

    // Vignette with energy influence
    float vignetteBase = 16.0 * q.x * q.y * (1.0 - q.x) * (1.0 - q.y);
    float vignettePower = 0.12 - audioEnergy * 0.05;
    float vignette = pow(vignetteBase, vignettePower);
    col *= vignette * 0.7 + 0.3;

    outColor = vec4(col, 1.0);
}