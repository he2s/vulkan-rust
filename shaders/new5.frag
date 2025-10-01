#version 450

// Push constants matching your application (SAME AS ORIGINAL)
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

// Quality settings - REDUCED for performance and blur
#ifndef QUALITY_LEVEL
#define QUALITY_LEVEL 2
#endif

#if QUALITY_LEVEL == 0
const float MAX_DISPERSE = 2.;
const float MAX_BOUNCE = 2.;
const float MAX_MARCH_STEPS = 150.;
#elif QUALITY_LEVEL == 1
const float MAX_DISPERSE = 3.;
const float MAX_BOUNCE = 2.;
const float MAX_MARCH_STEPS = 200.;
#else
const float MAX_DISPERSE = 4.;
const float MAX_BOUNCE = 3.;
const float MAX_MARCH_STEPS = 250.;
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

// Main distance field - BRUTALIST style with simpler geometry
vec2 map(vec3 p) {
    // Brutalist scale - larger, chunkier
    float scl = 1.8 - audioEnergy * 0.3;
    p /= scl;

    // Harsh angular rotation
    float rotAmount = 0.3 + audioBend * 0.4;
    pR(p.yz, rotAmount * PI);
    pR(p.xz, (-0.4 + audioMod * 0.3) * PI);

    // Simplified wave - less detail, more brutal
    float waveSpeed = mix(0.5, 1.5, audioMod);
    vec3 waveOffset = time * vec3(1, 2, 1) * PI * waveSpeed;

    // Reduced detail waves - chunkier distortion
    p += sin(p * 3. + waveOffset) * (0.15 + vertexEnergy * 0.1);

    // OSC inputs with stronger effect
    p.xy += vec2(pc.osc_ch1, pc.osc_ch2) * 0.1;

    vec3 p2 = p;

    // Remove fine detail - brutalist doesn't need it
    // p += sin3(p * 80.) * (0.0015 + audioEnergy * 0.001);

    // Brutalist box - chunky, angular
    float boxSize = 0.6 - audioEnergy * 0.1;
    float b = fBox(p, vec3(boxSize)) - 0.02; // Thicker edges

    // Simplified edge details - just one bold cut
    float d3 = 1e12;
    float rr = 0.01; // Thicker edges for brutalism
    p2 = abs(p2);
    d3 = fBox(p2 - vec3(0.4), vec3(0.05, 0.8, 0.8));

    // Harsh secondary cut - brutalist style
    float secondWaveAmp = 0.2 + audioEnergy * 0.1;
    p += sin(p * 2. + time * vec3(-1, 1, 0) * PI) * secondWaveAmp;

    // Larger, more brutal cuts
    float cutoutSize = 0.03 + audioBright * 0.02;
    b = smax(b, -vmin(abs(p)) + cutoutSize, 0.02);

    float d = b;

    d *= invertg;

    float id = 1.;

    // Simplified material IDs for performance
    if (d3 < d) {
        d = d3;
        id = 2.;
    }

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

// Simplified normal - less samples for blur effect
vec3 normal(in vec3 pos) {
    vec2 e = vec2(0.005, 0.0); // Larger epsilon for softer normals
    return normalize(vec3(
    map(pos + e.xyy).x - map(pos - e.xyy).x,
    map(pos + e.yxy).x - map(pos - e.yxy).x,
    map(pos + e.yyx).x - map(pos - e.yyx).x
    ));
}

// Hit structure
struct Hit {
    vec2 res;
    vec3 p;
    float len;
    float steps;
};

// Ray marching - OPTIMIZED for performance
Hit march(vec3 origin, vec3 rayDir, float invert, float maxDist, float understep) {
    vec3 p;
    float len = 0.;
    float dist = 0.;
    vec2 res = vec2(0.);
    vec2 candidate = vec2(0.);
    float steps = 0.;
    invertg = invert;

    // Much fewer steps for performance
    float maxSteps = MAX_MARCH_STEPS * 0.7;

    for(float i = 0.; i < MAX_MARCH_STEPS; i++) {
        if(i >= maxSteps) break;

        len += dist * 1.2; // Larger steps for blur/performance
        p = origin + len * rayDir;
        candidate = map(p);
        dist = candidate.x;
        steps += 1.;
        res = candidate;

        if(dist < 0.002) { // Larger threshold for softer edges
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

    // SIMPLIFIED hex pattern - removed for performance
    float h = 0.0;

    // Brutalist background - solid blocks of color
    vec3 BGCOL = getBGColor();
    vec3 bgCol = BGCOL;
    vec3 bgCol2 = bgCol * 0.5; // Darker variant

    // Simple gradient instead of hex pattern
    float gradient = dot(uv, vec2(0.7, 0.7)) * 0.5 + 0.5;
    bgCol = mix(bgCol, bgCol2, gradient);

    Hit hit, firstHit;
    vec2 res;
    vec3 p, rayDir, origin, sam, ref, raf, nor, camOrigin, camDir;
    float invert, ior, offset, extinctionDist, maxDist, firstLen, bounceCount, wavelength;

    vec3 col = vec3(0);

    invert = 1.;
    maxDist = 10.; // Reduced for performance

    // Simplified camera
    float fl = 15.;
    float camDist = 7.5 - audioEnergy * 1.5;

    camOrigin = vec3(
    bsMo.x * 3.,
    bsMo.y * 3.,
    camDist * fl
    );

    camDir = normalize(vec3(uv * 0.2, -fl));

    // First hit
    firstHit = march(camOrigin, camDir, invert, maxDist * fl, 0.8);
    firstLen = firstHit.len;

    float steps = 0.;

    // Much fewer dispersion samples for blur and performance
    float maxDisperse = MAX_DISPERSE;

    for(float disperse = 0.; disperse < MAX_DISPERSE; disperse++) {
        if(disperse >= maxDisperse) break;

        invert = 1.;
        sam = vec3(0);

        origin = camOrigin;
        rayDir = camDir;

        extinctionDist = 0.;
        wavelength = disperse / MAX_DISPERSE;

        // Remove randomness for more stable/brutal look

        bounceCount = 0.;
        vec3 nor;

        // Fewer bounces for performance
        float maxBounce = MAX_BOUNCE;

        for(float bounce = 0.; bounce < MAX_BOUNCE; bounce++) {
            if(bounce >= maxBounce) break;

            if(bounce == 0.) {
                hit = firstHit;
            } else {
                hit = march(origin, rayDir, invert, 1.5, 0.8);
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

            nor = normal(p);
            ref = reflect(rayDir, nor);

            if(res.y > 1.) {
                break;
            }

            // Simplified brutal shading
            sam += vec3(0.2) * (0.5 + audioEnergy * 0.2);
            sam *= vec3(0.8, 0.8, 0.85); // Industrial tint

            // Simplified refraction
            float iorBase = mix(1.3, 1.5, wavelength);
            iorBase += audioMod * 0.1;
            ior = invert < 0. ? iorBase : 1. / iorBase;

            raf = refract(rayDir, nor, ior);
            bool tif = raf == vec3(0);
            rayDir = tif ? ref : raf;
            offset = 0.02; // Larger offset for softer intersections
            origin = p + offset * rayDir;
            invert *= -1.;

            bounceCount = bounce;
        }

        if(res.y > 1.) {
            sam = vec3(0.1) * (1.0 + audioEnergy);
            sam *= vec3(0.8, 0.8, 0.85);
        }

        if(bounceCount == 0.) {
            col += sam * MAX_DISPERSE;
            break;
        }

        if(res.y < 2.) {
            sam += bgCol * 0.5; // Simplified environment
        }

        // Simplified extinction
        vec3 extinction = vec3(1.0);
        col += sam * extinction * spectrum(-wavelength + 0.25);
    }

    col /= MAX_DISPERSE;

    if(bounceCount == 0. && res.y == 0.) {
        col = bgCol;
    }

    // Brutalist post-processing
    col = pow(col, vec3(1.4)) * 1.8; // Harsh contrast

    // Simplified tonemapping for brutalist look
    col = col / (1.0 + col); // Reinhard tone mapping
    col = pow(col, vec3(0.7)); // Lift shadows for industrial look

    // Add grain for texture
    float grain = hash(fragUV + fract(iTime)) * 0.05;
    col += vec3(grain);

    // Harsh vignette for brutalist framing
    float vignette = 1.0 - length(uv) * 0.5;
    vignette = pow(vignette, 0.5);
    col *= vignette;

    // Reduced color saturation for industrial feel
    float gray = dot(col, vec3(0.299, 0.587, 0.114));
    col = mix(vec3(gray), col, 0.6 + audioEnergy * 0.2);

    outColor = vec4(col, 1.0);
}