#version 450

layout(push_constant) uniform PushConstants {
    float time;
    uint  mouse_x;
    uint  mouse_y;
    uint  mouse_pressed;
    float note_velocity;
    float pitch_bend;
    float cc1;
    float cc74;
    uint  note_count;
    uint  last_note;
} pc;

layout(location = 0) in vec2 fragUV;
layout(location = 1) in float vertexEnergy;
layout(location = 2) in vec3 worldPos;

layout(location = 0) out vec4 outColor;

// ---- Noise functions ----
float hash(vec3 p) {
    p = fract(p * vec3(443.8975, 397.2973, 491.1871));
    p += dot(p, p.zyx + 19.19);
    return fract(p.x * p.y * p.z);
}

float noise3D(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);

    float n = mix(
        mix(mix(hash(i + vec3(0,0,0)), hash(i + vec3(1,0,0)), f.x),
            mix(hash(i + vec3(0,1,0)), hash(i + vec3(1,1,0)), f.x), f.y),
        mix(mix(hash(i + vec3(0,0,1)), hash(i + vec3(1,0,1)), f.x),
            mix(hash(i + vec3(0,1,1)), hash(i + vec3(1,1,1)), f.x), f.y), f.z);

    return n;
}

// Fractal Brownian Motion
float fbm(vec3 p, int octaves, float lacunarity, float gain) {
    float amplitude = 0.5;
    float frequency = 1.0;
    float total = 0.0;
    float normalization = 0.0;

    for (int i = 0; i < octaves; i++) {
        float noiseValue = noise3D(p * frequency);
        total += noiseValue * amplitude;
        normalization += amplitude;
        amplitude *= gain;
        frequency *= lacunarity;
    }

    return total / normalization;
}

// Ridged noise for sharp features
float ridgedNoise(vec3 p) {
    float n = 1.0 - abs(noise3D(p) * 2.0 - 1.0);
    return n * n;
}

float ridgedFBM(vec3 p, int octaves) {
    float amplitude = 0.5;
    float frequency = 1.0;
    float total = 0.0;
    float normalization = 0.0;

    for (int i = 0; i < octaves; i++) {
        float noiseValue = ridgedNoise(p * frequency);
        total += noiseValue * amplitude;
        normalization += amplitude;
        amplitude *= 0.5;
        frequency *= 2.1;
    }

    return total / normalization;
}

// ---- Terrain SDF ----
float terrainMap(vec3 p) {
    // Audio reactive parameters
    float energy = pc.note_velocity;
    float lowFreq = pc.pitch_bend * 0.5 + 0.5;
    float midFreq = pc.cc1;
    float highFreq = pc.cc74;

    // Animated domain warping
    float warpSpeed = 0.1 + 0.2 * energy;
    vec3 warp = vec3(
        fbm(p * 0.3 + vec3(pc.time * warpSpeed, 0, 0), 3, 2.0, 0.5),
        fbm(p * 0.3 + vec3(0, pc.time * warpSpeed * 0.7, 0), 3, 2.0, 0.5),
        fbm(p * 0.3 + vec3(0, 0, pc.time * warpSpeed * 0.5), 3, 2.0, 0.5)
    ) * (0.5 + 0.5 * midFreq);

    vec3 wp = p + warp * 0.3;

    // Base terrain height
    float terrain = p.y;

    // Large scale features (mountains/valleys)
    float largeForms = ridgedFBM(wp * 0.2 + vec3(pc.time * 0.02, 0, 0), 4) * 2.5;
    largeForms *= 1.0 + 0.3 * lowFreq;

    // Medium details
    float mediumDetail = fbm(wp * 0.8, 5, 2.0, 0.5) * 0.8;
    mediumDetail *= 1.0 + 0.2 * midFreq;

    // Fine details (reactive to high frequencies)
    float fineDetail = fbm(wp * 3.0, 3, 2.5, 0.4) * 0.15;
    fineDetail *= 1.0 + 0.5 * highFreq;

    // Combine layers
    terrain += largeForms + mediumDetail + fineDetail;

    // Create overhangs and caves
    float caves = fbm(p * 0.5 + vec3(0, p.y * 0.3, 0), 4, 2.0, 0.5);
    caves = smoothstep(0.3, 0.7, caves) * 0.8;
    terrain -= caves * (0.5 + 0.5 * energy);

    return terrain;
}

// Softer version for faster marching
float terrainMapSoft(vec3 p) {
    float energy = pc.note_velocity;
    vec3 wp = p + vec3(pc.time * 0.05, 0, 0);
    float terrain = p.y + ridgedFBM(wp * 0.2, 3) * 2.0 * (1.0 + 0.3 * energy);
    return terrain;
}

// ---- Rendering ----
vec3 calcNormal(vec3 p) {
    const float eps = 0.01;
    vec3 n;
    n.x = terrainMap(p + vec3(eps, 0, 0)) - terrainMap(p - vec3(eps, 0, 0));
    n.y = terrainMap(p + vec3(0, eps, 0)) - terrainMap(p - vec3(0, eps, 0));
    n.z = terrainMap(p + vec3(0, 0, eps)) - terrainMap(p - vec3(0, 0, eps));
    return normalize(n);
}

float softshadow(vec3 ro, vec3 rd, float mint, float maxt, float k) {
    float res = 1.0;
    float t = mint;
    for(int i = 0; i < 16; i++) {
        float h = terrainMapSoft(ro + rd * t);
        res = min(res, k * h / t);
        t += clamp(h, 0.02, 0.2);
        if(h < 0.001 || t > maxt) break;
    }
    return clamp(res, 0.0, 1.0);
}

void main() {
    vec2 uv = (fragUV - 0.5) * 2.0;
    const vec2 resolution = vec2(800.0, 600.0);
    uv.x *= resolution.x / resolution.y;

    // Camera setup (reactive to audio)
    float camDist = 5.0 - 1.0 * pc.note_velocity;
    float camHeight = 2.0 + 1.0 * pc.cc1;
    vec3 ro = vec3(
        camDist * sin(pc.time * 0.1),
        camHeight,
        camDist * cos(pc.time * 0.1)
    );
    vec3 ta = vec3(0, 0.5, 0);

    // Camera matrix
    vec3 ww = normalize(ta - ro);
    vec3 uu = normalize(cross(ww, vec3(0, 1, 0)));
    vec3 vv = normalize(cross(uu, ww));
    vec3 rd = normalize(uv.x * uu + uv.y * vv + 1.5 * ww);

    // Raymarching
    float t = 0.0;
    float tmax = 50.0;
    vec3 col = vec3(0.02); // Dark background

    // Sky gradient
    float skyGrad = 1.0 - abs(rd.y);
    col += vec3(0.03, 0.04, 0.05) * skyGrad * 0.5;

    // March
    float material = 0.0;
    vec3 pos;
    bool hit = false;

    for(int i = 0; i < 128; i++) {
        pos = ro + rd * t;
        float h = terrainMap(pos);

        if(abs(h) < 0.001 * t) {
            hit = true;
            break;
        }

        // Atmospheric accumulation
        float density = exp(-pos.y * 0.2) * 0.01;
        col += vec3(0.05, 0.06, 0.07) * density * (1.0 - smoothstep(0.0, tmax, t));

        t += h * 0.5;
        if(t > tmax) break;
    }

    if(hit) {
        vec3 nor = calcNormal(pos);

        // Lighting
        vec3 lig1 = normalize(vec3(0.5, 0.8, -0.3));
        vec3 lig2 = normalize(vec3(-0.3, 0.4, 0.5));

        float dif1 = max(dot(nor, lig1), 0.0);
        float dif2 = max(dot(nor, lig2), 0.0) * 0.5;
        float amb = 0.1;

        // Shadows
        float sha1 = softshadow(pos + nor * 0.01, lig1, 0.01, 10.0, 8.0);

        // Material color (grayscale with subtle variation)
        vec3 mat = vec3(0.3 + 0.2 * fbm(pos * 2.0, 3, 2.0, 0.5));
        mat *= 1.0 + 0.3 * vertexEnergy;

        // Ridge highlighting
        float ridges = ridgedNoise(pos * 4.0);
        mat += vec3(0.1) * ridges * pc.cc74;

        // Combine lighting
        col = mat * (amb + dif1 * sha1 + dif2);

        // Depth fog
        float fog = 1.0 - exp(-t * 0.05);
        col = mix(col, vec3(0.02, 0.02, 0.03), fog);

        // Atmospheric scattering
        col += vec3(0.03, 0.04, 0.05) * fog * 0.5;
    }

    // Distance fog for everything
    float globalFog = 1.0 - exp(-t * 0.02);
    col = mix(col, vec3(0.01), globalFog * 0.3);

    // Subtle vignette
    float vignette = 1.0 - dot(fragUV - 0.5, fragUV - 0.5) * 0.5;
    col *= vignette;

    // Tone mapping and gamma
    col = col / (1.0 + col); // Reinhard tone mapping
    col = pow(col, vec3(1.0 / 2.2)); // Gamma correction

    outColor = vec4(col, 1.0);
}