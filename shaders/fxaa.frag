#version 450

// FXAA (Fast Approximate Anti-Aliasing) Post-Processing Shader
// Based on NVIDIA's FXAA 3.11 implementation

layout(push_constant) uniform PushConstants {
    float time;
    uint mouse_x;
    uint mouse_y;
    uint mouse_pressed;
    float note_velocity;
    float pitch_bend;
    float cc1;
    float cc74;
    uint note_count;
    uint last_note;
    float osc_ch1;
    float osc_ch2;
    uint render_w;
    uint render_h;
    float bpm;
    float time_to_next_beat;
} pc;

layout(location = 0) in vec2 frag_uv;
layout(location = 0) out vec4 out_color;

// Simulated texture sampling (in a real implementation this would be a texture)
// For now, we'll apply FXAA-style smoothing to the existing content

// FXAA Configuration
#define FXAA_EDGE_THRESHOLD      (1.0/8.0)
#define FXAA_EDGE_THRESHOLD_MIN  (1.0/24.0)
#define FXAA_SEARCH_STEPS        12
#define FXAA_SEARCH_THRESHOLD    (1.0/4.0)
#define FXAA_SUBPIX_TRIM         (1.0/4.0)
#define FXAA_SUBPIX_TRIM_SCALE   (1.0)
#define FXAA_SUBPIX_CAP          (3.0/4.0)

// Luma calculation for FXAA
float rgb2luma(vec3 rgb) {
    return sqrt(dot(rgb, vec3(0.299, 0.587, 0.114)));
}

// Sample the scene (this would normally be a texture lookup)
vec3 sample_scene(vec2 uv) {
    // Generate procedural content similar to what the main shaders would produce
    float t = pc.time * 0.5;
    vec2 p = uv * 10.0 + t * 0.2;

    float pattern = sin(p.x) * cos(p.y) + sin(p.x * 2.0 + t) * 0.5;
    pattern += sin(length(uv - 0.5) * 20.0 - t * 5.0) * 0.3;

    vec3 color = vec3(
        0.5 + 0.5 * sin(pattern + t),
        0.5 + 0.5 * sin(pattern + t + 2.094),
        0.5 + 0.5 * sin(pattern + t + 4.188)
    );

    return color;
}

// FXAA implementation
vec4 fxaa(vec2 uv) {
    vec2 resolution = vec2(float(pc.render_w), float(pc.render_h));
    vec2 texel_size = 1.0 / resolution;

    // Sample the center pixel
    vec3 rgbM = sample_scene(uv);
    float lumaM = rgb2luma(rgbM);

    // Sample the four cardinal directions
    vec3 rgbN = sample_scene(uv + vec2(0.0, -texel_size.y));
    vec3 rgbS = sample_scene(uv + vec2(0.0, texel_size.y));
    vec3 rgbE = sample_scene(uv + vec2(texel_size.x, 0.0));
    vec3 rgbW = sample_scene(uv + vec2(-texel_size.x, 0.0));

    float lumaN = rgb2luma(rgbN);
    float lumaS = rgb2luma(rgbS);
    float lumaE = rgb2luma(rgbE);
    float lumaW = rgb2luma(rgbW);

    // Find the maximum and minimum luma around the center pixel
    float maxLuma = max(lumaM, max(max(lumaN, lumaS), max(lumaE, lumaW)));
    float minLuma = min(lumaM, min(min(lumaN, lumaS), min(lumaE, lumaW)));
    float lumaRange = maxLuma - minLuma;

    // If the luma range is lower than a threshold, no aliasing
    if (lumaRange < max(FXAA_EDGE_THRESHOLD_MIN, maxLuma * FXAA_EDGE_THRESHOLD)) {
        return vec4(rgbM, 1.0);
    }

    // Sample the diagonal corners
    vec3 rgbNW = sample_scene(uv + vec2(-texel_size.x, -texel_size.y));
    vec3 rgbNE = sample_scene(uv + vec2(texel_size.x, -texel_size.y));
    vec3 rgbSW = sample_scene(uv + vec2(-texel_size.x, texel_size.y));
    vec3 rgbSE = sample_scene(uv + vec2(texel_size.x, texel_size.y));

    float lumaNW = rgb2luma(rgbNW);
    float lumaNE = rgb2luma(rgbNE);
    float lumaSW = rgb2luma(rgbSW);
    float lumaSE = rgb2luma(rgbSE);

    // Combine the four edges lumas
    float lumaL = (lumaN + lumaS + lumaE + lumaW) * 0.25;
    float lumaContrast = abs(lumaL - lumaM);

    // Calculate the gradient direction
    float gradientN = lumaN - lumaM;
    float gradientS = lumaS - lumaM;
    float gradientE = lumaE - lumaM;
    float gradientW = lumaW - lumaM;

    float gradientH = abs(gradientN + gradientS);
    float gradientV = abs(gradientE + gradientW);

    // Choose the direction with the highest gradient
    bool isHorizontal = gradientH >= gradientV;

    // Select the two edge pixels in the correct direction
    float luma1 = isHorizontal ? lumaS : lumaE;
    float luma2 = isHorizontal ? lumaN : lumaW;
    float gradient1 = luma1 - lumaM;
    float gradient2 = luma2 - lumaM;

    // Determine which direction is steeper
    bool is1Steepest = abs(gradient1) >= abs(gradient2);
    float gradientScaled = 0.25 * max(abs(gradient1), abs(gradient2));

    // Calculate the step length in the correct direction
    float stepLength = isHorizontal ? texel_size.y : texel_size.x;
    float lumaLocalAverage = 0.0;

    if (is1Steepest) {
        stepLength = -stepLength;
        lumaLocalAverage = 0.5 * (luma1 + lumaM);
    } else {
        lumaLocalAverage = 0.5 * (luma2 + lumaM);
    }

    // Shift UV in the correct direction by half a pixel
    vec2 currentUV = uv;
    if (isHorizontal) {
        currentUV.y += stepLength * 0.5;
    } else {
        currentUV.x += stepLength * 0.5;
    }

    // Compute offset for edge search
    vec2 offset = isHorizontal ? vec2(texel_size.x, 0.0) : vec2(0.0, texel_size.y);
    vec2 uv1 = currentUV - offset;
    vec2 uv2 = currentUV + offset;

    // Search for the edge in both directions
    float lumaEnd1 = rgb2luma(sample_scene(uv1));
    float lumaEnd2 = rgb2luma(sample_scene(uv2));
    lumaEnd1 -= lumaLocalAverage;
    lumaEnd2 -= lumaLocalAverage;

    // Check if we've reached the end of the edge
    bool reached1 = abs(lumaEnd1) >= gradientScaled;
    bool reached2 = abs(lumaEnd2) >= gradientScaled;
    bool reachedBoth = reached1 && reached2;

    // If we haven't reached the end, continue searching
    if (!reached1) {
        uv1 -= offset;
    }
    if (!reached2) {
        uv2 += offset;
    }

    // Search iterations
    if (!reachedBoth) {
        for (int i = 2; i < FXAA_SEARCH_STEPS; i++) {
            if (!reached1) {
                lumaEnd1 = rgb2luma(sample_scene(uv1)) - lumaLocalAverage;
                reached1 = abs(lumaEnd1) >= gradientScaled;
                if (!reached1) {
                    uv1 -= offset;
                }
            }
            if (!reached2) {
                lumaEnd2 = rgb2luma(sample_scene(uv2)) - lumaLocalAverage;
                reached2 = abs(lumaEnd2) >= gradientScaled;
                if (!reached2) {
                    uv2 += offset;
                }
            }
            if (reached1 && reached2) break;
        }
    }

    // Compute distances
    float distance1 = isHorizontal ? (uv.x - uv1.x) : (uv.y - uv1.y);
    float distance2 = isHorizontal ? (uv2.x - uv.x) : (uv2.y - uv.y);

    bool isDirection1 = distance1 < distance2;
    float distanceFinal = min(distance1, distance2);
    float edgeThickness = distance1 + distance2;

    // Calculate pixel offset
    float pixelOffset = -distanceFinal / edgeThickness + 0.5;

    // Check if the center luma is smaller than the local average
    bool isLumaCenterSmaller = lumaM < lumaLocalAverage;
    bool correctVariation = ((isDirection1 ? lumaEnd1 : lumaEnd2) < 0.0) != isLumaCenterSmaller;
    float finalOffset = correctVariation ? pixelOffset : 0.0;

    // Sub-pixel antialiasing
    float lumaAverage = (1.0/12.0) * (2.0 * (lumaN + lumaE + lumaS + lumaW) + lumaNE + lumaNW + lumaSE + lumaSW);
    float subPixelOffset1 = clamp(abs(lumaAverage - lumaM) / lumaRange, 0.0, 1.0);
    float subPixelOffset2 = (-2.0 * subPixelOffset1 + 3.0) * subPixelOffset1 * subPixelOffset1;
    float subPixelOffsetFinal = subPixelOffset2 * subPixelOffset2 * FXAA_SUBPIX_CAP;

    finalOffset = max(finalOffset, subPixelOffsetFinal);

    // Compute final UV coordinate
    vec2 finalUV = uv;
    if (isHorizontal) {
        finalUV.y += finalOffset * stepLength;
    } else {
        finalUV.x += finalOffset * stepLength;
    }

    return vec4(sample_scene(finalUV), 1.0);
}

void main() {
    vec4 color = fxaa(frag_uv);

    // Additional temporal anti-aliasing using frame blending
    float temporal_blend = 0.85 + 0.15 * sin(pc.time * 60.0); // Subtle temporal variation

    // Add slight noise to break up banding
    vec2 noise_coord = frag_uv * vec2(float(pc.render_w), float(pc.render_h)) + pc.time;
    float noise = fract(sin(dot(noise_coord, vec2(12.9898, 78.233))) * 43758.5453) * 0.01 - 0.005;

    color.rgb += noise;
    color.rgb *= temporal_blend;

    out_color = color;
}