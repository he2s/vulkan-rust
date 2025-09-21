#version 450

// === INPUTS FROM APPLICATION ===
// These values come from your audio/MIDI application
layout(push_constant) uniform PushConstants {
    float time;           // Current time in seconds
    uint  mouse_x;        // Mouse X position
    uint  mouse_y;        // Mouse Y position
    uint  mouse_pressed;  // Is mouse pressed?
    float note_velocity;  // How hard a note was hit (0-1)
    float pitch_bend;     // Pitch bend wheel (-1 to 1)
    float cc1;           // MIDI controller 1 (mod wheel)
    float cc74;          // MIDI controller 74 (filter cutoff)
    uint  note_count;    // Total notes played
    uint  last_note;     // Last MIDI note number
    float osc_ch1;       // Audio oscillator channel 1
    float osc_ch2;       // Audio oscillator channel 2
    uint  render_w;      // Window width
    uint  render_h;      // Window height
} pc;

// Inputs from vertex shader
layout(location = 0) in vec2 fragUV;        // UV coordinates (0-1)
layout(location = 1) in float vertexEnergy; // Audio energy at vertex
layout(location = 2) in vec3 worldPos;      // World position

// Output color
layout(location = 0) out vec4 outColor;

// === CONSTANTS ===
#define PI 3.14159265359
#define TAU (2.0 * PI)  // Full circle in radians

// === HELPER FUNCTIONS ===

// Simple 2D rotation matrix
mat2 rot(float angle) {
    float c = cos(angle);
    float s = sin(angle);
    return mat2(c, -s, s, c);
}

// Clamp value between 0 and 1
float saturate(float x) {
    return clamp(x, 0.0, 1.0);
}

// === COLOR PALETTE ===
// Creates rainbow colors based on a single input value
vec3 getRainbow(float t) {
    // Shift the input based on audio for variation
    t = t + pc.time * 0.2 + pc.note_velocity;

    // Create RGB sine waves offset by 120 degrees
    vec3 color;
    color.r = sin(t * TAU) * 0.5 + 0.5;
    color.g = sin(t * TAU + TAU/3.0) * 0.5 + 0.5;
    color.b = sin(t * TAU + 2.0*TAU/3.0) * 0.5 + 0.5;

    return color;
}

// === SIGNED DISTANCE FUNCTIONS (SDFs) ===
// These functions return the distance from a point to a shape
// Negative = inside, Positive = outside, Zero = on surface

// Sphere SDF - simplest 3D shape
float sdSphere(vec3 p, float radius) {
    return length(p) - radius;
}

// Box SDF
float sdBox(vec3 p, vec3 size) {
    vec3 q = abs(p) - size;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}

// Torus (donut) SDF
float sdTorus(vec3 p, float bigRadius, float smallRadius) {
    vec2 q = vec2(length(p.xz) - bigRadius, p.y);
    return length(q) - smallRadius;
}

// === SDF OPERATIONS ===

// Smooth minimum - blends two shapes together
float smoothMin(float a, float b, float smoothness) {
    float h = saturate(0.5 + 0.5 * (b - a) / smoothness);
    return mix(b, a, h) - smoothness * h * (1.0 - h);
}

// === MAIN SCENE SDF ===
// This defines all the geometry in our scene
float sceneSDF(vec3 p) {
    // Get audio parameters (0 to 1 range)
    float audioEnergy = pc.note_velocity;
    float modWheel = pc.cc1;

    // === ANIMATED TRANSFORMATIONS ===

    // Create a copy of position for each object
    vec3 p1 = p;
    vec3 p2 = p;
    vec3 p3 = p;

    // === OBJECT 1: ROTATING TORUS ===
    // Rotate around Y axis based on time
    p1.xz *= rot(pc.time);
    // Rotate around X axis based on audio
    p1.yz *= rot(pc.time * 0.7 + audioEnergy);
    // Create torus with audio-reactive size
    float torus = sdTorus(p1, 1.0 + audioEnergy * 0.3, 0.2);

    // === OBJECT 2: BOUNCING SPHERE ===
    // Move sphere up and down based on time and audio
    p2.y += sin(pc.time * 2.0) * 0.5 * (1.0 + audioEnergy);
    // Offset to the right
    p2.x += 2.0;
    // Create sphere
    float sphere = sdSphere(p2, 0.4 + modWheel * 0.2);

    // === OBJECT 3: SPINNING BOX ===
    // Offset to the left
    p3.x -= 2.0;
    // Rotate based on pitch bend
    p3.xy *= rot(pc.pitch_bend * PI);
    // Rotate over time
    p3.xz *= rot(pc.time * 1.5);
    // Create box
    float box = sdBox(p3, vec3(0.3, 0.3 + audioEnergy * 0.2, 0.3));

    // === COMBINE OBJECTS ===
    // Use smooth minimum to blend shapes together
    float scene = smoothMin(torus, sphere, 0.3);
    scene = smoothMin(scene, box, 0.3);

    // === DOMAIN REPETITION (optional cool effect) ===
    // Uncomment to see infinite copies of the scene!
    // p = mod(p + 3.0, 6.0) - 3.0;
    // scene = smoothMin(scene, sdSphere(p, 0.1), 0.2);

    return scene;
}

// === CALCULATE SURFACE NORMAL ===
// The normal is perpendicular to the surface - used for lighting
vec3 getNormal(vec3 p) {
    float epsilon = 0.001;
    vec3 normal;

    // Calculate gradient by sampling nearby points
    normal.x = sceneSDF(p + vec3(epsilon, 0, 0)) - sceneSDF(p - vec3(epsilon, 0, 0));
    normal.y = sceneSDF(p + vec3(0, epsilon, 0)) - sceneSDF(p - vec3(0, epsilon, 0));
    normal.z = sceneSDF(p + vec3(0, 0, epsilon)) - sceneSDF(p - vec3(0, 0, epsilon));

    return normalize(normal);
}

// === RAY MARCHING ===
// Step along a ray until we hit something
float rayMarch(vec3 rayOrigin, vec3 rayDirection) {
    float totalDistance = 0.0;
    const int MAX_STEPS = 64;    // Maximum iterations
    const float MAX_DIST = 20.0; // Maximum distance to march
    const float HIT_DIST = 0.001; // How close to consider a hit

    for(int i = 0; i < MAX_STEPS; i++) {
        // Current position along the ray
        vec3 currentPos = rayOrigin + rayDirection * totalDistance;

        // Get distance to nearest surface
        float distance = sceneSDF(currentPos);

        // Did we hit something?
        if(distance < HIT_DIST) {
            return totalDistance; // Return how far we traveled
        }

        // Move forward by the safe distance
        totalDistance += distance;

        // Gone too far?
        if(totalDistance > MAX_DIST) {
            break;
        }
    }

    return -1.0; // Didn't hit anything
}

// === SIMPLE LIGHTING ===
vec3 calculateLighting(vec3 position, vec3 normal, vec3 viewDir) {
    // Light direction (from above and to the right)
    vec3 lightDir = normalize(vec3(1.0, 2.0, -1.0));

    // Diffuse lighting (how much the surface faces the light)
    float diffuse = max(dot(normal, lightDir), 0.0);

    // Add some ambient light so we can see everything
    float ambient = 0.2;

    // Combine lighting
    float lighting = ambient + diffuse * 0.8;

    // Simple specular highlight (shiny spots)
    vec3 reflectDir = reflect(-lightDir, normal);
    float specular = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);

    return vec3(lighting) + vec3(specular * 0.5);
}

// === RENDER A SINGLE SAMPLE ===
// Separated out so we can call it multiple times for anti-aliasing
vec3 renderSample(vec2 uv, vec3 cameraPos) {
    // Create ray direction from camera through pixel
    vec3 rayDir = normalize(vec3(uv, 1.0));

    // === RAY MARCH THE SCENE ===
    float distance = rayMarch(cameraPos, rayDir);

    // Initialize with background color
    vec3 color = vec3(0.02, 0.02, 0.05);

    // Add some background gradient based on audio
    color += getRainbow(uv.y * 0.3 + pc.osc_ch1) * 0.1;

    // Did we hit something?
    if(distance > 0.0) {
        // Calculate hit position
        vec3 hitPos = cameraPos + rayDir * distance;

        // Get surface normal
        vec3 normal = getNormal(hitPos);

        // Calculate basic lighting
        vec3 lighting = calculateLighting(hitPos, normal, -rayDir);

        // === AUDIO-REACTIVE COLORING ===
        // Base color from position and audio
        vec3 surfaceColor = getRainbow(
        length(hitPos) * 0.3 +     // Color based on distance from origin
        pc.time * 0.1 +            // Animate colors over time
        pc.note_velocity * 2.0     // Shift colors with audio
        );

        // Modulate brightness with mod wheel
        surfaceColor *= 0.5 + pc.cc1 * 0.5;

        // Apply lighting to surface color
        color = surfaceColor * lighting;

        // Add rim lighting (edges glow)
        float rim = 1.0 - abs(dot(normal, -rayDir));
        rim = pow(rim, 2.0);
        color += getRainbow(pc.time * 0.5) * rim * 0.3;

        // === SIMPLE GLOW EFFECT ===
        // Add glow based on how close we got to surfaces during marching
        float glow = 1.0 / (1.0 + distance * distance * 0.1);
        color += getRainbow(pc.time * 0.3) * glow * 0.05 * pc.cc74;
    }

    return color;
}

// === MAIN RENDERING FUNCTION ===
void main() {
    // Convert UV coordinates to centered coordinates
    vec2 resolution = vec2(float(pc.render_w), float(pc.render_h));
    vec2 baseUV = (fragUV - 0.5) * 2.0;
    baseUV.x *= resolution.x / resolution.y; // Correct aspect ratio

    // === CAMERA SETUP ===
    // Position the camera
    vec3 cameraPos = vec3(0.0, 0.0, -5.0);

    // Add mouse control for camera rotation
    if(pc.mouse_pressed > 0u) {
        float mouseX = (float(pc.mouse_x) / resolution.x - 0.5) * TAU;
        float mouseY = (float(pc.mouse_y) / resolution.y - 0.5) * PI;

        // Rotate camera position around origin
        cameraPos.xz *= rot(mouseX);
        cameraPos.yz *= rot(mouseY);
    }

    // === ANTI-ALIASING ===
    // We'll sample the scene multiple times with slight offsets
    // This smooths out jagged edges (aliasing)

    vec3 color = vec3(0.0);

    // Choose AA quality (higher = smoother but slower)
    // 1 = no AA, 2 = 4x AA, 3 = 9x AA, 4 = 16x AA
    const int AA_SAMPLES = 2; // 4x anti-aliasing

    // Calculate pixel size for offset
    float pixelSize = 1.0 / resolution.y;

    // Sample multiple points within the pixel
    for(int x = 0; x < AA_SAMPLES; x++) {
        for(int y = 0; y < AA_SAMPLES; y++) {
            // Calculate sub-pixel offset
            vec2 offset = vec2(float(x), float(y)) / float(AA_SAMPLES) - 0.5;
            offset *= pixelSize;

            // Sample with offset
            vec2 sampleUV = baseUV + offset;

            // Accumulate color from this sample
            color += renderSample(sampleUV, cameraPos);
        }
    }

    // Average all samples
    color /= float(AA_SAMPLES * AA_SAMPLES);

    // === POST PROCESSING ===

    // Vignette (darken edges)
    float vignette = 1.0 - length(baseUV) * 0.4;
    color *= vignette;

    // Audio energy flash
    if(pc.note_velocity > 0.7) {
        color += vec3(0.1, 0.2, 0.3) * (pc.note_velocity - 0.7);
    }

    // Make sure colors stay in valid range
    color = clamp(color, 0.0, 1.0);

    // Output final color
    outColor = vec4(color, 1.0);
}