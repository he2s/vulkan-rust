#version 450

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
} pc;

layout(location = 0) in vec2 fragCoord;
layout(location = 0) out vec4 fragColor;

// Simple sphere distance function
float sphereSDF(vec3 p, float radius) {
    return length(p) - radius;
}

// Basic lighting
vec3 calculateNormal(vec3 p, float radius) {
    const float eps = 0.001;
    return normalize(vec3(
        sphereSDF(p + vec3(eps, 0, 0), radius) - sphereSDF(p - vec3(eps, 0, 0), radius),
        sphereSDF(p + vec3(0, eps, 0), radius) - sphereSDF(p - vec3(0, eps, 0), radius),
        sphereSDF(p + vec3(0, 0, eps), radius) - sphereSDF(p - vec3(0, 0, eps), radius)
    ));
}

void main() {
    vec2 uv = fragCoord;

    // Camera setup
    vec3 ro = vec3(0.0, 0.0, 3.0); // Ray origin
    vec3 rd = normalize(vec3(uv, -1.0)); // Ray direction

    // Sphere properties - react to audio/MIDI
    float baseRadius = 0.8;
    float pulseRadius = baseRadius + pc.note_velocity * 0.3; // React to MIDI velocity
    pulseRadius += pc.cc1 * 0.2; // React to audio mid frequencies

    // Sphere position - slight movement based on pitch bend
    vec3 spherePos = vec3(pc.pitch_bend * 0.3, sin(pc.time * 2.0) * 0.1, 0.0);

    // Raymarching
    float t = 0.0;
    const int MAX_STEPS = 64;
    const float MAX_DIST = 100.0;
    const float EPSILON = 0.001;

    for(int i = 0; i < MAX_STEPS; i++) {
        vec3 p = ro + rd * t;
        float d = sphereSDF(p - spherePos, pulseRadius);

        if(d < EPSILON || t > MAX_DIST) break;
        t += d;
    }

    vec3 color = vec3(0.0);

    if(t < MAX_DIST) {
        // Hit the sphere
        vec3 p = ro + rd * t;
        vec3 normal = calculateNormal(p - spherePos, pulseRadius);

        // Basic Phong lighting
        vec3 lightPos = vec3(2.0, 2.0, 2.0);
        vec3 lightDir = normalize(lightPos - p);
        vec3 viewDir = normalize(-rd);
        vec3 reflectDir = reflect(-lightDir, normal);

        // Base color - react to audio frequencies
        vec3 baseColor = vec3(
            0.3 + pc.cc74 * 0.7,  // High frequencies affect red
            0.5 + pc.cc1 * 0.5,   // Mid frequencies affect green
            0.8 + sin(pc.time + pc.note_velocity * 10.0) * 0.2 // Blue with time + MIDI
        );

        // Diffuse lighting
        float diff = max(dot(normal, lightDir), 0.0);

        // Specular lighting
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);

        // Rim lighting based on note count
        float rim = 1.0 - max(dot(viewDir, normal), 0.0);
        rim = pow(rim, 2.0) * (pc.note_count > 0u ? 1.0 : 0.3);

        color = baseColor * diff + vec3(1.0) * spec + vec3(0.2, 0.4, 1.0) * rim;

        // Add some glow based on audio level
        color += baseColor * pc.note_velocity * 0.5;
    }
    else {
        // Background - simple gradient
        color = mix(
            vec3(0.1, 0.1, 0.2),
            vec3(0.0, 0.0, 0.1),
            length(uv) * 0.5
        );

        // Add stars based on audio
        float stars = step(0.99, sin(uv.x * 100.0) * cos(uv.y * 100.0));
        color += stars * pc.cc74 * vec3(1.0);
    }

    fragColor = vec4(color, 1.0);
}