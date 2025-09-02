#version 450

// Push constants
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
} pc;

layout(location = 0) in vec2 frag_uv;
layout(location = 1) in vec2 frag_screen_pos;

layout(location = 0) out vec4 out_color;

const float PI = 3.14159265359;
const float TAU = PI * 2.0;

// === MATHEMATICAL HASH FUNCTIONS ===

float hash(float n) { return fract(sin(n) * 71234.5678); }
float hash2(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }

vec3 hash3(vec3 p) {
    p = fract(p * vec3(443.897, 441.423, 437.195));
    p += dot(p, p.yzx + 19.19);
    return fract((p.xxy + p.yzz) * p.zyx);
}

// === COMPLEX NUMBER OPERATIONS ===

vec2 complexMul(vec2 a, vec2 b) {
    return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

vec2 complexDiv(vec2 a, vec2 b) {
    float denom = dot(b, b);
    return vec2(dot(a, b), a.y * b.x - a.x * b.y) / denom;
}

vec2 complexExp(vec2 z) {
    return exp(z.x) * vec2(cos(z.y), sin(z.y));
}

// === FRACTAL FUNCTIONS ===

float mandelbulb(vec3 p, float power) {
    vec3 z = p;
    float dr = 1.0;
    float r = 0.0;

    for (int i = 0; i < 8; i++) {
        r = length(z);
        if (r > 2.0) break;

        float theta = acos(z.z / r);
        float phi = atan(z.y, z.x);
        dr = pow(r, power - 1.0) * power * dr + 1.0;

        float zr = pow(r, power);
        theta = theta * power;
        phi = phi * power;

        z = zr * vec3(sin(theta) * cos(phi), sin(phi) * sin(theta), cos(theta));
        z += p;
    }

    return 0.5 * log(r) * r / dr;
}

// === KALEIDOSCOPE TRANSFORM ===

vec2 kaleidoscope(vec2 p, float segments) {
    float angle = atan(p.y, p.x);
    float radius = length(p);

    float segment_angle = TAU / segments;
    angle = mod(angle, segment_angle);
    angle = abs(angle - segment_angle * 0.5);

    return radius * vec2(cos(angle), sin(angle));
}

// === CRYSTALLINE STRUCTURES ===

float crystal(vec3 p, float size) {
    p = abs(p);

    // Octahedron distance
    float octa = (p.x + p.y + p.z - size) * 0.57735027;

    // Box distance for cube crystals
    vec3 box = abs(p) - size;
    float cube = length(max(box, 0.0)) + min(max(box.x, max(box.y, box.z)), 0.0);

    // Combine for complex crystal
    return min(octa, cube);
}

float crystallineField(vec3 p, float time) {
    float result = 1000.0;

    // Multiple crystal layers
    for (int i = 0; i < 5; i++) {
        vec3 offset = hash3(vec3(float(i))) * 2.0 - 1.0;
        float size = 0.1 + float(i) * 0.05;

        vec3 crystal_p = p + offset * 0.5;

        // Rotation based on time and layer
        float rot = time * (0.5 + float(i) * 0.1);
        crystal_p.xz = mat2(cos(rot), -sin(rot), sin(rot), cos(rot)) * crystal_p.xz;

        float dist = crystal(crystal_p, size);
        result = min(result, dist);
    }

    return result;
}

// === VORONOI CELLS ===

vec3 voronoi(vec2 p) {
    vec2 n = floor(p);
    vec2 f = fract(p);

    float min_dist = 1.0;
    vec2 min_point = vec2(0.0);
    float second_dist = 1.0;

    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            vec2 neighbor = vec2(float(i), float(j));
            vec2 point = hash2(n + neighbor) * vec2(1.0);

            float dist = length(f - neighbor - point);

            if (dist < min_dist) {
                second_dist = min_dist;
                min_dist = dist;
                min_point = n + neighbor + point;
            } else if (dist < second_dist) {
                second_dist = dist;
            }
        }
    }

    return vec3(min_dist, second_dist - min_dist, hash2(min_point));
}

// === IRIDESCENT COLORING ===

vec3 iridescence(float angle, float thickness) {
    // Thin film interference simulation
    vec3 color = vec3(0.0);

    for (int i = 0; i < 3; i++) {
        float wavelength = 0.4 + float(i) * 0.1; // R, G, B wavelengths
        float interference = cos(thickness * TAU / wavelength + angle * float(i));
        color[i] = interference * 0.5 + 0.5;
    }

    return color;
}

// === JULIA SET FRACTAL ===

float julia(vec2 p, vec2 c, int iterations) {
    vec2 z = p;
    float escaped = 0.0;

    for (int i = 0; i < iterations; i++) {
        z = complexMul(z, z) + c;

        if (dot(z, z) > 4.0) {
            escaped = float(i) / float(iterations);
            break;
        }
    }

    return escaped;
}

void main() {
    vec2 fragCoord = frag_uv * vec2(float(pc.render_w), float(pc.render_h));
    vec2 resolution = vec2(float(pc.render_w), float(pc.render_h));

    // Audio-reactive parameters
    float audio_energy = (pc.note_velocity + pc.cc1 + pc.cc74 + pc.osc_ch1 + pc.osc_ch2) / 5.0;
    float crystal_time = pc.time * (0.3 + audio_energy * 0.2);

    // Centered UV coordinates
    vec2 uv = (fragCoord - resolution * 0.5) / resolution.y;

    // Mouse position
    vec2 mouse_pos = vec2(float(pc.mouse_x) / float(pc.render_w),
    float(pc.mouse_y) / float(pc.render_h)) * 2.0 - 1.0;
    mouse_pos.x *= float(pc.render_w) / float(pc.render_h);

    // === KALEIDOSCOPE TRANSFORM ===

    float kaleido_segments = 6.0 + floor(pc.cc1 * 10.0);
    vec2 kaleido_uv = kaleidoscope(uv, kaleido_segments);

    // Mouse pressed rotates kaleidoscope
    if (pc.mouse_pressed != 0) {
        float mouse_angle = atan(mouse_pos.y, mouse_pos.x);
        kaleido_uv = mat2(cos(mouse_angle), -sin(mouse_angle),
        sin(mouse_angle), cos(mouse_angle)) * kaleido_uv;
    }

    // === FRACTAL LAYERS ===

    vec3 fractal_color = vec3(0.0);

    // Layer 1: Julia set base
    vec2 julia_c = vec2(sin(crystal_time * 0.7) * 0.4, cos(crystal_time * 0.5) * 0.4);

    // MIDI notes affect Julia constant
    if (pc.note_count > 0) {
        float note_norm = float(pc.last_note) / 127.0;
        julia_c += vec2(sin(note_norm * TAU), cos(note_norm * TAU)) * 0.3 * pc.note_velocity;
    }

    // Pitch bend warps the fractal
    julia_c += vec2(pc.pitch_bend * 0.5, 0.0);

    float julia_val = julia(kaleido_uv * 2.0, julia_c, 50);

    // Layer 2: Mandelbulb in 3D
    vec3 ray_origin = vec3(kaleido_uv * 2.0, -2.0);
    vec3 ray_dir = normalize(vec3(kaleido_uv * 0.5, 1.0));

    float bulb_power = 8.0 + sin(crystal_time) * 2.0 * (1.0 + audio_energy);

    // Raymarch the mandelbulb
    float depth = 0.0;
    for (int i = 0; i < 32; i++) {
        vec3 p = ray_origin + ray_dir * depth;

        // Rotation based on OSC channels
        if (pc.osc_ch1 > 0.01) {
            float rot1 = crystal_time * pc.osc_ch1;
            p.xy = mat2(cos(rot1), -sin(rot1), sin(rot1), cos(rot1)) * p.xy;
        }
        if (pc.osc_ch2 > 0.01) {
            float rot2 = -crystal_time * pc.osc_ch2 * 1.3;
            p.yz = mat2(cos(rot2), -sin(rot2), sin(rot2), cos(rot2)) * p.yz;
        }

        float dist = mandelbulb(p, bulb_power);

        if (dist < 0.001) {
            float ao = 1.0 - float(i) / 32.0;
            fractal_color += vec3(0.2, 0.3, 0.5) * ao;
            break;
        }

        depth += dist * 0.8;
        if (depth > 5.0) break;
    }

    // === CRYSTALLINE OVERLAY ===

    vec3 crystal_color = vec3(0.0);

    // Voronoi cells for crystal facets
    vec3 vor = voronoi(kaleido_uv * 10.0 * (1.0 + pc.cc74));
    float facet = vor.x;
    float edge = vor.y;
    float cell_id = vor.z;

    // Crystal field distance
    vec3 crystal_p = vec3(kaleido_uv * 3.0, sin(crystal_time) * 0.5);
    float crystal_dist = crystallineField(crystal_p, crystal_time);

    // Iridescent coloring based on viewing angle
    float view_angle = atan(uv.y, uv.x) + crystal_time * 0.2;
    float thickness = (0.5 + 0.5 * sin(cell_id * 100.0)) * (1.0 + audio_energy);
    vec3 irid_color = iridescence(view_angle, thickness);

    // Combine crystal effects
    crystal_color = irid_color * (1.0 - facet * 2.0);
    crystal_color += vec3(1.0) * pow(edge, 3.0) * 0.5; // Edge highlights
    crystal_color *= exp(-crystal_dist * 5.0);

    // === ENERGY PARTICLES ===

    vec3 particle_color = vec3(0.0);

    // Spawn particles from MIDI notes
    for (uint i = 0u; i < min(pc.note_count, 10u); i++) {
        float particle_id = float(i) + float(pc.last_note) * 0.1;
        vec3 particle_pos = hash3(vec3(particle_id)) * 2.0 - 1.0;

        // Orbital motion
        float orbit_radius = 0.5 + hash(particle_id) * 0.3;
        float orbit_speed = 1.0 + hash(particle_id + 1.0) * 2.0;
        particle_pos.xy = orbit_radius * vec2(
        cos(crystal_time * orbit_speed + particle_id),
        sin(crystal_time * orbit_speed + particle_id)
        );

        float dist = length(vec3(kaleido_uv, 0.0) - particle_pos);
        float glow = exp(-dist * dist * 20.0) * pc.note_velocity;

        vec3 particle_hue = hash3(vec3(particle_id + 10.0));
        particle_color += particle_hue * glow * 2.0;
    }

    // === FINAL COMPOSITION ===

    vec3 final_color = vec3(0.0);

    // Background gradient
    vec3 bg_inner = vec3(0.02, 0.01, 0.03);
    vec3 bg_outer = vec3(0.0, 0.0, 0.01);
    final_color = mix(bg_inner, bg_outer, length(uv));

    // Add Julia set as base layer
    final_color += vec3(0.1, 0.2, 0.4) * julia_val;

    // Add mandelbulb
    final_color += fractal_color;

    // Add crystalline structures
    final_color += crystal_color * 0.7;

    // Add energy particles
    final_color += particle_color;

    // CC74 controls color richness
    float saturation = 0.5 + pc.cc74 * 0.5;
    vec3 gray = vec3(dot(final_color, vec3(0.299, 0.587, 0.114)));
    final_color = mix(gray, final_color, saturation);

    // Radial vignette with crystalline edge
    float vignette = 1.0 - length(uv) * 0.5;
    vignette = pow(vignette, 2.0);
    final_color *= vignette;

    // Subtle chromatic aberration at edges
    float aberration = length(uv) * 0.02;
    final_color.r *= 1.0 + aberration;
    final_color.b *= 1.0 - aberration;

    // Ensure minimum visibility
    final_color = max(final_color, vec3(0.001, 0.001, 0.002));

    out_color = vec4(final_color, 1.0);
}