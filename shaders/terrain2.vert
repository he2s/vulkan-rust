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
    float osc_ch1;
    float osc_ch2;
    uint render_w;
    uint render_h;
} pc;

layout(location = 0) out vec2 fragCoord;
layout(location = 1) out float dimension;
layout(location = 2) out vec2 uv;

const float PI = 3.14159265359;
const float PHI = 1.618033988749;

// 4D rotation matrix projection to 2D
vec2 rotate4D(vec2 p, float t) {
    // Simulate 4D rotation projected down to 2D
    float a = t * 0.7;
    float b = t * 1.3;
    float c = t * 0.9;

    // Clifford attractor-inspired transformation
    float x_new = sin(a * p.y) + c * cos(a * p.x);
    float y_new = sin(b * p.x) + c * cos(b * p.y);

    return vec2(x_new, y_new) * 0.5;
}

// Fractal kaleidoscope symmetry
vec2 kaleidoscope(vec2 p, int symmetry) {
    float angle = atan(p.y, p.x);
    float radius = length(p);

    // Create kaleidoscope symmetry
    float segment = PI * 2.0 / float(symmetry);
    angle = mod(angle, segment);
    if (angle > segment * 0.5) {
        angle = segment - angle;
    }

    // Apply fractal scaling
    radius = pow(radius, 1.0 + sin(pc.time * 0.01) * 0.3);

    return vec2(cos(angle), sin(angle)) * radius;
}

void main() {
    vec2 vertices[3] = vec2[3](
    vec2(-1.0, -1.0),
    vec2( 3.0, -1.0),
    vec2(-1.0,  3.0)
    );

    vec2 base_pos = vertices[gl_VertexIndex];
    uv = base_pos * 0.5 + 0.5;

    // HYPERDIMENSIONAL PROJECTION
    vec2 hyper_pos = base_pos;

    // Iterate through multiple dimensional projections
    for (int dim = 0; dim < 4; dim++) {
        float phase = float(dim) * PHI + pc.time * 0.01 * (0.3 + float(dim) * 0.1);
        hyper_pos = rotate4D(hyper_pos, phase);

        // MIDI modulates dimensional depth
        float depth = pc.note_velocity * float(dim) * 0.1;
        hyper_pos *= 1.0 + depth;
    }

    // KALEIDOSCOPE MIRRORS
    int symmetry = 3 + int(pc.cc1 * 13.0); // 3 to 16 fold symmetry
    vec2 kaleid_pos = kaleidoscope(base_pos, symmetry);

    // STRANGE ATTRACTOR INFLUENCE
    vec2 attractor = vec2(0.0);
    vec2 p = base_pos;
    for (int i = 0; i < 5; i++) {
        // Hénon map variation
        float x_new = 1.0 - 1.4 * p.x * p.x + p.y;
        float y_new = 0.3 * p.x;
        p = vec2(x_new, y_new) * 0.5;
        attractor += p * pow(0.5, float(i));
    }
    attractor *= pc.cc74 * 0.3;

    // TESSELLATION WARPING
    vec2 tessellation = vec2(0.0);
    float hex_x = base_pos.x * 3.464; // sqrt(3) * 2
    float hex_y = base_pos.y * 3.0;

    // Hexagonal grid coordinates
    float hx = floor(hex_x);
    float hy = floor(hex_y / 1.5);

    // Warp based on MIDI notes
    float note_warp = float(pc.last_note) / 127.0;
    tessellation = vec2(
    sin(hx * PHI + pc.time + note_warp * PI),
    cos(hy * PHI - pc.time + note_warp * PI)
    ) * 0.1 * pc.osc_ch1;

    // IMPOSSIBLE GEOMETRY: Penrose tiling influence
    float penrose_angle = atan(base_pos.y, base_pos.x);
    float penrose_r = length(base_pos);

    // Five-fold symmetry with golden ratio scaling
    for (int i = 0; i < 5; i++) {
        float angle = penrose_angle + float(i) * PI * 2.0 / 5.0;
        vec2 star_point = vec2(cos(angle), sin(angle)) * penrose_r;

        // Golden ratio spiral
        star_point *= pow(PHI, float(i) * 0.2 - pc.time * 0.1);
        tessellation += star_point * 0.05 * pc.osc_ch2;
    }

    // VERTEX ID MAGIC: Each vertex follows different dimensional rules
    float vertex_magic = float(gl_VertexIndex);
    vec2 magic_offset = vec2(
    sin(vertex_magic * PHI * 10.0 + pc.time * 2.0),
    cos(vertex_magic * PI * 7.0 - pc.time * 1.5)
    );

    // Make it respond to number of notes playing
    magic_offset *= float(pc.note_count) * 0.02;

    // MOUSE CREATES DIMENSIONAL RIFTS
    vec2 mouse = vec2(float(pc.mouse_x), float(pc.mouse_y)) / vec2(float(pc.render_w), float(pc.render_h));
    mouse = mouse * 2.0 - 1.0;

    vec2 rift = vec2(0.0);
    if (pc.mouse_pressed > 0) {
        vec2 to_mouse = mouse - base_pos;
        float rift_dist = length(to_mouse);

        // Create swirling dimensional rift
        float rift_angle = atan(to_mouse.y, to_mouse.x);
        float rift_swirl = rift_angle + 1.0 / (rift_dist + 0.1) * 2.0;

        rift = vec2(cos(rift_swirl), sin(rift_swirl)) * exp(-rift_dist) * 0.5;
    }

    // COMBINE ALL TRANSFORMATIONS
    vec2 total_displacement = hyper_pos - base_pos +
    (kaleid_pos - base_pos) * 0.3 +
    attractor +
    tessellation +
    magic_offset +
    rift;

    // Apply pitch bend as a reality-warping factor
    total_displacement = mix(total_displacement,
    total_displacement.yx * sign(pc.pitch_bend),
    abs(pc.pitch_bend));

    // Calculate dimension value for fragment shader
    dimension = length(hyper_pos) + length(attractor) * 2.0 + vertex_magic * 0.1;
    dimension = fract(dimension + pc.time * 0.0005);

    // Final position with creative clamping
    vec2 final_pos = base_pos + total_displacement * 0.6;

    // Wrap-around instead of hard clamping for more interesting visuals
    final_pos = fract(final_pos * 0.5 + 0.5) * 2.0 - 1.0;

    gl_Position = vec4(final_pos, 0.0, 1.0);
    fragCoord = final_pos;
}