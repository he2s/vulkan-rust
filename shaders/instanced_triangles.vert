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
    float bpm;
    float time_to_next_beat;
    uint max_points;
    uint fft_size;
    float audio_intensity;
    float bass_level;
    float mid_level;
    float high_level;
} pc;

// Instance data from compute shader
struct PointData {
    vec2 position;
    float size;
    float intensity;
    vec4 color;
    float rotation;
    uint type;
    vec2 velocity;
};

layout(set = 0, binding = 0, std430) restrict readonly buffer PointBuffer {
    PointData points[];
};

// Vertex attributes (base triangle shape)
layout(location = 0) in vec2 vertex_pos;
layout(location = 1) in vec2 vertex_uv;

// Outputs to fragment shader
layout(location = 0) out vec2 frag_uv;
layout(location = 1) out vec4 frag_color;
layout(location = 2) out float frag_intensity;
layout(location = 3) out float frag_distance_from_center;

// Rotation matrix
mat2 rotate(float angle) {
    float c = cos(angle);
    float s = sin(angle);
    return mat2(c, -s, s, c);
}

// Generate different shapes based on type
vec2 get_shape_vertex(uint shape_type, vec2 base_vertex) {
    if (shape_type == 0) {
        // Triangle - use as-is
        return base_vertex;
    } else if (shape_type == 1) {
        // Quad - stretch the triangle into a quad-like shape
        return base_vertex * 1.2;
    } else if (shape_type == 2) {
        // Star - create pointed shape
        float angle = atan(base_vertex.y, base_vertex.x);
        float radius = length(base_vertex);
        float star_factor = 1.0 + 0.3 * sin(angle * 5.0);
        return base_vertex * star_factor;
    } else {
        // Default to triangle
        return base_vertex;
    }
}

void main() {
    uint instance_id = gl_InstanceIndex;

    // Safety check
    if (instance_id >= pc.max_points) {
        gl_Position = vec4(0.0);
        return;
    }

    // Get instance data
    PointData point = points[instance_id];

    // Get the shaped vertex
    vec2 shaped_vertex = get_shape_vertex(point.type, vertex_pos);

    // Apply rotation
    vec2 rotated_vertex = rotate(point.rotation) * shaped_vertex;

    // Scale by size
    vec2 scaled_vertex = rotated_vertex * point.size;

    // Translate to final position
    vec2 final_pos = point.position + scaled_vertex;

    // Output final position
    gl_Position = vec4(final_pos, 0.0, 1.0);

    // Pass data to fragment shader
    frag_uv = vertex_uv;
    frag_color = point.color;
    frag_intensity = point.intensity;
    frag_distance_from_center = length(scaled_vertex);
}