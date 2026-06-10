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
    float osc_ch1;
    float osc_ch2;
    uint  render_w;
    uint  render_h;
} pc;

layout(location = 0) in vec2 fragUV;
layout(location = 1) in float vertexEnergy;
layout(location = 2) in vec3 worldPos;
layout(location = 0) out vec4 outColor;

#define TAU 6.2831853

float hash(vec2 p){ return fract(sin(dot(p, vec2(91.3, 47.7))) * 43758.5453); }

vec2 hash2(vec2 p){
    return fract(sin(vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)))) * 43758.5453);
}

// Voronoi distance to nearest cell center
vec2 voronoi(vec2 p){
    vec2 i = floor(p), f = fract(p);
    float d1 = 1e9, d2 = 1e9;
    vec2 cell = vec2(0);
    for (int y = -1; y <= 1; y++){
        for (int x = -1; x <= 1; x++){
            vec2 g = vec2(x, y);
            vec2 o = hash2(i + g);
            o = 0.5 + 0.5 * sin(pc.time * 0.7 + TAU * o);
            vec2 r = g + o - f;
            float d = dot(r, r);
            if (d < d1){ d2 = d1; d1 = d; cell = i + g; }
            else if (d < d2){ d2 = d; }
        }
    }
    return vec2(sqrt(d1), sqrt(d2) - sqrt(d1));
}

vec3 pal(float t){
    return 0.5 + 0.5 * cos(TAU * (vec3(1.0, 0.7, 0.5) * t + vec3(0.1, 0.3, 0.65)));
}

void main(){
    vec2 res = vec2(pc.render_w, pc.render_h);
    vec2 uv = (fragUV - 0.5) * vec2(res.x / res.y, 1.0) * 1.6;

    float t = pc.time * (0.5 + pc.cc1 * 0.7);

    // Multi-scale voronoi — bubbles within bubbles
    vec2 v1 = voronoi(uv * 4.0);
    vec2 v2 = voronoi(uv * 9.0 + v1.x * 0.5);
    vec2 v3 = voronoi(uv * 20.0 + v2.x * 0.3);

    // Cell edges (where d2 - d1 is small)
    float edge1 = smoothstep(0.15, 0.0, v1.y);
    float edge2 = smoothstep(0.10, 0.0, v2.y);
    float edge3 = smoothstep(0.06, 0.0, v3.y);

    // Bubble interiors brighten where d1 is small
    float bub1 = smoothstep(0.5, 0.0, v1.x);
    float bub2 = smoothstep(0.3, 0.0, v2.x);

    // Color — chaotic per cell
    vec3 col = pal(v1.x * 1.2 + t * 0.1);
    col = mix(col, pal(v2.x + t * 0.2 + 0.3), 0.5);

    col += pal(t + v3.x) * edge1 * 0.6;
    col += pal(t * 0.5 + v2.x) * edge2 * 0.4;
    col += pal(t * 0.7) * edge3 * 0.25;

    col *= 0.4 + bub1 * 0.5 + bub2 * 0.3;

    // Random "particle pops" — when voronoi cell suddenly brightens
    float pop = hash(floor(uv * 12.0) + floor(t * 4.0));
    pop = step(0.98, pop);
    col += vec3(1.0, 0.9, 0.7) * pop * pc.note_velocity * 2.0;

    // Glitch tearing
    float tear = step(0.97, hash(vec2(floor(uv.y * 80.0), floor(t * 10.0))));
    col *= 1.0 - tear * 0.4;

    // RGB channel shift
    float shift = pc.cc74 * 0.012;
    float rR = voronoi((uv + vec2(shift, 0.0)) * 4.0).y;
    float rB = voronoi((uv - vec2(shift, 0.0)) * 4.0).y;
    col.r = mix(col.r, smoothstep(0.15, 0.0, rR), 0.3);
    col.b = mix(col.b, smoothstep(0.15, 0.0, rB), 0.3);

    // Vignette
    col *= 1.0 - 0.3 * dot(uv * 0.5, uv * 0.5);

    outColor = vec4(col, 1.0);
}
