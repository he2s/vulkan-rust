// src/utils.rs

use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct PushConstants {
    pub time: f32,
    pub mouse_x: f32,
    pub mouse_y: f32,
    pub mouse_pressed: f32,
    pub resolution_x: f32,
    pub resolution_y: f32,
    pub note_velocity: f32,
    pub pitch_bend: f32,
    pub modulation: f32,
    pub expression: f32,
    pub note_count: f32,
    pub last_note: f32,
}

// ===== SHADER FILES =====
// Create these in a shaders/ directory in your project

// shaders/fullscreen.vert
pub const FULLSCREEN_VERT: &str = r#"
#version 450

layout(push_constant) uniform PushConstants {
    float time;
    float mouseX;
    float mouseY;
    float mousePressed;
    float resolutionX;
    float resolutionY;
    float noteVelocity;
    float pitchBend;
    float modulation;
    float expression;
    float noteCount;
    float lastNote;
} pc;

layout(location = 0) out vec2 fragCoord;
layout(location = 1) out vec2 resolution;

vec2 positions[3] = vec2[](
    vec2(-1.0, -1.0),
    vec2( 3.0, -1.0),
    vec2(-1.0,  3.0)
);

void main() {
    vec2 pos = positions[gl_VertexIndex];
    gl_Position = vec4(pos, 0.0, 1.0);
    fragCoord = (pos * 0.5 + 0.5) * vec2(pc.resolutionX, pc.resolutionY);
    resolution = vec2(pc.resolutionX, pc.resolutionY);
}
"#;

// shaders/torus.frag
pub const TORUS_FRAG: &str = r#"
#version 450

layout(location = 0) in vec2 fragCoord;
layout(location = 1) in vec2 resolution;

layout(location = 0) out vec4 outColor;

layout(push_constant) uniform PushConstants {
    float time;
    float mouseX;
    float mouseY;
    float mousePressed;
    float resolutionX;
    float resolutionY;
    float noteVelocity;
    float pitchBend;
    float modulation;
    float expression;
    float noteCount;
    float lastNote;
} pc;

// 4D Torus visualization
vec4 getTorus(vec3 p, float t) {
    float r1 = 2.0 + 0.5 * sin(t + pc.modulation * 3.14159);
    float r2 = 1.0 + 0.3 * pc.noteVelocity;
    
    vec2 q = vec2(length(p.xz) - r1, p.y);
    float d = length(q) - r2;
    
    vec3 col = vec3(0.5) + 0.5 * sin(vec3(0.0, 2.0, 4.0) + t + pc.lastNote * 0.1);
    col *= 1.0 + pc.noteVelocity * 0.5;
    
    return vec4(col, smoothstep(0.1, 0.0, d));
}

void main() {
    vec2 uv = (fragCoord - 0.5 * resolution) / min(resolution.x, resolution.y);
    
    float t = pc.time;
    vec3 ro = vec3(5.0 * cos(t * 0.3), 2.0, 5.0 * sin(t * 0.3));
    vec3 ta = vec3(0.0, 0.0, 0.0);
    vec3 ww = normalize(ta - ro);
    vec3 uu = normalize(cross(ww, vec3(0.0, 1.0, 0.0)));
    vec3 vv = normalize(cross(uu, ww));
    
    vec3 rd = normalize(uv.x * uu + uv.y * vv + 1.5 * ww);
    
    vec3 col = vec3(0.0);
    
    // Raymarch
    float td = 0.0;
    for(int i = 0; i < 64; i++) {
        vec3 p = ro + rd * td;
        vec4 res = getTorus(p, t);
        
        if(res.w > 0.01) {
            col = mix(col, res.rgb, res.w);
            if(res.w > 0.95) break;
        }
        
        td += 0.1;
        if(td > 20.0) break;
    }
    
    // Background gradient
    vec3 bg = mix(
        vec3(0.1, 0.0, 0.2),
        vec3(0.0, 0.1, 0.3),
        uv.y * 0.5 + 0.5
    );
    bg *= 1.0 + pc.expression * 0.3;
    
    col = mix(bg, col, step(0.01, length(col)));
    
    // Vignette
    float vignette = 1.0 - length(uv) * 0.5;
    col *= vignette;
    
    outColor = vec4(col, 1.0);
}
"#;

// shaders/terrain.frag  
pub const TERRAIN_FRAG: &str = r#"
#version 450

layout(location = 0) in vec2 fragCoord;
layout(location = 1) in vec2 resolution;

layout(location = 0) out vec4 outColor;

layout(push_constant) uniform PushConstants {
    float time;
    float mouseX;
    float mouseY;
    float mousePressed;
    float resolutionX;
    float resolutionY;
    float noteVelocity;
    float pitchBend;
    float modulation;
    float expression;
    float noteCount;
    float lastNote;
} pc;

// Simple noise function
float noise(vec2 p) {
    return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}

// Fractal brownian motion
float fbm(vec2 p) {
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;
    
    for(int i = 0; i < 6; i++) {
        value += amplitude * noise(p * frequency);
        frequency *= 2.0;
        amplitude *= 0.5;
    }
    
    return value;
}

void main() {
    vec2 uv = fragCoord / resolution;
    vec2 p = (fragCoord - 0.5 * resolution) / min(resolution.x, resolution.y);
    
    // Terrain waves
    float t = pc.time * 0.2;
    p *= 3.0 + pc.modulation * 2.0;
    p.y += t;
    
    float terrain = fbm(p + vec2(sin(t), cos(t * 0.7)));
    terrain += fbm(p * 2.0 + vec2(cos(t * 1.3), sin(t * 0.9))) * 0.5;
    terrain = pow(terrain, 1.5 + pc.noteVelocity);
    
    // Color based on height and MIDI input
    vec3 col = mix(
        vec3(0.1, 0.2, 0.4),
        vec3(0.8, 0.6, 0.3),
        terrain
    );
    
    col = mix(col, vec3(1.0, 0.9, 0.7), 
        pow(terrain, 3.0) * pc.noteVelocity);
    
    // Add note reactive glow
    if(pc.noteCount > 0.0) {
        float noteEffect = exp(-length(p - vec2(
            pc.pitchBend, 
            sin(pc.lastNote * 0.1)
        )) * 2.0);
        col += vec3(0.5, 0.3, 0.8) * noteEffect * pc.noteVelocity;
    }
    
    // Output
    outColor = vec4(col, 1.0);
}
"#;

// shaders/crystal.frag
pub const CRYSTAL_FRAG: &str = r#"
#version 450

layout(location = 0) in vec2 fragCoord;
layout(location = 1) in vec2 resolution;

layout(location = 0) out vec4 outColor;

layout(push_constant) uniform PushConstants {
    float time;
    float mouseX;
    float mouseY;
    float mousePressed;
    float resolutionX;
    float resolutionY;
    float noteVelocity;
    float pitchBend;
    float modulation;
    float expression;
    float noteCount;
    float lastNote;
} pc;

// Crystal lattice pattern
float crystal(vec3 p) {
    p = abs(p);
    float d = max(p.x, max(p.y, p.z)) - 1.0;
    
    // Add detail
    float s = 1.0;
    for(int i = 0; i < 3; i++) {
        p = abs(p) - 0.5 * s;
        p = p.yzx;
        s *= 0.5;
        d = min(d, max(p.x, max(p.y, p.z)) * s);
    }
    
    return d;
}

void main() {
    vec2 uv = (fragCoord - 0.5 * resolution) / min(resolution.x, resolution.y);
    
    float t = pc.time * 0.5;
    
    // Camera
    vec3 ro = vec3(
        3.0 * cos(t + pc.pitchBend), 
        2.0 * sin(t * 0.7), 
        3.0 * sin(t + pc.pitchBend)
    );
    vec3 ta = vec3(0.0);
    
    vec3 ww = normalize(ta - ro);
    vec3 uu = normalize(cross(ww, vec3(0.0, 1.0, 0.0)));
    vec3 vv = cross(uu, ww);
    
    vec3 rd = normalize(uv.x * uu + uv.y * vv + 1.5 * ww);
    
    // Raymarch the crystal
    float d = 0.0;
    vec3 col = vec3(0.0);
    
    for(int i = 0; i < 100; i++) {
        vec3 p = ro + rd * d;
        
        // Rotate based on MIDI
        float angle = t + pc.modulation * 3.14159;
        p.xz = mat2(cos(angle), -sin(angle), sin(angle), cos(angle)) * p.xz;
        
        float dist = crystal(p);
        
        if(dist < 0.001) {
            // Crystal color
            col = vec3(0.3, 0.6, 0.9);
            col += vec3(0.5, 0.3, 0.1) * pc.noteVelocity;
            
            // Iridescence
            float fresnel = pow(1.0 - dot(-rd, normalize(p)), 2.0);
            col += vec3(0.8, 0.3, 0.9) * fresnel * (1.0 + pc.expression);
            
            break;
        }
        
        d += dist * 0.8;
        if(d > 10.0) break;
    }
    
    // Background
    vec3 bg = vec3(0.05, 0.05, 0.1) * (1.0 + uv.y * 0.5);
    col = mix(bg, col, step(0.01, length(col)));
    
    // Note flash
    if(pc.noteVelocity > 0.1) {
        col += vec3(1.0, 0.5, 0.0) * 
            exp(-d * 0.5) * pc.noteVelocity * 0.3;
    }
    
    outColor = vec4(col, 1.0);
}
"#;

// shaders/plasma.frag
pub const PLASMA_FRAG: &str = r#"
#version 450

layout(location = 0) in vec2 fragCoord;
layout(location = 1) in vec2 resolution;

layout(location = 0) out vec4 outColor;

layout(push_constant) uniform PushConstants {
    float time;
    float mouseX;
    float mouseY;
    float mousePressed;
    float resolutionX;
    float resolutionY;
    float noteVelocity;
    float pitchBend;
    float modulation;
    float expression;
    float noteCount;
    float lastNote;
} pc;

void main() {
    vec2 uv = fragCoord / resolution;
    vec2 p = (fragCoord - 0.5 * resolution) / min(resolution.x, resolution.y);
    
    float t = pc.time;
    
    // Plasma effect
    float plasma = 0.0;
    
    // Multiple sine waves
    plasma += sin(p.x * 10.0 + t * 2.0);
    plasma += sin(p.y * 10.0 + t * 1.5);
    plasma += sin((p.x + p.y) * 10.0 + t * 3.0) * 0.5;
    plasma += sin(length(p * 20.0) - t * 4.0) * 0.5;
    
    // MIDI modulation
    plasma += sin(p.x * 30.0 * (1.0 + pc.modulation) + t * 5.0) * pc.noteVelocity;
    plasma += cos(p.y * 20.0 * (1.0 + pc.expression) - t * 3.0) * pc.noteVelocity;
    
    plasma = plasma / 3.0;
    
    // Color palette
    vec3 col = vec3(0.5) + 0.5 * vec3(
        sin(plasma * 3.14159 + pc.pitchBend),
        sin(plasma * 3.14159 + 2.0 + pc.lastNote * 0.01),
        sin(plasma * 3.14159 + 4.0)
    );
    
    // Brightness based on MIDI activity
    col *= 0.8 + 0.5 * pc.noteVelocity;
    
    // Add reactive circles for notes
    if(pc.noteCount > 0.0) {
        vec2 notePos = vec2(
            sin(pc.lastNote * 0.1) * 0.5,
            cos(pc.lastNote * 0.15) * 0.5
        );
        float noteDist = length(p - notePos);
        float noteGlow = exp(-noteDist * 10.0) * pc.noteVelocity;
        col += vec3(1.0, 0.5, 0.0) * noteGlow;
    }
    
    outColor = vec4(col, 1.0);
}
"#;