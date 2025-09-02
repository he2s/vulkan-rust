#version 450
layout(push_constant) uniform PushConstants {
    float time; uint mouse_x; uint mouse_y; uint mouse_pressed;
    float note_velocity; float pitch_bend; float cc1; float cc74;
    uint note_count; uint last_note; float osc_ch1; float osc_ch2;
    uint render_w; uint render_h;
} pc;

layout(location=0) in vec2 frag_uv;
layout(location=1) in vec2 frag_screen_pos;
layout(location=0) out vec4 out_color;

const float TAU=6.28318530718;

float smin(float a, float b, float k){
    float h = clamp(0.5 + 0.5*(b-a)/k, 0.0, 1.0);
    return mix(b, a, h) - k*h*(1.0-h);
}

float map(vec3 p){
    float a = (pc.note_velocity + pc.osc_ch1 + pc.osc_ch2)*0.5;
    vec3 c1 = vec3( 0.9*sin(pc.time*0.7), 0.5*cos(pc.time*0.4+1.0), 0.9*cos(pc.time*0.5));
    vec3 c2 = vec3( 0.9*sin(pc.time*0.6+2.0), 0.6*cos(pc.time*0.45+2.3), 0.9*cos(pc.time*0.38+1.7));
    vec3 c3 = vec3( 0.9*sin(pc.time*0.55+3.1), 0.55*cos(pc.time*0.52+0.8), 0.9*cos(pc.time*0.33+0.3));

    float r = 0.5 + 0.25*sin(pc.time*0.9 + a*3.0);
    float d1 = length(p - c1) - r;
    float d2 = length(p - c2) - r;
    float d3 = length(p - c3) - r;

    float k = 0.6 + 0.4*a; // blobby blend
    return smin(smin(d1,d2,k), d3, k);
}

vec3 getNormal(vec3 p){
    vec2 e=vec2(1e-3,0.0);
    return normalize(vec3(
    map(p+vec3(e.x,e.y,e.y)) - map(p-vec3(e.x,e.y,e.y)),
    map(p+vec3(e.y,e.x,e.y)) - map(p-vec3(e.y,e.x,e.y)),
    map(p+vec3(e.y,e.y,e.x)) - map(p-vec3(e.y,e.y,e.x))
    ));
}

float raymarch(vec3 ro, vec3 rd, out vec3 pos){
    float t=0.0;
    for(int i=0;i<110;i++){
        pos = ro + rd*t;
        float d = map(pos);
        if(d<0.001) return t;
        t += d*0.9;
        if(t>40.0) break;
    }
    return 1e9;
}

mat3 lookAt(vec3 ro, vec3 ta){
    vec3 f=normalize(ta-ro);
    vec3 r=normalize(cross(vec3(0,1,0), f));
    vec3 u=cross(f,r);
    return mat3(r,u,f);
}

void main(){
    vec2 q=(frag_uv-0.5)*2.0;
    float aspect=float(pc.render_w)/float(pc.render_h);
    q.x*=aspect;

    float a=(pc.note_velocity+pc.osc_ch1+pc.osc_ch2)*0.5;
    float t=pc.time;

    // Camera orbits; mouse nudges target
    vec3 ro = vec3(4.5*sin(t*0.2+pc.pitch_bend*1.5), 2.2+0.5*sin(t*0.5), 4.5*cos(t*0.2+pc.pitch_bend*1.5));
    vec3 ta = vec3(0.0);
    if(pc.mouse_pressed!=0u){
        vec2 m=vec2(float(pc.mouse_x)/float(pc.render_w), float(pc.mouse_y)/float(pc.render_h));
        m=(m-0.5)*2.0; m.x*=aspect;
        ta += vec3(m.x, m.y*0.6, 0.0);
    }

    mat3 cam = lookAt(ro, ta);
    vec3 rd = normalize(cam * normalize(vec3(q, 1.6)));

    vec3 pos; float d = raymarch(ro, rd, pos);

    vec3 col = vec3(0.02,0.03,0.05); // background
    if(d<1e9){
        vec3 n = getNormal(pos);
        vec3 l = normalize(vec3(0.3, 0.9, 0.2));
        float diff = clamp(dot(n,l), 0.0, 1.0);
        float rim = pow(1.0-max(0.0,dot(n,-rd)), 4.0);
        float subsurface = exp(-max(0.0, map(pos + n*0.02))*60.0);

        // Hue palette
        float hue = (pc.note_count>0u?float(pc.last_note)/127.0:0.35) + 0.05*t + 0.1*a;
        vec3 pal = vec3(
        0.5+0.5*sin(TAU*(hue+0.00)+pc.osc_ch1*TAU),
        0.5+0.5*sin(TAU*(hue+0.33)-pc.osc_ch2*TAU),
        0.5+0.5*sin(TAU*(hue+0.66)+a*3.14159)
        );

        float em = 1.2*subsurface + 0.6*rim + 0.3;
        col = pal*(0.35 + 0.9*diff) + em*pal;

        // Screen-space bloom-ish line
        col += 0.05*sin((frag_screen_pos.y)*3.14159*1.7 + t*18.0)*(0.4+a);
    }

    // CCs: gamma & swap
    float gamma = 1.0 + pc.cc1*0.9;
    col = pow(col, vec3(gamma));
    col = mix(col, col.bgr, clamp(pc.cc74,0.0,1.0)*0.5);

    // Vignette
    float vig = smoothstep(1.6, 0.25, length(q));
    col *= mix(0.7, 1.7, vig);

    out_color = vec4(clamp(col,0.0,3.0), 1.0);
}
