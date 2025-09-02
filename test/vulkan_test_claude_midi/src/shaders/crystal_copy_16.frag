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

const float PI=3.14159265359, TAU=6.28318530718;

void main(){
    vec2 uv=(frag_uv-0.5)*2.0;
    float aspect=float(pc.render_w)/float(pc.render_h);
    uv.x*=aspect;

    float audio=(pc.note_velocity+pc.osc_ch1+pc.osc_ch2)*0.5;
    float t=pc.time*(1.0+audio);

    // Center & mouse move of the trap
    vec2 trap = vec2(0.0);
    if(pc.mouse_pressed!=0u){
        vec2 m = vec2(float(pc.mouse_x), float(pc.mouse_y))/vec2(float(pc.render_w), float(pc.render_h));
        m = (m-0.5)*2.0; m.x*=aspect;
        trap = mix(vec2(0.0), m, 0.8);
    }
    trap += 0.25*vec2(sin(t*0.7+pc.pitch_bend*3.0), cos(t*0.53-pc.pitch_bend*2.0));

    // Simple orbit-iterate: z -> z^2 + c
    vec2 c = uv*0.9 + 0.2*vec2(sin(t*0.3), cos(t*0.2));
    vec2 z = c;
    float mDist = 1e6;
    float iterGlow = 0.0;
    const int ITERS = 24;
    for(int i=0;i<ITERS;i++){
        // z^2 + c
        z = vec2(z.x*z.x - z.y*z.y, 2.0*z.x*z.y) + c;
        // orbit trap distance to moving point
        float d = length(z - trap);
        mDist = min(mDist, d);
        iterGlow += exp(-d*12.0);
        if(dot(z,z)>16.0) break;
    }

    // Map distances to color
    float hue = (pc.note_count>0u ? float(pc.last_note)/127.0 : 0.2) + t*0.03;
    vec3 base = vec3(
    0.5+0.5*sin(TAU*(hue+0.00) + pc.osc_ch1*TAU),
    0.5+0.5*sin(TAU*(hue+0.33) - pc.osc_ch2*TAU),
    0.5+0.5*sin(TAU*(hue+0.66) + audio*PI)
    );

    float edge = 1.0/(0.02 + mDist*10.0);
    float glow = iterGlow*(0.6+audio) + edge*2.2;

    // CC controls alter contrast and channel swap
    float ccMix = clamp(pc.cc1, 0.0, 1.0);
    float ccGamma = 1.0 + pc.cc74*0.8;
    vec3 col = pow(base*glow, vec3(ccGamma));
    col = mix(col, col.bgr, ccMix*0.7);

    // Soft vignette for depth
    float vig = smoothstep(1.6, 0.2, length(uv));
    col *= mix(0.7, 1.6, vig);

    out_color = vec4(clamp(col,0.0,3.0), 1.0);
}
