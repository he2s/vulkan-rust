#version 450

// ---------- Push constants / IO (from your setup) ----------
layout(push_constant) uniform PushConstants {
    float time; uint mouse_x; uint mouse_y; uint mouse_pressed;
    float note_velocity; float pitch_bend; float cc1; float cc74;
    uint  note_count; uint last_note; float osc_ch1; float osc_ch2;
    uint  render_w; uint render_h;
} pc;

layout(location=0) in vec2 frag_uv;          // 0..1
layout(location=1) in vec2 frag_screen_pos;  // optional
layout(location=0) out vec4 out_color;

// ---------- Helpers to mimic Shadertoy uniforms ----------
#define iTime        (pc.time)
#define iResolution  vec3(float(pc.render_w), float(pc.render_h), 1.0)
vec4   iMouse = vec4(float(pc.mouse_x), float(pc.render_h) - float(pc.mouse_y),
float(pc.mouse_pressed), 0.0);

// Optional: expose your audio/MIDI as “beat”/“audioLevel” for FX
float audioLevel() {
    return clamp((pc.note_velocity + pc.osc_ch1 + pc.osc_ch2) * 0.5, 0.0, 1.0);
}
float beat() {
    // simple tempo-ish proxy you can remap if needed
    return fract(iTime * mix(1.0, 2.0, audioLevel()));
}

// ---------- (OPTIONAL) Texture channels ----------
// If the Shadertoy uses iChannel0..3, add descriptor sets & samplers here.
// Example bindings (uncomment and hook in your pipeline):
// layout(set=0, binding=0) uniform sampler2D iChannel0;
// layout(set=0, binding=1) uniform sampler2D iChannel1;
// layout(set=0, binding=2) uniform sampler2D iChannel2;
// layout(set=0, binding=3) uniform sampler2D iChannel3;

// For 0..1 UV sampling (like Shadertoy’s texture()) helpers:
// vec4 tex0(vec2 uv){ return texture(iChannel0, uv); }
// vec4 tex1(vec2 uv){ return texture(iChannel1, uv); }
// ...

float hash31( vec3 p ) // randomization function
{
    p = vec3( dot(p,vec3(127.1,311.7, 74.7)),
    dot(p,vec3(269.5,183.3,246.1)),
    dot(p,vec3(113.5,271.9,124.6)));

    p = fract(sin(p)*43758.5453);

    return fract(p.x + p.y + p.z);
}


float width = 0.1; // line width





// ---------- Shadertoy entrypoint to be inserted ----------
    // ===== PASTE THE SHADERTOY BODY HERE =====
    // Keep its logic intact. It expects:
    //   - fragCoord in pixels
    //   - iTime, iResolution, iMouse, iChannelN (if used)
    //   - Any extra defines/macros remain fine

    void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
int j;
for(j=0;j<3;j+=1) // loop over rgb channels
{
    float t = iTime + 0.04*float(j);
    vec2 pos = 2.*vec2(sin(t*0.5)+0.1*t,cos(t*0.5)+0.3*t); // movement
    vec3 col = vec3(1.); // background

    // Loop over parallax layers
    for(float i=10.;i<17.; i+=0.75)
    {
        vec2 uv = pos+(0.2*sin(t)*sin(t)+0.5)*((20. - 0.7*i)*(fragCoord-.5*iResolution.xy)/iResolution.y);
        vec2 gv = (fract(uv)-0.5);
        vec2 id = floor(uv);
        vec3 col2 = (vec3(1.0)*(11./(i+5.))); // change this to make line colors more interesting
        float mask1 = 1.; // Used for hard edges
        float mask2 = 1.; // used for glows / "shadows"

        if (hash31(vec3(id.x,id.y,i))>0.75) // Intersections
        {
            float a1 = smoothstep(-0.01,0.01,width-(abs(gv.x)));
            float b1 = smoothstep(-0.01,0.01,width-(abs(gv.y)));
            float a2 = smoothstep(-0.3,0.3,width-(abs(gv.x)));
            float b2 = smoothstep(-0.3,0.3,width-(abs(gv.y)));
            mask1 = a1 + b1 - a1*b1;
            mask2 = a2 + b2 - a2*b2;
        }
        else // Diagonals
        {
            gv.x *= (float(hash31(vec3(i,id.x,id.y))>0.5)-0.5)*2.; // Flip half of them
            mask1 = smoothstep(-0.01,0.01,width-abs(gv.x+gv.y-0.5*sign(gv.x+gv.y+0.01)));
            mask2 = smoothstep(-0.3,0.3,width-abs(gv.x+gv.y-0.5*sign(gv.x+gv.y+0.01)));

        }

        // Output to screen
        col = - 0.2*mask2 + 0.9*(col2.r*col2.r+col2.g*col2.g+col2.b*col2.b + col2*col2)*col2*mask1 + col*(1.-mask1);

        // Enavble to show grid:
        //if (abs(gv.x)>0.475 || abs(gv.y)>0.475) col.r=1.;

    }
    fragColor[j] = col[j];
}

    //// Example placeholder:
    //vec2 uv = fragCoord / iResolution.xy;
    //vec3 col = vec3(uv, 0.5 + 0.5*sin(iTime));
    //// Make it a little audio-reactive by default:
    //col *= mix(0.85, 1.35, audioLevel());
    //fragColor = vec4(col, 1.0);
}

// ---------- Main ----------
void main() {
    // Recreate Shadertoy’s fragCoord from your varying
    vec2 fragCoord = frag_uv * iResolution.xy;

    // Call the Shadertoy entry and then apply your CC/gamma pipeline
    vec4 col; mainImage(col, fragCoord);

    // Your existing brightness/gamma mapping
    float bright = mix(0.85, 1.55, clamp(pc.cc74, 0.0, 1.0));
    float gammaC = 1.0 + 0.7 * clamp(pc.cc1, 0.0, 1.0);
    col.rgb = pow(max(col.rgb * bright, 0.0), vec3(gammaC));

    // MIDI note tint (from your setup)
    if(pc.note_count > 0u) {
        float noteTint = float(pc.last_note) / 127.0;
        vec3 tintColor = vec3(
        sin(noteTint * 3.14159265) * 0.2,
        sin(noteTint * 3.14159265 + 3.14159265/3.0) * 0.2,
        sin(noteTint * 3.14159265 + 2.0*3.14159265/3.0) * 0.2
        );
        col.rgb += tintColor * pc.note_velocity;
    }

    out_color = vec4(clamp(col.rgb, 0.0, 3.0), 1.0);
}
