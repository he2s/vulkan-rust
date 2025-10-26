# Shader Variations Guide

## Overview
Created **84 shader variations** from 9 enabled fragment shaders, providing a wide variety of visual effects!

## new22.frag Variations (10 total)

### Specialized Variations
1. **new22_var1.frag** - Electric Blue
   - Cool blue color palette
   - Great for calm, ambient visuals

2. **new22_var2.frag** - Hot Red
   - Warm fiery red palette
   - Slower color cycling for dramatic effect

3. **new22_var3.frag** - Hyperspeed
   - 2x animation speed
   - 2x camera speed
   - Intense, fast-paced visuals

4. **new22_var4.frag** - Slow Motion
   - 0.5x speed for dreamy effect
   - Meditative, flowing visuals

5. **new22_var5.frag** - SuperGlow
   - Intense glowing lines
   - Bright core with expanded glow radius

6. **new22_var6.frag** - Minimal
   - Fewer, thinner lines
   - Subtle, refined aesthetic

7. **new22_var7.frag** - Rainbow
   - Full spectrum color cycling
   - Fast rainbow transitions

8. **new22_var8.frag** - Chaotic
   - Wild, unpredictable camera movement
   - Crazy roll effects

9. **new22_var9.frag** - Dense
   - Maximum line density
   - Complex, intricate patterns

10. **new22_var10.frag** - Psychedelic
    - Extreme colors and rapid shifts
    - Boosted saturation
    - Ultimate visual intensity

## Standard Variations (v1-v8) for Other Shaders

Applied to: beauty.frag, new10.frag, new21.frag, new2.frag, new3.frag, new8.frag, wildbeauty7.frag, wildbeauty.frag

### v1 - Faster
- 1.5x animation speed
- Energetic and dynamic

### v2 - Slower
- 0.6x speed
- Dreamy, meditative feel

### v3 - Red Shift
- Warm color palette
- Red/orange/yellow emphasis

### v4 - Blue Shift
- Cool color palette
- Blue/cyan emphasis

### v5 - Cyan Shift
- Matrix-like green/cyan palette
- Tech/digital aesthetic

### v6 - High Contrast
- Vivid, saturated colors
- Boosted intensity

### v7 - Soft
- Gentle pastel colors
- Low contrast, soothing

### v8 - Hyper
- Extreme everything
- 2.5x speed + 2x intensity
- Maximum visual impact

## Usage

All variations use the same vertex shader (`fullscreen.vert`) and respond to the same MIDI/audio controls as the original shaders.

Simply reference them in your config.toml like:

```toml
[shader.presets.my_preset]
name = "My Custom Preset"
enabled = true
geometry_type = "fullscreen"
vertex = "fullscreen.vert"
fragment = "new22_var5.frag"  # SuperGlow variant
```

## Tips

- **Performance**: All variations have the same performance characteristics as their base shader (new22 variations are optimized)
- **Mixing**: Try different variations for different moods
- **Live Performance**: Map different presets to MIDI notes for live switching
- **Experimentation**: Each variation reacts differently to audio input

Enjoy exploring the 84 different visual experiences!
