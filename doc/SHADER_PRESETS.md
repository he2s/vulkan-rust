# Vulkan Rust Shader Presets

## Available Presets

### **Simple Presets (Original Complexity)**
- **`torus`** - Fullscreen torus rendering with audio-reactive parameters
- **`terrain`** - Terrain-based effects
- **`stars`** - Star field effects
- **`crystal`** - **Simple geometry shader** with basic triangles and audio reactivity

### **Advanced Presets**
- **`weirdcrystal`** - **Complex geometry shader** with:
  - 3 dynamic shape modes (fractal spikes, twisted polygons, chaotic triangles)
  - Advanced audio reactivity with FBM noise
  - Psychedelic fragment shader with antialiasing
  - Up to 32 vertices per primitive

- **`computeparticles`** - **GPU compute shader** optimization:
  - 50,000 particles generated on GPU
  - 4 different audio-reactive patterns
  - Instanced triangle rendering
  - Maximum performance

## Usage Examples

```bash
# Simple geometry shader (basic triangles)
cargo run -- --preset crystal

# Complex weird geometry shader
cargo run -- --preset weirdcrystal

# High-performance compute particles
cargo run -- --preset computeparticles

# Traditional fullscreen effects
cargo run -- --preset torus
cargo run -- --preset terrain
cargo run -- --preset stars
```

## Shader Organization

### Crystal vs WeirdCrystal
- **Crystal**: Uses `simple.geom` + `simple_geometry.frag` for basic effects
- **WeirdCrystal**: Uses `example.geom` + `weird_crystal.frag` for complex effects

This separation allows users to choose their preferred level of visual complexity while maintaining performance on different hardware.