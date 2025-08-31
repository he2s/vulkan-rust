# Vulkan MIDI Visualizer - Refactored Architecture

A high-performance, modular Vulkan-based audio/MIDI reactive visualizer with clean architecture and separation of concerns.

## Project Structure

```
vulkan-visualizer/
├── Cargo.toml                 # Project dependencies
├── config.toml                # Runtime configuration (auto-generated)
├── README.md                  # This file
│
├── src/
│   ├── main.rs               # Entry point and initialization
│   ├── app.rs                # Application lifecycle management
│   ├── config.rs             # Configuration structures and loading
│   ├── utils.rs              # Shared utilities and types
│   │
│   ├── renderer/             # Graphics subsystem (modular components)
│   │   ├── mod.rs           # Main renderer orchestration
│   │   ├── context.rs       # Vulkan instance/device management
│   │   ├── swapchain.rs     # Swapchain management
│   │   ├── pipeline.rs      # Pipeline creation and management
│   │   ├── command.rs       # Command buffer recording
│   │   ├── sync.rs          # Synchronization primitives
│   │   ├── framebuffer.rs   # Framebuffer management
│   │   └── shader.rs        # Shader loading and compilation
│   │
│   └── input/               # Input subsystem
│       ├── mod.rs          # Input system orchestration
│       ├── midi.rs         # MIDI input handling
│       └── audio.rs        # Audio input and FFT analysis
│
└── shaders/                 # GLSL shader sources
    ├── fullscreen.vert     # Fullscreen triangle vertex shader
    ├── torus.frag         # 4D torus visualization
    ├── terrain.frag       # Organic terrain
    ├── crystal.frag       # Crystalline structures
    └── plasma.frag        # Plasma effect
```

## Key Improvements Over Original

### 1. **Modular Architecture**
- **Single Responsibility**: Each module has one clear purpose
- **Dependency Injection**: Components receive dependencies rather than creating them
- **Clean Interfaces**: Well-defined boundaries between modules

### 2. **Resource Management**
- **RAII Pattern**: Automatic cleanup via Drop traits
- **Arc-based Sharing**: Safe resource sharing between components
- **Explicit Lifetimes**: Clear ownership and borrowing

### 3. **Error Handling**
- **Result Types**: Consistent error propagation
- **Custom Error Types**: Domain-specific error information
- **Graceful Recovery**: Automatic swapchain recreation, shader reload

### 4. **Input System**
- **Unified Interface**: Single system for all input types
- **Channel-based Communication**: Efficient thread-safe message passing
- **Async/Await**: Non-blocking input processing

### 5. **Renderer Design**
- **Hot-swappable Shaders**: Runtime shader switching
- **Dynamic State**: Viewport/scissor updates without pipeline recreation
- **Optimized Command Recording**: Reusable command buffers

## Building and Running

### Prerequisites
- Rust 1.70+
- Vulkan SDK 1.2+
- C++ compiler (for shaderc)

### Build
```bash
# Debug build
cargo build

# Release build (recommended for performance)
cargo build --release
```

### Run
```bash
# With default configuration
cargo run --release

# With custom settings
cargo run --release -- --fullscreen --shader plasma

# Enable validation layers
cargo run -- --debug
```

## Configuration

The application generates a `config.toml` file on first run:

```toml
[window]
width = 1280
height = 720
title = "Vulkan MIDI Visualizer"
fullscreen = false
resizable = true

[graphics]
vsync = true
validation_layers = false  # Auto-enabled in debug builds
msaa_samples = 1

[shader]
preset = "torus"  # Options: torus, terrain, crystal, plasma, custom
hot_reload = true

[midi]
enabled = true
auto_connect = true
# port_name = "Optional specific MIDI port"

[audio]
enabled = false
# device_name = "Optional specific audio device"
# sample_rate = 48000
fft_size = 2048
```

## Controls

- **F11** - Toggle fullscreen
- **ESC** - Exit (or exit fullscreen if in fullscreen mode)
- **TAB** - Cycle through shader presets
- **R** - Reload current shader
- **1-4** - Select shader preset directly

## Architecture Benefits

### Maintainability
- Easy to add new shader presets
- Simple to extend input systems
- Clear separation makes debugging straightforward

### Performance
- Minimal allocations in hot path
- Efficient command buffer reuse
- Optimized synchronization

### Extensibility
- Add compute shaders easily
- Integrate new input sources
- Support multiple windows/surfaces

## Adding Custom Shaders

1. Create your fragment shader in `shaders/`
2. Add to `ShaderPreset` enum in `config.rs`
3. Update shader loading in `shader.rs`
4. Rebuild and select via config or runtime

## Troubleshooting

### Validation Errors
Enable validation layers:
```bash
cargo run -- --debug
```

### Performance Issues
- Ensure release build: `cargo build --release`
- Check vsync setting in config
- Reduce shader complexity

### MIDI Not Working
- Check MIDI device is connected before starting
- Specify port name in config if multiple devices
- Enable MIDI debug logging: `RUST_LOG=vulkan_visualizer=debug`

## Future Enhancements

- [ ] Compute shader integration for particles
- [ ] Multi-window support
- [ ] Network-based input (OSC, WebSocket)
- [ ] Shader hot-reload from file system
- [ ] Recording/export capabilities
- [ ] GUI overlay with egui
- [ ] Scripting support (Lua/Wasm)

## License

MIT - See LICENSE file for details