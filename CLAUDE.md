# vulkan-rust

Real-time GLSL pixel-shader visualiser. Vulkan (ash) + winit + egui overlay + MIDI/OSC/audio-FFT inputs feeding push constants. Hot-reloadable shader presets.

You are working with a peer. Be terse, opinionated, right. Sycophancy is noise. The diff is the artefact; the prose around it should be small.

---

## Code rules

**Minimum diff.** Don't refactor surroundings. Don't "improve" working code. Don't add what wasn't asked. Three near-duplicate lines beat a premature abstraction.

**No wallpaper.** No `Result` swallowed silently, no `if x.is_some()` on values that can't be `None`, no `unwrap_or_else(|| panic!(...))` smoke screens. Validate at boundaries (config load, file IO, swapchain acquire). Trust internal code. Errors propagate.

**Functions over structs-with-methods.** Reach for a type when an invariant has to be enforced, not because "things should be objects." Same for traits — extract on the second caller, not the first.

**Comments are for *why*.** Not what, not how. No file-header banners. No `// ============` dividers. No "// TODO: clean up later" left in. `fn parse_otlp_traces` doesn't need a doc saying "parses OTLP traces".

**Names are the documentation.**

**Don't narrate.** No "Now let's…", no recap, no "I noticed and fixed…". I read the diff. End-of-turn summary is one sentence.

---

## Hot-path posture (per frame, every frame)

`App::window_event` → `Gfx::draw` runs at vsync. Treat anything inside `draw`, `record_command_buffer`, `get_push_constants`, `InputManager::get_frame_state` as hot.

- **No allocations in the draw loop.** No `Vec::new`, no `format!`, no `String::from`, no `to_vec()`, no `collect()`. Pre-size with `with_capacity` at init, `clear()` to reuse.
- **`PushConstants` is `#[repr(C)]` and is bit-copied to GPU memory.** Field order and size matter — must match GLSL `layout(push_constant) uniform`. Touch one side, touch the other. Total stays ≤ 128 bytes on guaranteed hardware; we currently use 96.
- **No new `String` per frame for window titles, FPS, BPM.** `update_window_title` already throttles to every 30 frames — keep it that way.
- **No locks in the draw loop.** If `InputManager` ever needs sharing, use lock-free (atomics, SPSC ringbuffer, `parking_lot::Mutex::try_lock`). Never `std::sync::Mutex`.
- **`println!` is a debug crutch.** It locks stdout. Acceptable for `FPS` tick and shader-switch logs. Not in `draw`.
- **Iterators over indexed loops** when bounds are obvious; indexed when there's a measured reason. Either way: no per-frame `iter().collect::<Vec<_>>()`.

If unsure whether a perf change matters, measure. Don't guess. Run `/perf-check` after non-trivial draw-path edits.

---

## Vulkan rules

- **Validation layers must stay clean** when `validation_layers = true`. Any new VUID in the log is a regression — fix before moving on.
- **Sync is non-negotiable.** Every resource that crosses a frame boundary needs a fence or semaphore. `image_available` (acquire→color attachment), `render_finished` (submit→present), `in_flight` (fence: CPU waits before reusing command buffer). Don't add a fourth sync primitive without writing down what it protects.
- **Before destroying anything, `device_wait_idle()`.** Pipelines, swapchain, framebuffers, shader modules — all of them. `recreate_swapchain` and `recreate_pipeline` already do this; new teardown paths must too.
- **Cleanup order is reverse of creation.** `Drop for Gfx` is the canonical order. If you add a resource, add its destroy *before* the resources it depends on.
- **Push constants > UBO > descriptor set** when the data is small (≤ 128 B) and per-draw. UBO when shared across draws. Descriptor sets when bindless or per-material.
- **`unsafe` needs a `// SAFETY:` block.** Stating the invariants the caller must uphold and why each holds here. Most Vulkan calls are unsafe because they require external synchronisation — say so.
- **Suboptimal / OutOfDate KHR** at `acquire_next_image` or `queue_present` triggers swapchain recreation. Never panic on those — they're normal during window events.

---

## GLSL / shader rules

Shaders live in `shaders/`. Fullscreen presets use `fullscreen.vert` + `<name>.frag`. Geometry-shader presets use `points.vert` + `*.geom` + `*.frag`. Compute uses `point_generator.comp`.

- **`PushConstants` layout in GLSL must mirror Rust exactly.** Field order, types (`float` ↔ `f32`, `uint` ↔ `u32`), no implicit padding. When in doubt, std430-pack on the Rust side already.
- **No `#version` mismatches.** Stick to one GLSL version per project unless a preset needs otherwise — call it out in the preset's comment in `config.toml`.
- **Variant naming: `<base>_v<n>.frag`.** Don't invent new schemes. Variants are cheap; long-form names aren't.
- **Compile errors live in `shaderc`.** The error message names the file as `shader` (we don't pass the path). When debugging, copy the offending shader text into a `.frag` and reload — `F5` does this.

---

## Project map

```
src/
  main.rs                   App, event loop, push-constant assembly
  lib.rs                    Module roots
  gfx.rs                    Vulkan top-level: Gfx; VulkanPipeline, VulkanCommands,
                            VulkanSync live here as inline structs; draw, record
  graphics/
    mod.rs                  GeometryMode, PushConstants, Vertex, InstanceData
    vulkan/                 context.rs, swapchain.rs; VulkanBuffers in mod.rs
    shaders/                mod.rs: ShaderSources loader + compile_to_spirv (shaderc)
    egui_overlay.rs         egui-on-Vulkan integration
  input/                    MIDI (midir), audio (cpal+rustfft), OSC, manager
  processing/filters.rs     Audio post-processing (level RMS, low/mid/high bands)
  state/                    FrameState (snapshot handed to push-constant assembly)
  config/                   TOML config + clap CLI args, schema, presets
  utils/                    Device lister
  audio.rs                  AudioState, BeatState, AudioLevels (core shared types;
                            capture/FFT lives in input/audio.rs)

shaders/                    GLSL sources. Compiled at preset-load time via shaderc.
config.toml                 Active preset, MIDI port, audio device, OSC port, tap-tempo
build.sh                    Docker MUSL static build (not the dev loop)
Renderdoc.rdc               Existing capture — open in RenderDoc for reference frame
```

Key invariant: a "preset" = entry in `[shader.presets.<key>]` in `config.toml` with `enabled = true`, plus matching files in `shaders/`. `enabled = false` presets are skipped at startup.

---

## Build / run / debug

**Dev loop (Windows / Linux):**
```
cargo run --release                       # always release; debug Vulkan is too slow
cargo run --release -- --list-devices     # enumerate MIDI + audio
cargo check                               # fast feedback while editing Rust
cargo clippy                              # before declaring a Rust change done
```

`shaderc` builds from source. On a clean checkout the first build pulls CMake-built native libs. If CMake errors on policy, the workaround that's been baked into `.claude/settings.local.json` is:
```
CMAKE_POLICY_VERSION_MINIMUM=3.5 cargo build
```

**Runtime controls:**
- `F5` reload + recompile current preset's shaders (preserves window/swapchain)
- `Tab` cycle to next enabled preset
- `F11` toggle borderless fullscreen (forces swapchain recreate)
- `H` toggle egui overlay (preset list, BPM, FFT bands)
- `Space` tap tempo
- `Esc` exit (or exit fullscreen first)

**Debug:**
- `validation_layers = true` in `config.toml` for any code work. Off only for benching.
- RenderDoc: `Renderdoc.rdc` in the repo is a reference frame. Open it in qrenderdoc; capture a new one via `vkconfig` or RenderDoc's "Launch Application" pointing at the built `vulkan-rust.exe`. See `.claude/skills/renderdoc-capture/SKILL.md`.
- FPS prints every 10 seconds via `App::update_fps_tracking`. The window title shows BPM.
- Profiling: no in-tree profiler. Don't add tracing-style spans in the draw path without measuring overhead.

---

## Workflow

- When ambiguous, ask one sharp question.
- For non-trivial design choices, invoke `/grill-me` *before* writing code. The project variant lives in `.claude/skills/grill-me/`.
- Before declaring a perf change done, `/perf-check`.
- For pipeline/swapchain/shader hot-reload work, `.claude/skills/shader-hot-reload/SKILL.md` documents the safe teardown order.
- Don't create planning docs, summary markdowns, or session notes unless asked.
- Don't commit unless asked. Don't push unless asked. Don't open PRs unless asked.

## When you don't know

Say so. Read the code. Run the thing. `cargo doc --open -p ash`. Guessing API surface from memory and pattern-matching to "what Vulkan usually looks like" is how `vk::ImageLayout::GENERAL` shows up in code that needed `COLOR_ATTACHMENT_OPTIMAL`.