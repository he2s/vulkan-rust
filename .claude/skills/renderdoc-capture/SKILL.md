---
name: renderdoc-capture
description: Capture and inspect a frame of this renderer with RenderDoc. Translates RenderDoc findings (pipeline state, descriptor bindings, push-constant values, draw counts, SPIR-V disasm, pixel-history) back to Rust/GLSL locations. Use when the user invokes /renderdoc-capture, when debugging a visual artifact, a black screen, suspicious GPU timing, or a validation error that names a specific draw call.
---

# renderdoc-capture

The repo has `Renderdoc.rdc` already — a reference capture. New captures go next to it (untracked by default; gitignore it or move it to a `captures/` dir).

## When to use this

- Visual artefact you can't explain from code: missing geometry, wrong colours, garbage in one channel.
- Black screen with no validation errors.
- "Why is `cc1` not affecting the shader?" → confirm the push-constant value at the moment of the draw.
- Frame is too slow and `cargo flamegraph` blames `vkQueueSubmit` (GPU is the bottleneck, not CPU).
- Validation error names a specific draw or pipeline — capture, navigate to it, read the state.

## Capture

**Option 1 — RenderDoc UI (recommended for one-offs):**
1. Open `qrenderdoc.exe`.
2. *File → Launch Application*.
3. Executable: `target\release\vulkan-rust.exe` (build first: `cargo build --release`).
4. Working dir: the repo root (so `config.toml` and `shaders/` resolve).
5. Launch. Press `F12` (or PrintScreen) in the running app to capture a frame.
6. Double-click the new capture in the Captures list — opens the analyser.

**Option 2 — `vkconfig` / `VK_LAYER_RENDERDOC_Capture`:**
- Set `VK_INSTANCE_LAYERS=VK_LAYER_RENDERDOC_Capture` in env, run normally, press `F12`.
- Useful when launching from an IDE.

**Option 3 — Programmatic (in-app):** not currently set up; would need `renderdoc` crate. Don't add unless asked.

## Navigating a capture — what to check

### The Event Browser (left panel)
Each frame in this renderer is small: one `vkBeginCommandBuffer` → one `vkCmdBeginRenderPass` → 1–3 draws (depending on geometry mode) → optional egui draws → `vkCmdEndRenderPass` → `vkEndCommandBuffer` → submit → present.

If you see *more* than that — something is dispatching extra work each frame. That's a bug.

### Pipeline state (right panel) at the main draw
- **VS / GS / FS**: confirms which shader modules are bound. The names from RenderDoc will be hashes (we don't set debug names — flag as a fix-it if it bothers you).
- **Push constants**: lists every byte. Compare against `PushConstants` field order in `src/graphics/mod.rs`. If a field reads as garbage, you've likely mismatched alignment with the GLSL side.
- **Descriptor sets**: only `ComputeGenerated` preset uses descriptors. For fullscreen presets, expect "no descriptor sets bound" — that's correct.
- **Vertex inputs**: `Trivial` mode = 3 verts, position + uv. `InstancedTriangles` = 6 indices × 10000 instances. `GeometryShader` = 400 points, no vertex inputs. `ComputeGenerated` = 3 indices × 50000 instances.
- **Viewport / scissor**: should match the swapchain extent shown in the app's window title or FPS log.

### Texture viewer
- Open the colour attachment after the main draw → is it the expected shader output before egui composites?
- Open after egui draws → did egui clobber anything?

### Mesh viewer (for non-Trivial geometry)
- `InstancedTriangles` and `ComputeGenerated`: confirm instance data is what you expect. If the compute shader is writing garbage, this is where you'll see it first.
- `GeometryShader`: see the points going in; see the expanded triangles after the geometry stage.

### Pixel history (right-click a pixel)
- Why is this pixel black? Lists every draw that touched it and what it wrote.
- Common find: blending state wrong, or the wrong shader is bound for the active preset.

## Mapping findings back to code

| RenderDoc says... | Look at... |
|--|--|
| Wrong shader bound | `VulkanPipeline::create_graphics_pipeline` in `src/gfx.rs`, and `ShaderSources::load_from_config` |
| Wrong push-constant value | `App::get_push_constants` in `src/main.rs`, and `InputManager::get_frame_state` |
| Wrong vertex inputs | The match-on-`geometry_mode` block in `create_graphics_pipeline` (around line 733 of `src/gfx.rs`) |
| Compute output is garbage | `shaders/point_generator.comp` and the descriptor binding in `create_compute_pipeline` |
| Validation error references draw #N | Use the Event Browser to find draw #N's source line in the command buffer recording (`record_command_buffer` in `src/gfx.rs`) |
| Pipeline layout missing geometry-stage push constants | Known latent issue — see `shader-hot-reload` skill, "geometry mode change" gotcha |

## What captures are good for and what they're not

**Good for:**
- "Is this push-constant value what I think it is?" → yes, read it.
- "Which draw produced this pixel?" → pixel history.
- "Is the vertex input correct?" → mesh viewer.
- "Is the pipeline I think is bound actually bound?" → pipeline state.

**Bad for:**
- CPU-side timing (use `tracy`, `cargo flamegraph`, or `src/utils/profiling.rs`).
- Multi-frame analysis (RenderDoc is single-frame; for trends use `vkpmctl` or NSight).
- Triggering on a specific input event — captures fire on the F12 keystroke; if the bug is transient you'll need to hold a chord or modify the F12 handler.

## Reference frame

`Renderdoc.rdc` in the repo root is a reference capture from an earlier preset. Open it side-by-side with a new capture to compare:
- Pipeline state at the main draw
- Push-constant byte layout (you can spot a layout drift here)
- Number and order of vkCmd calls per frame

If a regression shows up, "draw it the same way it used to" is a viable strategy — just match the reference capture's pipeline state.

## Output rules

When the user pastes RenderDoc findings (state, disasm, errors), translate to file:line in our code. Don't restate what RenderDoc said. Say "this maps to `src/gfx.rs:489`; the fix is …".
