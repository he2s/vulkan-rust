---
name: verify-frame
description: Build, run, and visually verify the renderer after a change. Confirms a shader or pipeline change actually produces the intended frame instead of relying on "it compiled". Use when the user invokes /verify-frame, after a non-trivial change to shaders, the draw path, or push-constant assembly, or before declaring any visual change done.
---

# verify-frame

`cargo check` passing means "the types line up." It does not mean "the renderer draws the right thing." For visual work, you have to *look*.

## When to invoke

- Edited a `.frag`, `.vert`, `.geom`, or `.comp` shader.
- Changed `PushConstants` (either side: Rust or GLSL).
- Changed anything in `record_command_buffer`, `create_graphics_pipeline`, or `create_compute_pipeline`.
- Added or modified a preset in `config.toml`.

## The loop

1. **Build.** `cargo build --release`. (Debug builds with `shaderc` are too slow for visual iteration.)
2. **Pick the preset.** Edit `config.toml`'s `[shader] active_preset = "..."` to the one your change targets. If the preset doesn't exist or is `enabled = false`, fix that first.
3. **Run.** `cargo run --release`.
4. **Look.**
   - Does it render at all, or black screen?
   - FPS log every 10s — within expected range (60 FPS with vsync, hundreds without)?
   - Validation layer log clean? (Run with `validation_layers = true` in `config.toml`. Any new VUID is a regression.)
5. **Cycle inputs.**
   - `Tab` to other presets — did you break presets you didn't touch?
   - `F5` to reload — does the shader recompile cleanly?
   - `F11` fullscreen → back → does swapchain recreate without artefacts?
   - If MIDI/OSC/audio is wired up, drive the relevant push-constant field (`cc1`, `note_velocity`, `bass_level`) and confirm the shader visibly reacts.
6. **Capture.** If anything looks off and code-reading doesn't reveal it, capture in RenderDoc (`F12` if running under it). See `renderdoc-capture` skill.
7. **Compare to reference.** If you have a reference image / `Renderdoc.rdc`, eyeball or side-by-side. (We don't currently have automated visual diff — if the user wants it, propose `image` crate + perceptual hash comparison; do not implement unless asked.)

## Failure modes and what they mean

| Symptom | Likely cause |
|--|--|
| Black screen, validation clean | Push-constant field mismatch (Rust ↔ GLSL); time/bpm reading as zero; uniform branch that's always false; cleared but never drawn (check `clear_values` in `record_command_buffer`) |
| Black screen, validation noisy | Pipeline layout / descriptor binding mismatch — read the VUID |
| Wrong colours, glitchy | Push-constant byte alignment drift; instance data wrong; geometry-mode mismatch (vertex inputs don't line up with what the shader expects) |
| Crashes on `F5` reload | Teardown ordering in `recreate_pipeline` — see `shader-hot-reload` skill |
| Crashes on `F11` fullscreen | `recreate_swapchain` failed; check the retry count in the log; on Linux this is common, on Windows it's usually a real bug |
| FPS drops sharply after the change | Run `/perf-check`. Likely per-frame alloc or descriptor churn |
| Validation error referencing a specific draw # | Open RenderDoc; navigate the event browser to that draw; read pipeline state |
| Egui overlay disappeared | `H` to toggle; if still gone, check `EguiOverlay::render` and the order it's called in `record_command_buffer` |

## Reporting

When the user asks "did this work?", reply with:
- ✓ what you verified visually (preset name, what reacted to what input)
- ✗ what you couldn't verify (e.g. "MIDI not connected — couldn't drive `cc1`")
- Validation layer status: clean / list of new VUIDs
- Average FPS in the last 10s log line

If you didn't actually run the binary (you only `cargo check`'d), say that. Don't claim verification you didn't do.

## What this skill does NOT do

- Automated screenshot diffing — not set up. Propose it if the user wants it; don't add without asking.
- Headless rendering — would need a swapchain replacement (offscreen image + readback). Not currently supported.
- Performance benchmarking — that's `/perf-check`'s job.
