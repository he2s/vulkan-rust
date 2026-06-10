---
name: perf-check
description: Renderer-specific performance audit of a Rust+GLSL diff. Flags per-frame allocations, lock contention, descriptor churn, redundant pipeline rebuilds, push-constant bloat, GLSL branch-heavy hot loops, and SPIR-V compile cost at runtime. Cites file:line. Use when the user invokes /perf-check, after any change to the draw path, push-constant assembly, input pipeline, or shader compilation, or before declaring a perf-sensitive change done.
---

# perf-check — vulkan-rust

This is not generic Rust perf review. It's the things that bite a real-time renderer running at vsync.

## When to invoke

- After a change to: `gfx.rs::draw`, `record_command_buffer`, `get_push_constants`, `App::window_event` (RedrawRequested arm), `InputManager::get_frame_state`, `processing/filters.rs`.
- Any new code under `graphics/vulkan/` that runs per-frame.
- Any GLSL change that adds loops, conditionals, or texture sampling to a hot fragment shader.
- Before saying "this is faster" — measure or stay quiet.

## Process

1. **Read the diff first.** `git diff main...HEAD` or the staged delta. Don't audit the whole repo.
2. **Classify each hunk** by where it runs:
   - **STARTUP** — once, in `App::resumed` / `Gfx::new`. Anything goes here.
   - **PRESET-LOAD** — `recreate_pipeline` / `recreate_swapchain`. Allowed to allocate, must `device_wait_idle` first.
   - **HOT** — `draw`, `record_command_buffer`, `get_push_constants`, `update_fps_tracking`, `about_to_wait`. Per-frame at vsync. Default budget: zero allocations, zero locks.
3. **Apply the checklist** below to HOT hunks. Be ruthless. Cite `file:line` for every finding.
4. **Output**: a flat list of findings, each with severity (`bug` / `perf` / `nit`), location, what's wrong, suggested fix. No prose padding.

## Checklist — Rust hot path

- [ ] Per-frame `Vec::new` / `String::new` / `format!` / `to_string()` / `collect()` / `to_vec()` — any of these inside `draw` or its callees?
- [ ] `clone()` of `Vec`, `String`, `Arc` in hot path. `Arc::clone` is cheap but not free; `Vec::clone` is a heap copy.
- [ ] `HashMap` / `HashSet` access in hot path with default `RandomState`. Use `FxHashMap` if it's actually on the hot path.
- [ ] `Mutex` from `std::sync`. Use `parking_lot::Mutex` if a lock is unavoidable; prefer atomics or a SPSC channel.
- [ ] `println!` / `eprintln!` in the draw path. They lock stdout. (FPS log every 10s is fine.)
- [ ] `Instant::now()` called more than once per frame for the same logical "now".
- [ ] Index re-computed in a tight loop (`frame_state.midi.notes[frame_state.midi.last_note as usize]` is fine once; not 1000×).
- [ ] `Option::unwrap_or_else(|| <expensive>)` where the some-case is hot. Use `match` and pre-compute the default at init.
- [ ] `if let Some(...)` chains that re-borrow self three times — does borrow-checker force a clone?
- [ ] Bounds check elision worth doing? (Rare. Only when measured.)

## Checklist — Vulkan-specific

- [ ] **Descriptor churn**: are descriptor sets being allocated *per frame* instead of once at pipeline create? `vkAllocateDescriptorSets` is slow; reuse the set, update bindings with `vkUpdateDescriptorSets` if the buffer changes.
- [ ] **Pipeline rebuild in draw path**: anything that calls `recreate_pipeline` from inside `draw`? That's a bug, not perf — pipeline create is ms, not μs.
- [ ] **SPIR-V compile inside draw path**: `shaderc::Compiler::new()` and `compile_into_spirv` are *expensive*. They belong in preset-load only. Verify they're not reachable from `draw`.
- [ ] **`device_wait_idle` per frame**: only acceptable at teardown / recreate. In `draw` it would serialise CPU and GPU and tank framerate.
- [ ] **`queue_wait_idle` in submit path**: `gfx.rs` currently has `queue_wait_idle` before signaling semaphores (to dodge VUID-vkQueueSubmit-pSignalSemaphores-00067). That's a stop-gap; flag it as a perf cost (~0.1–1 ms depending on GPU) and suggest the correct fix (binary semaphore reset via VK_KHR_synchronization2 or unsignaled-at-create — investigate).
- [ ] **Push-constant size**: ≤ 128 bytes total? `PushConstants` is currently 96 B. Adding more pushes you over the guaranteed minimum.
- [ ] **Sparse arrays in push constants**: a 128-entry array used for 4 notes is 512 B wasted bandwidth per draw. Use indices, not dense arrays.
- [ ] **Pipeline barrier scope**: `cmd_pipeline_barrier` with stage flags broader than needed (`ALL_COMMANDS`) stalls more pipeline stages than required.
- [ ] **`reset_command_buffer` per frame** is correct (we use `RESET_COMMAND_BUFFER` pool flag). `reset_command_pool` would be faster for multi-buffer pools but isn't always worth it.
- [ ] **Validation layers in release benches**: validation costs 10–50% framerate. Confirm `validation_layers = false` for any bench number cited.
- [ ] **vsync skews measurements**: any "FPS" claim with `vsync = true` is meaningless above 60 Hz. Disable for benching.

## Checklist — GLSL hot path

- [ ] **Branches on per-pixel uniforms** are cheap (uniform control flow). Branches on `gl_FragCoord` or sampled values are dynamic and cost. Flag deeply nested per-pixel `if`.
- [ ] **`pow(x, y)` with constant `y`**: `pow(x, 2.0)` is `x*x`. The compiler usually catches this; verify on AMD.
- [ ] **Trig in inner loops**: `sin`/`cos`/`atan` per-pixel inside a march loop. Tabulate or simplify.
- [ ] **`length()` vs `dot()` for distance comparisons**: compare squared distances when possible to skip a sqrt.
- [ ] **Texture sampling in a loop**: gather all samples into temps before the loop ends if the compiler can't hoist.
- [ ] **`discard` cost**: kills early-z and can disable HiZ on some GPUs. Acceptable in stylised shaders; flag in performance-critical ones.
- [ ] **Raymarching iteration counts**: hard-coded `for (int i = 0; i < 128; i++)` — is 128 needed, or is 32 + better step size enough?
- [ ] **`mix(a, b, t)` with `t` clamped to 0/1**: the compiler can't always elide. Hoist when known.

## Output format

```
PERF-CHECK SUMMARY

[bug]  shaders/new10.frag:42      — pow(x, 2.0) → x*x (likely already folded, but verify on AMD)
[perf] src/gfx.rs:489             — queue_wait_idle in submit path costs ~0.5ms/frame. Real fix: re-create image_available semaphore unsignaled.
[perf] src/main.rs:281            — get_push_constants reads frame_state.midi.notes[last_note] without bounds elision; sub-μs but in HOT path. Index is already u32→usize cast — fine as-is.
[nit]  src/processing/filters.rs:88 — `vec.clone()` in audio callback. Audio runs on its own thread, not the draw path. Allocation budget there is looser but not unlimited.
```

No prose around the list. Findings only. If there's nothing wrong, say "No findings." in one line and stop.
