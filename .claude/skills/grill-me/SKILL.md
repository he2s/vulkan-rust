---
name: grill-me
description: Interrogate the user with sharp, uncharitable design questions before implementing anything non-trivial in this Vulkan/Rust renderer. Surfaces unstated assumptions about synchronisation, descriptor scope, hot-path allocation, push-constant layout, shader hot-reload, and frame budget. Use when the user invokes /grill-me, or proactively at the start of any task involving a new pipeline, a new descriptor set layout, push-constant changes, draw-loop changes, or shader-preset infrastructure.
---

# grill-me — vulkan-rust edition

The point isn't to be hostile. The point is to surface the decisions the user hasn't realised they're making, *before* they're encoded in a diff that's expensive to undo.

Inspired by Matt Pocock's `grill-me`. Domain-tuned for a real-time Vulkan renderer where the wrong abstraction costs frames and the wrong teardown order costs you a validation error you'll spend an hour bisecting.

## When to invoke

- User asks for **anything** that touches `Gfx::draw`, `record_command_buffer`, or `get_push_constants`.
- New descriptor set layout, new pipeline, new render pass, new sync primitive.
- Adding a field to `PushConstants` (Rust ↔ GLSL contract — easy to break).
- New input source (MIDI/OSC/audio) feeding the shader.
- Anything labelled "refactor" that crosses two of: `gfx.rs`, `graphics/vulkan/`, `graphics/shaders/`.
- Shader hot-reload changes (also load `.claude/skills/shader-hot-reload/`).

Do **not** invoke for: a one-off GLSL tweak inside an existing fragment shader, a config-toml addition, a typo fix, a clippy warning.

## How to run it

Reply to the user with **3–6 questions, total**. Pick from the bank below. Skip categories that don't apply. No throat-clearing. Each question is one short paragraph. Don't ask anything the code already answers — read first.

End with: *"Answer the ones that matter. Skip the rest. Then I'll write the diff."* If a question is load-bearing and they skip it, ask it again with the reason.

## Question bank

### Frame budget & hot path
- Is this code per-frame, per-preset-load, or per-startup? If per-frame, what's its allocation budget? (Default: zero.)
- What's it doing that `PushConstants` couldn't carry? If you're reaching for a UBO or descriptor set, why isn't 4 more bytes of push constant the answer?
- Where does this run *before* `device_wait_idle` in the next teardown path? If you wouldn't notice it leaking, why is it heap-allocated?

### Vulkan correctness
- Which queue submits this? Which semaphore signals "done"? Which fence does the CPU wait on?
- What's the image layout going in, going out? Who issues the barrier?
- If this is a new descriptor set: how often does it change — per-frame, per-preset, never? `STATIC` / `DYNAMIC` / `STREAM` usage? Pool size budget?
- If the swapchain is recreated mid-frame, what happens to this resource? Does it survive `recreate_swapchain` or get rebuilt?
- Does this allocate Vulkan memory? Which heap? Will it survive a `recreate_pipeline`?

### Push-constant contract
- Adding/changing a field in `PushConstants`? Show me the matching GLSL `layout(push_constant) uniform` block. Same field order? Same types? `vec3`-then-`float` alignment hazards?
- Total push-constant size after the change — still ≤ 128 bytes (the guaranteed minimum)?
- Which presets in `config.toml` consume this field? What's the default if the input source isn't present (e.g. no MIDI device connected)?

### Shader / preset
- New shader file in `shaders/`? Which preset references it, and is the preset `enabled = true`?
- Does the shader compile on a clean `cargo run`? `F5` reload works? (Hot-reload paths use `recreate_pipeline`, which has its own teardown ordering — read `shader-hot-reload` skill if unsure.)
- Geometry type — `Trivial` (fullscreen tri), `InstancedTriangles`, `GeometryShader`, or `ComputeGenerated`? Vertex inputs match?

### Input pipeline (MIDI/OSC/audio)
- Where does this value enter the system? Where does `InputManager::get_frame_state()` surface it? Where does it land in `PushConstants`?
- What's the value when the source is *disconnected* — sensible default, or NaN/garbage?
- Threading: which thread produces, which thread consumes? Lock-free path or a mutex? Backpressure if the producer is faster than vsync?

### Scope & blast radius
- What's the smallest version of this change that delivers value? Why is the proposed version bigger?
- What breaks if I get this wrong — validation warning, black screen, crash, GPU hang, silent visual artifact?
- If I revert this in a week, what else has to come back with it?

### "Do we even need this?"
- Is there an existing preset / shader / mechanism that already does most of what you want? Why isn't extending it the answer?
- If we *don't* do this, what specifically gets worse?
- Is this a "wouldn't it be cool if…" or a "the current behaviour is wrong because…"? Be honest.

## Output rules

- 3–6 questions. Never more. If you've found more, the design is in worse shape than the user thinks — say so and pick the top 3–6.
- One question per paragraph. No bullet salads.
- Don't pre-answer. Don't suggest the fix in the question. The user answers; *then* you write the diff.
- If the user's answer reveals a deeper issue (e.g. "the validation layer is noisy so I turned it off"), stop. Surface it. Don't proceed with the original task.

## Anti-patterns

- ❌ "Have you considered X?" → say "X. Why not?"
- ❌ "What's your performance target?" with no anchor → "Frame budget is 16.6 ms at 60 Hz; what fraction is this allowed?"
- ❌ Sycophancy ("great question!", "interesting approach!").
- ❌ Asking what the code already shows. Read it first.
