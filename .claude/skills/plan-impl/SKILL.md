---
name: plan-impl
description: Write a tight, one-page implementation plan before non-trivial work in this Vulkan/Rust renderer. Forces the author to name invariants touched, hot-path implications, validation-layer risk, and rollback. Use when the user invokes /plan-impl, or proactively for any task that touches more than one of: gfx.rs, graphics/vulkan/, graphics/shaders/, PushConstants, descriptor sets, or sync primitives.
---

# plan-impl

A plan is not a planning document. It's a 1-page contract between you and the diff. If it's longer than 1 page, the change is too big; split it.

This is **not** a Notion brief. It's a thinking-tool to catch the failure modes that show up after you've written 300 lines and notice they don't compose.

## When to invoke

- Anything that grill-me flagged as needing decisions.
- Multi-file diffs that cross `gfx.rs` ↔ `graphics/vulkan/*` ↔ `graphics/shaders/*`.
- Adding a new geometry mode, a new pipeline variant, a new input source.
- Changing the `PushConstants` layout.
- Refactors of `App` lifecycle or window-event handling.

Do **not** invoke for: a GLSL tweak inside one preset, a config-only change, a clippy/lint fix, a typo.

## The template

Reply with exactly this skeleton, filled in. Anything you can't fill in is a question for the user — ask it instead of guessing.

```
## Goal
<One sentence. What works after this that doesn't work before. No "improve", no "refactor" — say what behaviour changes.>

## Non-goals
<Bullet list of things this change is NOT doing. Crucial for scope. Examples: "not rewriting the swapchain teardown", "not changing the MIDI input path", "not adding a UI".>

## Invariants touched
<For each, name the invariant and how this change preserves or modifies it.>
- PushConstants layout (Rust ↔ GLSL parity): <preserved | modified — show the diff to both sides>
- Vulkan resource cleanup order in Drop for Gfx: <unchanged | new resource added; cleanup added at line X>
- device_wait_idle before destroy: <yes, at recreate_pipeline / recreate_swapchain>
- Validation-layer-clean run: <yes | known new VUID at <site> with justification>
- Per-frame zero-allocation in draw path: <yes | no — and the measured cost>

## Files to change
<Path-by-path, one line each. Be specific. "Refactor renderer" is not a file change.>
- src/gfx.rs: <what>
- src/graphics/mod.rs: <what — e.g. add field to PushConstants>
- shaders/<name>.frag: <what — mirror the push_constant field>
- config.toml: <new preset entry if applicable>

## Hot-path implications
<Does this code run in draw / record_command_buffer / get_push_constants / acquire / submit / present?>
- New per-frame allocations: <none | <what>>
- New per-frame branches in tight loops: <none | <where>>
- New locks: <none | <which lock, scope, contention expected>>

## Rollback
<How to undo this in one commit if it ships broken. Configurable behind a config flag? Behind a new preset? Single-file revert?>

## Open questions
<Things you genuinely don't know. Ask them before writing code, not after.>
```

## Output rules

- 1 page maximum. If it overflows, split the work.
- No prose padding. No "in this section we will…". Bullets and short sentences.
- "TBD" is allowed in `Open questions` only. Everywhere else, fill it in or ask.
- Show the plan, **stop**, wait for the user. Don't start coding until they say go.

## What this skill does NOT do

- It doesn't write the code.
- It doesn't get committed as a `.md` file in the repo. Reply text only, unless the user explicitly asks for a file.
- It doesn't track progress across days. Use TaskCreate for that.
