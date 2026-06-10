---
name: shader-hot-reload
description: Codifies the safe pipeline / swapchain / shader-module teardown order for hot-reload in this renderer. Catches validation errors and use-after-free bugs in recreate_pipeline and recreate_swapchain. Use when the user invokes /shader-hot-reload, when touching Gfx::recreate_pipeline, Gfx::recreate_swapchain, VulkanPipeline::cleanup_*, or when adding a new Vulkan resource that has to survive (or be rebuilt on) preset switches and window resizes.
---

# shader-hot-reload

Hot-reload is "destroy then rebuild." Get the order wrong, get a validation error or a GPU hang. This skill spells out the order.

## When to read this

- Editing `Gfx::recreate_pipeline` or `Gfx::recreate_swapchain` in `src/gfx.rs`.
- Editing `VulkanPipeline::cleanup_framebuffers`, `cleanup_pipeline`, or `Drop for Gfx`.
- Adding a new Vulkan resource — you need to decide which teardown path owns its destroy.
- Validation layer is complaining after `F5` (reload), `Tab` (preset cycle), or `F11` (fullscreen).

## The invariant

> **Before destroying any Vulkan resource the GPU might still be using: `device_wait_idle()`.**
> No exceptions. Yes, it's expensive. Yes, it serialises CPU and GPU. Yes, it's the right answer for hot-reload paths because hot-reload happens at human speed (~once per second at most).

Both `recreate_pipeline` and `recreate_swapchain` already call `device_wait_idle()` first. Any new teardown path must too.

## Teardown order — `Drop for Gfx` (canonical)

This is the order things are destroyed in `Drop for Gfx`. Memorise it; mirror it for partial-teardown paths.

```
1.  device_wait_idle()
2.  destroy_fence(in_flight)
3.  destroy_semaphore(render_finished)
4.  destroy_semaphore(image_available)
5.  free_command_buffers(commands.buffers)
6.  destroy_command_pool(commands.pool)
7.  pipeline.cleanup_framebuffers(device)   ← per-swapchain-image framebuffers
8.  pipeline.cleanup_pipeline(device)       ← the vk::Pipeline itself
9.  destroy_pipeline_layout(pipeline.pipeline_layout)
10. destroy_render_pass(pipeline.render_pass)
11. buffers.cleanup(device)
12. swapchain.cleanup(device)               ← views first, then swapchain
13. destroy_surface(context.surface)
14. destroy_device(context.device)
15. destroy_instance(context.instance)
```

**Reverse-dependency rule**: child resources die before parents. Framebuffers depend on the render pass; framebuffers go first. Pipeline depends on pipeline layout; pipeline goes first.

## `recreate_swapchain` — what gets rebuilt

Triggered by: window resize, `F11` fullscreen toggle, `ERROR_OUT_OF_DATE_KHR` / `SUBOPTIMAL_KHR` at acquire/present.

```
1. device_wait_idle()
2. pipeline.cleanup_framebuffers(device)
3. swapchain.cleanup(device)
4. VulkanSwapchain::new(...)        ← retried up to 3× on Linux
5. pipeline.recreate_framebuffers(device, &swapchain)
6. state.current_extent = None
```

What survives: render pass, pipeline, pipeline layout, descriptor sets, command pool, sync primitives, vertex/index/instance buffers, shader modules (already destroyed at pipeline-create time anyway).

**Gotcha**: if you ever make the render pass depend on the swapchain format and the format changes between recreations (rare, but happens when moving to an HDR monitor), you must also recreate the render pass — and therefore the pipeline. Currently we don't handle this. Add a check + log if you touch this code.

## `recreate_pipeline` — what gets rebuilt

Triggered by: `F5` (shader reload), `Tab` (preset cycle).

```
1. device_wait_idle()
2. pipeline.cleanup_pipeline(device)    ← only the vk::Pipeline; layout + render pass stay
3. pipeline.create_pipeline(...)         ← recompiles GLSL → SPIR-V → vk::ShaderModule → vk::Pipeline
                                            shader modules are destroyed at the end of create_pipeline.
```

What survives: render pass, pipeline layout (push constants + descriptor set layouts), framebuffers, swapchain, descriptor sets (if any), command pool, sync, buffers.

**Gotcha — geometry mode change**: if the new preset changes `geometry_mode` (e.g. `Trivial` → `GeometryShader`), the **pipeline layout's push-constant stage flags differ** (geometry shader needs `VERTEX | GEOMETRY | FRAGMENT`, others need `VERTEX | FRAGMENT`). Currently `recreate_pipeline` reuses the existing `pipeline_layout`. If you switch from a `Trivial` preset to a `GeometryShader` preset at runtime, the geometry-stage push constant won't be visible. Either:
- (a) recreate the pipeline layout too on geometry-mode change, or
- (b) always create the pipeline layout with `VERTEX | GEOMETRY | FRAGMENT | COMPUTE` stages (overhead is zero — the flags only affect validation).

This is a real latent bug. Flag it if relevant.

## Adding a new Vulkan resource — decision flow

For each new resource, answer:

1. **Lifetime?**
   - Lives for the whole app → create in `Gfx::new`, destroy in `Drop for Gfx`.
   - Per-pipeline (changes on preset switch) → create in `VulkanPipeline::create_pipeline`, destroy in `cleanup_pipeline` or at the start of `create_pipeline`.
   - Per-swapchain-image (changes on resize) → create in `recreate_framebuffers` (or extend it), destroy in `cleanup_framebuffers`.
   - Per-frame (e.g. dynamic descriptor) → use a ring buffer of pre-allocated resources, never create/destroy in `draw`.

2. **What does it depend on?** Destroy children before parents. Update `Drop for Gfx` to keep the order valid.

3. **Does the validation layer report a `VUID-vkDestroy...-...-...` after your change?** That's not a "warning"; it's "you destroyed something the GPU was still using." Add `device_wait_idle()` and reread the ordering.

## Validation-error decoder

When `validation_layers = true` and you see:

| Pattern | Cause | Fix |
|--|--|--|
| `VUID-vkDestroyPipeline-pipeline-00765` | Destroyed pipeline while GPU was using it | `device_wait_idle` before destroy |
| `VUID-vkAcquireNextImageKHR-semaphore-01779` | Reused `image_available` semaphore while still pending | Use per-frame-in-flight semaphores, or `queue_wait_idle` (current workaround) |
| `VUID-vkQueueSubmit-pSignalSemaphores-00067` | Signaling a semaphore that's already signaled | The current `queue_wait_idle` before submit prevents this; if you remove that, you need binary semaphore reset |
| `VUID-vkDestroyFramebuffer-framebuffer-00892` | Destroyed framebuffer mid-frame | Recreate path missing `device_wait_idle`, or wrong order |
| `VUID-vkCmdBeginRenderPass-framebuffer-...` | Framebuffer destroyed before render pass that owns it | Cleanup order wrong; framebuffers before render pass |

If you see a VUID not in this table, look it up at `vulkan.lunarg.com/doc/view/latest/windows/<VUID>.html` — never silently ignore.

## Output

When invoked, do **one** of:

- (a) Audit the user's proposed change against the ordering above. Cite `file:line` for each violation.
- (b) Walk the user through the safe teardown for a new resource they're adding.
- (c) Decode a validation error they've pasted, using the table above and the diff context.

Be terse. The user has the code open.
