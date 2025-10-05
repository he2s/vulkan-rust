// Graphics module - contains the main Gfx struct and all Vulkan graphics structures
// Extracted from main.rs for better modularity

use anyhow::{Result, anyhow};
use ash::{Entry, vk, khr::{surface, swapchain}};
use std::{
    cell::RefCell,
    ffi::{CString, c_char},
};
use winit::window::Window;

// Re-export graphics types from graphics module
use crate::graphics::{
    VulkanContext, VulkanSwapchain, VulkanBuffers, GeometryMode, PushConstants,
    Vertex, InstanceData, PointData, ShaderSources
};
use crate::config::config::{ShaderConfig, GeometryType};

// ============================================================================
// VULKAN STRUCTURES
// Core Vulkan wrapper structures used by the Gfx interface
// ============================================================================

pub struct VulkanPipeline {
    pub render_pass: vk::RenderPass,
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    pub framebuffers: Vec<vk::Framebuffer>,
    pub geometry_mode: GeometryMode,
    // Compute pipeline for point generation
    pub compute_pipeline_layout: vk::PipelineLayout,
    pub compute_pipeline: vk::Pipeline,
    #[allow(dead_code)]
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    #[allow(dead_code)]
    pub descriptor_pool: vk::DescriptorPool,
    pub descriptor_set: vk::DescriptorSet,
    #[allow(dead_code)]
    pub use_compute_generation: bool,
}

pub struct VulkanCommands {
    pub pool: vk::CommandPool,
    pub buffers: Vec<vk::CommandBuffer>,
    pub frame_index: RefCell<usize>,
}

pub struct VulkanSync {
    image_available: vk::Semaphore,
    render_finished: vk::Semaphore,
    in_flight: vk::Fence,
}

#[derive(Default)]
pub struct VulkanState {
    current_extent: Option<vk::Extent2D>,
}

// ============================================================================
// GFX MAIN STRUCTURE
// The main graphics interface that combines all Vulkan components
// ============================================================================

pub struct Gfx {
    pub context: VulkanContext,
    pub swapchain: VulkanSwapchain,
    pub pipeline: VulkanPipeline,
    pub buffers: VulkanBuffers,
    pub commands: VulkanCommands,
    pub sync: VulkanSync,
    pub state: VulkanState,
    pub vsync: bool,
}

// ============================================================================
// IMPLEMENTATIONS
// Implementation blocks for all the graphics structures
// ============================================================================

impl Gfx {
    /// # Safety
    /// This function is unsafe because it creates Vulkan resources and calls unsafe Vulkan functions.
    /// The caller must ensure that the window handle is valid and that Vulkan is properly initialized.
    pub unsafe fn new(window: &Window, shader_config: &ShaderConfig, vsync: bool, validation_layers: bool) -> Result<Self> {
        let context = unsafe { VulkanContext::new(window, validation_layers)? };
        let swapchain = unsafe { VulkanSwapchain::new(&context, window, vsync)? };
        let buffers = unsafe { VulkanBuffers::new(&context)? };
        let pipeline = unsafe { VulkanPipeline::new(&context, &swapchain, shader_config, &buffers)? };
        let commands = unsafe { VulkanCommands::new(&context)? };
        let sync = unsafe { VulkanSync::new(&context)? };

        Ok(Self {
            context,
            swapchain,
            pipeline,
            buffers,
            commands,
            sync,
            state: VulkanState::default(),
            vsync,
        })
    }

    /// # Safety
    /// This function is unsafe because it calls unsafe Vulkan functions to destroy and recreate swapchain resources.
    /// The caller must ensure that all operations on the swapchain have completed before calling this function.
    pub unsafe fn recreate_swapchain(&mut self, window: &Window) -> Result<()> {
        unsafe { self.context.device.device_wait_idle()? };

        unsafe { self.pipeline.cleanup_framebuffers(&self.context.device) };
        unsafe { self.swapchain.cleanup(&self.context.device) };

        self.swapchain = unsafe { VulkanSwapchain::new(&self.context, window, self.vsync)? };
        unsafe { self.pipeline
            .recreate_framebuffers(&self.context.device, &self.swapchain)? };

        self.state.current_extent = None;

        println!(
            "Swapchain recreated: {}x{}",
            self.swapchain.extent.width, self.swapchain.extent.height
        );
        Ok(())
    }

    /// # Safety
    /// This function is unsafe because it calls unsafe Vulkan functions to destroy and recreate pipeline resources.
    /// The caller must ensure that all operations using the pipeline have completed before calling this function.
    pub unsafe fn recreate_pipeline(&mut self, shader_config: &ShaderConfig) -> Result<()> {
        unsafe { self.context.device.device_wait_idle()? };

        unsafe { self.pipeline.cleanup_pipeline(&self.context.device) };

        unsafe { self.pipeline
            .create_pipeline(&self.context, &self.swapchain, shader_config)? };

        println!("Pipeline recreated successfully");
        Ok(())
    }

    /// # Safety
    /// This function is unsafe because it calls numerous unsafe Vulkan functions for command buffer recording and submission.
    /// The caller must ensure that Vulkan resources are properly initialized and that synchronization is handled correctly.
    pub unsafe fn draw(
        &mut self,
        push_constants: &PushConstants,
        egui_primitives: Option<&Vec<egui::ClippedPrimitive>>,
        egui_overlay: Option<&mut crate::graphics::egui_overlay::EguiOverlay>,
    ) -> Result<bool> {
        unsafe { self.context
            .device
            .wait_for_fences(&[self.sync.in_flight], true, u64::MAX)? };
        unsafe { self.context.device.reset_fences(&[self.sync.in_flight])? };

        let (image_index, needs_recreation) = unsafe { self.acquire_next_image()? };
        if needs_recreation {
            return Ok(true);
        }

        let cmd_buffer = self.commands.get_current_buffer();
        unsafe { self.record_command_buffer(cmd_buffer, image_index, push_constants, egui_primitives, egui_overlay)? };

        unsafe { self.submit_commands(cmd_buffer)? };

        unsafe { self.present_image(image_index) }
    }

    unsafe fn acquire_next_image(&self) -> Result<(u32, bool)> {
        match unsafe { self.swapchain.loader.acquire_next_image(
            self.swapchain.swapchain,
            u64::MAX,
            self.sync.image_available,
            vk::Fence::null(),
        ) } {
            Ok((index, suboptimal)) => Ok((index, suboptimal)),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => Ok((0, true)),
            Err(e) => Err(anyhow!("Failed to acquire image: {:?}", e)),
        }
    }

    unsafe fn record_command_buffer(
        &mut self,
        cmd_buffer: vk::CommandBuffer,
        image_index: u32,
        push_constants: &PushConstants,
        egui_primitives: Option<&Vec<egui::ClippedPrimitive>>,
        egui_overlay: Option<&mut crate::graphics::egui_overlay::EguiOverlay>,
    ) -> Result<()> {
        unsafe {
            self.context
                .device
                .reset_command_buffer(cmd_buffer, vk::CommandBufferResetFlags::empty())?;
        }

        unsafe {
            self.context
                .device
                .begin_command_buffer(cmd_buffer, &vk::CommandBufferBeginInfo::default())?;
        }

        let clear_values = [vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        }];

        let render_area = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: self.swapchain.extent,
        };

        let render_pass_begin = vk::RenderPassBeginInfo {
            render_pass: self.pipeline.render_pass,
            framebuffer: self.pipeline.framebuffers[image_index as usize],
            render_area,
            clear_value_count: clear_values.len() as u32,
            p_clear_values: clear_values.as_ptr(),
            ..Default::default()
        };

        unsafe {
            self.context.device.cmd_begin_render_pass(
                cmd_buffer,
                &render_pass_begin,
                vk::SubpassContents::INLINE,
            );
        }
        unsafe {
            self.context.device.cmd_bind_pipeline(
                cmd_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline.pipeline,
            );
        }

        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: self.swapchain.extent.width as f32,
            height: self.swapchain.extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        unsafe {
            self.context
                .device
                .cmd_set_viewport(cmd_buffer, 0, &[viewport]);
        }
        unsafe {
            self.context
                .device
                .cmd_set_scissor(cmd_buffer, 0, &[render_area]);
        }

        let push_constant_stages = match self.pipeline.geometry_mode {
            GeometryMode::GeometryShader => {
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::GEOMETRY | vk::ShaderStageFlags::FRAGMENT
            }
            _ => {
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT
            }
        };

        unsafe {
            self.context.device.cmd_push_constants(
                cmd_buffer,
                self.pipeline.pipeline_layout,
                push_constant_stages,
                0,
                std::slice::from_raw_parts(
                    push_constants as *const PushConstants as *const u8,
                    std::mem::size_of::<PushConstants>(),
                ),
            );
        }

        // Choose drawing method based on geometry mode
        match self.pipeline.geometry_mode {
            GeometryMode::Trivial => {
                // Optimized fullscreen triangle for fragment-only shaders
                let vertex_buffers = [self.buffers.trivial_vertex_buffer];
                let offsets = [0];
                unsafe {
                    self.context.device.cmd_bind_vertex_buffers(cmd_buffer, 0, &vertex_buffers, &offsets);
                    self.context.device.cmd_draw(cmd_buffer, 3, 1, 0, 0);
                }
            }
            GeometryMode::ComputeGenerated => {
            // Dispatch compute shader first to generate particles
            unsafe {
                self.context.device.cmd_bind_pipeline(
                    cmd_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    self.pipeline.compute_pipeline,
                );
            }

            // Bind descriptor set for compute shader
            unsafe {
                self.context.device.cmd_bind_descriptor_sets(
                    cmd_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    self.pipeline.compute_pipeline_layout,
                    0,
                    &[self.pipeline.descriptor_set],
                    &[],
                );
            }

            // Push constants for compute shader
            unsafe {
                self.context.device.cmd_push_constants(
                    cmd_buffer,
                    self.pipeline.compute_pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    std::slice::from_raw_parts(
                        push_constants as *const PushConstants as *const u8,
                        std::mem::size_of::<PushConstants>(),
                    ),
                );
            }

            // Dispatch compute shader (50,000 points, 64 threads per workgroup)
            unsafe {
                self.context.device.cmd_dispatch(cmd_buffer, 50000_u32.div_ceil(64), 1, 1);
            }

            // Memory barrier to ensure compute writes complete before vertex reading
            let barrier = vk::MemoryBarrier {
                src_access_mask: vk::AccessFlags::SHADER_WRITE,
                dst_access_mask: vk::AccessFlags::VERTEX_ATTRIBUTE_READ,
                ..Default::default()
            };

            unsafe {
                self.context.device.cmd_pipeline_barrier(
                    cmd_buffer,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::VERTEX_INPUT,
                    vk::DependencyFlags::empty(),
                    &[barrier],
                    &[],
                    &[],
                );
            }

            // Switch back to graphics pipeline for rendering
            unsafe {
                self.context.device.cmd_bind_pipeline(
                    cmd_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipeline.pipeline,
                );
            }

            // Bind descriptor set for vertex shader
            unsafe {
                self.context.device.cmd_bind_descriptor_sets(
                    cmd_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipeline.pipeline_layout,
                    0,
                    &[self.pipeline.descriptor_set],
                    &[],
                );
            }

            // Bind vertex buffer (triangle vertices)
            let vertex_buffers = [self.buffers.vertex_buffer];
            let offsets = [0];
            unsafe {
                self.context.device.cmd_bind_vertex_buffers(cmd_buffer, 0, &vertex_buffers, &offsets);
            }

            // Bind index buffer
            unsafe {
                self.context.device.cmd_bind_index_buffer(
                    cmd_buffer,
                    self.buffers.index_buffer,
                    0,
                    vk::IndexType::UINT16
                );
            }

            // Draw instanced triangles: 3 indices per triangle, 50,000 instances
            unsafe {
                self.context.device.cmd_draw_indexed(cmd_buffer, 3, 50000, 0, 0, 0);
            }

            }
            GeometryMode::GeometryShader => {
            // For geometry shader: draw points that will be expanded into triangles
            // No need for index buffer or instance buffer
            unsafe {
                self.context.device.cmd_draw(cmd_buffer, 400, 1, 0, 0); // 20x20 grid of points
            }
            }
            GeometryMode::InstancedTriangles => {
            // Traditional indexed drawing with instances
            // Bind vertex and instance buffers
            let vertex_buffers = [self.buffers.vertex_buffer];
            let instance_buffers = [self.buffers.instance_buffer];
            let offsets = [0];

            unsafe {
                self.context.device.cmd_bind_vertex_buffers(cmd_buffer, 0, &vertex_buffers, &offsets);
            }
            unsafe {
                self.context.device.cmd_bind_vertex_buffers(cmd_buffer, 1, &instance_buffers, &offsets);
            }

            // Bind index buffer
            unsafe {
                self.context.device.cmd_bind_index_buffer(
                    cmd_buffer,
                    self.buffers.index_buffer,
                    0,
                    vk::IndexType::UINT16
                );
            }

            // Draw indexed: 6 indices per rectangle, 10,000 instances (40,000 vertices instead of 60,000)
            unsafe {
                self.context.device.cmd_draw_indexed(cmd_buffer, 6, 10000, 0, 0, 0);
            }
            }
        }

        // Render egui overlay if primitives exist
        if let (Some(primitives), Some(overlay)) = (egui_primitives, egui_overlay) {
            if !primitives.is_empty() {
                unsafe {
                    overlay.render(
                        &self.context.instance,
                        &self.context.device,
                        self.context.physical_device,
                        cmd_buffer,
                        self.swapchain.extent,
                        primitives,
                    )?;
                }
            }
        }

        unsafe {
            self.context.device.cmd_end_render_pass(cmd_buffer);
        }
        unsafe {
            self.context.device.end_command_buffer(cmd_buffer)?;
        }

        Ok(())
    }

    unsafe fn submit_commands(&self, cmd_buffer: vk::CommandBuffer) -> Result<()> {
        // Ensure queue is idle before signaling semaphores to avoid VUID-vkQueueSubmit-pSignalSemaphores-00067
        unsafe { self.context.device.queue_wait_idle(self.context.queue)? };

        let wait_semaphores = [self.sync.image_available];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let signal_semaphores = [self.sync.render_finished];
        let command_buffers = [cmd_buffer];

        let submit_info = vk::SubmitInfo {
            wait_semaphore_count: wait_semaphores.len() as u32,
            p_wait_semaphores: wait_semaphores.as_ptr(),
            p_wait_dst_stage_mask: wait_stages.as_ptr(),
            command_buffer_count: command_buffers.len() as u32,
            p_command_buffers: command_buffers.as_ptr(),
            signal_semaphore_count: signal_semaphores.len() as u32,
            p_signal_semaphores: signal_semaphores.as_ptr(),
            ..Default::default()
        };

        unsafe { self.context.device.queue_submit(
            self.context.queue,
            &[submit_info],
            self.sync.in_flight,
        )? };
        Ok(())
    }

    unsafe fn present_image(&self, image_index: u32) -> Result<bool> {
        let swapchains = [self.swapchain.swapchain];
        let wait_semaphores = [self.sync.render_finished];
        let image_indices = [image_index];

        let present_info = vk::PresentInfoKHR {
            wait_semaphore_count: wait_semaphores.len() as u32,
            p_wait_semaphores: wait_semaphores.as_ptr(),
            swapchain_count: swapchains.len() as u32,
            p_swapchains: swapchains.as_ptr(),
            p_image_indices: image_indices.as_ptr(),
            ..Default::default()
        };

        match unsafe { self
            .swapchain
            .loader
            .queue_present(self.context.queue, &present_info) }
        {
            Ok(_) => Ok(false),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) | Err(vk::Result::SUBOPTIMAL_KHR) => Ok(true),
            Err(e) => Err(anyhow!("Failed to present: {:?}", e)),
        }
    }
}

impl VulkanPipeline {
    unsafe fn new(
        context: &VulkanContext,
        swapchain: &VulkanSwapchain,
        shader_config: &ShaderConfig,
        buffers: &VulkanBuffers,
    ) -> Result<Self> {
        let render_pass = unsafe { Self::create_render_pass(&context.device, swapchain.format)? };

        // Check if we should use compute generation
        let use_compute_generation = shader_config.presets
            .get(&shader_config.active_preset)
            .map(|p| p.geometry_type == GeometryType::Compute)
            .unwrap_or(false);

        // Determine geometry mode from shader sources first
        let shader_sources = ShaderSources::load_from_config(shader_config)?;
        let geometry_mode = shader_sources.determine_geometry_mode();

        // Create compute pipeline and descriptor sets first if needed
        let (compute_pipeline_layout, compute_pipeline, descriptor_set_layout, descriptor_pool, descriptor_set) =
            if use_compute_generation {
                unsafe { Self::create_compute_pipeline(&context.device, buffers, shader_config)? }
            } else {
                (vk::PipelineLayout::null(), vk::Pipeline::null(), vk::DescriptorSetLayout::null(),
                 vk::DescriptorPool::null(), vk::DescriptorSet::null())
            };

        // Create graphics pipeline layout with optional descriptor set layout
        let pipeline_layout = if use_compute_generation {
            unsafe { Self::create_graphics_pipeline_layout(&context.device, Some(descriptor_set_layout), geometry_mode)? }
        } else {
            unsafe { Self::create_graphics_pipeline_layout(&context.device, None, geometry_mode)? }
        };

        let (pipeline, _) = unsafe { Self::create_graphics_pipeline(
            &context.device,
            render_pass,
            pipeline_layout,
            swapchain,
            shader_config,
        )? };
        let framebuffers = unsafe { Self::create_framebuffers(&context.device, render_pass, swapchain)? };

        Ok(Self {
            render_pass,
            pipeline_layout,
            pipeline,
            framebuffers,
            geometry_mode,
            compute_pipeline_layout,
            compute_pipeline,
            descriptor_set_layout,
            descriptor_pool,
            descriptor_set,
            use_compute_generation,
        })
    }

    unsafe fn create_render_pass(
        device: &ash::Device,
        format: vk::Format,
    ) -> Result<vk::RenderPass> {
        let color_attachment = vk::AttachmentDescription {
            format,
            samples: vk::SampleCountFlags::TYPE_1,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
            ..Default::default()
        };

        let color_attachment_ref = vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        };

        let subpass = vk::SubpassDescription {
            pipeline_bind_point: vk::PipelineBindPoint::GRAPHICS,
            color_attachment_count: 1,
            p_color_attachments: &color_attachment_ref,
            ..Default::default()
        };

        let create_info = vk::RenderPassCreateInfo {
            attachment_count: 1,
            p_attachments: &color_attachment,
            subpass_count: 1,
            p_subpasses: &subpass,
            ..Default::default()
        };

        Ok(unsafe { device.create_render_pass(&create_info, None)? })
    }

    unsafe fn create_graphics_pipeline_layout(
        device: &ash::Device,
        descriptor_set_layout: Option<vk::DescriptorSetLayout>,
        geometry_mode: GeometryMode,
    ) -> Result<vk::PipelineLayout> {
        let stage_flags = match geometry_mode {
            GeometryMode::GeometryShader => {
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::GEOMETRY | vk::ShaderStageFlags::FRAGMENT
            }
            _ => {
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT
            }
        };

        let push_constant_range = vk::PushConstantRange {
            stage_flags,
            offset: 0,
            size: std::mem::size_of::<PushConstants>() as u32,
        };

        let create_info = if let Some(layout) = descriptor_set_layout {
            vk::PipelineLayoutCreateInfo {
                set_layout_count: 1,
                p_set_layouts: &layout,
                push_constant_range_count: 1,
                p_push_constant_ranges: &push_constant_range,
                ..Default::default()
            }
        } else {
            vk::PipelineLayoutCreateInfo {
                push_constant_range_count: 1,
                p_push_constant_ranges: &push_constant_range,
                ..Default::default()
            }
        };

        Ok(unsafe { device.create_pipeline_layout(&create_info, None)? })
    }

    unsafe fn create_graphics_pipeline(
        device: &ash::Device,
        render_pass: vk::RenderPass,
        pipeline_layout: vk::PipelineLayout,
        _swapchain: &VulkanSwapchain,
        shader_config: &ShaderConfig,
    ) -> Result<(vk::Pipeline, GeometryMode)> {
        println!("Loading shader preset: {}", shader_config.active_preset);
        let shader_sources = ShaderSources::load_from_config(shader_config)?;

        println!("Compiling shaders...");
        let vert_code = unsafe { Self::compile_shader(&shader_sources.vertex, shaderc::ShaderKind::Vertex)? };
        let frag_code = unsafe {
            Self::compile_shader(&shader_sources.fragment, shaderc::ShaderKind::Fragment)?
        };

        let vert_module = unsafe { Self::create_shader_module(device, &vert_code)? };
        let frag_module = unsafe { Self::create_shader_module(device, &frag_code)? };

        let (geom_module, _has_geometry) = if let Some(ref geometry_source) = shader_sources.geometry {
            let geom_code = unsafe { Self::compile_shader(geometry_source, shaderc::ShaderKind::Geometry)? };
            (Some(unsafe { Self::create_shader_module(device, &geom_code)? }), true)
        } else {
            (None, false)
        };

        let entry_name = CString::new("main")?;
        let mut shader_stages = vec![
            vk::PipelineShaderStageCreateInfo {
                stage: vk::ShaderStageFlags::VERTEX,
                module: vert_module,
                p_name: entry_name.as_ptr(),
                ..Default::default()
            },
        ];

        if let Some(geom_mod) = geom_module {
            shader_stages.push(vk::PipelineShaderStageCreateInfo {
                stage: vk::ShaderStageFlags::GEOMETRY,
                module: geom_mod,
                p_name: entry_name.as_ptr(),
                ..Default::default()
            });
        }

        shader_stages.push(vk::PipelineShaderStageCreateInfo {
            stage: vk::ShaderStageFlags::FRAGMENT,
            module: frag_module,
            p_name: entry_name.as_ptr(),
            ..Default::default()
        });

        // Determine geometry mode
        let geometry_mode = shader_sources.determine_geometry_mode();

        let (vertex_binding_descriptions, vertex_attribute_descriptions) = match geometry_mode {
            GeometryMode::Trivial => {
                // Simple vertex input for fullscreen triangle
                let bindings = [
                    vk::VertexInputBindingDescription {
                        binding: 0,
                        stride: std::mem::size_of::<Vertex>() as u32,
                        input_rate: vk::VertexInputRate::VERTEX,
                    },
                ];
                let attributes = [
                    vk::VertexInputAttributeDescription {
                        location: 0,
                        binding: 0,
                        format: vk::Format::R32G32_SFLOAT,
                        offset: 0,
                    },
                    vk::VertexInputAttributeDescription {
                        location: 1,
                        binding: 0,
                        format: vk::Format::R32G32_SFLOAT,
                        offset: 8,
                    },
                ];
                (bindings.to_vec(), attributes.to_vec())
            }
            GeometryMode::GeometryShader => {
                // No vertex input needed for geometry shader points
                (vec![], vec![])
            }
            _ => {
                // Complex vertex input for instanced rendering
                let bindings = [
                    vk::VertexInputBindingDescription {
                        binding: 0,
                        stride: std::mem::size_of::<Vertex>() as u32,
                        input_rate: vk::VertexInputRate::VERTEX,
                    },
                    vk::VertexInputBindingDescription {
                        binding: 1,
                        stride: std::mem::size_of::<InstanceData>() as u32,
                        input_rate: vk::VertexInputRate::INSTANCE,
                    },
                ];
                let attributes = [
                    vk::VertexInputAttributeDescription {
                        location: 0,
                        binding: 0,
                        format: vk::Format::R32G32_SFLOAT,
                        offset: 0,
                    },
                    vk::VertexInputAttributeDescription {
                        location: 1,
                        binding: 0,
                        format: vk::Format::R32G32_SFLOAT,
                        offset: 8,
                    },
                    vk::VertexInputAttributeDescription {
                        location: 2,
                        binding: 1,
                        format: vk::Format::R32G32_SFLOAT,
                        offset: 0,
                    },
                    vk::VertexInputAttributeDescription {
                        location: 3,
                        binding: 1,
                        format: vk::Format::R32G32_SFLOAT,
                        offset: 8,
                    },
                    vk::VertexInputAttributeDescription {
                        location: 4,
                        binding: 1,
                        format: vk::Format::R32_SFLOAT,
                        offset: 16,
                    },
                    vk::VertexInputAttributeDescription {
                        location: 5,
                        binding: 1,
                        format: vk::Format::R32_SFLOAT,
                        offset: 20,
                    },
                    vk::VertexInputAttributeDescription {
                        location: 6,
                        binding: 1,
                        format: vk::Format::R32_UINT,
                        offset: 24,
                    },
                ];
                (bindings.to_vec(), attributes.to_vec())
            }
        };

        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo {
            vertex_binding_description_count: vertex_binding_descriptions.len() as u32,
            p_vertex_binding_descriptions: vertex_binding_descriptions.as_ptr(),
            vertex_attribute_description_count: vertex_attribute_descriptions.len() as u32,
            p_vertex_attribute_descriptions: vertex_attribute_descriptions.as_ptr(),
            ..Default::default()
        };
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo {
            topology: match geometry_mode {
                GeometryMode::GeometryShader => vk::PrimitiveTopology::POINT_LIST,
                _ => vk::PrimitiveTopology::TRIANGLE_LIST,
            },
            ..Default::default()
        };

        let viewport_state = vk::PipelineViewportStateCreateInfo {
            viewport_count: 1,
            scissor_count: 1,
            ..Default::default()
        };

        let rasterizer = vk::PipelineRasterizationStateCreateInfo {
            polygon_mode: vk::PolygonMode::FILL,
            cull_mode: vk::CullModeFlags::NONE, // safer for fullscreen triangle
            front_face: vk::FrontFace::COUNTER_CLOCKWISE,
            line_width: 1.0,
            ..Default::default()
        };

        let multisampling = vk::PipelineMultisampleStateCreateInfo {
            rasterization_samples: vk::SampleCountFlags::TYPE_1,
            ..Default::default()
        };

        let color_blend_attachment = vk::PipelineColorBlendAttachmentState {
            color_write_mask: vk::ColorComponentFlags::RGBA,
            ..Default::default()
        };

        let color_blending = vk::PipelineColorBlendStateCreateInfo {
            attachment_count: 1,
            p_attachments: &color_blend_attachment,
            ..Default::default()
        };

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state = vk::PipelineDynamicStateCreateInfo {
            dynamic_state_count: dynamic_states.len() as u32,
            p_dynamic_states: dynamic_states.as_ptr(),
            ..Default::default()
        };

        let pipeline_info = vk::GraphicsPipelineCreateInfo {
            stage_count: shader_stages.len() as u32,
            p_stages: shader_stages.as_ptr(),
            p_vertex_input_state: &vertex_input_info,
            p_input_assembly_state: &input_assembly,
            p_viewport_state: &viewport_state,
            p_rasterization_state: &rasterizer,
            p_multisample_state: &multisampling,
            p_color_blend_state: &color_blending,
            p_dynamic_state: &dynamic_state,
            layout: pipeline_layout,
            render_pass,
            subpass: 0,
            ..Default::default()
        };

        let pipelines = unsafe { device
            .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None) }
            .map_err(|e| e.1)?;

        unsafe { device.destroy_shader_module(vert_module, None) };
        if let Some(geom_mod) = geom_module {
            unsafe { device.destroy_shader_module(geom_mod, None) };
        }
        unsafe { device.destroy_shader_module(frag_module, None) };

        Ok((pipelines[0], geometry_mode))
    }

    unsafe fn compile_shader(source: &str, kind: shaderc::ShaderKind) -> Result<Vec<u32>> {
        let compiler =
            shaderc::Compiler::new().ok_or_else(|| anyhow!("Failed to create shader compiler"))?;

        let result = compiler
            .compile_into_spirv(source, kind, "shader", "main", None)
            .map_err(|e| anyhow!("Shader compilation failed: {}", e))?;

        Ok(result.as_binary().to_vec())
    }

    unsafe fn create_shader_module(device: &ash::Device, code: &[u32]) -> Result<vk::ShaderModule> {
        let create_info = vk::ShaderModuleCreateInfo {
            code_size: code.len() * 4,
            p_code: code.as_ptr(),
            ..Default::default()
        };
        Ok(unsafe { device.create_shader_module(&create_info, None)? })
    }

    unsafe fn create_framebuffers(
        device: &ash::Device,
        render_pass: vk::RenderPass,
        swapchain: &VulkanSwapchain,
    ) -> Result<Vec<vk::Framebuffer>> {
        swapchain
            .views
            .iter()
            .map(|&view| {
                let create_info = vk::FramebufferCreateInfo {
                    render_pass,
                    attachment_count: 1,
                    p_attachments: &view,
                    width: swapchain.extent.width,
                    height: swapchain.extent.height,
                    layers: 1,
                    ..Default::default()
                };
                unsafe { device.create_framebuffer(&create_info, None) }
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(Into::into)
    }

    unsafe fn create_compute_pipeline(
        device: &ash::Device,
        buffers: &VulkanBuffers,
        shader_config: &ShaderConfig,
    ) -> Result<(vk::PipelineLayout, vk::Pipeline, vk::DescriptorSetLayout, vk::DescriptorPool, vk::DescriptorSet)> {
        // Create descriptor set layout for storage buffer
        let binding = vk::DescriptorSetLayoutBinding {
            binding: 0,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            ..Default::default()
        };

        let layout_info = vk::DescriptorSetLayoutCreateInfo {
            binding_count: 1,
            p_bindings: &binding,
            ..Default::default()
        };

        let descriptor_set_layout = unsafe { device.create_descriptor_set_layout(&layout_info, None)? };

        // Create pipeline layout with push constants and descriptor set
        let push_constant_range = vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            offset: 0,
            size: std::mem::size_of::<PushConstants>() as u32,
        };

        let pipeline_layout_info = vk::PipelineLayoutCreateInfo {
            set_layout_count: 1,
            p_set_layouts: &descriptor_set_layout,
            push_constant_range_count: 1,
            p_push_constant_ranges: &push_constant_range,
            ..Default::default()
        };

        let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None)? };

        // Compile compute shader
        let active_preset = shader_config.presets
            .get(&shader_config.active_preset)
            .ok_or_else(|| anyhow!("Active preset '{}' not found", shader_config.active_preset))?;

        let compute_filename = active_preset.compute
            .as_ref()
            .ok_or_else(|| anyhow!("No compute shader specified for preset '{}'", shader_config.active_preset))?;

        let compute_source = std::fs::read_to_string(format!("shaders/{}", compute_filename))
            .map_err(|e| anyhow!("Failed to load compute shader '{}': {}", compute_filename, e))?;
        let compute_code = unsafe { Self::compile_shader(&compute_source, shaderc::ShaderKind::Compute)? };
        let compute_module = unsafe { Self::create_shader_module(device, &compute_code)? };

        // Create compute pipeline
        let entry_name = CString::new("main")?;
        let pipeline_info = vk::ComputePipelineCreateInfo {
            stage: vk::PipelineShaderStageCreateInfo {
                stage: vk::ShaderStageFlags::COMPUTE,
                module: compute_module,
                p_name: entry_name.as_ptr(),
                ..Default::default()
            },
            layout: pipeline_layout,
            ..Default::default()
        };

        let pipeline = unsafe { device
            .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None) }
            .map_err(|e| e.1)?[0];

        // Create descriptor pool
        let pool_size = vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
        };

        let pool_info = vk::DescriptorPoolCreateInfo {
            flags: vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET,
            max_sets: 1,
            pool_size_count: 1,
            p_pool_sizes: &pool_size,
            ..Default::default()
        };

        let descriptor_pool = unsafe { device.create_descriptor_pool(&pool_info, None)? };

        // Allocate descriptor set
        let alloc_info = vk::DescriptorSetAllocateInfo {
            descriptor_pool,
            descriptor_set_count: 1,
            p_set_layouts: &descriptor_set_layout,
            ..Default::default()
        };

        let descriptor_sets = unsafe { device.allocate_descriptor_sets(&alloc_info)? };
        let descriptor_set = descriptor_sets[0];

        // Update descriptor set with storage buffer
        let buffer_info = vk::DescriptorBufferInfo {
            buffer: buffers.point_storage_buffer,
            offset: 0,
            range: vk::WHOLE_SIZE,
        };

        let write_descriptor_set = vk::WriteDescriptorSet {
            dst_set: descriptor_set,
            dst_binding: 0,
            dst_array_element: 0,
            descriptor_count: 1,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            p_buffer_info: &buffer_info,
            ..Default::default()
        };

        unsafe { device.update_descriptor_sets(&[write_descriptor_set], &[]) };

        // Cleanup shader module
        unsafe { device.destroy_shader_module(compute_module, None) };

        Ok((pipeline_layout, pipeline, descriptor_set_layout, descriptor_pool, descriptor_set))
    }

    unsafe fn create_pipeline(
        &mut self,
        context: &VulkanContext,
        swapchain: &VulkanSwapchain,
        shader_config: &ShaderConfig,
    ) -> Result<()> {
        let shader_sources = ShaderSources::load_from_config(shader_config)?;
        let geometry_mode = shader_sources.determine_geometry_mode();
        let (pipeline, _) = unsafe { Self::create_graphics_pipeline(
            &context.device,
            self.render_pass,
            self.pipeline_layout,
            swapchain,
            shader_config,
        )? };
        self.pipeline = pipeline;
        self.geometry_mode = geometry_mode;
        Ok(())
    }

    unsafe fn recreate_framebuffers(
        &mut self,
        device: &ash::Device,
        swapchain: &VulkanSwapchain,
    ) -> Result<()> {
        self.framebuffers = unsafe { Self::create_framebuffers(device, self.render_pass, swapchain)? };
        Ok(())
    }

    unsafe fn cleanup_framebuffers(&mut self, device: &ash::Device) {
        for &framebuffer in &self.framebuffers {
            unsafe { device.destroy_framebuffer(framebuffer, None) };
        }
        self.framebuffers.clear();
    }

    unsafe fn cleanup_pipeline(&mut self, device: &ash::Device) {
        unsafe { device.destroy_pipeline(self.pipeline, None) };
    }
}

impl VulkanCommands {
    unsafe fn new(context: &VulkanContext) -> Result<Self> {
        let pool_info = vk::CommandPoolCreateInfo {
            flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            queue_family_index: context.queue_family_index,
            ..Default::default()
        };

        let pool = unsafe { context.device.create_command_pool(&pool_info, None)? };

        let alloc_info = vk::CommandBufferAllocateInfo {
            command_pool: pool,
            level: vk::CommandBufferLevel::PRIMARY,
            command_buffer_count: 2,
            ..Default::default()
        };

        let buffers = unsafe { context.device.allocate_command_buffers(&alloc_info)? };

        Ok(Self {
            pool,
            buffers,
            frame_index: RefCell::new(0),
        })
    }

    fn get_current_buffer(&self) -> vk::CommandBuffer {
        let mut idx = self.frame_index.borrow_mut();
        let cmd = self.buffers[*idx % self.buffers.len()];
        *idx = (*idx + 1) % self.buffers.len();
        cmd
    }
}

impl VulkanSync {
    unsafe fn new(context: &VulkanContext) -> Result<Self> {
        let semaphore_info = vk::SemaphoreCreateInfo::default();
        let fence_info = vk::FenceCreateInfo {
            flags: vk::FenceCreateFlags::SIGNALED,
            ..Default::default()
        };

        Ok(Self {
            image_available: unsafe { context.device.create_semaphore(&semaphore_info, None)? },
            render_finished: unsafe { context.device.create_semaphore(&semaphore_info, None)? },
            in_flight: unsafe { context.device.create_fence(&fence_info, None)? },
        })
    }
}

impl Drop for Gfx {
    fn drop(&mut self) {
        unsafe {
            let _ = self.context.device.device_wait_idle();

            self.context.device.destroy_fence(self.sync.in_flight, None);
            self.context
                .device
                .destroy_semaphore(self.sync.render_finished, None);
            self.context
                .device
                .destroy_semaphore(self.sync.image_available, None);

            self.context
                .device
                .free_command_buffers(self.commands.pool, &self.commands.buffers);
            self.context
                .device
                .destroy_command_pool(self.commands.pool, None);

            self.pipeline.cleanup_framebuffers(&self.context.device);
            self.pipeline.cleanup_pipeline(&self.context.device);
            self.context
                .device
                .destroy_pipeline_layout(self.pipeline.pipeline_layout, None);
            self.context
                .device
                .destroy_render_pass(self.pipeline.render_pass, None);

            self.buffers.cleanup(&self.context.device);
            self.swapchain.cleanup(&self.context.device);

            self.context
                .surface_loader
                .destroy_surface(self.context.surface, None);
            self.context.device.destroy_device(None);
            self.context.instance.destroy_instance(None);
        }
    }
}