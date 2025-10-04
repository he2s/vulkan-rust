// ============================================================================
// MODULAR ARCHITECTURE
// Refactored into clean module structure for better performance and maintainability
// ============================================================================

// Module declarations
mod audio;
mod beat_detection;
mod config;
mod graphics;
mod input;
mod processing;
mod state;
mod utils;

// Core application imports
use crate::config::config::{
    Args, AudioConfig, Config, ShaderConfig, ShaderPreset, GeometryType,
    load_or_create_config, print_startup_info,
};
use crate::graphics::{PushConstants, GeometryMode, Vertex, InstanceData, PointData, ShaderSources};
use crate::input::midi::{MidiConfig, MidiManager};
use crate::input::osc::{OscConfig, OscManager};
use crate::input::InputManager;
use crate::audio::{AudioLevels, AudioState, BeatState};
use crate::state::FrameState;
use crate::utils::DeviceLister;

// External crate imports
use anyhow::{Result, anyhow};
use clap::Parser;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use midir::{Ignore, MidiInput};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use ash::{Entry, vk, khr::{surface, swapchain}};
use std::{
    cell::RefCell,
    collections::VecDeque,
    ffi::{CString, CStr, c_char},
    fs,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use winit::{
    application::ApplicationHandler,
    event::{ElementState, KeyEvent, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Fullscreen, Window},
};

// ============================================================================
// CORE TYPES AND CONSTANTS
// This section contains fundamental constants and data structures used throughout the application
// These will remain here as they're shared across multiple modules
// ============================================================================

// constants
const FRAME_TIME_VSYNC: Duration = Duration::from_millis(16);
const FRAME_TIME_NO_VSYNC: Duration = Duration::from_millis(1);

// state management
// FrameState moved to state::FrameState








// ============================================================================
// GRAPHICS MODULE
// This section contains all Vulkan graphics structures and implementations
// Will be moved to graphics.rs in future modularization with sub-modules for each component
// ============================================================================

// vulkan graphics structures
pub struct VulkanContext {
    _entry: Entry,
    instance: ash::Instance,
    surface_loader: surface::Instance,
    surface: vk::SurfaceKHR,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    queue_family_index: u32,
    queue: vk::Queue,
}

pub struct VulkanSwapchain {
    loader: swapchain::Device,
    swapchain: vk::SwapchainKHR,
    extent: vk::Extent2D,
    format: vk::Format,
    #[allow(dead_code)]
    images: Vec<vk::Image>,
    views: Vec<vk::ImageView>,
}

pub struct VulkanBuffers {
    // Complex geometry buffers
    vertex_buffer: vk::Buffer,
    vertex_memory: vk::DeviceMemory,
    index_buffer: vk::Buffer,
    index_memory: vk::DeviceMemory,
    instance_buffer: vk::Buffer,
    instance_memory: vk::DeviceMemory,
    // Storage buffer for compute-generated points
    point_storage_buffer: vk::Buffer,
    point_storage_memory: vk::DeviceMemory,

    // Trivial geometry buffers (optimized fullscreen quad)
    trivial_vertex_buffer: vk::Buffer,
    trivial_vertex_memory: vk::DeviceMemory,
}

pub struct VulkanPipeline {
    render_pass: vk::RenderPass,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    framebuffers: Vec<vk::Framebuffer>,
    geometry_mode: GeometryMode,
    // Compute pipeline for point generation
    compute_pipeline_layout: vk::PipelineLayout,
    compute_pipeline: vk::Pipeline,
    #[allow(dead_code)]
    descriptor_set_layout: vk::DescriptorSetLayout,
    #[allow(dead_code)]
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    #[allow(dead_code)]
    use_compute_generation: bool,
}

pub struct VulkanCommands {
    pool: vk::CommandPool,
    buffers: Vec<vk::CommandBuffer>,
    frame_index: RefCell<usize>,
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

pub struct Gfx {
    context: VulkanContext,
    swapchain: VulkanSwapchain,
    pipeline: VulkanPipeline,
    buffers: VulkanBuffers,
    commands: VulkanCommands,
    sync: VulkanSync,
    state: VulkanState,
    vsync: bool,
}

// Gfx implementation - main graphics interface
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

// VulkanContext implementation - handles Vulkan instance and device creation
impl VulkanContext {
    unsafe fn new(window: &Window, validation_layers: bool) -> Result<Self> {
        let entry = Entry::linked();
        let display_handle = window.display_handle()?.as_raw();
        let window_handle = window.window_handle()?.as_raw();
        let required_extensions =
            ash_window::enumerate_required_extensions(display_handle)?;

        let instance = unsafe { Self::create_instance(&entry, required_extensions, validation_layers)? };
        let surface =
            unsafe { ash_window::create_surface(&entry, &instance, display_handle, window_handle, None)? };
        let surface_loader = surface::Instance::new(&entry, &instance);
        let (physical_device, queue_family_index) =
            unsafe { Self::select_physical_device(&instance, &surface_loader, surface)? };
        let (device, queue) =
            unsafe { Self::create_logical_device(&instance, physical_device, queue_family_index)? };

        Ok(Self {
            _entry: entry,
            instance,
            surface_loader,
            surface,
            physical_device,
            device,
            queue_family_index,
            queue,
        })
    }

    unsafe fn create_instance(
        entry: &Entry,
        required_extensions: &[*const c_char],
        validation_layers: bool,
    ) -> Result<ash::Instance> {
        let app_name = CString::new("vulkan-pixel-shader")?;

        let layer_names: Vec<CString> = if validation_layers {
            vec![CString::new("VK_LAYER_KHRONOS_validation")?]
        } else {
            vec![]
        };

        let layer_name_pointers: Vec<*const c_char> =
            layer_names.iter().map(|name| name.as_ptr()).collect();

        let app_info = vk::ApplicationInfo {
            p_application_name: app_name.as_ptr(),
            application_version: vk::make_api_version(0, 1, 0, 0),
            p_engine_name: app_name.as_ptr(),
            engine_version: vk::make_api_version(0, 1, 0, 0),
            api_version: vk::make_api_version(0, 1, 2, 0),
            ..Default::default()
        };

        let create_info = vk::InstanceCreateInfo {
            p_application_info: &app_info,
            enabled_layer_count: layer_name_pointers.len() as u32,
            pp_enabled_layer_names: layer_name_pointers.as_ptr(),
            enabled_extension_count: required_extensions.len() as u32,
            pp_enabled_extension_names: required_extensions.as_ptr(),
            ..Default::default()
        };

        Ok(unsafe { entry.create_instance(&create_info, None)? })
    }

    unsafe fn select_physical_device(
        instance: &ash::Instance,
        surface_loader: &surface::Instance,
        surface: vk::SurfaceKHR,
    ) -> Result<(vk::PhysicalDevice, u32)> {
        let physical_devices = unsafe { instance.enumerate_physical_devices()? };

        for device in physical_devices {
            let queue_families = unsafe { instance.get_physical_device_queue_family_properties(device) };

            for (index, queue_family) in queue_families.iter().enumerate() {
                let index = index as u32;

                if queue_family.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                    && unsafe { surface_loader.get_physical_device_surface_support(device, index, surface)? }
                {
                    return Ok((device, index));
                }
            }
        }

        Err(anyhow!("No suitable GPU found"))
    }

    unsafe fn create_logical_device(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        queue_family_index: u32,
    ) -> Result<(ash::Device, vk::Queue)> {
        let queue_priorities = [1.0f32];
        let queue_info = vk::DeviceQueueCreateInfo {
            queue_family_index,
            queue_count: 1,
            p_queue_priorities: queue_priorities.as_ptr(),
            ..Default::default()
        };

        let device_extensions = [swapchain::NAME.as_ptr()];
        let device_create_info = vk::DeviceCreateInfo {
            queue_create_info_count: 1,
            p_queue_create_infos: &queue_info,
            enabled_extension_count: device_extensions.len() as u32,
            pp_enabled_extension_names: device_extensions.as_ptr(),
            ..Default::default()
        };

        let device = unsafe { instance.create_device(physical_device, &device_create_info, None)? };
        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        Ok((device, queue))
    }
}

// VulkanSwapchain implementation - handles swapchain creation and management
impl VulkanSwapchain {
    unsafe fn new(context: &VulkanContext, window: &Window, vsync: bool) -> Result<Self> {
        let loader = swapchain::Device::new(&context.instance, &context.device);
        let surface_caps = unsafe { context
            .surface_loader
            .get_physical_device_surface_capabilities(context.physical_device, context.surface)? };
        let formats = unsafe { context
            .surface_loader
            .get_physical_device_surface_formats(context.physical_device, context.surface)? };

        let chosen = Self::choose_surface_format(&formats);
        let format = chosen.format;
        let extent = Self::choose_extent(&surface_caps, window);
        let image_count = Self::choose_image_count(&surface_caps);
        let present_mode = unsafe { Self::choose_present_mode(context, vsync) };

        let create_info = vk::SwapchainCreateInfoKHR {
            surface: context.surface,
            min_image_count: image_count,
            image_format: format,
            image_color_space: chosen.color_space,
            image_extent: extent,
            image_array_layers: 1,
            image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
            image_sharing_mode: vk::SharingMode::EXCLUSIVE,
            pre_transform: surface_caps.current_transform,
            composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
            present_mode,
            clipped: vk::TRUE,
            ..Default::default()
        };

        let swapchain = unsafe { loader.create_swapchain(&create_info, None)? };
        let images = unsafe { loader.get_swapchain_images(swapchain)? };
        let views = unsafe { Self::create_image_views(&context.device, &images, format)? };

        Ok(Self {
            loader,
            swapchain,
            extent,
            format,
            images,
            views,
        })
    }

    fn choose_surface_format(formats: &[vk::SurfaceFormatKHR]) -> vk::SurfaceFormatKHR {
        formats
            .iter()
            .find(|f| {
                f.format == vk::Format::B8G8R8A8_SRGB
                    && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .copied()
            .unwrap_or_else(|| formats[0])
    }

    unsafe fn choose_present_mode(context: &VulkanContext, vsync: bool) -> vk::PresentModeKHR {
        if vsync {
            return vk::PresentModeKHR::FIFO; // always available
        }
        let modes = unsafe { context
            .surface_loader
            .get_physical_device_surface_present_modes(context.physical_device, context.surface) }
            .unwrap_or_default();
        if modes.contains(&vk::PresentModeKHR::MAILBOX) {
            vk::PresentModeKHR::MAILBOX
        } else if modes.contains(&vk::PresentModeKHR::IMMEDIATE) {
            vk::PresentModeKHR::IMMEDIATE
        } else {
            vk::PresentModeKHR::FIFO
        }
    }

    fn choose_extent(caps: &vk::SurfaceCapabilitiesKHR, window: &Window) -> vk::Extent2D {
        if caps.current_extent.width != u32::MAX {
            caps.current_extent
        } else {
            let size = window.inner_size();
            vk::Extent2D {
                width: size.width.max(1),
                height: size.height.max(1),
            }
        }
    }

    fn choose_image_count(caps: &vk::SurfaceCapabilitiesKHR) -> u32 {
        let desired = caps.min_image_count + 1;
        if caps.max_image_count > 0 && desired > caps.max_image_count {
            caps.max_image_count
        } else {
            desired
        }
    }

    unsafe fn create_image_views(
        device: &ash::Device,
        images: &[vk::Image],
        format: vk::Format,
    ) -> Result<Vec<vk::ImageView>> {
        images
            .iter()
            .map(|&image| {
                let create_info = vk::ImageViewCreateInfo {
                    image,
                    view_type: vk::ImageViewType::TYPE_2D,
                    format,
                    subresource_range: vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    ..Default::default()
                };
                unsafe { device.create_image_view(&create_info, None) }
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(Into::into)
    }

    unsafe fn cleanup(&mut self, device: &ash::Device) {
        for &view in &self.views {
            unsafe { device.destroy_image_view(view, None) };
        }
        unsafe { self.loader.destroy_swapchain(self.swapchain, None) };
    }
}

// VulkanBuffers implementation - handles vertex, index, and storage buffers
impl VulkanBuffers {
    unsafe fn new(context: &VulkanContext) -> Result<Self> {
        // Triangle vertices for instanced rendering
        let vertices = [
            Vertex { pos: [0.0, 0.5], uv: [0.5, 0.0] },     // Top
            Vertex { pos: [-0.43, -0.25], uv: [0.0, 1.0] }, // Bottom left
            Vertex { pos: [0.43, -0.25], uv: [1.0, 1.0] },  // Bottom right
        ];

        // Triangle indices (single triangle)
        let indices: [u16; 3] = [0, 1, 2];

        // Create vertex buffer with simple host-visible memory for now
        let (vertex_buffer, vertex_memory) = unsafe { Self::create_buffer(
            &context.device,
            context.physical_device,
            &context.instance,
            &vertices,
            vk::BufferUsageFlags::VERTEX_BUFFER,
        )? };

        // Create index buffer
        let (index_buffer, index_memory) = unsafe { Self::create_buffer(
            &context.device,
            context.physical_device,
            &context.instance,
            &indices,
            vk::BufferUsageFlags::INDEX_BUFFER,
        )? };

        // Generate 10,000 rectangle instances with pre-computed rotations
        let mut instances = Vec::with_capacity(10000);
        let grid_size = 100; // 100x100 grid
        let spacing = 0.02;

        for y in 0..grid_size {
            for x in 0..grid_size {
                let offset_x = (x as f32 - grid_size as f32 * 0.5) * spacing;
                let offset_y = (y as f32 - grid_size as f32 * 0.5) * spacing;

                // Pre-compute base rotation (will be animated with time in shader)
                let base_rotation = (x + y) as f32 * 0.1;

                instances.push(InstanceData {
                    offset: [offset_x, offset_y],
                    scale: [0.008, 0.008], // Small rectangles
                    rotation_cos: base_rotation.cos(),
                    rotation_sin: base_rotation.sin(),
                    color_index: (x + y) % 16,
                    _padding: 0,
                });
            }
        }

        // Create instance buffer with simple host-visible memory for now
        let (instance_buffer, instance_memory) = unsafe { Self::create_buffer(
            &context.device,
            context.physical_device,
            &context.instance,
            &instances,
            vk::BufferUsageFlags::VERTEX_BUFFER,
        )? };

        // Create storage buffer for compute-generated points (50,000 points)
        const MAX_POINTS: usize = 50000;
        let point_data = vec![PointData {
            position: [0.0, 0.0],
            size: 0.01,
            intensity: 1.0,
            color: [1.0, 1.0, 1.0, 1.0],
            rotation: 0.0,
            point_type: 0,
            velocity: [0.0, 0.0],
        }; MAX_POINTS];

        let (point_storage_buffer, point_storage_memory) = unsafe { Self::create_buffer(
            &context.device,
            context.physical_device,
            &context.instance,
            &point_data,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::VERTEX_BUFFER,
        )? };

        // Create optimized trivial geometry (fullscreen triangle)
        let trivial_vertices = [
            Vertex { pos: [-1.0, -1.0], uv: [0.0, 0.0] },
            Vertex { pos: [3.0, -1.0], uv: [2.0, 0.0] },
            Vertex { pos: [-1.0, 3.0], uv: [0.0, 2.0] },
        ];

        let (trivial_vertex_buffer, trivial_vertex_memory) = unsafe { Self::create_buffer(
            &context.device,
            context.physical_device,
            &context.instance,
            &trivial_vertices,
            vk::BufferUsageFlags::VERTEX_BUFFER,
        )? };

        Ok(Self {
            vertex_buffer,
            vertex_memory,
            index_buffer,
            index_memory,
            instance_buffer,
            instance_memory,
            point_storage_buffer,
            point_storage_memory,
            trivial_vertex_buffer,
            trivial_vertex_memory,
        })
    }

    #[allow(dead_code)]
    unsafe fn create_gpu_buffer<T>(
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        instance: &ash::Instance,
        data: &[T],
        usage: vk::BufferUsageFlags,
    ) -> Result<(vk::Buffer, vk::DeviceMemory)> {
        let buffer_size = std::mem::size_of_val(data) as vk::DeviceSize;

        // Create staging buffer (CPU-accessible)
        let staging_buffer_info = vk::BufferCreateInfo {
            size: buffer_size,
            usage: vk::BufferUsageFlags::TRANSFER_SRC,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };

        let staging_buffer = unsafe { device.create_buffer(&staging_buffer_info, None)? };
        let staging_mem_requirements = unsafe { device.get_buffer_memory_requirements(staging_buffer) };

        let mem_properties = unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let staging_memory_type = Self::find_memory_type(
            staging_mem_requirements.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            &mem_properties,
        )?;

        let staging_alloc_info = vk::MemoryAllocateInfo {
            allocation_size: staging_mem_requirements.size,
            memory_type_index: staging_memory_type,
            ..Default::default()
        };

        let staging_memory = unsafe { device.allocate_memory(&staging_alloc_info, None)? };
        unsafe { device.bind_buffer_memory(staging_buffer, staging_memory, 0)? };

        // Copy data to staging buffer
        let data_ptr = unsafe { device.map_memory(
            staging_memory,
            0,
            buffer_size,
            vk::MemoryMapFlags::empty(),
        )? };

        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                data_ptr as *mut u8,
                buffer_size as usize,
            );
        }

        unsafe { device.unmap_memory(staging_memory) };

        // Create GPU-local buffer
        let buffer_info = vk::BufferCreateInfo {
            size: buffer_size,
            usage,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };

        let buffer = unsafe { device.create_buffer(&buffer_info, None)? };
        let mem_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

        let memory_type = Self::find_memory_type(
            mem_requirements.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            &mem_properties,
        )?;

        let alloc_info = vk::MemoryAllocateInfo {
            allocation_size: mem_requirements.size,
            memory_type_index: memory_type,
            ..Default::default()
        };

        let buffer_memory = unsafe { device.allocate_memory(&alloc_info, None)? };
        unsafe { device.bind_buffer_memory(buffer, buffer_memory, 0)? };

        // Copy from staging to GPU buffer
        unsafe { Self::copy_buffer(device, staging_buffer, buffer, buffer_size)? };

        // Cleanup staging resources
        unsafe { device.destroy_buffer(staging_buffer, None) };
        unsafe { device.free_memory(staging_memory, None) };

        Ok((buffer, buffer_memory))
    }

    #[allow(dead_code)]
    unsafe fn copy_buffer(
        _device: &ash::Device,
        _src_buffer: vk::Buffer,
        _dst_buffer: vk::Buffer,
        _size: vk::DeviceSize,
    ) -> Result<()> {
        // Note: In a real application, you'd want to use a dedicated transfer queue
        // For simplicity, we're using a simple synchronous copy
        // This would need the command pool and queue from VulkanContext
        Ok(())
    }

    unsafe fn create_buffer<T>(
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        instance: &ash::Instance,
        data: &[T],
        usage: vk::BufferUsageFlags,
    ) -> Result<(vk::Buffer, vk::DeviceMemory)> {
        let buffer_size = std::mem::size_of_val(data) as vk::DeviceSize;

        let buffer_info = vk::BufferCreateInfo {
            size: buffer_size,
            usage,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };

        let buffer = unsafe { device.create_buffer(&buffer_info, None)? };
        let mem_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

        let mem_properties = unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let memory_type = Self::find_memory_type(
            mem_requirements.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            &mem_properties,
        )?;

        let alloc_info = vk::MemoryAllocateInfo {
            allocation_size: mem_requirements.size,
            memory_type_index: memory_type,
            ..Default::default()
        };

        let buffer_memory = unsafe { device.allocate_memory(&alloc_info, None)? };
        unsafe { device.bind_buffer_memory(buffer, buffer_memory, 0)? };

        // Copy data to buffer
        let data_ptr = unsafe { device.map_memory(
            buffer_memory,
            0,
            buffer_size,
            vk::MemoryMapFlags::empty(),
        )? };

        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                data_ptr as *mut u8,
                buffer_size as usize,
            );
        }

        unsafe { device.unmap_memory(buffer_memory) };

        Ok((buffer, buffer_memory))
    }

    fn find_memory_type(
        type_filter: u32,
        properties: vk::MemoryPropertyFlags,
        mem_properties: &vk::PhysicalDeviceMemoryProperties,
    ) -> Result<u32> {
        for i in 0..mem_properties.memory_type_count {
            if (type_filter & (1 << i)) != 0
                && mem_properties.memory_types[i as usize].property_flags.contains(properties)
            {
                return Ok(i);
            }
        }
        Err(anyhow!("Failed to find suitable memory type"))
    }

    unsafe fn cleanup(&mut self, device: &ash::Device) {
        unsafe { device.destroy_buffer(self.vertex_buffer, None) };
        unsafe { device.free_memory(self.vertex_memory, None) };
        unsafe { device.destroy_buffer(self.index_buffer, None) };
        unsafe { device.free_memory(self.index_memory, None) };
        unsafe { device.destroy_buffer(self.instance_buffer, None) };
        unsafe { device.free_memory(self.instance_memory, None) };
        unsafe { device.destroy_buffer(self.point_storage_buffer, None) };
        unsafe { device.free_memory(self.point_storage_memory, None) };
        unsafe { device.destroy_buffer(self.trivial_vertex_buffer, None) };
        unsafe { device.free_memory(self.trivial_vertex_memory, None) };
    }
}

// VulkanPipeline implementation - handles render and compute pipeline creation
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

// VulkanCommands implementation - handles command buffers
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

// VulkanSync implementation - handles synchronization objects
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

// Drop implementation for cleanup
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

// ============================================================================
// INPUT MODULE
// This section handles all input management including MIDI, audio, and OSC
// Will be moved to input.rs in future modularization
// ============================================================================


// ============================================================================
// APPLICATION MODULE
// This section contains the main application structure and window event handling
// Will be moved to app.rs in future modularization
// ============================================================================

// main app
pub struct App {
    window: Option<Window>,
    gfx: Option<Gfx>,
    overlay: Option<crate::graphics::egui_overlay::EguiOverlay>,
    start_time: Option<Instant>,
    mouse_pos: (f64, f64),
    mouse_pressed: bool,
    input_manager: InputManager,
    config: Config,
    is_fullscreen: bool,
    current_shader_index: usize,
    shader_presets: Vec<String>,
    #[allow(dead_code)]
    frame_times: VecDeque<Instant>,
    last_fps_log: Instant,
    #[allow(dead_code)]
    frame_count: u64,
    frame_count_since_log: u32,
    cached_window_size: (u32, u32),

    // Tap tempo functionality
    tap_times: VecDeque<Instant>,
    manual_bpm: Option<f32>,
    manual_tempo_mode: bool,
    last_tap_display: Option<Instant>,
}

impl App {
    pub fn new(config: Config) -> Self {
        let shader_presets: Vec<String> = config.shader.presets
            .iter()
            .filter(|(_, preset)| preset.enabled)
            .map(|(key, _)| key.clone())
            .collect();

        let current_shader_index = shader_presets
            .iter()
            .position(|p| *p == config.shader.active_preset)
            .unwrap_or(0);

        let now = Instant::now();

        // Extract values before moving config
        let is_fullscreen = config.window.fullscreen;
        let window_size = (config.window.width, config.window.height);
        let tap_capacity = config.tap_tempo.max_tap_history + 2;

        Self {
            window: None,
            gfx: None,
            overlay: None,
            start_time: None,
            mouse_pos: (0.0, 0.0),
            mouse_pressed: false,
            input_manager: InputManager::new(),
            is_fullscreen,
            current_shader_index,
            shader_presets,
            config,

            frame_times: VecDeque::with_capacity(1000), // Store up to 1000 recent frame times
            last_fps_log: now,
            frame_count: 0,
            frame_count_since_log: 0,
            cached_window_size: window_size,

            // Tap tempo initialization
            tap_times: VecDeque::with_capacity(tap_capacity),
            manual_bpm: None,
            manual_tempo_mode: false,
            last_tap_display: None,
        }
    }

    #[inline]
    fn update_fps_tracking(&mut self) {
        self.frame_count_since_log += 1;

        let now = Instant::now();
        let elapsed = now.duration_since(self.last_fps_log);

        if elapsed.as_secs() >= 10 {
            let fps = self.frame_count_since_log as f64 / elapsed.as_secs_f64();

            println!(
                "Average FPS: {:.1} ({} frames in {:.1}s)",
                fps,
                self.frame_count_since_log,
                elapsed.as_secs_f64()
            );

            self.last_fps_log = now;
            self.frame_count_since_log = 0;
        }
    }

    fn toggle_fullscreen(&mut self) {
        if let Some(window) = &self.window {
            self.is_fullscreen = !self.is_fullscreen;

            let fullscreen = if self.is_fullscreen {
                Some(Fullscreen::Borderless(window.current_monitor()))
            } else {
                None
            };

            window.set_fullscreen(fullscreen);
            println!(
                "Toggled fullscreen: {}",
                if self.is_fullscreen { "ON" } else { "OFF" }
            );
        }
    }

    fn cycle_shader(&mut self) {
        if !self.config.shader.allow_runtime_switching {
            println!("Runtime shader switching is disabled in config");
            return;
        }

        self.current_shader_index = (self.current_shader_index + 1) % self.shader_presets.len();
        let new_preset = self.shader_presets[self.current_shader_index].clone();
        self.config.shader.active_preset = new_preset.clone();

        if let Some(gfx) = &mut self.gfx {
            println!("Switching to shader: {new_preset}");
            if let Err(e) = unsafe { gfx.recreate_pipeline(&self.config.shader) } {
                eprintln!("Failed to switch shader: {e}");
            }
        }
    }

    fn reload_shaders(&mut self) {
        if let Some(gfx) = &mut self.gfx {
            println!("Reloading and recompiling shaders...");
            match unsafe { gfx.recreate_pipeline(&self.config.shader) } {
                Ok(()) => {
                    println!("Shaders reloaded successfully!");
                }
                Err(e) => {
                    eprintln!("Failed to reload shaders: {e}");
                    println!("Check your shader files for compilation errors.");
                }
            }
        }
    }

    fn handle_tap_tempo(&mut self) {
        let now = Instant::now();
        let tap_config = &self.config.tap_tempo;

        // Add this tap to our history
        self.tap_times.push_back(now);

        // Keep only recent taps (using config values)
        while let Some(&first_tap) = self.tap_times.front() {
            if self.tap_times.len() > tap_config.max_tap_history
                || now.duration_since(first_tap).as_secs() > tap_config.tap_timeout_seconds {
                self.tap_times.pop_front();
            } else {
                break;
            }
        }

        // Calculate BPM if we have enough taps
        if self.tap_times.len() >= 2 {
            let mut intervals = Vec::new();
            for i in 1..self.tap_times.len() {
                let interval = self.tap_times[i].duration_since(self.tap_times[i-1]).as_secs_f32();
                intervals.push(interval);
            }

            // Calculate average interval
            let avg_interval = intervals.iter().sum::<f32>() / intervals.len() as f32;

            // Convert to BPM (beats per minute)
            let bpm = 60.0 / avg_interval;

            // Only accept reasonable BPM values (using config values)
            if bpm >= tap_config.min_bpm && bpm <= tap_config.max_bpm {
                self.manual_bpm = Some(bpm);
                self.manual_tempo_mode = true;
                self.last_tap_display = Some(now);

                // Set the manual BPM in the audio system
                self.input_manager.set_manual_bpm(bpm);

                println!("Tap tempo: {:.1} BPM (manual mode activated)", bpm);
            } else {
                println!("Tap tempo: {:.1} BPM out of range ({:.1}-{:.1})",
                    bpm, tap_config.min_bpm, tap_config.max_bpm);
            }
        } else {
            println!("Tap tempo: waiting for more taps...");
        }
    }

    fn get_push_constants(&mut self, elapsed: f32, w: u32, h: u32) -> PushConstants {
        let frame_state = self.input_manager.get_frame_state();

        let note_velocity = if frame_state.midi.note_count > 0 {
            frame_state.midi.notes[frame_state.midi.last_note as usize]
        } else {
            0.0
        };

        let blended_velocity = note_velocity.max(frame_state.audio_levels.level_rms);
        let blended_pitch_bend = frame_state
            .midi
            .pitch_bend
            .max(frame_state.audio_levels.low * 2.0 - 1.0);
        let blended_cc1 = frame_state.midi.controllers[1].max(frame_state.audio_levels.mid);
        let blended_cc74 = frame_state.midi.controllers[74].max(frame_state.audio_levels.high);

        PushConstants {
            time: elapsed,
            mouse_x: self.mouse_pos.0 as u32,
            mouse_y: self.mouse_pos.1 as u32,
            mouse_pressed: if self.mouse_pressed { 1 } else { 0 },
            note_velocity: blended_velocity,
            pitch_bend: blended_pitch_bend,
            cc1: blended_cc1,
            cc74: blended_cc74,
            note_count: frame_state.midi.note_count,
            last_note: frame_state.midi.last_note as u32,
            osc_ch1: frame_state.osc.channel1,
            osc_ch2: frame_state.osc.channel2,
            render_w: w,
            render_h: h,
            bpm: frame_state.beat.bpm,
            time_to_next_beat: frame_state.beat.time_to_next_beat,
            time_since_last_beat: frame_state.beat.time_since_last_beat,
            beats_per_bar: frame_state.beat.beats_per_bar,
            max_points: 50000,
            fft_size: 1024, // Could be configurable
            audio_intensity: frame_state.audio_levels.level_rms,
            bass_level: frame_state.audio_levels.low,
            mid_level: frame_state.audio_levels.mid,
            high_level: frame_state.audio_levels.high,
        }
    }

    fn print_controls(&self) {
        println!("Controls:");
        println!("  H     - Toggle preset overlay");
        println!("  F11   - Toggle fullscreen");
        println!("  F5    - Reload and recompile shaders");
        println!("  ESC   - Exit (or exit fullscreen)");
        println!("  SPACE - Tap tempo (set manual BPM)");
        if self.config.shader.allow_runtime_switching {
            println!("  TAB   - Cycle shaders");
        }
    }

    fn update_window_title(&mut self) {
        // Only update title every 30 frames (~0.5 seconds at 60fps) to avoid spam
        if !self.frame_count.is_multiple_of(30) {
            return;
        }

        if let Some(window) = &self.window {
            let frame_state = self.input_manager.get_frame_state();
            let base_title = &self.config.window.title;
            let bpm = frame_state.beat.bpm;

            let title = if self.manual_tempo_mode {
                format!("{} - {:.1} BPM (Manual)", base_title, bpm)
            } else {
                format!("{} - {:.1} BPM (Auto)", base_title, bpm)
            };

            window.set_title(&title);
        }
    }
}

impl Drop for App {
    fn drop(&mut self) {
        // Clean up egui before Vulkan context is destroyed
        if let (Some(overlay), Some(gfx)) = (&mut self.overlay, &self.gfx) {
            unsafe {
                overlay.destroy(&gfx.context.device);
            }
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let fullscreen = if self.is_fullscreen {
            Some(Fullscreen::Borderless(None))
        } else {
            None
        };

        let mut attributes = Window::default_attributes()
            .with_title(&self.config.window.title)
            .with_resizable(self.config.window.resizable)
            .with_fullscreen(fullscreen);

        if !self.is_fullscreen {
            attributes = attributes.with_inner_size(winit::dpi::LogicalSize::new(
                self.config.window.width as f64,
                self.config.window.height as f64,
            ));
        }

        let window = event_loop
            .create_window(attributes)
            .expect("Failed to create window");
        let gfx = unsafe {
            Gfx::new(&window, &self.config.shader, self.config.graphics.vsync, self.config.graphics.validation_layers)
                .expect("Failed to initialize Vulkan")
        };

        let mut overlay = crate::graphics::egui_overlay::EguiOverlay::new(&window)
            .expect("Failed to initialize egui overlay");

        // Initialize egui Vulkan resources
        unsafe {
            overlay
                .init_vulkan(
                    &gfx.context.instance,
                    &gfx.context.device,
                    gfx.context.physical_device,
                    gfx.pipeline.render_pass,
                    gfx.commands.pool,
                    gfx.context.queue,
                )
                .expect("Failed to initialize egui Vulkan resources");
        }

        self.window = Some(window);
        self.gfx = Some(gfx);
        self.overlay = Some(overlay);
        self.start_time = Some(Instant::now());

        self.input_manager.setup_midi(&self.config.midi);
        self.input_manager.setup_audio(&self.config.audio);
        self.input_manager.setup_osc(&self.config.osc);

        self.print_controls();
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _: winit::window::WindowId,
        event: WindowEvent,
    ) {
        // Let egui handle events only when overlay is visible
        let egui_consumed = if let (Some(window), Some(overlay)) = (&self.window, &mut self.overlay) {
            if overlay.show_overlay {
                overlay.handle_event(window, &event)
            } else {
                false
            }
        } else {
            false
        };

        // If egui consumed the event, don't process it further
        if egui_consumed {
            return;
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::Resized(new_size) => {
                if new_size.width > 0 && new_size.height > 0 {
                    // Update cached window size
                    self.cached_window_size = (new_size.width, new_size.height);
                    println!("Window resized to {}x{}", new_size.width, new_size.height);
                    if let (Some(gfx), Some(window)) = (&mut self.gfx, &self.window)
                        && let Err(e) = unsafe { gfx.recreate_swapchain(window) } {
                            eprintln!("Failed to recreate swapchain: {e}");
                            event_loop.exit();
                        }
                }
            }

            WindowEvent::KeyboardInput {
                event:
                KeyEvent {
                    physical_key,
                    state: ElementState::Pressed,
                    ..
                },
                ..
            } => match physical_key {
                PhysicalKey::Code(KeyCode::F11) => self.toggle_fullscreen(),
                PhysicalKey::Code(KeyCode::Escape) => {
                    if self.is_fullscreen {
                        self.toggle_fullscreen();
                    } else {
                        event_loop.exit();
                    }
                }
                PhysicalKey::Code(KeyCode::Tab) => self.cycle_shader(),
                PhysicalKey::Code(KeyCode::F5) => self.reload_shaders(),
                PhysicalKey::Code(KeyCode::Space) => self.handle_tap_tempo(),
                PhysicalKey::Code(KeyCode::KeyH) => {
                    if let Some(overlay) = &mut self.overlay {
                        overlay.toggle_overlay();
                    }
                }
                _ => {}
            },

            WindowEvent::CursorMoved { position, .. } => {
                self.mouse_pos = (position.x, position.y);
            }

            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => {
                self.mouse_pressed = state == ElementState::Pressed;
            }

            WindowEvent::RedrawRequested => {
                self.update_fps_tracking();
                self.update_window_title();
                if let Some(start_time) = &self.start_time {
                    let elapsed = start_time.elapsed().as_secs_f32();
                    // Use cached window size to avoid system call
                    let (width, height) = self.cached_window_size;
                    let push_constants =
                        self.get_push_constants(elapsed, width.max(1), height.max(1));

                    // Update egui overlay and get primitives
                    let egui_primitives = if let (Some(window), Some(overlay)) = (&self.window, &mut self.overlay) {
                        let current_preset = &self.shader_presets[self.current_shader_index];
                        let (_output, primitives) = overlay.run_ui(window, current_preset);
                        Some(primitives)
                    } else {
                        None
                    };

                    if let Some(gfx) = &mut self.gfx {
                        match unsafe { gfx.draw(&push_constants, egui_primitives.as_ref(), self.overlay.as_mut()) } {
                            Ok(true) => {
                                // Swapchain needs recreation
                                if let Some(window) = &self.window
                                    && let Err(e) = unsafe { gfx.recreate_swapchain(window) } {
                                        eprintln!("Failed to recreate swapchain: {e}");
                                        event_loop.exit();
                                    }
                            }
                            Ok(false) => {
                                // Draw succeeded normally
                            }
                            Err(e) => {
                                eprintln!("Draw error: {e}");
                                event_loop.exit();
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let frame_time = if self.config.graphics.vsync {
            FRAME_TIME_VSYNC
        } else {
            FRAME_TIME_NO_VSYNC
        };

        event_loop.set_control_flow(ControlFlow::WaitUntil(Instant::now() + frame_time));

        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

// ============================================================================
// MAIN FUNCTION
// This is the application entry point and will remain here
// ============================================================================

// main
fn main() -> Result<()> {
    env_logger::init();

    let args = Args::parse();

    if args.list_devices {
        DeviceLister::list_all_devices();
        return Ok(());
    }

    let mut config = load_or_create_config(&args.config)?;
    config.merge_with_args(&args);

    print_startup_info(&config);

    let event_loop = EventLoop::new().expect("Failed to create event loop");
    let frame_time = if config.graphics.vsync {
        FRAME_TIME_VSYNC
    } else {
        FRAME_TIME_NO_VSYNC
    };
    event_loop.set_control_flow(ControlFlow::WaitUntil(Instant::now() + frame_time));

    let mut app = App::new(config);
    let _ = event_loop.run_app(&mut app);

    Ok(())
}