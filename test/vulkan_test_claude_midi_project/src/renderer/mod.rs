// src/renderer/mod.rs

mod context;
mod swapchain;
mod pipeline;
mod command;
mod sync;
mod framebuffer;
mod shader;

pub use context::VulkanContext;
pub use swapchain::Swapchain;
pub use pipeline::Pipeline;
pub use command::CommandPool;
pub use sync::FrameSync;
pub use shader::{ShaderManager, ShaderSource};

use anyhow::Result;
use ash::vk;
use std::sync::Arc;
use winit::window::Window;
use tracing::{info, debug};

use crate::config::{GraphicsConfig, ShaderConfig};
use crate::utils::PushConstants;

pub struct Renderer {
    context: Arc<VulkanContext>,
    surface: vk::SurfaceKHR,
    swapchain: Swapchain,
    render_pass: vk::RenderPass,
    pipeline: Pipeline,
    framebuffers: Vec<vk::Framebuffer>,
    command_pool: CommandPool,
    sync: FrameSync,
    shader_manager: ShaderManager,
    frame_count: u64,
}

impl Renderer {
    pub fn new(window: &Window, graphics: &GraphicsConfig, shader: &ShaderConfig) -> Result<Self> {
        info!("Initializing Vulkan renderer");

        // Create core context
        let context = Arc::new(VulkanContext::new(window, graphics.validation_layers)?);

        // Create surface
        let surface = context.create_surface(window)?;

        // Create swapchain
        let swapchain = Swapchain::new(
            Arc::clone(&context),
            surface,
            window,
            graphics.vsync,
        )?;

        // Create render pass
        let render_pass = create_render_pass(&context.device(), swapchain.format())?;

        // Create shader manager
        let shader_manager = ShaderManager::new(shader.clone())?;

        // Create pipeline
        let pipeline = Pipeline::new(
            Arc::clone(&context),
            render_pass,
            swapchain.extent(),
            &shader_manager,
        )?;

        // Create framebuffers
        let framebuffers = framebuffer::create_framebuffers(
            &context.device(),
            swapchain.image_views(),
            render_pass,
            swapchain.extent(),
        )?;

        // Create command pool
        let command_pool = CommandPool::new(
            Arc::clone(&context),
            context.queue_family_index(),
        )?;

        // Create synchronization
        let sync = FrameSync::new(Arc::clone(&context))?;

        info!("Renderer initialized successfully");

        Ok(Self {
            context,
            surface,
            swapchain,
            render_pass,
            pipeline,
            framebuffers,
            command_pool,
            sync,
            shader_manager,
            frame_count: 0,
        })
    }

    pub fn draw(&mut self, push_constants: &PushConstants) -> Result<DrawResult> {
        self.frame_count += 1;
        debug!("Drawing frame {}", self.frame_count);

        // Wait for previous frame
        self.sync.wait()?;

        // Acquire next image
        let (image_index, suboptimal) = match self.swapchain.acquire_next_image(&self.sync) {
            Ok(result) => result,
            Err(SwapchainError::OutOfDate) => {
                return Ok(DrawResult::NeedsRecreation);
            }
            Err(e) => return Err(e.into()),
        };

        // Record command buffer
        let cmd = self.command_pool.begin()?;
        self.record_commands(cmd, image_index, push_constants)?;
        self.command_pool.end(cmd)?;

        // Submit and present
        self.submit(cmd)?;

        match self.swapchain.present(&self.sync, image_index) {
            Ok(_) => {
                if suboptimal {
                    Ok(DrawResult::Suboptimal)
                } else {
                    Ok(DrawResult::Success)
                }
            }
            Err(SwapchainError::OutOfDate) => Ok(DrawResult::NeedsRecreation),
            Err(e) => Err(e.into()),
        }
    }

    pub fn resize(&mut self, window: &Window) -> Result<()> {
        info!("Resizing renderer");

        // Wait for GPU to finish
        unsafe { self.context.device().device_wait_idle()?; }

        // Recreate swapchain
        self.swapchain.recreate(window)?;

        // Recreate framebuffers
        self.recreate_framebuffers()?;

        info!("Renderer resized to {}x{}", 
            self.swapchain.extent().width,
            self.swapchain.extent().height
        );

        Ok(())
    }

    pub fn reload_shaders(&mut self) -> Result<()> {
        info!("Reloading shaders");

        // Wait for GPU to finish
        unsafe { self.context.device().device_wait_idle()?; }

        // Reload shader sources
        self.shader_manager.reload()?;

        // Recreate pipeline
        self.pipeline.recreate(
            self.render_pass,
            self.swapchain.extent(),
            &self.shader_manager,
        )?;

        info!("Shaders reloaded successfully");
        Ok(())
    }

    pub fn switch_shader_preset(&mut self, preset: crate::config::ShaderPreset) -> Result<()> {
        info!("Switching to shader preset: {:?}", preset);

        self.shader_manager.switch_preset(preset)?;
        self.reload_shaders()
    }

    fn record_commands(
        &self,
        cmd: vk::CommandBuffer,
        image_index: u32,
        push_constants: &PushConstants,
    ) -> Result<()> {
        unsafe {
            let device = self.context.device();

            // Begin render pass
            let clear_values = [vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
            }];

            let render_pass_begin = vk::RenderPassBeginInfo::default()
                .render_pass(self.render_pass)
                .framebuffer(self.framebuffers[image_index as usize])
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: self.swapchain.extent(),
                })
                .clear_values(&clear_values);

            device.cmd_begin_render_pass(cmd, &render_pass_begin, vk::SubpassContents::INLINE);

            // Bind pipeline
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline.handle());

            // Set dynamic state
            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: self.swapchain.extent().width as f32,
                height: self.swapchain.extent().height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            };
            device.cmd_set_viewport(cmd, 0, &[viewport]);

            let scissor = vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.swapchain.extent(),
            };
            device.cmd_set_scissor(cmd, 0, &[scissor]);

            // Push constants
            let constants_bytes = bytemuck::bytes_of(push_constants);
            device.cmd_push_constants(
                cmd,
                self.pipeline.layout(),
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0,
                constants_bytes,
            );

            // Draw
            device.cmd_draw(cmd, 3, 1, 0, 0);

            // End render pass
            device.cmd_end_render_pass(cmd);
        }

        Ok(())
    }

    fn submit(&self, cmd: vk::CommandBuffer) -> Result<()> {
        let wait_semaphores = [self.sync.image_available()];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let signal_semaphores = [self.sync.render_finished()];
        let command_buffers = [cmd];

        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(&command_buffers)
            .signal_semaphores(&signal_semaphores);

        unsafe {
            self.context.device().queue_submit(
                self.context.queue(),
                &[submit_info],
                self.sync.in_flight(),
            )?;
        }

        Ok(())
    }

    fn recreate_framebuffers(&mut self) -> Result<()> {
        // Destroy old framebuffers
        for fb in self.framebuffers.drain(..) {
            unsafe {
                self.context.device().destroy_framebuffer(fb, None);
            }
        }

        // Create new framebuffers
        self.framebuffers = framebuffer::create_framebuffers(
            &self.context.device(),
            self.swapchain.image_views(),
            self.render_pass,
            self.swapchain.extent(),
        )?;

        Ok(())
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            let _ = self.context.device().device_wait_idle();

            // Destroy framebuffers
            for fb in &self.framebuffers {
                self.context.device().destroy_framebuffer(*fb, None);
            }

            // Destroy render pass
            self.context.device().destroy_render_pass(self.render_pass, None);

            // Destroy surface
            self.context.destroy_surface(self.surface);
        }
    }
}

pub enum DrawResult {
    Success,
    Suboptimal,
    NeedsRecreation,
}

#[derive(Debug, thiserror::Error)]
pub enum SwapchainError {
    #[error("Swapchain is out of date")]
    OutOfDate,

    #[error("Vulkan error: {0}")]
    VulkanError(#[from] vk::Result),
}

fn create_render_pass(device: &ash::Device, format: vk::Format) -> Result<vk::RenderPass> {
    let color_attachment = vk::AttachmentDescription::default()
        .format(format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

    let color_attachment_ref = vk::AttachmentReference::default()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

    let subpass = vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(std::slice::from_ref(&color_attachment_ref));

    let dependency = vk::SubpassDependency::default()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .src_access_mask(vk::AccessFlags::empty())
        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);

    let attachments = [color_attachment];
    let subpasses = [subpass];
    let dependencies = [dependency];

    let create_info = vk::RenderPassCreateInfo::default()
        .attachments(&attachments)
        .subpasses(&subpasses)
        .dependencies(&dependencies);

    unsafe { Ok(device.create_render_pass(&create_info, None)?) }
}