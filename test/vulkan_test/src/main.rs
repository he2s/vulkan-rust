use anyhow::{anyhow, Result};
use ash::{vk, Entry};
use ash::khr::{surface, swapchain};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::ffi::CString;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowAttributes},
};

struct VulkanRenderer {
    _entry: Entry,
    instance: ash::Instance,
    surface_loader: surface::Instance,
    surface: vk::SurfaceKHR,
    device: ash::Device,
    graphics_queue: vk::Queue,
    swapchain: SwapchainData,
    render_pass: vk::RenderPass,
    framebuffers: Vec<vk::Framebuffer>,
    commands: CommandData,
    sync: SyncObjects,
}

struct SwapchainData {
    loader: swapchain::Device,
    handle: vk::SwapchainKHR,
    extent: vk::Extent2D,
    _views: Vec<vk::ImageView>, // Keep views alive for framebuffers
}

struct CommandData {
    pool: vk::CommandPool,
    buffers: Vec<vk::CommandBuffer>,
}

struct SyncObjects {
    image_acquired: vk::Semaphore,
    render_finished: vk::Semaphore,
    frame_fence: vk::Fence,
}

impl VulkanRenderer {
    unsafe fn new(window: &Window) -> Result<Self> {
        let entry = Entry::linked();
        let instance = Self::create_instance(&entry, window)?;
        let (surface, surface_loader) = Self::create_surface(&entry, &instance, window)?;
        
        // Find suitable device and queue family
        let (physical_device, queue_family_idx) = Self::find_suitable_device(&instance, &surface_loader, surface)?;
        
        // Create logical device and get queue
        let (device, graphics_queue) = Self::create_logical_device(&instance, physical_device, queue_family_idx)?;
        
        // Create swapchain and related resources
        let swapchain = Self::create_swapchain_data(&instance, &device, &surface_loader, surface, physical_device, window)?;
        
        // Create render pipeline components
        let render_pass = Self::create_render_pass(&device, swapchain.get_format())?;
        let framebuffers = Self::create_framebuffers(&device, &swapchain, render_pass)?;
        
        // Create command submission resources
        let commands = Self::create_command_data(&device, queue_family_idx, &framebuffers, render_pass, swapchain.extent)?;
        let sync = Self::create_sync_objects(&device)?;

        Ok(Self {
            _entry: entry,
            instance,
            surface_loader,
            surface,
            device,
            graphics_queue,
            swapchain,
            render_pass,
            framebuffers,
            commands,
            sync,
        })
    }

    unsafe fn create_instance(entry: &Entry, window: &Window) -> Result<ash::Instance> {
        let required_extensions = ash_window::enumerate_required_extensions(
            window.display_handle()?.as_raw()
        )?.to_vec();

        let app_name = CString::new("Vulkan Renderer")?;
        let app_info = vk::ApplicationInfo {
            s_type: vk::StructureType::APPLICATION_INFO,
            p_application_name: app_name.as_ptr(),
            p_engine_name: app_name.as_ptr(),
            api_version: vk::make_api_version(0, 1, 2, 0),
            ..Default::default()
        };

        let create_info = vk::InstanceCreateInfo {
            s_type: vk::StructureType::INSTANCE_CREATE_INFO,
            p_application_info: &app_info,
            enabled_extension_count: required_extensions.len() as u32,
            pp_enabled_extension_names: required_extensions.as_ptr(),
            ..Default::default()
        };

        Ok(entry.create_instance(&create_info, None)?)
    }

    unsafe fn create_surface(entry: &Entry, instance: &ash::Instance, window: &Window) 
        -> Result<(vk::SurfaceKHR, surface::Instance)> {
        let surface = ash_window::create_surface(
            entry,
            instance,
            window.display_handle()?.as_raw(),
            window.window_handle()?.as_raw(),
            None,
        )?;
        let surface_loader = surface::Instance::new(entry, instance);
        Ok((surface, surface_loader))
    }

    unsafe fn find_suitable_device(
        instance: &ash::Instance,
        surface_loader: &surface::Instance,
        surface: vk::SurfaceKHR,
    ) -> Result<(vk::PhysicalDevice, u32)> {
        for physical_device in instance.enumerate_physical_devices()? {
            let queue_families = instance.get_physical_device_queue_family_properties(physical_device);
            
            for (index, properties) in queue_families.iter().enumerate() {
                let has_graphics = properties.queue_flags.contains(vk::QueueFlags::GRAPHICS);
                let supports_surface = surface_loader
                    .get_physical_device_surface_support(physical_device, index as u32, surface)?;
                
                if has_graphics && supports_surface {
                    return Ok((physical_device, index as u32));
                }
            }
        }
        Err(anyhow!("No suitable graphics device found"))
    }

    unsafe fn create_logical_device(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        queue_family_index: u32,
    ) -> Result<(ash::Device, vk::Queue)> {
        let queue_priorities = [1.0f32];
        let queue_create_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&queue_priorities);

        let required_extensions = [swapchain::NAME.as_ptr()];
        let queue_create_infos = [queue_create_info];
        
        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&required_extensions);

        let device = instance.create_device(physical_device, &device_create_info, None)?;
        let graphics_queue = device.get_device_queue(queue_family_index, 0);
        
        Ok((device, graphics_queue))
    }

    unsafe fn create_swapchain_data(
        instance: &ash::Instance,
        device: &ash::Device,
        surface_loader: &surface::Instance,
        surface: vk::SurfaceKHR,
        physical_device: vk::PhysicalDevice,
        window: &Window,
    ) -> Result<SwapchainData> {
        // Query surface capabilities
        let capabilities = surface_loader.get_physical_device_surface_capabilities(physical_device, surface)?;
        let formats = surface_loader.get_physical_device_surface_formats(physical_device, surface)?;
        
        // Choose format (prefer sRGB)
        let surface_format = formats
            .iter()
            .find(|format| format.format == vk::Format::B8G8R8A8_SRGB)
            .unwrap_or(&formats[0]);
        
        // Determine extent
        let extent = if capabilities.current_extent.width == u32::MAX {
            let window_size = window.inner_size();
            vk::Extent2D {
                width: window_size.width,
                height: window_size.height,
            }
        } else {
            capabilities.current_extent
        };

        // Calculate image count
        let min_image_count = capabilities.min_image_count + 1;
        let image_count = if capabilities.max_image_count > 0 {
            min_image_count.min(capabilities.max_image_count)
        } else {
            min_image_count
        };

        let swapchain_loader = swapchain::Device::new(instance, device);
        let swapchain_create_info = vk::SwapchainCreateInfoKHR {
            s_type: vk::StructureType::SWAPCHAIN_CREATE_INFO_KHR,
            surface,
            min_image_count: image_count,
            image_format: surface_format.format,
            image_color_space: surface_format.color_space,
            image_extent: extent,
            image_array_layers: 1,
            image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
            image_sharing_mode: vk::SharingMode::EXCLUSIVE,
            pre_transform: capabilities.current_transform,
            composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
            present_mode: vk::PresentModeKHR::FIFO,
            clipped: vk::TRUE,
            old_swapchain: vk::SwapchainKHR::null(),
            ..Default::default()
        };

        let swapchain_handle = swapchain_loader.create_swapchain(&swapchain_create_info, None)?;
        let swapchain_images = swapchain_loader.get_swapchain_images(swapchain_handle)?;
        
        // Create image views
        let image_views = Self::create_image_views(device, &swapchain_images, surface_format.format)?;

        Ok(SwapchainData {
            loader: swapchain_loader,
            handle: swapchain_handle,
            extent,
            _views: image_views,
        })
    }

    unsafe fn create_image_views(
        device: &ash::Device,
        images: &[vk::Image],
        format: vk::Format,
    ) -> Result<Vec<vk::ImageView>> {
        images
            .iter()
            .map(|&image| {
                let subresource_range = vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                };

                let create_info = vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format)
                    .subresource_range(subresource_range);

                device.create_image_view(&create_info, None).map_err(Into::into)
            })
            .collect()
    }

    unsafe fn create_render_pass(device: &ash::Device, format: vk::Format) -> Result<vk::RenderPass> {
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

        let attachments = [color_attachment];
        let color_attachments = [color_attachment_ref];
        
        let subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_attachments);

        let subpasses = [subpass];
        let create_info = vk::RenderPassCreateInfo::default()
            .attachments(&attachments)
            .subpasses(&subpasses);

        Ok(device.create_render_pass(&create_info, None)?)
    }

    unsafe fn create_framebuffers(
        device: &ash::Device,
        swapchain: &SwapchainData,
        render_pass: vk::RenderPass,
    ) -> Result<Vec<vk::Framebuffer>> {
        swapchain._views
            .iter()
            .map(|&image_view| {
                let attachments = [image_view];
                let create_info = vk::FramebufferCreateInfo::default()
                    .render_pass(render_pass)
                    .attachments(&attachments)
                    .width(swapchain.extent.width)
                    .height(swapchain.extent.height)
                    .layers(1);

                device.create_framebuffer(&create_info, None).map_err(Into::into)
            })
            .collect()
    }

    unsafe fn create_command_data(
        device: &ash::Device,
        queue_family_index: u32,
        framebuffers: &[vk::Framebuffer],
        render_pass: vk::RenderPass,
        extent: vk::Extent2D,
    ) -> Result<CommandData> {
        // Create command pool
        let pool_create_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family_index);
        let command_pool = device.create_command_pool(&pool_create_info, None)?;

        // Allocate command buffers
        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(framebuffers.len() as u32);
        let command_buffers = device.allocate_command_buffers(&allocate_info)?;

        // Record clear commands
        Self::record_clear_commands(device, &command_buffers, framebuffers, render_pass, extent)?;

        Ok(CommandData {
            pool: command_pool,
            buffers: command_buffers,
        })
    }

    unsafe fn record_clear_commands(
        device: &ash::Device,
        command_buffers: &[vk::CommandBuffer],
        framebuffers: &[vk::Framebuffer],
        render_pass: vk::RenderPass,
        extent: vk::Extent2D,
    ) -> Result<()> {
        let clear_color = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.1, 0.2, 0.3, 1.0], // Nice blue color
            },
        };
        let clear_values = [clear_color];

        for (command_buffer, &framebuffer) in command_buffers.iter().zip(framebuffers) {
            let begin_info = vk::CommandBufferBeginInfo::default();
            device.begin_command_buffer(*command_buffer, &begin_info)?;

            let render_area = vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent,
            };

            let render_pass_begin_info = vk::RenderPassBeginInfo::default()
                .render_pass(render_pass)
                .framebuffer(framebuffer)
                .render_area(render_area)
                .clear_values(&clear_values);

            device.cmd_begin_render_pass(
                *command_buffer,
                &render_pass_begin_info,
                vk::SubpassContents::INLINE,
            );
            device.cmd_end_render_pass(*command_buffer);
            device.end_command_buffer(*command_buffer)?;
        }

        Ok(())
    }

    unsafe fn create_sync_objects(device: &ash::Device) -> Result<SyncObjects> {
        let semaphore_create_info = vk::SemaphoreCreateInfo::default();
        let fence_create_info = vk::FenceCreateInfo::default()
            .flags(vk::FenceCreateFlags::SIGNALED);

        Ok(SyncObjects {
            image_acquired: device.create_semaphore(&semaphore_create_info, None)?,
            render_finished: device.create_semaphore(&semaphore_create_info, None)?,
            frame_fence: device.create_fence(&fence_create_info, None)?,
        })
    }

    unsafe fn render_frame(&self) -> Result<()> {
        // Wait for previous frame to finish
        self.device.wait_for_fences(&[self.sync.frame_fence], true, u64::MAX)?;
        self.device.reset_fences(&[self.sync.frame_fence])?;

        // Acquire next image from swapchain
        let (image_index, _) = self.swapchain.loader.acquire_next_image(
            self.swapchain.handle,
            u64::MAX,
            self.sync.image_acquired,
            vk::Fence::null(),
        )?;

        // Submit rendering commands
        let wait_semaphores = [self.sync.image_acquired];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let command_buffers = [self.commands.buffers[image_index as usize]];
        let signal_semaphores = [self.sync.render_finished];

        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(&command_buffers)
            .signal_semaphores(&signal_semaphores);

        self.device.queue_submit(
            self.graphics_queue,
            &[submit_info],
            self.sync.frame_fence,
        )?;

        // Present the rendered image
        let swapchains = [self.swapchain.handle];
        let image_indices = [image_index];
        
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&signal_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        self.swapchain.loader.queue_present(self.graphics_queue, &present_info)?;
        Ok(())
    }
}

impl SwapchainData {
    fn get_format(&self) -> vk::Format {
        vk::Format::B8G8R8A8_SRGB // We know this from creation
    }
}

impl Drop for VulkanRenderer {
    fn drop(&mut self) {
        unsafe {
            // Wait for all operations to complete
            self.device.device_wait_idle().ok();

            // Clean up in reverse order of creation
            self.device.destroy_fence(self.sync.frame_fence, None);
            self.device.destroy_semaphore(self.sync.render_finished, None);
            self.device.destroy_semaphore(self.sync.image_acquired, None);
            
            self.device.free_command_buffers(self.commands.pool, &self.commands.buffers);
            self.device.destroy_command_pool(self.commands.pool, None);
            
            for framebuffer in &self.framebuffers {
                self.device.destroy_framebuffer(*framebuffer, None);
            }
            
            self.device.destroy_render_pass(self.render_pass, None);
            
            for view in &self.swapchain._views {
                self.device.destroy_image_view(*view, None);
            }
            
            self.swapchain.loader.destroy_swapchain(self.swapchain.handle, None);
            self.surface_loader.destroy_surface(self.surface, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

#[derive(Default)]
struct VulkanApp {
    window: Option<Window>,
    renderer: Option<VulkanRenderer>,
}

impl ApplicationHandler for VulkanApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = event_loop
            .create_window(
                Window::default_attributes()
                    .with_title("Clean Vulkan Renderer")
                    .with_inner_size(winit::dpi::LogicalSize::new(1024.0, 768.0)),
            )
            .expect("Failed to create window");

        let renderer = unsafe { 
            VulkanRenderer::new(&window).expect("Failed to initialize Vulkan renderer") 
        };

        self.window = Some(window);
        self.renderer = Some(renderer);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _: winit::window::WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => {
                if let Some(renderer) = &self.renderer {
                    unsafe {
                        if let Err(e) = renderer.render_frame() {
                            eprintln!("Render error: {}", e);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

fn main() -> Result<()> {
    env_logger::init();
    println!("Starting Vulkan renderer...");

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut VulkanApp::default())?;
    
    Ok(())
}
