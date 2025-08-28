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

struct VulkanCtx {
    _entry: Entry,
    instance: ash::Instance,
    surface_loader: surface::Instance,
    surface: vk::SurfaceKHR,
    device: ash::Device,
    queue: vk::Queue,
    swapchain_loader: swapchain::Device,
    swapchain: vk::SwapchainKHR,
    extent: vk::Extent2D,
    render_pass: vk::RenderPass,
    framebuffers: Vec<vk::Framebuffer>,
    cmd_pool: vk::CommandPool,
    cmd_bufs: Vec<vk::CommandBuffer>,
    sync: SyncObjects,
}

struct SyncObjects {
    img_acq: vk::Semaphore,
    render_done: vk::Semaphore,
    fence: vk::Fence,
}

impl VulkanCtx {
    unsafe fn new(window: &Window) -> Result<Self> {
        let entry = Entry::linked();
        let instance = Self::create_instance(&entry, window)?;
        let (surface, surface_loader) = Self::create_surface(&entry, &instance, window)?;
        let (pdev, qfi) = Self::select_device(&instance, &surface_loader, surface)?;
        let (device, queue) = Self::create_device(&instance, pdev, qfi)?;
        let (swapchain_loader, swapchain, format, extent) = 
            Self::create_swapchain(&instance, &device, &surface_loader, surface, pdev, window)?;
        let render_pass = Self::create_render_pass(&device, format)?;
        let (views, framebuffers) = Self::create_framebuffers(&device, swapchain_loader.get_swapchain_images(swapchain)?, format, extent, render_pass)?;
        let (cmd_pool, cmd_bufs) = Self::create_commands(&device, qfi, &framebuffers, render_pass, extent)?;
        let sync = Self::create_sync(&device)?;

        Ok(Self {
            _entry: entry, instance, surface_loader, surface, device, queue,
            swapchain_loader, swapchain, extent, render_pass, framebuffers,
            cmd_pool, cmd_bufs, sync,
        })
    }

    unsafe fn create_instance(entry: &Entry, window: &Window) -> Result<ash::Instance> {
        let ext_names = ash_window::enumerate_required_extensions(
            window.display_handle()?.as_raw()
        )?.to_vec();

        let app_name = CString::new("optimized-vulkan")?;
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
            enabled_extension_count: ext_names.len() as u32,
            pp_enabled_extension_names: ext_names.as_ptr(),
            ..Default::default()
        };

        Ok(entry.create_instance(&create_info, None)?)
    }

    unsafe fn create_surface(entry: &Entry, instance: &ash::Instance, window: &Window) 
        -> Result<(vk::SurfaceKHR, surface::Instance)> {
        let surface = ash_window::create_surface(
            entry, instance, 
            window.display_handle()?.as_raw(),
            window.window_handle()?.as_raw(),
            None
        )?;
        Ok((surface, surface::Instance::new(entry, instance)))
    }

    unsafe fn select_device(instance: &ash::Instance, surface_loader: &surface::Instance, surface: vk::SurfaceKHR) 
        -> Result<(vk::PhysicalDevice, u32)> {
        for pdev in instance.enumerate_physical_devices()? {
            for (i, props) in instance.get_physical_device_queue_family_properties(pdev).iter().enumerate() {
                let supports_graphics = props.queue_flags.contains(vk::QueueFlags::GRAPHICS);
                let supports_surface = surface_loader.get_physical_device_surface_support(pdev, i as u32, surface)?;
                
                if supports_graphics && supports_surface {
                    return Ok((pdev, i as u32));
                }
            }
        }
        Err(anyhow!("No suitable device found"))
    }

    unsafe fn create_device(instance: &ash::Instance, pdev: vk::PhysicalDevice, qfi: u32) 
        -> Result<(ash::Device, vk::Queue)> {
        let priorities = [1.0f32];
        let queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(qfi)
            .queue_priorities(&priorities);

        let extensions = [swapchain::NAME.as_ptr()];
        let queue_infos = [queue_info];
        let device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&extensions);

        let device = instance.create_device(pdev, &device_info, None)?;
        let queue = device.get_device_queue(qfi, 0);
        Ok((device, queue))
    }

    unsafe fn create_swapchain(instance: &ash::Instance, device: &ash::Device, 
                             surface_loader: &surface::Instance, surface: vk::SurfaceKHR, 
                             pdev: vk::PhysicalDevice, window: &Window) 
        -> Result<(swapchain::Device, vk::SwapchainKHR, vk::Format, vk::Extent2D)> {
        let formats = surface_loader.get_physical_device_surface_formats(pdev, surface)?;
        let caps = surface_loader.get_physical_device_surface_capabilities(pdev, surface)?;
        
        let format = formats.iter()
            .find(|f| f.format == vk::Format::B8G8R8A8_SRGB)
            .unwrap_or(&formats[0]).format;
        
        let extent = if caps.current_extent.width == u32::MAX {
            let size = window.inner_size();
            vk::Extent2D { width: size.width, height: size.height }
        } else { caps.current_extent };

        let swapchain_loader = swapchain::Device::new(instance, device);
        let image_count = (caps.min_image_count + 1).min(if caps.max_image_count > 0 { caps.max_image_count } else { u32::MAX });
        
        let create_info = vk::SwapchainCreateInfoKHR {
            s_type: vk::StructureType::SWAPCHAIN_CREATE_INFO_KHR,
            surface,
            min_image_count: image_count,
            image_format: format,
            image_color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
            image_extent: extent,
            image_array_layers: 1,
            image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
            image_sharing_mode: vk::SharingMode::EXCLUSIVE,
            pre_transform: caps.current_transform,
            composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
            present_mode: vk::PresentModeKHR::FIFO,
            clipped: vk::TRUE,
            old_swapchain: vk::SwapchainKHR::null(),
            ..Default::default()
        };

        let swapchain = swapchain_loader.create_swapchain(&create_info, None)?;
        Ok((swapchain_loader, swapchain, format, extent))
    }

    unsafe fn create_render_pass(device: &ash::Device, format: vk::Format) -> Result<vk::RenderPass> {
        let attachment = vk::AttachmentDescription {
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

        let color_ref = vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        };

        let subpass = vk::SubpassDescription {
            pipeline_bind_point: vk::PipelineBindPoint::GRAPHICS,
            color_attachment_count: 1,
            p_color_attachments: &color_ref,
            ..Default::default()
        };

        let create_info = vk::RenderPassCreateInfo {
            s_type: vk::StructureType::RENDER_PASS_CREATE_INFO,
            attachment_count: 1,
            p_attachments: &attachment,
            subpass_count: 1,
            p_subpasses: &subpass,
            ..Default::default()
        };

        Ok(device.create_render_pass(&create_info, None)?)
    }

    unsafe fn create_framebuffers(device: &ash::Device, images: Vec<vk::Image>, format: vk::Format, 
                                extent: vk::Extent2D, render_pass: vk::RenderPass) 
        -> Result<(Vec<vk::ImageView>, Vec<vk::Framebuffer>)> {
        let mut views = Vec::new();
        let mut framebuffers = Vec::new();

        for image in images {
            let subresource_range = vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            };
            
            let view_info = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format)
                .subresource_range(subresource_range);
            let view = device.create_image_view(&view_info, None)?;
            views.push(view);

            let attachments = [view];
            let fb_info = vk::FramebufferCreateInfo::default()
                .render_pass(render_pass)
                .attachments(&attachments)
                .width(extent.width)
                .height(extent.height)
                .layers(1);
            framebuffers.push(device.create_framebuffer(&fb_info, None)?);
        }
        Ok((views, framebuffers))
    }

    unsafe fn create_commands(device: &ash::Device, qfi: u32, framebuffers: &[vk::Framebuffer], 
                            render_pass: vk::RenderPass, extent: vk::Extent2D) 
        -> Result<(vk::CommandPool, Vec<vk::CommandBuffer>)> {
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(qfi);
        let cmd_pool = device.create_command_pool(&pool_info, None)?;

        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(cmd_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(framebuffers.len() as u32);
        let cmd_bufs = device.allocate_command_buffers(&alloc_info)?;

        let clear_values = [vk::ClearValue { 
            color: vk::ClearColorValue { float32: [0.0, 0.2, 0.4, 1.0] }
        }];
        
        for (i, &cmd_buf) in cmd_bufs.iter().enumerate() {
            let begin_info = vk::CommandBufferBeginInfo::default();
            device.begin_command_buffer(cmd_buf, &begin_info)?;

            let render_area = vk::Rect2D { 
                offset: vk::Offset2D { x: 0, y: 0 }, 
                extent 
            };
            let render_pass_info = vk::RenderPassBeginInfo::default()
                .render_pass(render_pass)
                .framebuffer(framebuffers[i])
                .render_area(render_area)
                .clear_values(&clear_values);

            device.cmd_begin_render_pass(cmd_buf, &render_pass_info, vk::SubpassContents::INLINE);
            device.cmd_end_render_pass(cmd_buf);
            device.end_command_buffer(cmd_buf)?;
        }
        Ok((cmd_pool, cmd_bufs))
    }

    unsafe fn create_sync(device: &ash::Device) -> Result<SyncObjects> {
        let sem_info = vk::SemaphoreCreateInfo {
            s_type: vk::StructureType::SEMAPHORE_CREATE_INFO,
            ..Default::default()
        };
        let fence_info = vk::FenceCreateInfo {
            s_type: vk::StructureType::FENCE_CREATE_INFO,
            flags: vk::FenceCreateFlags::SIGNALED,
            ..Default::default()
        };

        Ok(SyncObjects {
            img_acq: device.create_semaphore(&sem_info, None)?,
            render_done: device.create_semaphore(&sem_info, None)?,
            fence: device.create_fence(&fence_info, None)?,
        })
    }

    unsafe fn draw(&self) -> Result<()> {
        self.device.wait_for_fences(&[self.sync.fence], true, u64::MAX)?;
        self.device.reset_fences(&[self.sync.fence])?;

        let (idx, _) = self.swapchain_loader.acquire_next_image(
            self.swapchain, u64::MAX, self.sync.img_acq, vk::Fence::null())?;

        let wait_semaphores = [self.sync.img_acq];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let command_buffers = [self.cmd_bufs[idx as usize]];
        let signal_semaphores = [self.sync.render_done];
        
        let submit_info = vk::SubmitInfo {
            s_type: vk::StructureType::SUBMIT_INFO,
            wait_semaphore_count: 1,
            p_wait_semaphores: wait_semaphores.as_ptr(),
            p_wait_dst_stage_mask: wait_stages.as_ptr(),
            command_buffer_count: 1,
            p_command_buffers: command_buffers.as_ptr(),
            signal_semaphore_count: 1,
            p_signal_semaphores: signal_semaphores.as_ptr(),
            ..Default::default()
        };

        self.device.queue_submit(self.queue, &[submit_info], self.sync.fence)?;

        let swapchains = [self.swapchain];
        let image_indices = [idx];
        let present_info = vk::PresentInfoKHR {
            s_type: vk::StructureType::PRESENT_INFO_KHR,
            wait_semaphore_count: 1,
            p_wait_semaphores: signal_semaphores.as_ptr(),
            swapchain_count: 1,
            p_swapchains: swapchains.as_ptr(),
            p_image_indices: image_indices.as_ptr(),
            ..Default::default()
        };

        self.swapchain_loader.queue_present(self.queue, &present_info)?;
        Ok(())
    }
}

impl Drop for VulkanCtx {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().ok();
            self.device.destroy_fence(self.sync.fence, None);
            self.device.destroy_semaphore(self.sync.render_done, None);
            self.device.destroy_semaphore(self.sync.img_acq, None);
            self.device.free_command_buffers(self.cmd_pool, &self.cmd_bufs);
            self.device.destroy_command_pool(self.cmd_pool, None);
            for &fb in &self.framebuffers { self.device.destroy_framebuffer(fb, None); }
            self.device.destroy_render_pass(self.render_pass, None);
            self.swapchain_loader.destroy_swapchain(self.swapchain, None);
            self.surface_loader.destroy_surface(self.surface, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

#[derive(Default)]
struct App {
    window: Option<Window>,
    vulkan: Option<VulkanCtx>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = event_loop.create_window(
            Window::default_attributes()
                .with_title("Optimized Vulkan")
                .with_inner_size(winit::dpi::LogicalSize::new(800.0, 600.0))
        ).expect("Failed to create window");

        let vulkan = unsafe { VulkanCtx::new(&window).expect("Failed to init Vulkan") };
        self.window = Some(window);
        self.vulkan = Some(vulkan);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: winit::window::WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => {
                if let Some(vulkan) = &self.vulkan {
                    unsafe { vulkan.draw().unwrap(); }
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
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut App::default())?;
    Ok(())
}
