use anyhow::{anyhow, Result};
use ash::{vk, Entry};
use ash::khr::{surface, swapchain};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle};
use std::{ffi::CString, ptr};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowAttributes},
};

struct Gfx {
    _entry: ash::Entry,
    instance: ash::Instance,
    surface_loader: surface::Instance,
    surface: vk::SurfaceKHR,
    pdev: vk::PhysicalDevice,
    device: ash::Device,
    queue: vk::Queue,
    qfi: u32,
    swapchain_loader: swapchain::Device,
    swapchain: vk::SwapchainKHR,
    format: vk::Format,
    extent: vk::Extent2D,
    views: Vec<vk::ImageView>,
    render_pass: vk::RenderPass,
    framebuffers: Vec<vk::Framebuffer>,
    cmd_pool: vk::CommandPool,
    cmd_bufs: Vec<vk::CommandBuffer>,
    img_acq: vk::Semaphore,
    render_done: vk::Semaphore,
    inflight: vk::Fence,
}

impl Gfx {
    unsafe fn new(window: &Window) -> Result<Self> {
        let entry = Entry::linked();

        // --- winit 0.30: fallible handles → Raw*Handle
        let display_handle: RawDisplayHandle = window.display_handle()?.as_raw();
        let window_handle:  RawWindowHandle  = window.window_handle()?.as_raw();

        // --- Extensions required for this platform (Result in ash-window 0.13)
        let ext_names = ash_window::enumerate_required_extensions(display_handle)?.to_vec();

        // --- Instance (no builders)
        let app_name = CString::new("vulkan-clear-latest")?;
        let app_info = vk::ApplicationInfo {
            s_type: vk::StructureType::APPLICATION_INFO,
            p_next: ptr::null(),
            p_application_name: app_name.as_ptr(),
            application_version: 0,
            p_engine_name: app_name.as_ptr(),
            engine_version: 0,
            api_version: vk::make_api_version(0, 1, 2, 0),
        };

        let ici = vk::InstanceCreateInfo {
            s_type: vk::StructureType::INSTANCE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::InstanceCreateFlags::empty(),
            p_application_info: &app_info,
            enabled_layer_count: 0,
            pp_enabled_layer_names: ptr::null(),
            enabled_extension_count: ext_names.len() as u32,
            pp_enabled_extension_names: ext_names.as_ptr(),
        };

        let instance = entry.create_instance(&ici, None)?;

        // --- Surface + loader
        let surface = ash_window::create_surface(&entry, &instance, display_handle, window_handle, None)?;
        let surface_loader = surface::Instance::new(&entry, &instance);

        // --- Pick physical device + queue family (graphics + present)
        let pdev = instance.enumerate_physical_devices()?
            .into_iter()
            .find(|&pd| {
                instance.get_physical_device_queue_family_properties(pd)
                    .iter().enumerate().any(|(i, q)| {
                        q.queue_flags.contains(vk::QueueFlags::GRAPHICS) &&
                        surface_loader.get_physical_device_surface_support(pd, i as u32, surface).unwrap_or(false)
                    })
            })
            .ok_or_else(|| anyhow!("No suitable GPU"))?;

        let (qfi, _) = instance.get_physical_device_queue_family_properties(pdev)
            .iter().enumerate()
            .find(|(i, q)| {
                q.queue_flags.contains(vk::QueueFlags::GRAPHICS) &&
                surface_loader.get_physical_device_surface_support(pdev, *i as u32, surface).unwrap_or(false)
            })
            .map(|(i, q)| (i as u32, q))
            .ok_or_else(|| anyhow!("No queue family"))?;

        // --- Logical device
        let priorities = [1.0f32];
        let qci = vk::DeviceQueueCreateInfo {
            s_type: vk::StructureType::DEVICE_QUEUE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::DeviceQueueCreateFlags::empty(),
            queue_family_index: qfi,
            queue_count: 1,
            p_queue_priorities: priorities.as_ptr(),
        };

        let dev_exts = [swapchain::NAME.as_ptr()];
        let dci = vk::DeviceCreateInfo {
            s_type: vk::StructureType::DEVICE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::DeviceCreateFlags::empty(),
            queue_create_info_count: 1,
            p_queue_create_infos: &qci,
            enabled_layer_count: 0,
            pp_enabled_layer_names: ptr::null(),
            enabled_extension_count: dev_exts.len() as u32,
            pp_enabled_extension_names: dev_exts.as_ptr(),
            p_enabled_features: ptr::null(),
        };

        let device = instance.create_device(pdev, &dci, None)?;
        let queue = device.get_device_queue(qfi, 0);

        // --- Swapchain
        let swapchain_loader = swapchain::Device::new(&instance, &device);

        let formats = surface_loader.get_physical_device_surface_formats(pdev, surface)?;
        let surface_format = formats.iter()
            .find(|f| f.format == vk::Format::B8G8R8A8_SRGB)
            .or_else(|| formats.first())
            .ok_or_else(|| anyhow!("No surface formats"))?;
        let caps = surface_loader.get_physical_device_surface_capabilities(pdev, surface)?;
        let present_modes = surface_loader.get_physical_device_surface_present_modes(pdev, surface)?;
        let present_mode = present_modes.into_iter()
            .find(|m| *m == vk::PresentModeKHR::MAILBOX)
            .unwrap_or(vk::PresentModeKHR::FIFO);

        let extent = if caps.current_extent.width == u32::MAX {
            let s = window.inner_size();
            vk::Extent2D { width: s.width, height: s.height }
        } else { caps.current_extent };

        let desired = caps.min_image_count + 1;
        let min_image_count = if caps.max_image_count > 0 {
            desired.min(caps.max_image_count)
        } else { desired };

        let mut sci = vk::SwapchainCreateInfoKHR {
            s_type: vk::StructureType::SWAPCHAIN_CREATE_INFO_KHR,
            p_next: ptr::null(),
            flags: vk::SwapchainCreateFlagsKHR::empty(),
            surface,
            min_image_count,
            image_format: surface_format.format,
            image_color_space: surface_format.color_space,
            image_extent: extent,
            image_array_layers: 1,
            image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
            image_sharing_mode: vk::SharingMode::EXCLUSIVE,
            queue_family_index_count: 0,
            p_queue_family_indices: ptr::null(),
            pre_transform: caps.current_transform,
            composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
            present_mode,
            clipped: vk::TRUE,
            old_swapchain: vk::SwapchainKHR::null(),
        };

        let swapchain = swapchain_loader.create_swapchain(&sci, None)?;
        let images = swapchain_loader.get_swapchain_images(swapchain)?;
        let format = surface_format.format;

        // --- Image views
        let mut views = Vec::with_capacity(images.len());
        for &img in &images {
            let sub = vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            };
            let ivci = vk::ImageViewCreateInfo {
                s_type: vk::StructureType::IMAGE_VIEW_CREATE_INFO,
                p_next: ptr::null(),
                flags: vk::ImageViewCreateFlags::empty(),
                image: img,
                view_type: vk::ImageViewType::TYPE_2D,
                format,
                components: vk::ComponentMapping {
                    r: vk::ComponentSwizzle::IDENTITY,
                    g: vk::ComponentSwizzle::IDENTITY,
                    b: vk::ComponentSwizzle::IDENTITY,
                    a: vk::ComponentSwizzle::IDENTITY,
                },
                subresource_range: sub,
            };
            views.push(device.create_image_view(&ivci, None)?);
        }

        // --- Render pass (clear → present)
        let color_att = vk::AttachmentDescription {
            flags: vk::AttachmentDescriptionFlags::empty(),
            format,
            samples: vk::SampleCountFlags::TYPE_1,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
        };
        let color_ref = vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        };
        let subpass = vk::SubpassDescription {
            flags: vk::SubpassDescriptionFlags::empty(),
            pipeline_bind_point: vk::PipelineBindPoint::GRAPHICS,
            input_attachment_count: 0,
            p_input_attachments: ptr::null(),
            color_attachment_count: 1,
            p_color_attachments: &color_ref,
            p_resolve_attachments: ptr::null(),
            p_depth_stencil_attachment: ptr::null(),
            preserve_attachment_count: 0,
            p_preserve_attachments: ptr::null(),
        };
        let rpci = vk::RenderPassCreateInfo {
            s_type: vk::StructureType::RENDER_PASS_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::RenderPassCreateFlags::empty(),
            attachment_count: 1,
            p_attachments: &color_att,
            subpass_count: 1,
            p_subpasses: &subpass,
            dependency_count: 0,
            p_dependencies: ptr::null(),
        };
        let render_pass = device.create_render_pass(&rpci, None)?;

        // --- Framebuffers
        let mut framebuffers = Vec::with_capacity(views.len());
        for &v in &views {
            let attachments = [v];
            let fbci = vk::FramebufferCreateInfo {
                s_type: vk::StructureType::FRAMEBUFFER_CREATE_INFO,
                p_next: ptr::null(),
                flags: vk::FramebufferCreateFlags::empty(),
                render_pass,
                attachment_count: attachments.len() as u32,
                p_attachments: attachments.as_ptr(),
                width: extent.width,
                height: extent.height,
                layers: 1,
            };
            framebuffers.push(device.create_framebuffer(&fbci, None)?);
        }

        // --- Command pool + buffers
        let cpci = vk::CommandPoolCreateInfo {
            s_type: vk::StructureType::COMMAND_POOL_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::CommandPoolCreateFlags::empty(),
            queue_family_index: qfi,
        };
        let cmd_pool = device.create_command_pool(&cpci, None)?;
        let cbai = vk::CommandBufferAllocateInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
            p_next: ptr::null(),
            command_pool: cmd_pool,
            level: vk::CommandBufferLevel::PRIMARY,
            command_buffer_count: framebuffers.len() as u32,
        };
        let cmd_bufs = device.allocate_command_buffers(&cbai)?;

        // --- Record a clear pass per framebuffer
        for (i, &cb) in cmd_bufs.iter().enumerate() {
            let begin = vk::CommandBufferBeginInfo {
                s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
                p_next: ptr::null(),
                flags: vk::CommandBufferUsageFlags::empty(),
                p_inheritance_info: ptr::null(),
            };
            device.begin_command_buffer(cb, &begin)?;

            let clear = vk::ClearValue {
                color: vk::ClearColorValue { float32: [0.0, 0.2, 0.25, 1.0] },
            };
            let clear_vals = [clear];

            let rpbi = vk::RenderPassBeginInfo {
                s_type: vk::StructureType::RENDER_PASS_BEGIN_INFO,
                p_next: ptr::null(),
                render_pass,
                framebuffer: framebuffers[i],
                render_area: vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent,
                },
                clear_value_count: clear_vals.len() as u32,
                p_clear_values: clear_vals.as_ptr(),
            };
            device.cmd_begin_render_pass(cb, &rpbi, vk::SubpassContents::INLINE);
            device.cmd_end_render_pass(cb);
            device.end_command_buffer(cb)?;
        }

        // --- Sync
        let sem_ci = vk::SemaphoreCreateInfo {
            s_type: vk::StructureType::SEMAPHORE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::SemaphoreCreateFlags::empty(),
        };
        let img_acq = device.create_semaphore(&sem_ci, None)?;
        let render_done = device.create_semaphore(&sem_ci, None)?;
        let fence_ci = vk::FenceCreateInfo {
            s_type: vk::StructureType::FENCE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::FenceCreateFlags::SIGNALED,
        };
        let inflight = device.create_fence(&fence_ci, None)?;

        Ok(Self {
            _entry: entry,
            instance,
            surface_loader,
            surface,
            pdev,
            device,
            queue,
            qfi,
            swapchain_loader,
            swapchain,
            format,
            extent,
            views,
            render_pass,
            framebuffers,
            cmd_pool,
            cmd_bufs,
            img_acq,
            render_done,
            inflight,
        })
    }

    unsafe fn draw(&self) -> Result<()> {
        self.device.wait_for_fences(&[self.inflight], true, u64::MAX)?;
        self.device.reset_fences(&[self.inflight])?;

        let (image_index, _) = self
            .swapchain_loader
            .acquire_next_image(self.swapchain, u64::MAX, self.img_acq, vk::Fence::null())?;

        let wait_sems = [self.img_acq];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let signal_sems = [self.render_done];

        let submit = vk::SubmitInfo {
            s_type: vk::StructureType::SUBMIT_INFO,
            p_next: ptr::null(),
            wait_semaphore_count: wait_sems.len() as u32,
            p_wait_semaphores: wait_sems.as_ptr(),
            p_wait_dst_stage_mask: wait_stages.as_ptr(),
            command_buffer_count: 1,
            p_command_buffers: &self.cmd_bufs[image_index as usize],
            signal_semaphore_count: signal_sems.len() as u32,
            p_signal_semaphores: signal_sems.as_ptr(),
        };
        self.device.queue_submit(self.queue, &[submit], self.inflight)?;

        let scs = [self.swapchain];
        let idx = [image_index];
        let present = vk::PresentInfoKHR {
            s_type: vk::StructureType::PRESENT_INFO_KHR,
            p_next: ptr::null(),
            wait_semaphore_count: signal_sems.len() as u32,
            p_wait_semaphores: signal_sems.as_ptr(),
            swapchain_count: scs.len() as u32,
            p_swapchains: scs.as_ptr(),
            p_image_indices: idx.as_ptr(),
            p_results: ptr::null_mut(),
        };
        self.swapchain_loader.queue_present(self.queue, &present)?;
        Ok(())
    }
}

impl Drop for Gfx {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().ok();
            self.device.destroy_semaphore(self.render_done, None);
            self.device.destroy_semaphore(self.img_acq, None);
            self.device.destroy_fence(self.inflight, None);
            self.device.free_command_buffers(self.cmd_pool, &self.cmd_bufs);
            self.device.destroy_command_pool(self.cmd_pool, None);
            for &fb in &self.framebuffers { self.device.destroy_framebuffer(fb, None); }
            self.device.destroy_render_pass(self.render_pass, None);
            for &iv in &self.views { self.device.destroy_image_view(iv, None); }
            self.swapchain_loader.destroy_swapchain(self.swapchain, None);
            self.surface_loader.destroy_surface(self.surface, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

// -------------- winit 0.30 application scaffold --------------
#[derive(Default)]
struct App {
    window: Option<Window>,
    gfx: Option<Gfx>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attrs: WindowAttributes = Window::default_attributes()
            .with_title("Vulkan (ash 0.38, winit 0.30) — builder-free")
            .with_inner_size(winit::dpi::LogicalSize::new(960.0, 540.0));

        let window = event_loop.create_window(attrs).expect("create window");
        let gfx = unsafe { Gfx::new(&window).expect("init Vulkan") };
        self.window = Some(window);
        self.gfx = Some(gfx);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => {
                if let Some(gfx) = &self.gfx {
                    unsafe { gfx.draw().unwrap() };
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(w) = &self.window { w.request_redraw(); }
    }
}

fn main() -> Result<()> {
    env_logger::init();

    let event_loop = EventLoop::new().expect("create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::default();
    event_loop.run_app(&mut app);

    #[allow(unreachable_code)]
    Ok(())
}

