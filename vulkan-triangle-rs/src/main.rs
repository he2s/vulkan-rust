use ash::vk;
use std::ffi::CString;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
    raw_window_handle::{HasDisplayHandle, HasWindowHandle},
};

struct VulkanApp {
    entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    physical_device: vk::PhysicalDevice,
    queue_family_index: u32,
    queue: vk::Queue,
    surface_loader: ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
    swapchain_loader: ash::khr::swapchain::Device,
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_image_views: Vec<vk::ImageView>,
    render_pass: vk::RenderPass,
    pipeline_layout: vk::PipelineLayout,
    graphics_pipeline: vk::Pipeline,
    framebuffers: Vec<vk::Framebuffer>,
    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,
    current_frame: usize,
    swapchain_extent: vk::Extent2D,
}

const MAX_FRAMES_IN_FLIGHT: usize = 2;

const VERTEX_SHADER_CODE: &[u8] = include_bytes!("../shaders/vert.spv");
const FRAGMENT_SHADER_CODE: &[u8] = include_bytes!("../shaders/frag.spv");

impl VulkanApp {
    fn new(window: &Window) -> Self {
        let entry = unsafe { ash::Entry::load() }.expect("Failed to load Vulkan entry points");

        // Create instance
        let app_name = CString::new("Vulkan Triangle").unwrap();
        let engine_name = CString::new("No Engine").unwrap();
        let app_info = vk::ApplicationInfo {
            p_application_name: app_name.as_ptr(),
            application_version: vk::make_api_version(0, 1, 0, 0),
            p_engine_name: engine_name.as_ptr(),
            engine_version: vk::make_api_version(0, 1, 0, 0),
            api_version: vk::make_api_version(0, 1, 0, 0),
            ..Default::default()
        };

        let extension_names = ash_window::enumerate_required_extensions(window.display_handle().unwrap().as_raw()).unwrap();

        // Add validation layers in debug builds
        #[cfg(debug_assertions)]
        let layer_names = [CString::new("VK_LAYER_KHRONOS_validation").unwrap()];
        #[cfg(debug_assertions)]
        let layers_names_raw: Vec<*const i8> = layer_names
            .iter()
            .map(|raw_name| raw_name.as_ptr())
            .collect();

        let create_info = vk::InstanceCreateInfo {
            p_application_info: &app_info,
            enabled_extension_count: extension_names.len() as u32,
            pp_enabled_extension_names: extension_names.as_ptr(),
            #[cfg(debug_assertions)]
            enabled_layer_count: layers_names_raw.len() as u32,
            #[cfg(debug_assertions)]
            pp_enabled_layer_names: layers_names_raw.as_ptr(),
            #[cfg(not(debug_assertions))]
            enabled_layer_count: 0,
            #[cfg(not(debug_assertions))]
            pp_enabled_layer_names: std::ptr::null(),
            ..Default::default()
        };

        let instance = unsafe { entry.create_instance(&create_info, None).unwrap() };

        // Create surface
        let surface = unsafe {
            ash_window::create_surface(
                &entry,
                &instance,
                window.display_handle().unwrap().as_raw(),
                window.window_handle().unwrap().as_raw(),
                None
            ).unwrap()
        };
        let surface_loader = ash::khr::surface::Instance::new(&entry, &instance);

        // Pick physical device
        let physical_devices = unsafe { instance.enumerate_physical_devices().unwrap() };
        let physical_device = physical_devices[0];

        // Find queue family
        let queue_family_properties = unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        let queue_family_index = queue_family_properties
            .iter()
            .enumerate()
            .find_map(|(index, info)| {
                let supports_graphic_and_surface = info.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                    && unsafe { surface_loader.get_physical_device_surface_support(physical_device, index as u32, surface).unwrap() };
                if supports_graphic_and_surface {
                    Some(index as u32)
                } else {
                    None
                }
            })
            .unwrap();

        // Create logical device
        let device_extension_names_raw = [ash::khr::swapchain::NAME.as_ptr()];
        let priorities = [1.0];
        let queue_info = vk::DeviceQueueCreateInfo {
            queue_family_index,
            queue_count: 1,
            p_queue_priorities: priorities.as_ptr(),
            ..Default::default()
        };

        let device_create_info = vk::DeviceCreateInfo {
            queue_create_info_count: 1,
            p_queue_create_infos: &queue_info,
            enabled_extension_count: device_extension_names_raw.len() as u32,
            pp_enabled_extension_names: device_extension_names_raw.as_ptr(),
            ..Default::default()
        };

        let device = unsafe { instance.create_device(physical_device, &device_create_info, None).unwrap() };
        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        // Create swapchain
        let surface_capabilities = unsafe { surface_loader.get_physical_device_surface_capabilities(physical_device, surface).unwrap() };
        let surface_formats = unsafe { surface_loader.get_physical_device_surface_formats(physical_device, surface).unwrap() };

        let surface_format = surface_formats[0];
        let present_mode = vk::PresentModeKHR::FIFO;

        // Fix extent validation
        let extent = if surface_capabilities.current_extent.width != u32::MAX {
            surface_capabilities.current_extent
        } else {
            vk::Extent2D {
                width: 800,
                height: 600,
            }
        };

        // Fix image count calculation
        let mut desired_image_count = surface_capabilities.min_image_count + 1;
        if surface_capabilities.max_image_count > 0 {
            desired_image_count = desired_image_count.min(surface_capabilities.max_image_count);
        }

        let swapchain_loader = ash::khr::swapchain::Device::new(&instance, &device);
        let swapchain_create_info = vk::SwapchainCreateInfoKHR {
            surface,
            min_image_count: desired_image_count,
            image_format: surface_format.format,
            image_color_space: surface_format.color_space,
            image_extent: extent,
            image_array_layers: 1,
            image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
            image_sharing_mode: vk::SharingMode::EXCLUSIVE,
            pre_transform: surface_capabilities.current_transform,
            composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
            present_mode,
            clipped: vk::TRUE,
            ..Default::default()
        };

        let swapchain = unsafe { swapchain_loader.create_swapchain(&swapchain_create_info, None).unwrap() };
        let swapchain_images = unsafe { swapchain_loader.get_swapchain_images(swapchain).unwrap() };

        // Create image views
        let swapchain_image_views: Vec<vk::ImageView> = swapchain_images
            .iter()
            .map(|&image| {
                let create_view_info = vk::ImageViewCreateInfo {
                    image,
                    view_type: vk::ImageViewType::TYPE_2D,
                    format: surface_format.format,
                    components: vk::ComponentMapping {
                        r: vk::ComponentSwizzle::IDENTITY,
                        g: vk::ComponentSwizzle::IDENTITY,
                        b: vk::ComponentSwizzle::IDENTITY,
                        a: vk::ComponentSwizzle::IDENTITY,
                    },
                    subresource_range: vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    ..Default::default()
                };
                unsafe { device.create_image_view(&create_view_info, None).unwrap() }
            })
            .collect();

        // Create render pass
        let color_attachment = vk::AttachmentDescription {
            format: surface_format.format,
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

        let dependency = vk::SubpassDependency {
            src_subpass: vk::SUBPASS_EXTERNAL,
            dst_subpass: 0,
            src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            src_access_mask: vk::AccessFlags::empty(),
            dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            ..Default::default()
        };

        let render_pass_info = vk::RenderPassCreateInfo {
            attachment_count: 1,
            p_attachments: &color_attachment,
            subpass_count: 1,
            p_subpasses: &subpass,
            dependency_count: 1,
            p_dependencies: &dependency,
            ..Default::default()
        };

        let render_pass = unsafe { device.create_render_pass(&render_pass_info, None).unwrap() };

        // Create shader modules
        let vert_shader_module = Self::create_shader_module(&device, VERTEX_SHADER_CODE);
        let frag_shader_module = Self::create_shader_module(&device, FRAGMENT_SHADER_CODE);

        let main_function_name = CString::new("main").unwrap();
        let vert_shader_stage_info = vk::PipelineShaderStageCreateInfo {
            stage: vk::ShaderStageFlags::VERTEX,
            module: vert_shader_module,
            p_name: main_function_name.as_ptr(),
            ..Default::default()
        };

        let frag_shader_stage_info = vk::PipelineShaderStageCreateInfo {
            stage: vk::ShaderStageFlags::FRAGMENT,
            module: frag_shader_module,
            p_name: main_function_name.as_ptr(),
            ..Default::default()
        };

        let shader_stages = vec![vert_shader_stage_info, frag_shader_stage_info];

        // Create graphics pipeline
        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default();
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo {
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            primitive_restart_enable: vk::FALSE,
            ..Default::default()
        };

        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: extent.width as f32,
            height: extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };

        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent,
        };

        let viewport_state = vk::PipelineViewportStateCreateInfo {
            viewport_count: 1,
            p_viewports: &viewport,
            scissor_count: 1,
            p_scissors: &scissor,
            ..Default::default()
        };

        let rasterizer = vk::PipelineRasterizationStateCreateInfo {
            depth_clamp_enable: vk::FALSE,
            rasterizer_discard_enable: vk::FALSE,
            polygon_mode: vk::PolygonMode::FILL,
            line_width: 1.0,
            cull_mode: vk::CullModeFlags::BACK,
            front_face: vk::FrontFace::CLOCKWISE,
            depth_bias_enable: vk::FALSE,
            ..Default::default()
        };

        let multisampling = vk::PipelineMultisampleStateCreateInfo {
            sample_shading_enable: vk::FALSE,
            rasterization_samples: vk::SampleCountFlags::TYPE_1,
            ..Default::default()
        };

        let color_blend_attachment = vk::PipelineColorBlendAttachmentState {
            color_write_mask: vk::ColorComponentFlags::R
                | vk::ColorComponentFlags::G
                | vk::ColorComponentFlags::B
                | vk::ColorComponentFlags::A,
            blend_enable: vk::FALSE,
            ..Default::default()
        };

        let color_blending = vk::PipelineColorBlendStateCreateInfo {
            logic_op_enable: vk::FALSE,
            attachment_count: 1,
            p_attachments: &color_blend_attachment,
            ..Default::default()
        };

        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default();
        let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None).unwrap() };

        let pipeline_info = vk::GraphicsPipelineCreateInfo {
            stage_count: shader_stages.len() as u32,
            p_stages: shader_stages.as_ptr(),
            p_vertex_input_state: &vertex_input_info,
            p_input_assembly_state: &input_assembly,
            p_viewport_state: &viewport_state,
            p_rasterization_state: &rasterizer,
            p_multisample_state: &multisampling,
            p_color_blend_state: &color_blending,
            layout: pipeline_layout,
            render_pass,
            subpass: 0,
            ..Default::default()
        };

        let graphics_pipeline = unsafe {
            device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .unwrap()[0]
        };

        // Clean up shader modules
        unsafe {
            device.destroy_shader_module(vert_shader_module, None);
            device.destroy_shader_module(frag_shader_module, None);
        }

        // Create framebuffers
        let framebuffers: Vec<vk::Framebuffer> = swapchain_image_views
            .iter()
            .map(|&image_view| {
                let attachments = [image_view];
                let framebuffer_info = vk::FramebufferCreateInfo {
                    render_pass,
                    attachment_count: 1,
                    p_attachments: attachments.as_ptr(),
                    width: extent.width,
                    height: extent.height,
                    layers: 1,
                    ..Default::default()
                };
                unsafe { device.create_framebuffer(&framebuffer_info, None).unwrap() }
            })
            .collect();

        // Create command pool
        let pool_info = vk::CommandPoolCreateInfo {
            flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            queue_family_index,
            ..Default::default()
        };
        let command_pool = unsafe { device.create_command_pool(&pool_info, None).unwrap() };

        // Create command buffers
        let alloc_info = vk::CommandBufferAllocateInfo {
            command_pool,
            level: vk::CommandBufferLevel::PRIMARY,
            command_buffer_count: MAX_FRAMES_IN_FLIGHT as u32,
            ..Default::default()
        };
        let command_buffers = unsafe { device.allocate_command_buffers(&alloc_info).unwrap() };

        // Create sync objects
        let semaphore_info = vk::SemaphoreCreateInfo::default();
        let fence_info = vk::FenceCreateInfo {
            flags: vk::FenceCreateFlags::SIGNALED,
            ..Default::default()
        };

        let mut image_available_semaphores = vec![];
        let mut render_finished_semaphores = vec![];
        let mut in_flight_fences = vec![];

        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            image_available_semaphores.push(unsafe { device.create_semaphore(&semaphore_info, None).unwrap() });
            render_finished_semaphores.push(unsafe { device.create_semaphore(&semaphore_info, None).unwrap() });
            in_flight_fences.push(unsafe { device.create_fence(&fence_info, None).unwrap() });
        }

        Self {
            entry,
            instance,
            device,
            physical_device,
            queue_family_index,
            queue,
            surface_loader,
            surface,
            swapchain_loader,
            swapchain,
            swapchain_images,
            swapchain_image_views,
            render_pass,
            pipeline_layout,
            graphics_pipeline,
            framebuffers,
            command_pool,
            command_buffers,
            image_available_semaphores,
            render_finished_semaphores,
            in_flight_fences,
            current_frame: 0,
            swapchain_extent: extent,
        }
    }

    fn create_shader_module(device: &ash::Device, code: &[u8]) -> vk::ShaderModule {
        let code_u32: Vec<u32> = code
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        let create_info = vk::ShaderModuleCreateInfo {
            code_size: code.len(),
            p_code: code_u32.as_ptr(),
            ..Default::default()
        };
        unsafe { device.create_shader_module(&create_info, None).unwrap() }
    }

    fn record_command_buffer(&self, command_buffer: vk::CommandBuffer, image_index: u32) {
        let begin_info = vk::CommandBufferBeginInfo::default();
        unsafe { self.device.begin_command_buffer(command_buffer, &begin_info).unwrap() };

        let clear_values = [vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        }];

        let render_pass_info = vk::RenderPassBeginInfo {
            render_pass: self.render_pass,
            framebuffer: self.framebuffers[image_index as usize],
            render_area: vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.swapchain_extent,
            },
            clear_value_count: clear_values.len() as u32,
            p_clear_values: clear_values.as_ptr(),
            ..Default::default()
        };

        unsafe {
            self.device.cmd_begin_render_pass(command_buffer, &render_pass_info, vk::SubpassContents::INLINE);
            self.device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, self.graphics_pipeline);
            self.device.cmd_draw(command_buffer, 3, 1, 0, 0);
            self.device.cmd_end_render_pass(command_buffer);
            self.device.end_command_buffer(command_buffer).unwrap();
        }
    }

    fn draw_frame(&mut self) {
        unsafe {
            self.device
                .wait_for_fences(&[self.in_flight_fences[self.current_frame]], true, u64::MAX)
                .unwrap();
            self.device.reset_fences(&[self.in_flight_fences[self.current_frame]]).unwrap();

            let image_index = self
                .swapchain_loader
                .acquire_next_image(
                    self.swapchain,
                    u64::MAX,
                    self.image_available_semaphores[self.current_frame],
                    vk::Fence::null(),
                )
                .unwrap()
                .0;

            self.device
                .reset_command_buffer(
                    self.command_buffers[self.current_frame],
                    vk::CommandBufferResetFlags::empty(),
                )
                .unwrap();

            self.record_command_buffer(self.command_buffers[self.current_frame], image_index);

            let wait_semaphores = [self.image_available_semaphores[self.current_frame]];
            let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let signal_semaphores = [self.render_finished_semaphores[self.current_frame]];

            let submit_info = vk::SubmitInfo {
                wait_semaphore_count: wait_semaphores.len() as u32,
                p_wait_semaphores: wait_semaphores.as_ptr(),
                p_wait_dst_stage_mask: wait_stages.as_ptr(),
                command_buffer_count: 1,
                p_command_buffers: &self.command_buffers[self.current_frame],
                signal_semaphore_count: signal_semaphores.len() as u32,
                p_signal_semaphores: signal_semaphores.as_ptr(),
                ..Default::default()
            };

            self.device
                .queue_submit(self.queue, &[submit_info], self.in_flight_fences[self.current_frame])
                .unwrap();

            let swapchains = [self.swapchain];
            let image_indices = [image_index];
            let present_info = vk::PresentInfoKHR {
                wait_semaphore_count: signal_semaphores.len() as u32,
                p_wait_semaphores: signal_semaphores.as_ptr(),
                swapchain_count: swapchains.len() as u32,
                p_swapchains: swapchains.as_ptr(),
                p_image_indices: image_indices.as_ptr(),
                ..Default::default()
            };

            self.swapchain_loader.queue_present(self.queue, &present_info).unwrap();

            self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
        }
    }
}

impl Drop for VulkanApp {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();

            for i in 0..MAX_FRAMES_IN_FLIGHT {
                self.device.destroy_semaphore(self.image_available_semaphores[i], None);
                self.device.destroy_semaphore(self.render_finished_semaphores[i], None);
                self.device.destroy_fence(self.in_flight_fences[i], None);
            }

            self.device.destroy_command_pool(self.command_pool, None);

            for framebuffer in &self.framebuffers {
                self.device.destroy_framebuffer(*framebuffer, None);
            }

            self.device.destroy_pipeline(self.graphics_pipeline, None);
            self.device.destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_render_pass(self.render_pass, None);

            for image_view in &self.swapchain_image_views {
                self.device.destroy_image_view(*image_view, None);
            }

            self.swapchain_loader.destroy_swapchain(self.swapchain, None);
            self.device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);
            self.instance.destroy_instance(None);
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    let window = WindowBuilder::new()
        .with_title("Vulkan Triangle")
        .with_inner_size(winit::dpi::LogicalSize::new(800, 600))
        .build(&event_loop)
        .unwrap();

    let mut app = VulkanApp::new(&window);

    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);

        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => elwt.exit(),
            Event::AboutToWait => {
                app.draw_frame();
            }
            _ => {}
        }
    }).unwrap();
}