use anyhow::{anyhow, Result};
use ash::{vk, Entry};
use ash::khr::{surface, swapchain};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::{ffi::CString, ptr, time::Instant, sync::{Arc, Mutex}};
use winit::{
    application::ApplicationHandler,
    event::{WindowEvent, ElementState, MouseButton},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowAttributes},
};
use midir::{MidiInput, Ignore};

// MIDI state to share between threads
#[derive(Clone, Debug)]
struct MidiState {
    notes: [f32; 128],        // Note velocities (0.0 = off, 0.0-1.0 = on)
    controllers: [f32; 128],  // CC values (0.0-1.0)
    pitch_bend: f32,          // Pitch bend (-1.0 to 1.0)
    last_note: u8,            // Last note played
    note_count: u32,          // Number of notes currently pressed
}

impl Default for MidiState {
    fn default() -> Self {
        Self {
            notes: [0.0; 128],
            controllers: [0.5; 128], // Start CCs at middle position
            pitch_bend: 0.0,
            last_note: 60, // Middle C
            note_count: 0,
        }
    }
}

// Push constants structure (must match shader layout)
#[repr(C)]
#[derive(Clone, Copy)]
struct PushConstants {
    time: f32,
    mouse_x: u32,
    mouse_y: u32,
    mouse_pressed: u32,
    // MIDI data
    note_velocity: f32,       // Velocity of last played note
    pitch_bend: f32,          // Pitch bend value
    cc1: f32,                 // Modulation wheel (CC1)
    cc74: f32,                // Filter cutoff (CC74)
    note_count: u32,          // Number of notes pressed
    last_note: u32,           // Last note number
}

struct Gfx {
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
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    framebuffers: Vec<vk::Framebuffer>,
    cmd_pool: vk::CommandPool,
    cmd_bufs: Vec<vk::CommandBuffer>,
    sync: (vk::Semaphore, vk::Semaphore, vk::Fence), // img_acq, render_done, inflight
}

impl Gfx {
    unsafe fn new(window: &Window) -> Result<Self> {
        let entry = Entry::linked();
        let display_handle = window.display_handle()?.as_raw();
        let window_handle = window.window_handle()?.as_raw();
        let ext_names = ash_window::enumerate_required_extensions(display_handle)?.to_vec();

        // Instance
        let app_name = CString::new("vulkan-pixel-shader")?;
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
            enabled_extension_count: ext_names.len() as u32,
            pp_enabled_extension_names: ext_names.as_ptr(),
            ..Default::default()
        };

        let instance = entry.create_instance(&create_info, None)?;

        // Surface
        let surface = ash_window::create_surface(&entry, &instance, display_handle, window_handle, None)?;
        let surface_loader = surface::Instance::new(&entry, &instance);

        // Physical device + queue family
        let (pdev, qfi) = instance.enumerate_physical_devices()?
            .into_iter()
            .find_map(|pd| {
                instance.get_physical_device_queue_family_properties(pd)
                    .iter().enumerate()
                    .find(|(i, q)| {
                        q.queue_flags.contains(vk::QueueFlags::GRAPHICS) &&
                        surface_loader.get_physical_device_surface_support(pd, *i as u32, surface).unwrap_or(false)
                    })
                    .map(|(i, _)| (pd, i as u32))
            })
            .ok_or_else(|| anyhow!("No suitable GPU found"))?;

        // Logical device
        let queue_priorities = [1.0f32];
        let queue_info = vk::DeviceQueueCreateInfo {
            queue_family_index: qfi,
            queue_count: 1,
            p_queue_priorities: queue_priorities.as_ptr(),
            ..Default::default()
        };

        let device_extensions = [swapchain::NAME.as_ptr()];
        let device_info = vk::DeviceCreateInfo {
            queue_create_info_count: 1,
            p_queue_create_infos: &queue_info,
            enabled_extension_count: device_extensions.len() as u32,
            pp_enabled_extension_names: device_extensions.as_ptr(),
            ..Default::default()
        };

        let device = instance.create_device(pdev, &device_info, None)?;
        let queue = device.get_device_queue(qfi, 0);

        // Swapchain
        let swapchain_loader = swapchain::Device::new(&instance, &device);
        let caps = surface_loader.get_physical_device_surface_capabilities(pdev, surface)?;
        let formats = surface_loader.get_physical_device_surface_formats(pdev, surface)?;
        
        let format = formats.iter()
            .find(|f| f.format == vk::Format::B8G8R8A8_SRGB)
            .unwrap_or(&formats[0])
            .format;

        let extent = if caps.current_extent.width != u32::MAX {
            caps.current_extent
        } else {
            let size = window.inner_size();
            vk::Extent2D { width: size.width, height: size.height }
        };

        let image_count = (caps.min_image_count + 1).min(
            if caps.max_image_count > 0 { caps.max_image_count } else { u32::MAX }
        );

        let swapchain_info = vk::SwapchainCreateInfoKHR {
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
            ..Default::default()
        };

        let swapchain = swapchain_loader.create_swapchain(&swapchain_info, None)?;
        let images = swapchain_loader.get_swapchain_images(swapchain)?;

        // Image views
        let views: Result<Vec<_>> = images.iter().map(|&img| {
            let view_info = vk::ImageViewCreateInfo {
                image: img,
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
            device.create_image_view(&view_info, None).map_err(|e| anyhow::Error::from(e))
        }).collect();
        let views = views?;

        // Render pass
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

        let color_ref = vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            ..Default::default()
        };

        let subpass = vk::SubpassDescription {
            pipeline_bind_point: vk::PipelineBindPoint::GRAPHICS,
            color_attachment_count: 1,
            p_color_attachments: &color_ref,
            ..Default::default()
        };

        let render_pass_info = vk::RenderPassCreateInfo {
            attachment_count: 1,
            p_attachments: &color_attachment,
            subpass_count: 1,
            p_subpasses: &subpass,
            ..Default::default()
        };

        let render_pass = device.create_render_pass(&render_pass_info, None)?;

        // Shaders & Pipeline
        let vert_code = Self::compile_shader(include_str!("shaders/fullscreen.vert"), shaderc::ShaderKind::Vertex)?;
        let frag_code = Self::compile_shader(include_str!("shaders/gradient.frag"), shaderc::ShaderKind::Fragment)?;
        
        let vert_module = Self::create_shader_module(&device, &vert_code)?;
        let frag_module = Self::create_shader_module(&device, &frag_code)?;

        let entry_name = CString::new("main")?;
        let stages = [
            vk::PipelineShaderStageCreateInfo {
                stage: vk::ShaderStageFlags::VERTEX,
                module: vert_module,
                p_name: entry_name.as_ptr(),
                ..Default::default()
            },
            vk::PipelineShaderStageCreateInfo {
                stage: vk::ShaderStageFlags::FRAGMENT,
                module: frag_module,
                p_name: entry_name.as_ptr(),
                ..Default::default()
            },
        ];

        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo {
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            ..Default::default()
        };

        let viewport = vk::Viewport {
            x: 0.0, y: 0.0,
            width: extent.width as f32,
            height: extent.height as f32,
            min_depth: 0.0, max_depth: 1.0,
        };
        let scissor = vk::Rect2D {
            extent,
            ..Default::default()
        };
        let viewport_state = vk::PipelineViewportStateCreateInfo {
            viewport_count: 1,
            p_viewports: &viewport,
            scissor_count: 1,
            p_scissors: &scissor,
            ..Default::default()
        };

        let rasterizer = vk::PipelineRasterizationStateCreateInfo {
            polygon_mode: vk::PolygonMode::FILL,
            cull_mode: vk::CullModeFlags::BACK,
            front_face: vk::FrontFace::CLOCKWISE,
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

        // Push constants for uniforms
        let push_constant_range = vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            offset: 0,
            size: std::mem::size_of::<PushConstants>() as u32,
        };

        let pipeline_layout = device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo {
                push_constant_range_count: 1,
                p_push_constant_ranges: &push_constant_range,
                ..Default::default()
            }, None
        )?;

        let pipeline_info = vk::GraphicsPipelineCreateInfo {
            stage_count: stages.len() as u32,
            p_stages: stages.as_ptr(),
            p_vertex_input_state: &vertex_input,
            p_input_assembly_state: &input_assembly,
            p_viewport_state: &viewport_state,
            p_rasterization_state: &rasterizer,
            p_multisample_state: &multisampling,
            p_color_blend_state: &color_blending,
            layout: pipeline_layout,
            render_pass,
            ..Default::default()
        };

        let pipelines = device.create_graphics_pipelines(vk::PipelineCache::null(), 
            &[pipeline_info], None).map_err(|e| e.1)?;
        let pipeline = pipelines[0];

        device.destroy_shader_module(vert_module, None);
        device.destroy_shader_module(frag_module, None);

        // Framebuffers
        let framebuffers: Result<Vec<_>> = views.iter().map(|&view| {
            let fb_info = vk::FramebufferCreateInfo {
                render_pass,
                attachment_count: 1,
                p_attachments: &view,
                width: extent.width,
                height: extent.height,
                layers: 1,
                ..Default::default()
            };
            device.create_framebuffer(&fb_info, None).map_err(|e| anyhow::Error::from(e))
        }).collect();
        let framebuffers = framebuffers?;

        // Commands - allocate with RESET flag for re-recording
        let cmd_pool = device.create_command_pool(
            &vk::CommandPoolCreateInfo {
                flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
                queue_family_index: qfi,
                ..Default::default()
            }, None
        )?;

        let cmd_bufs = device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo {
                command_pool: cmd_pool,
                level: vk::CommandBufferLevel::PRIMARY,
                command_buffer_count: framebuffers.len() as u32,
                ..Default::default()
            }
        )?;

        // Don't pre-record command buffers - we'll record them each frame
        // to update push constants

        // Sync objects
        let sem_info = vk::SemaphoreCreateInfo::default();
        let fence_info = vk::FenceCreateInfo {
            flags: vk::FenceCreateFlags::SIGNALED,
            ..Default::default()
        };
        
        let sync = (
            device.create_semaphore(&sem_info, None)?,
            device.create_semaphore(&sem_info, None)?,
            device.create_fence(&fence_info, None)?
        );

        // Clean up views (not needed after framebuffers)
        for view in views { device.destroy_image_view(view, None); }

        Ok(Self {
            _entry: entry, instance, surface_loader, surface, device, queue,
            swapchain_loader, swapchain, extent, render_pass, pipeline, pipeline_layout,
            framebuffers, cmd_pool, cmd_bufs, sync,
        })
    }

    unsafe fn compile_shader(source: &str, kind: shaderc::ShaderKind) -> Result<Vec<u32>> {
        let compiler = shaderc::Compiler::new().ok_or_else(|| anyhow!("Failed to create compiler"))?;
        let result = compiler.compile_into_spirv(source, kind, "shader", "main", None)
            .map_err(|e| anyhow!("Shader compilation failed: {}", e))?;
        Ok(result.as_binary().to_vec())
    }

    unsafe fn create_shader_module(device: &ash::Device, code: &[u32]) -> Result<vk::ShaderModule> {
        let create_info = vk::ShaderModuleCreateInfo {
            code_size: code.len() * 4,
            p_code: code.as_ptr(),
            ..Default::default()
        };
        Ok(device.create_shader_module(&create_info, None)?)
    }

    unsafe fn draw(&self, push_constants: &PushConstants) -> Result<()> {
        self.device.wait_for_fences(&[self.sync.2], true, u64::MAX)?;
        self.device.reset_fences(&[self.sync.2])?;

        let (image_index, _) = self.swapchain_loader
            .acquire_next_image(self.swapchain, u64::MAX, self.sync.0, vk::Fence::null())?;

        // Re-record command buffer with updated push constants
        let cb = self.cmd_bufs[image_index as usize];
        self.device.begin_command_buffer(cb, &vk::CommandBufferBeginInfo::default())?;
        
        let clear_values = [vk::ClearValue {
            color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] }
        }];
        
        let scissor = vk::Rect2D {
            extent: self.extent,
            ..Default::default()
        };
        
        let render_pass_begin = vk::RenderPassBeginInfo {
            render_pass: self.render_pass,
            framebuffer: self.framebuffers[image_index as usize],
            render_area: scissor,
            clear_value_count: clear_values.len() as u32,
            p_clear_values: clear_values.as_ptr(),
            ..Default::default()
        };
        
        self.device.cmd_begin_render_pass(cb, &render_pass_begin, vk::SubpassContents::INLINE);
        self.device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
        
        // Push constants to shader
        self.device.cmd_push_constants(
            cb,
            self.pipeline_layout,
            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            0,
            std::slice::from_raw_parts(
                push_constants as *const PushConstants as *const u8,
                std::mem::size_of::<PushConstants>()
            )
        );
        
        self.device.cmd_draw(cb, 3, 1, 0, 0);
        self.device.cmd_end_render_pass(cb);
        self.device.end_command_buffer(cb)?;

        let submit_info = vk::SubmitInfo {
            wait_semaphore_count: 1,
            p_wait_semaphores: &self.sync.0,
            p_wait_dst_stage_mask: &vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            command_buffer_count: 1,
            p_command_buffers: &self.cmd_bufs[image_index as usize],
            signal_semaphore_count: 1,
            p_signal_semaphores: &self.sync.1,
            ..Default::default()
        };

        self.device.queue_submit(self.queue, &[submit_info], self.sync.2)?;

        let present_info = vk::PresentInfoKHR {
            wait_semaphore_count: 1,
            p_wait_semaphores: &self.sync.1,
            swapchain_count: 1,
            p_swapchains: &self.swapchain,
            p_image_indices: &image_index,
            ..Default::default()
        };

        self.swapchain_loader.queue_present(self.queue, &present_info)?;
        Ok(())
    }
}

impl Drop for Gfx {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().ok();
            self.device.destroy_fence(self.sync.2, None);
            self.device.destroy_semaphore(self.sync.1, None);
            self.device.destroy_semaphore(self.sync.0, None);
            self.device.free_command_buffers(self.cmd_pool, &self.cmd_bufs);
            self.device.destroy_command_pool(self.cmd_pool, None);
            for &fb in &self.framebuffers { self.device.destroy_framebuffer(fb, None); }
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_pipeline_layout(self.pipeline_layout, None);
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
    gfx: Option<Gfx>,
    start_time: Option<Instant>,
    mouse_pos: (f64, f64),
    mouse_pressed: bool,
    midi_state: Arc<Mutex<MidiState>>,
    _midi_connection: Option<midir::MidiInputConnection<()>>,
}

impl App {
    fn setup_midi(&mut self) {
        match self.try_setup_midi() {
            Ok(connection) => {
                self._midi_connection = Some(connection);
                println!("MIDI input connected successfully!");
            }
            Err(e) => {
                eprintln!("MIDI setup failed: {}. Continuing without MIDI input.", e);
            }
        }
    }
    
    fn try_setup_midi(&mut self) -> Result<midir::MidiInputConnection<()>, Box<dyn std::error::Error>> {
        let mut midi_in = MidiInput::new("Vulkan MIDI Visualizer")?;
        midi_in.ignore(Ignore::None);
        
        let in_ports = midi_in.ports();
        if in_ports.is_empty() {
            return Err("No MIDI input ports available".into());
        }
        
        // Auto-select first available port
        let in_port = &in_ports[1];
        let port_name = midi_in.port_name(in_port)?;
        println!("Connecting to MIDI port: {}", port_name);
        
        let midi_state = Arc::clone(&self.midi_state);
        
        let connection = midi_in.connect(in_port, "vulkan-visualizer", move |_timestamp, message, _| {
            Self::handle_midi_message(&midi_state, message);
        }, ())?;
        
        Ok(connection)
    }
    
    fn handle_midi_message(midi_state: &Arc<Mutex<MidiState>>, message: &[u8]) {
        if message.is_empty() {
            return;
        }
        
        let mut state = midi_state.lock().unwrap();
        let status = message[0];
        
        match status & 0xF0 {
            0x80 => {
                // Note Off
                if message.len() >= 3 {
                    let note = message[1] as usize;
                    if note < 128 {
                        if state.notes[note] > 0.0 {
                            state.note_count = state.note_count.saturating_sub(1);
                        }
                        state.notes[note] = 0.0;
                        println!("Note Off: {} (Count: {})", note, state.note_count);
                    }
                }
            }
            0x90 => {
                // Note On
                if message.len() >= 3 {
                    let note = message[1] as usize;
                    let velocity = message[2];
                    
                    if note < 128 {
                        if velocity == 0 {
                            // Velocity 0 = Note Off
                            if state.notes[note] > 0.0 {
                                state.note_count = state.note_count.saturating_sub(1);
                            }
                            state.notes[note] = 0.0;
                            println!("Note Off: {} (Count: {})", note, state.note_count);
                        } else {
                            // Note On
                            if state.notes[note] == 0.0 {
                                state.note_count += 1;
                            }
                            state.notes[note] = velocity as f32 / 127.0;
                            state.last_note = note as u8;
                            println!("Note On: {} Velocity: {} (Count: {})", note, velocity, state.note_count);
                        }
                    }
                }
            }
            0xB0 => {
                // Control Change
                if message.len() >= 3 {
                    let controller = message[1] as usize;
                    let value = message[2];
                    
                    if controller < 128 {
                        state.controllers[controller] = value as f32 / 127.0;
                        println!("CC{}: {}", controller, value);
                    }
                }
            }
            0xE0 => {
                // Pitch Bend
                if message.len() >= 3 {
                    let bend_value = (message[2] as u16) << 7 | (message[1] as u16);
                    // Convert from 0-16383 to -1.0 to 1.0
                    state.pitch_bend = (bend_value as f32 / 8192.0) - 1.0;
                    println!("Pitch Bend: {:.3}", state.pitch_bend);
                }
            }
            _ => {
                // Ignore other MIDI messages for now
            }
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = event_loop.create_window(
            Window::default_attributes()
                .with_title("Vulkan MIDI Pixel Shader")
                .with_inner_size(winit::dpi::LogicalSize::new(800.0, 600.0))
        ).expect("Failed to create window");
        
        let gfx = unsafe { Gfx::new(&window).expect("Failed to initialize Vulkan") };
        self.window = Some(window);
        self.gfx = Some(gfx);
        self.start_time = Some(Instant::now());
        
        // Initialize MIDI
        self.setup_midi();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: winit::window::WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::CursorMoved { position, .. } => {
                self.mouse_pos = (position.x, position.y);
            }
            WindowEvent::MouseInput { state, button: MouseButton::Left, .. } => {
                self.mouse_pressed = state == ElementState::Pressed;
            }
            WindowEvent::RedrawRequested => {
                if let (Some(gfx), Some(start_time)) = (&self.gfx, &self.start_time) {
                    let elapsed = start_time.elapsed().as_secs_f32();
                    
                    // Get current MIDI state
                    let midi_state = self.midi_state.lock().unwrap();
                    let note_velocity = if midi_state.note_count > 0 {
                        midi_state.notes[midi_state.last_note as usize]
                    } else {
                        0.0
                    };
                    
                    let push_constants = PushConstants {
                        time: elapsed,
                        mouse_x: self.mouse_pos.0 as u32,
                        mouse_y: self.mouse_pos.1 as u32,
                        mouse_pressed: if self.mouse_pressed { 1 } else { 0 },
                        // MIDI values
                        note_velocity,
                        pitch_bend: midi_state.pitch_bend,
                        cc1: midi_state.controllers[1],   // Modulation wheel
                        cc74: midi_state.controllers[74], // Filter cutoff
                        note_count: midi_state.note_count,
                        last_note: midi_state.last_note as u32,
                    };
                    
                    drop(midi_state); // Release the lock
                    
                    if let Err(e) = unsafe { gfx.draw(&push_constants) } {
                        eprintln!("Draw error: {}", e);
                        event_loop.exit();
                    }
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _: &ActiveEventLoop) {
        if let Some(window) = &self.window { window.request_redraw(); }
    }
}

fn main() -> Result<()> {
    env_logger::init();
    let event_loop = EventLoop::new().expect("Failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::default();
    event_loop.run_app(&mut app);
    Ok(())
}
