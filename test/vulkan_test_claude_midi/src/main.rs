use anyhow::{anyhow, Result};
use ash::{vk, Entry};
use ash::khr::{surface, swapchain};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::{ffi::CString, ptr, time::{Instant, Duration}, sync::{Arc, Mutex}, path::Path, fs, collections::VecDeque};
use winit::{
    application::ApplicationHandler,
    event::{WindowEvent, ElementState, MouseButton, KeyEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowAttributes, Fullscreen},
    keyboard::{PhysicalKey, KeyCode},
};
use midir::{MidiInput, Ignore};
use clap::Parser;
use serde::{Deserialize, Serialize};

// NEW: audio imports
use cpal::{traits::{DeviceTrait, HostTrait, StreamTrait}};
use rustfft::{FftPlanner, num_complex::Complex32};

#[derive(Parser)]
#[command(name = "vulkan-midi-visualizer")]
#[command(about = "A MIDI-reactive Vulkan pixel shader visualizer")]
struct Args {
    /// Path to configuration file
    #[arg(short, long, default_value = "config.toml")]
    config: String,

    /// Start in fullscreen mode
    #[arg(short, long)]
    fullscreen: bool,

    /// Window width (ignored in fullscreen)
    #[arg(long)]
    width: Option<u32>,

    /// Window height (ignored in fullscreen)
    #[arg(long)]
    height: Option<u32>,

    /// Window title
    #[arg(long)]
    title: Option<String>,

    /// Override shader from config
    #[arg(long)]
    shader: Option<String>,
}

#[derive(Deserialize, Serialize)]
struct Config {
    #[serde(default)]
    window: WindowConfig,
    #[serde(default)]
    midi: MidiConfig,
    #[serde(default)]
    graphics: GraphicsConfig,
    #[serde(default)]
    audio: AudioConfig,
    #[serde(default)]
    shader: ShaderConfig,
}

#[derive(Deserialize, Serialize)]
struct WindowConfig {
    #[serde(default = "default_width")]
    width: u32,
    #[serde(default = "default_height")]
    height: u32,
    #[serde(default = "default_title")]
    title: String,
    #[serde(default)]
    fullscreen: bool,
    #[serde(default = "default_true")]
    resizable: bool,
}

#[derive(Deserialize, Serialize)]
struct MidiConfig {
    #[serde(default = "default_true")]
    enabled: bool,
    #[serde(default)]
    auto_connect: bool,
    #[serde(default)]
    port_name: Option<String>,
}

#[derive(Deserialize, Serialize)]
struct GraphicsConfig {
    #[serde(default = "default_true")]
    vsync: bool,
    #[serde(default = "default_true")]
    validation_layers: bool,
}

#[derive(Deserialize, Serialize)]
struct AudioConfig {
    #[serde(default = "default_true")]
    enabled: bool,
    #[serde(default)]
    device_name: Option<String>,
    #[serde(default)]
    sample_rate: Option<u32>,
}

// NEW: Shader configuration
#[derive(Deserialize, Serialize, Clone)]
struct ShaderConfig {
    #[serde(default = "default_shader_preset")]
    preset: ShaderPreset,
    #[serde(default)]
    custom_vertex_path: Option<String>,
    #[serde(default)]
    custom_fragment_path: Option<String>,
    #[serde(default = "default_true")]
    allow_runtime_switching: bool,
}

#[derive(Deserialize, Serialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "lowercase")]
enum ShaderPreset {
    Torus,      // Original 4D torus
    Terrain,    // Organic terrain
    Crystal,    // Crystalline patterns
    Custom,     // Use custom paths
}

fn default_shader_preset() -> ShaderPreset { ShaderPreset::Torus }
fn default_width() -> u32 { 800 }
fn default_height() -> u32 { 600 }
fn default_title() -> String { "Vulkan MIDI Pixel Shader".to_string() }
fn default_true() -> bool { true }

impl Default for Config {
    fn default() -> Self {
        Self {
            window: WindowConfig::default(),
            midi: MidiConfig::default(),
            graphics: GraphicsConfig::default(),
            audio: AudioConfig::default(),
            shader: ShaderConfig::default(),
        }
    }
}

impl Default for WindowConfig {
    fn default() -> Self {
        Self {
            width: default_width(),
            height: default_height(),
            title: default_title(),
            fullscreen: false,
            resizable: default_true(),
        }
    }
}

impl Default for MidiConfig {
    fn default() -> Self {
        Self {
            enabled: default_true(),
            auto_connect: true,
            port_name: None,
        }
    }
}

impl Default for GraphicsConfig {
    fn default() -> Self {
        Self {
            vsync: default_true(),
            validation_layers: cfg!(debug_assertions),
        }
    }
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            device_name: None,
            sample_rate: None,
        }
    }
}

impl Default for ShaderConfig {
    fn default() -> Self {
        Self {
            preset: default_shader_preset(),
            custom_vertex_path: None,
            custom_fragment_path: None,
            allow_runtime_switching: true,
        }
    }
}

impl Config {
    fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = fs::read_to_string(path)?;
        Ok(toml::from_str(&content)?)
    }

    fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let content = toml::to_string_pretty(self)?;
        fs::write(path, content)?;
        Ok(())
    }

    fn merge_with_args(&mut self, args: &Args) {
        if args.fullscreen {
            self.window.fullscreen = true;
        }
        if let Some(width) = args.width {
            self.window.width = width;
        }
        if let Some(height) = args.height {
            self.window.height = height;
        }
        if let Some(ref title) = args.title {
            self.window.title = title.clone();
        }
        if let Some(ref shader) = args.shader {
            self.shader.preset = match shader.as_str() {
                "torus" => ShaderPreset::Torus,
                "terrain" => ShaderPreset::Terrain,
                "crystal" => ShaderPreset::Crystal,
                "custom" => ShaderPreset::Custom,
                _ => {
                    eprintln!("Unknown shader preset '{}', using default", shader);
                    self.shader.preset.clone()
                }
            };
        }
    }
}

// Shader source management
struct ShaderSources {
    vertex: String,
    fragment: String,
}

impl ShaderSources {
    fn load_preset(preset: &ShaderPreset) -> Result<Self> {
        match preset {
            ShaderPreset::Torus => Ok(Self {
                vertex: include_str!("shaders/fullscreen.vert").to_string(),
                fragment: include_str!("shaders/gradient.frag").to_string(),
            }),
            ShaderPreset::Terrain => Ok(Self {
                vertex: include_str!("shaders/terrain.vert").to_string(),
                fragment: include_str!("shaders/terrain.frag").to_string(),
            }),
            ShaderPreset::Crystal => Ok(Self {
                vertex: include_str!("shaders/crystal.vert").to_string(),
                fragment: include_str!("shaders/crystal.frag").to_string(),
            }),
            ShaderPreset::Custom => Err(anyhow!("Custom shader requires paths")),
        }
    }

    fn load_from_files(vertex_path: &str, fragment_path: &str) -> Result<Self> {
        Ok(Self {
            vertex: fs::read_to_string(vertex_path)?,
            fragment: fs::read_to_string(fragment_path)?,
        })
    }

    fn load_from_config(config: &ShaderConfig) -> Result<Self> {
        if config.preset == ShaderPreset::Custom {
            match (&config.custom_vertex_path, &config.custom_fragment_path) {
                (Some(vert), Some(frag)) => Self::load_from_files(vert, frag),
                _ => Err(anyhow!("Custom shader preset requires both vertex and fragment paths")),
            }
        } else {
            Self::load_preset(&config.preset)
        }
    }
}

// MIDI state to share between threads
#[derive(Clone, Debug)]
struct MidiState {
    notes: [f32; 128],
    controllers: [f32; 128],
    pitch_bend: f32,
    last_note: u8,
    note_count: u32,
}

impl Default for MidiState {
    fn default() -> Self {
        Self {
            notes: [0.0; 128],
            controllers: [0.5; 128],
            pitch_bend: 0.0,
            last_note: 60,
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
    note_velocity: f32,
    pitch_bend: f32,
    cc1: f32,
    cc74: f32,
    note_count: u32,
    last_note: u32,
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
    views: Vec<vk::ImageView>,
    render_pass: vk::RenderPass,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    framebuffers: Vec<vk::Framebuffer>,
    cmd_pool: vk::CommandPool,
    cmd_bufs: Vec<vk::CommandBuffer>,
    sync: (vk::Semaphore, vk::Semaphore, vk::Fence),
    frame_index: std::cell::RefCell<usize>,

    // Store physical device and queue family for pipeline recreation
    physical_device: vk::PhysicalDevice,
    queue_family_index: u32,
}

impl Gfx {
    unsafe fn new(window: &Window, shader_config: &ShaderConfig) -> Result<Self> {
        let entry = Entry::linked();
        let display_handle = window.display_handle()?.as_raw();
        let window_handle = window.window_handle()?.as_raw();
        let ext_names = ash_window::enumerate_required_extensions(display_handle)?.to_vec();

        // Instance with validation layers
        let app_name = CString::new("vulkan-pixel-shader")?;

        let layer_names = if cfg!(debug_assertions) {
            vec![CString::new("VK_LAYER_KHRONOS_validation").unwrap()]
        } else {
            vec![]
        };
        let layer_name_pointers: Vec<*const i8> = layer_names.iter().map(|name| name.as_ptr()).collect();

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

        // Load shader sources based on config
        println!("Loading shader preset: {:?}", shader_config.preset);
        let shader_sources = ShaderSources::load_from_config(shader_config)?;

        // Compile and create pipeline
        println!("Compiling shaders...");
        let vert_code = Self::compile_shader(&shader_sources.vertex, shaderc::ShaderKind::Vertex)?;
        let frag_code = Self::compile_shader(&shader_sources.fragment, shaderc::ShaderKind::Fragment)?;

        println!("Creating shader modules...");
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

        // Commands
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
                command_buffer_count: 2,
                ..Default::default()
            }
        )?;

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

        Ok(Self {
            _entry: entry, instance, surface_loader, surface, device, queue,
            swapchain_loader, swapchain, extent, views, render_pass, pipeline, pipeline_layout,
            framebuffers, cmd_pool, cmd_bufs, sync,
            frame_index: std::cell::RefCell::new(0),
            physical_device: pdev,
            queue_family_index: qfi,
        })
    }

    unsafe fn recreate_pipeline(&mut self, shader_config: &ShaderConfig) -> Result<()> {
        self.device.device_wait_idle()?;

        // Destroy old pipeline
        self.device.destroy_pipeline(self.pipeline, None);

        // Load new shader sources
        println!("Switching to shader preset: {:?}", shader_config.preset);
        let shader_sources = ShaderSources::load_from_config(shader_config)?;

        // Compile new shaders
        let vert_code = Self::compile_shader(&shader_sources.vertex, shaderc::ShaderKind::Vertex)?;
        let frag_code = Self::compile_shader(&shader_sources.fragment, shaderc::ShaderKind::Fragment)?;

        let vert_module = Self::create_shader_module(&self.device, &vert_code)?;
        let frag_module = Self::create_shader_module(&self.device, &frag_code)?;

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

        // Recreate pipeline with same configuration but new shaders
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo {
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            ..Default::default()
        };

        let viewport = vk::Viewport {
            x: 0.0, y: 0.0,
            width: self.extent.width as f32,
            height: self.extent.height as f32,
            min_depth: 0.0, max_depth: 1.0,
        };
        let scissor = vk::Rect2D {
            extent: self.extent,
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

        let pipeline_info = vk::GraphicsPipelineCreateInfo {
            stage_count: stages.len() as u32,
            p_stages: stages.as_ptr(),
            p_vertex_input_state: &vertex_input,
            p_input_assembly_state: &input_assembly,
            p_viewport_state: &viewport_state,
            p_rasterization_state: &rasterizer,
            p_multisample_state: &multisampling,
            p_color_blend_state: &color_blending,
            layout: self.pipeline_layout,
            render_pass: self.render_pass,
            ..Default::default()
        };

        let pipelines = self.device.create_graphics_pipelines(vk::PipelineCache::null(),
                                                              &[pipeline_info], None).map_err(|e| e.1)?;
        self.pipeline = pipelines[0];

        self.device.destroy_shader_module(vert_module, None);
        self.device.destroy_shader_module(frag_module, None);

        println!("Pipeline recreated successfully");
        Ok(())
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

        let mut frame_index = self.frame_index.borrow_mut();
        let current_cmd_buf = self.cmd_bufs[*frame_index % 2];
        *frame_index += 1;

        self.device.begin_command_buffer(current_cmd_buf, &vk::CommandBufferBeginInfo::default())?;

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

        self.device.cmd_begin_render_pass(current_cmd_buf, &render_pass_begin, vk::SubpassContents::INLINE);
        self.device.cmd_bind_pipeline(current_cmd_buf, vk::PipelineBindPoint::GRAPHICS, self.pipeline);

        self.device.cmd_push_constants(
            current_cmd_buf,
            self.pipeline_layout,
            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            0,
            std::slice::from_raw_parts(
                push_constants as *const PushConstants as *const u8,
                std::mem::size_of::<PushConstants>()
            )
        );

        self.device.cmd_draw(current_cmd_buf, 3, 1, 0, 0);
        self.device.cmd_end_render_pass(current_cmd_buf);
        self.device.end_command_buffer(current_cmd_buf)?;

        let submit_info = vk::SubmitInfo {
            wait_semaphore_count: 1,
            p_wait_semaphores: &self.sync.0,
            p_wait_dst_stage_mask: &vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            command_buffer_count: 1,
            p_command_buffers: &current_cmd_buf,
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
            for &view in &self.views { self.device.destroy_image_view(view, None); }
            self.swapchain_loader.destroy_swapchain(self.swapchain, None);
            self.surface_loader.destroy_surface(self.surface, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

// ---- Audio analysis state ----
#[derive(Clone, Debug)]
struct AudioState {
    ring: VecDeque<f32>,
    capacity: usize,
    last_sample_rate: u32,
    level_rms: f32,
    low: f32,
    mid: f32,
    high: f32,
}

impl AudioState {
    fn new() -> Self {
        Self {
            ring: VecDeque::with_capacity(4096),
            capacity: 4096,
            last_sample_rate: 48000,
            level_rms: 0.0,
            low: 0.0, mid: 0.0, high: 0.0,
        }
    }

    fn push_samples(&mut self, samples: &[f32], sr: u32) {
        self.last_sample_rate = sr;
        for &s in samples {
            if self.ring.len() == self.capacity { self.ring.pop_front(); }
            self.ring.push_back(s);
        }
    }

    fn snapshot(&mut self) {
        let n: usize = 1024;
        if self.ring.is_empty() {
            self.level_rms = 0.0; self.low = 0.0; self.mid = 0.0; self.high = 0.0;
            return;
        }
        let take = n.min(self.ring.len());
        let mut buf: Vec<f32> = self.ring.iter().rev().take(take).cloned().collect();
        buf.reverse();

        let rms = (buf.iter().map(|x| x * x).sum::<f32>() / (buf.len() as f32)).sqrt();

        let fft_len = buf.len().next_power_of_two().max(256).min(2048);
        buf.resize(fft_len, 0.0);
        for (i, x) in buf.iter_mut().enumerate() {
            let w = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (fft_len as f32)).cos());
            *x *= w as f32;
        }
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(fft_len);
        let mut spectrum: Vec<Complex32> = buf.into_iter().map(|v| Complex32::new(v, 0.0)).collect();
        fft.process(&mut spectrum);

        let sr = self.last_sample_rate as f32;
        let bin_hz = sr / (fft_len as f32);
        let mut low = 0.0;
        let mut mid = 0.0;
        let mut high = 0.0;
        for (i, c) in spectrum.iter().enumerate().take(fft_len/2) {
            let f = i as f32 * bin_hz;
            let mag = c.norm();
            if f < 20.0 { continue; }
            if f <= 250.0 { low += mag; }
            else if f <= 2000.0 { mid += mag; }
            else if f <= 8000.0 { high += mag; }
        }
        let norm = |x: f32| (x / 1000.0).min(1.0);
        let a = 0.7;
        self.level_rms = a * self.level_rms + (1.0 - a) * rms.min(1.0);
        self.low = a * self.low + (1.0 - a) * norm(low);
        self.mid = a * self.mid + (1.0 - a) * norm(mid);
        self.high = a * self.high + (1.0 - a) * norm(high);
    }
}

struct App {
    window: Option<Window>,
    gfx: Option<Gfx>,
    start_time: Option<Instant>,
    mouse_pos: (f64, f64),
    mouse_pressed: bool,
    midi_state: Arc<Mutex<MidiState>>,
    _midi_connection: Option<midir::MidiInputConnection<()>>,
    audio_state: Arc<Mutex<AudioState>>,
    _audio_stream: Option<cpal::Stream>,
    config: Config,
    is_fullscreen: bool,
    current_shader_index: usize,
    shader_presets: Vec<ShaderPreset>,
}

impl App {
    fn new(config: Config) -> Self {
        let is_fullscreen = config.window.fullscreen;
        let shader_presets = vec![
            ShaderPreset::Torus,
            ShaderPreset::Terrain,
            ShaderPreset::Crystal,
        ];

        Self {
            window: None,
            gfx: None,
            start_time: None,
            mouse_pos: (0.0, 0.0),
            mouse_pressed: false,
            midi_state: Arc::new(Mutex::new(MidiState::default())),
            _midi_connection: None,
            audio_state: Arc::new(Mutex::new(AudioState::new())),
            _audio_stream: None,
            config,
            is_fullscreen,
            current_shader_index: 0,
            shader_presets,
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
            println!("Toggled fullscreen: {}", if self.is_fullscreen { "ON" } else { "OFF" });
        }
    }

    fn cycle_shader(&mut self) {
        if !self.config.shader.allow_runtime_switching {
            println!("Runtime shader switching is disabled in config");
            return;
        }

        self.current_shader_index = (self.current_shader_index + 1) % self.shader_presets.len();
        let new_preset = self.shader_presets[self.current_shader_index].clone();

        // Update config with new preset
        self.config.shader.preset = new_preset.clone();

        // Recreate pipeline with new shader
        if let Some(gfx) = &mut self.gfx {
            println!("Switching to shader: {:?}", new_preset);
            if let Err(e) = unsafe { gfx.recreate_pipeline(&self.config.shader) } {
                eprintln!("Failed to switch shader: {}", e);
            }
        }
    }

    fn setup_midi(&mut self) {
        if !self.config.midi.enabled {
            println!("MIDI disabled in configuration");
            return;
        }

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

    fn setup_audio(&mut self) {
        if !self.config.audio.enabled {
            println!("Audio input disabled in configuration");
            return;
        }

        let host = cpal::default_host();
        let mut device = match &self.config.audio.device_name {
            Some(substr) => {
                let dev = host.input_devices().ok()
                    .and_then(|mut it| it.find(|d| d.name().map(|n| n.contains(substr)).unwrap_or(false)));
                dev.or_else(|| host.default_input_device())
            }
            None => host.default_input_device(),
        };

        let device = match device.take() {
            Some(d) => d,
            None => { eprintln!("No input audio device found"); return; }
        };

        let supported = match device.supported_input_configs() {
            Ok(cfgs) => cfgs.collect::<Vec<_>>(),
            Err(e) => { eprintln!("Failed to query audio formats: {}", e); return; }
        };
        if supported.is_empty() {
            eprintln!("No supported audio input configs");
            return;
        }

        let desired_sr = self.config.audio.sample_rate;
        let mut config = supported[0].with_max_sample_rate().config();
        for cfg in &supported {
            if cfg.sample_format() == cpal::SampleFormat::F32 {
                if let Some(req) = desired_sr {
                    if cfg.min_sample_rate().0 <= req && req <= cfg.max_sample_rate().0 {
                        config = cfg.clone().with_sample_rate(cpal::SampleRate(req)).config();
                        break;
                    }
                } else {
                    config = cfg.with_max_sample_rate().config();
                    break;
                }
            }
        }

        println!("Using audio device: {}", device.name().unwrap_or_else(|_| "Unknown".into()));
        println!("Audio config: {:?}Hz, {:?}ch", config.sample_rate.0, config.channels);

        let audio_state = Arc::clone(&self.audio_state);
        let channels = config.channels as usize;

        let stream = match device.build_input_stream(
            &config,
            move |data: &[f32], _| {
                let mut mono: Vec<f32> = Vec::with_capacity(data.len()/channels + 1);
                for frame in data.chunks_exact(channels) {
                    let s = frame.iter().copied().sum::<f32>() / (channels as f32);
                    mono.push(s);
                }
                if let Ok(mut st) = audio_state.lock() {
                    st.push_samples(&mono, config.sample_rate.0);
                }
            },
            move |err| {
                eprintln!("Audio input error: {}", err);
            },
            None
        ) {
            Ok(s) => s,
            Err(e) => { eprintln!("Failed to build audio input stream: {}", e); return; }
        };

        if let Err(e) = stream.play() {
            eprintln!("Failed to start audio stream: {}", e);
            return;
        }
        self._audio_stream = Some(stream);
        println!("Audio input connected successfully!");
    }

    fn try_setup_midi(&mut self) -> Result<midir::MidiInputConnection<()>, Box<dyn std::error::Error>> {
        let mut midi_in = MidiInput::new("Vulkan MIDI Visualizer")?;
        midi_in.ignore(Ignore::None);

        let in_ports = midi_in.ports();
        if in_ports.is_empty() {
            return Err("No MIDI input ports available".into());
        }

        let selected_port = if let Some(ref target_name) = self.config.midi.port_name {
            in_ports.iter().find(|&port| {
                midi_in.port_name(port).map_or(false, |name| name.contains(target_name))
            }).unwrap_or(&in_ports[0])
        } else {
            &in_ports[0]
        };

        let port_name = midi_in.port_name(selected_port)?;
        println!("Connecting to MIDI port: {}", port_name);

        let midi_state = Arc::clone(&self.midi_state);

        let connection = midi_in.connect(selected_port, "vulkan-visualizer", move |_timestamp, message, _| {
            Self::handle_midi_message(&midi_state, message);
        }, ())?;

        Ok(connection)
    }

    fn handle_midi_message(midi_state: &Arc<Mutex<MidiState>>, message: &[u8]) {
        if message.is_empty() {
            return;
        }

        let mut state = match midi_state.lock() {
            Ok(state) => state,
            Err(_) => return,
        };

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
                            if state.notes[note] > 0.0 {
                                state.note_count = state.note_count.saturating_sub(1);
                            }
                            state.notes[note] = 0.0;
                            println!("Note Off: {} (Count: {})", note, state.note_count);
                        } else {
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
                    state.pitch_bend = (bend_value as f32 / 8192.0) - 1.0;
                    println!("Pitch Bend: {:.3}", state.pitch_bend);
                }
            }
            _ => {}
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
                self.config.window.height as f64
            ));
        }

        let window = event_loop.create_window(attributes).expect("Failed to create window");

        let gfx = unsafe { Gfx::new(&window, &self.config.shader).expect("Failed to initialize Vulkan") };
        self.window = Some(window);
        self.gfx = Some(gfx);
        self.start_time = Some(Instant::now());

        // Set initial shader index based on config
        self.current_shader_index = self.shader_presets.iter()
            .position(|p| *p == self.config.shader.preset)
            .unwrap_or(0);

        self.setup_midi();
        self.setup_audio();

        println!("Controls:");
        println!("  F11 - Toggle fullscreen");
        println!("  ESC - Exit (or exit fullscreen)");
        if self.config.shader.allow_runtime_switching {
            println!("  TAB - Cycle shaders");
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: winit::window::WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput { event: KeyEvent { physical_key, state: ElementState::Pressed, .. }, .. } => {
                match physical_key {
                    PhysicalKey::Code(KeyCode::F11) => {
                        self.toggle_fullscreen();
                    }
                    PhysicalKey::Code(KeyCode::Escape) => {
                        if self.is_fullscreen {
                            self.toggle_fullscreen();
                        } else {
                            event_loop.exit();
                        }
                    }
                    PhysicalKey::Code(KeyCode::Tab) => {
                        self.cycle_shader();
                    }
                    _ => {}
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.mouse_pos = (position.x, position.y);
            }
            WindowEvent::MouseInput { state, button: MouseButton::Left, .. } => {
                self.mouse_pressed = state == ElementState::Pressed;
            }
            WindowEvent::RedrawRequested => {
                if let (Some(gfx), Some(start_time)) = (&self.gfx, &self.start_time) {
                    let elapsed = start_time.elapsed().as_secs_f32();

                    let midi_state = match self.midi_state.lock() {
                        Ok(state) => state,
                        Err(_) => return,
                    };

                    let (mut level, mut band_low, mut band_mid, mut band_high) = (0.0f32, 0.0, 0.0, 0.0);
                    if let Ok(mut a) = self.audio_state.lock() {
                        a.snapshot();
                        level = a.level_rms;
                        band_low = a.low;
                        band_mid = a.mid;
                        band_high = a.high;
                    }

                    let note_velocity = if midi_state.note_count > 0 {
                        midi_state.notes[midi_state.last_note as usize]
                    } else {
                        0.0
                    };

                    let blended_velocity = note_velocity.max(level);
                    let blended_pitch_bend = midi_state.pitch_bend.max(band_low * 2.0 - 1.0);
                    let blended_cc1 = midi_state.controllers[1].max(band_mid);
                    let blended_cc74 = midi_state.controllers[74].max(band_high);

                    let push_constants = PushConstants {
                        time: elapsed,
                        mouse_x: self.mouse_pos.0 as u32,
                        mouse_y: self.mouse_pos.1 as u32,
                        mouse_pressed: if self.mouse_pressed { 1 } else { 0 },
                        note_velocity: blended_velocity,
                        pitch_bend: blended_pitch_bend,
                        cc1: blended_cc1,
                        cc74: blended_cc74,
                        note_count: midi_state.note_count,
                        last_note: midi_state.last_note as u32,
                    };

                    drop(midi_state);

                    if let Err(e) = unsafe { gfx.draw(&push_constants) } {
                        eprintln!("Draw error: {}", e);
                        event_loop.exit();
                    }
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let frame_time = if self.config.graphics.vsync {
            Duration::from_millis(16)
        } else {
            Duration::from_millis(1)
        };

        event_loop.set_control_flow(ControlFlow::WaitUntil(Instant::now() + frame_time));

        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

fn main() -> Result<()> {
    env_logger::init();

    let args = Args::parse();

    let mut config = if Path::new(&args.config).exists() {
        match Config::load_from_file(&args.config) {
            Ok(config) => {
                println!("Loaded configuration from: {}", args.config);
                config
            }
            Err(e) => {
                eprintln!("Failed to load config file '{}': {}", args.config, e);
                println!("Using default configuration");
                Config::default()
            }
        }
    } else {
        println!("Config file '{}' not found, creating default config", args.config);
        let default_config = Config::default();
        if let Err(e) = default_config.save_to_file(&args.config) {
            eprintln!("Failed to save default config: {}", e);
        } else {
            println!("Default configuration saved to: {}", args.config);
        }
        default_config
    };

    config.merge_with_args(&args);

    println!("Starting Vulkan MIDI Pixel Shader");
    println!("Window: {}x{} - {}",
             config.window.width,
             config.window.height,
             if config.window.fullscreen { "Fullscreen" } else { "Windowed" });
    println!("Shader: {:?}", config.shader.preset);
    println!("MIDI: {}", if config.midi.enabled { "Enabled" } else { "Disabled" });
    println!("Audio: {}", if config.audio.enabled { "Enabled" } else { "Disabled" });

    let event_loop = EventLoop::new().expect("Failed to create event loop");

    let frame_time = if config.graphics.vsync {
        Duration::from_millis(16)
    } else {
        Duration::from_millis(1)
    };

    event_loop.set_control_flow(ControlFlow::WaitUntil(Instant::now() + frame_time));

    let mut app = App::new(config);
    event_loop.run_app(&mut app);
    Ok(())
}