use anyhow::{anyhow, Result};
use ash::{vk, Entry};
use ash::khr::{surface, swapchain};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::{
    ffi::{c_char, CString},
    ptr,
    time::{Duration, Instant},
    sync::{Arc, Mutex},
    path::Path,
    fs,
    collections::{VecDeque, HashMap},
    cell::RefCell
};
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

// Audio imports
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use rustfft::{FftPlanner, num_complex::Complex32, Fft};

// Constants
const DEFAULT_WIDTH: u32 = 800;
const DEFAULT_HEIGHT: u32 = 600;
const DEFAULT_TITLE: &str = "Vulkan MIDI Pixel Shader";
const FRAME_TIME_VSYNC: Duration = Duration::from_millis(16);
const FRAME_TIME_NO_VSYNC: Duration = Duration::from_millis(1);
const MAX_NOTES: usize = 128;
const MAX_CONTROLLERS: usize = 128;
const AUDIO_RING_CAPACITY: usize = 4096;
const FFT_SAMPLE_SIZE: usize = 1024;
const DEFAULT_SAMPLE_RATE: u32 = 48000;
const MAX_FFT_SIZE: usize = 2048;
const MIN_FFT_SIZE: usize = 256;

// ==================== Configuration Types ====================

#[derive(Parser)]
#[command(name = "vulkan-midi-visualizer")]
#[command(about = "A MIDI-reactive Vulkan pixel shader visualizer")]
pub struct Args {
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

#[derive(Deserialize, Serialize, Default)]
pub struct Config {
    #[serde(default)]
    pub window: WindowConfig,
    #[serde(default)]
    pub midi: MidiConfig,
    #[serde(default)]
    pub graphics: GraphicsConfig,
    #[serde(default)]
    pub audio: AudioConfig,
    #[serde(default)]
    pub shader: ShaderConfig,
}

#[derive(Deserialize, Serialize)]
pub struct WindowConfig {
    #[serde(default = "default_width")]
    pub width: u32,
    #[serde(default = "default_height")]
    pub height: u32,
    #[serde(default = "default_title")]
    pub title: String,
    #[serde(default)]
    pub fullscreen: bool,
    #[serde(default = "default_true")]
    pub resizable: bool,
}

#[derive(Deserialize, Serialize)]
pub struct MidiConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub auto_connect: bool,
    #[serde(default)]
    pub port_name: Option<String>,
}

#[derive(Deserialize, Serialize)]
pub struct GraphicsConfig {
    #[serde(default = "default_true")]
    pub vsync: bool,
    #[serde(default = "default_validation_layers")]
    pub validation_layers: bool,
}

#[derive(Deserialize, Serialize)]
pub struct AudioConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub device_name: Option<String>,
    #[serde(default)]
    pub sample_rate: Option<u32>,
}

#[derive(Deserialize, Serialize, Clone)]
pub struct ShaderConfig {
    #[serde(default = "default_shader_preset")]
    pub preset: ShaderPreset,
    #[serde(default)]
    pub custom_vertex_path: Option<String>,
    #[serde(default)]
    pub custom_fragment_path: Option<String>,
    #[serde(default = "default_true")]
    pub allow_runtime_switching: bool,
}

#[derive(Deserialize, Serialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ShaderPreset {
    Torus,
    Terrain,
    Crystal,
    Custom,
}

// Default value functions
const fn default_width() -> u32 { DEFAULT_WIDTH }
const fn default_height() -> u32 { DEFAULT_HEIGHT }
fn default_title() -> String { DEFAULT_TITLE.to_string() }
const fn default_true() -> bool { true }
const fn default_shader_preset() -> ShaderPreset { ShaderPreset::Torus }
const fn default_validation_layers() -> bool { cfg!(debug_assertions) }

// Default implementations
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
            validation_layers: default_validation_layers(),
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
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = fs::read_to_string(path)?;
        Ok(toml::from_str(&content)?)
    }

    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let content = toml::to_string_pretty(self)?;
        fs::write(path, content)?;
        Ok(())
    }

    pub fn merge_with_args(&mut self, args: &Args) {
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
            self.shader.preset = Self::parse_shader_preset(shader);
        }
    }

    fn parse_shader_preset(shader_str: &str) -> ShaderPreset {
        match shader_str {
            "torus" => ShaderPreset::Torus,
            "terrain" => ShaderPreset::Terrain,
            "crystal" => ShaderPreset::Crystal,
            "custom" => ShaderPreset::Custom,
            _ => {
                eprintln!("Unknown shader preset '{}', using default", shader_str);
                ShaderPreset::Torus
            }
        }
    }
}

// ==================== Shader Management ====================

pub struct ShaderSources {
    pub vertex: String,
    pub fragment: String,
}

impl ShaderSources {
    pub fn load_preset(preset: &ShaderPreset) -> Result<Self> {
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

    pub fn load_from_files(vertex_path: &str, fragment_path: &str) -> Result<Self> {
        Ok(Self {
            vertex: fs::read_to_string(vertex_path)?,
            fragment: fs::read_to_string(fragment_path)?,
        })
    }

    pub fn load_from_config(config: &ShaderConfig) -> Result<Self> {
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

// ==================== Optimized State Management ====================

#[derive(Clone, Debug)]
pub struct MidiState {
    pub notes: [f32; MAX_NOTES],
    pub controllers: [f32; MAX_CONTROLLERS],
    pub pitch_bend: f32,
    pub last_note: u8,
    pub note_count: u32,
}

impl Default for MidiState {
    fn default() -> Self {
        Self {
            notes: [0.0; MAX_NOTES],
            controllers: [0.5; MAX_CONTROLLERS],
            pitch_bend: 0.0,
            last_note: 60,
            note_count: 0,
        }
    }
}

// PERFORMANCE OPTIMIZATION #2: Frame state snapshot to reduce mutex lock frequency
#[derive(Clone, Debug)]
pub struct FrameState {
    pub midi: MidiState,
    pub audio_levels: AudioLevels,
}

#[derive(Clone, Copy, Debug)]
pub struct AudioLevels {
    pub level_rms: f32,
    pub low: f32,
    pub mid: f32,
    pub high: f32,
}

impl Default for AudioLevels {
    fn default() -> Self {
        Self {
            level_rms: 0.0,
            low: 0.0,
            mid: 0.0,
            high: 0.0,
        }
    }
}

// OPTIMIZATION #1 & #3: Cached FFT planner and pre-allocated buffers
pub struct AudioState {
    ring: VecDeque<f32>,
    capacity: usize,
    last_sample_rate: u32,
    pub levels: AudioLevels,

    // OPTIMIZATION: Cached FFT components
    fft_planner: FftPlanner<f32>,
    fft_cache: HashMap<usize, Arc<dyn Fft<f32>>>,

    // OPTIMIZATION: Pre-allocated buffers to avoid runtime allocations
    processing_buffer: Vec<Complex32>,
    windowing_buffer: Vec<f32>,
    mono_conversion_buffer: Vec<f32>,
}

// Manual Debug implementation since FftPlanner doesn't implement Debug
impl std::fmt::Debug for AudioState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AudioState")
            .field("ring_len", &self.ring.len())
            .field("capacity", &self.capacity)
            .field("last_sample_rate", &self.last_sample_rate)
            .field("levels", &self.levels)
            .field("fft_cache_size", &self.fft_cache.len())
            .field("processing_buffer_capacity", &self.processing_buffer.capacity())
            .field("windowing_buffer_capacity", &self.windowing_buffer.capacity())
            .field("mono_conversion_buffer_capacity", &self.mono_conversion_buffer.capacity())
            .finish()
    }
}
impl AudioState {
    pub fn new() -> Self {
        Self {
            ring: VecDeque::with_capacity(AUDIO_RING_CAPACITY),
            capacity: AUDIO_RING_CAPACITY,
            last_sample_rate: DEFAULT_SAMPLE_RATE,
            levels: AudioLevels::default(),

            // Initialize cached components
            fft_planner: FftPlanner::<f32>::new(),
            fft_cache: HashMap::with_capacity(8), // Cache common FFT sizes

            // Pre-allocate buffers with generous capacity
            processing_buffer: Vec::with_capacity(MAX_FFT_SIZE),
            windowing_buffer: Vec::with_capacity(MAX_FFT_SIZE),
            mono_conversion_buffer: Vec::with_capacity(1024),
        }
    }

    pub fn push_samples(&mut self, samples: &[f32], sample_rate: u32) {
        self.last_sample_rate = sample_rate;
        for &sample in samples {
            if self.ring.len() == self.capacity {
                self.ring.pop_front();
            }
            self.ring.push_back(sample);
        }
    }

    // OPTIMIZATION: Combine analysis and level extraction to reduce lock time
    pub fn analyze_and_get_levels(&mut self) -> AudioLevels {
        if self.ring.is_empty() {
            self.levels = AudioLevels::default();
            return self.levels;
        }

        let take = FFT_SAMPLE_SIZE.min(self.ring.len());

        // Reuse windowing buffer to avoid allocation
        self.windowing_buffer.clear();
        self.windowing_buffer.extend(self.ring.iter().rev().take(take));
        self.windowing_buffer.reverse();

        // Calculate RMS
        let rms = self.calculate_rms(&self.windowing_buffer);

        // Perform frequency analysis with cached FFT
        let (low, mid, high) = self.perform_cached_fft_analysis();

        // Apply smoothing
        self.apply_smoothing(rms, low, mid, high);

        self.levels
    }

    fn calculate_rms(&self, buffer: &[f32]) -> f32 {
        let sum_squares: f32 = buffer.iter().map(|x| x * x).sum();
        (sum_squares / buffer.len() as f32).sqrt()
    }

    // OPTIMIZATION: Use cached FFT planner and reuse buffers
    fn perform_cached_fft_analysis(&mut self) -> (f32, f32, f32) {
        let fft_len = self.windowing_buffer.len().next_power_of_two().max(MIN_FFT_SIZE).min(MAX_FFT_SIZE);
        self.windowing_buffer.resize(fft_len, 0.0);

        // Apply windowing function in-place
        Self::apply_hann_window(&mut self.windowing_buffer);

        // Get or create cached FFT (MAJOR OPTIMIZATION)
        let fft = self.fft_cache.entry(fft_len).or_insert_with(|| {
            self.fft_planner.plan_fft_forward(fft_len)
        }).clone();

        // Reuse processing buffer
        self.processing_buffer.clear();
        self.processing_buffer.extend(
            self.windowing_buffer.iter().map(|&v| Complex32::new(v, 0.0))
        );

        fft.process(&mut self.processing_buffer);

        self.analyze_frequency_bands(fft_len)
    }

    fn apply_hann_window(buffer: &mut [f32]) {
        let len = buffer.len() as f32;
        for (i, sample) in buffer.iter_mut().enumerate() {
            let window = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / len).cos());
            *sample *= window;
        }
    }

    fn analyze_frequency_bands(&self, fft_len: usize) -> (f32, f32, f32) {
        let sample_rate = self.last_sample_rate as f32;
        let bin_hz = sample_rate / fft_len as f32;
        let mut low = 0.0;
        let mut mid = 0.0;
        let mut high = 0.0;

        for (i, complex) in self.processing_buffer.iter().enumerate().take(fft_len / 2) {
            let frequency = i as f32 * bin_hz;
            let magnitude = complex.norm();

            if frequency < 20.0 {
                continue;
            }

            match frequency {
                f if f <= 250.0 => low += magnitude,
                f if f <= 2000.0 => mid += magnitude,
                f if f <= 8000.0 => high += magnitude,
                _ => {}
            }
        }

        (low, mid, high)
    }

    fn apply_smoothing(&mut self, rms: f32, low: f32, mid: f32, high: f32) {
        const SMOOTHING_FACTOR: f32 = 0.7;
        let normalize = |x: f32| (x / 1000.0).min(1.0);

        self.levels.level_rms = SMOOTHING_FACTOR * self.levels.level_rms + (1.0 - SMOOTHING_FACTOR) * rms.min(1.0);
        self.levels.low = SMOOTHING_FACTOR * self.levels.low + (1.0 - SMOOTHING_FACTOR) * normalize(low);
        self.levels.mid = SMOOTHING_FACTOR * self.levels.mid + (1.0 - SMOOTHING_FACTOR) * normalize(mid);
        self.levels.high = SMOOTHING_FACTOR * self.levels.high + (1.0 - SMOOTHING_FACTOR) * normalize(high);
    }

    // OPTIMIZATION: Pre-allocated buffer for mono conversion
    pub fn convert_to_mono_optimized(&mut self, data: &[f32], channels: usize) -> &[f32] {
        self.mono_conversion_buffer.clear();
        self.mono_conversion_buffer.reserve(data.len() / channels + 1);

        for frame in data.chunks_exact(channels) {
            let sample = frame.iter().sum::<f32>() / channels as f32;
            self.mono_conversion_buffer.push(sample);
        }

        &self.mono_conversion_buffer
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

// ==================== Vulkan Graphics ====================

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
    images: Vec<vk::Image>,
    views: Vec<vk::ImageView>,
}

pub struct VulkanPipeline {
    render_pass: vk::RenderPass,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    framebuffers: Vec<vk::Framebuffer>,
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

// OPTIMIZATION: Cache dynamic state to avoid redundant updates
#[derive(Default)]
pub struct VulkanState {
    current_extent: Option<vk::Extent2D>,
}

pub struct Gfx {
    context: VulkanContext,
    swapchain: VulkanSwapchain,
    pipeline: VulkanPipeline,
    commands: VulkanCommands,
    sync: VulkanSync,
    state: VulkanState, // OPTIMIZATION: Track state to avoid redundant operations
}

impl Gfx {
    pub unsafe fn new(window: &Window, shader_config: &ShaderConfig) -> Result<Self> {
        let context = VulkanContext::new(window)?;
        let swapchain = VulkanSwapchain::new(&context, window)?;
        let pipeline = VulkanPipeline::new(&context, &swapchain, shader_config)?;
        let commands = VulkanCommands::new(&context)?;
        let sync = VulkanSync::new(&context)?;

        Ok(Self {
            context,
            swapchain,
            pipeline,
            commands,
            sync,
            state: VulkanState::default(),
        })
    }

    pub unsafe fn recreate_swapchain(&mut self, window: &Window) -> Result<()> {
        self.context.device.device_wait_idle()?;

        // Clean up old swapchain
        self.pipeline.cleanup_framebuffers(&self.context.device);
        self.swapchain.cleanup(&self.context.device);

        // Create new swapchain
        self.swapchain = VulkanSwapchain::new(&self.context, window)?;
        self.pipeline.recreate_framebuffers(&self.context.device, &self.swapchain)?;

        // Reset cached state
        self.state.current_extent = None;

        println!("Swapchain recreated: {}x{}", self.swapchain.extent.width, self.swapchain.extent.height);
        Ok(())
    }

    pub unsafe fn recreate_pipeline(&mut self, shader_config: &ShaderConfig) -> Result<()> {
        self.context.device.device_wait_idle()?;

        // Clean up old pipeline
        self.pipeline.cleanup_pipeline(&self.context.device);

        // Create new pipeline
        self.pipeline.create_pipeline(&self.context, &self.swapchain, shader_config)?;

        println!("Pipeline recreated successfully");
        Ok(())
    }

    pub unsafe fn draw(&mut self, push_constants: &PushConstants) -> Result<bool> {
        // Wait for previous frame
        self.context.device.wait_for_fences(&[self.sync.in_flight], true, u64::MAX)?;
        self.context.device.reset_fences(&[self.sync.in_flight])?;

        // Acquire next image
        let (image_index, needs_recreation) = self.acquire_next_image()?;
        if needs_recreation {
            return Ok(true);
        }

        // Record command buffer
        let cmd_buffer = self.commands.get_current_buffer();
        self.record_command_buffer(cmd_buffer, image_index, push_constants)?;

        // Submit command buffer
        self.submit_commands(cmd_buffer)?;

        // Present
        self.present_image(image_index)
    }

    unsafe fn acquire_next_image(&self) -> Result<(u32, bool)> {
        match self.swapchain.loader.acquire_next_image(
            self.swapchain.swapchain,
            u64::MAX,
            self.sync.image_available,
            vk::Fence::null(),
        ) {
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
    ) -> Result<()> {
        self.context.device.begin_command_buffer(cmd_buffer, &vk::CommandBufferBeginInfo::default())?;

        let clear_values = [vk::ClearValue {
            color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] }
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

        self.context.device.cmd_begin_render_pass(cmd_buffer, &render_pass_begin, vk::SubpassContents::INLINE);
        self.context.device.cmd_bind_pipeline(cmd_buffer, vk::PipelineBindPoint::GRAPHICS, self.pipeline.pipeline);

        // OPTIMIZATION: Only update dynamic state when changed
        if Some(self.swapchain.extent) != self.state.current_extent {
            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: self.swapchain.extent.width as f32,
                height: self.swapchain.extent.height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            };
            self.context.device.cmd_set_viewport(cmd_buffer, 0, &[viewport]);
            self.context.device.cmd_set_scissor(cmd_buffer, 0, &[render_area]);
            self.state.current_extent = Some(self.swapchain.extent);
        }

        // Push constants
        self.context.device.cmd_push_constants(
            cmd_buffer,
            self.pipeline.pipeline_layout,
            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            0,
            std::slice::from_raw_parts(
                push_constants as *const PushConstants as *const u8,
                std::mem::size_of::<PushConstants>()
            )
        );

        self.context.device.cmd_draw(cmd_buffer, 3, 1, 0, 0);
        self.context.device.cmd_end_render_pass(cmd_buffer);
        self.context.device.end_command_buffer(cmd_buffer)?;

        Ok(())
    }

    unsafe fn submit_commands(&self, cmd_buffer: vk::CommandBuffer) -> Result<()> {
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

        self.context.device.queue_submit(self.context.queue, &[submit_info], self.sync.in_flight)?;
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

        match self.swapchain.loader.queue_present(self.context.queue, &present_info) {
            Ok(_) => Ok(false),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) | Err(vk::Result::SUBOPTIMAL_KHR) => Ok(true),
            Err(e) => Err(anyhow!("Failed to present: {:?}", e)),
        }
    }
}

// Implement VulkanContext
impl VulkanContext {
    unsafe fn new(window: &Window) -> Result<Self> {
        let entry = Entry::linked();
        let display_handle = window.display_handle()?.as_raw();
        let window_handle = window.window_handle()?.as_raw();
        let required_extensions = ash_window::enumerate_required_extensions(display_handle)?.to_vec();

        let instance = Self::create_instance(&entry, &required_extensions)?;
        let surface = ash_window::create_surface(&entry, &instance, display_handle, window_handle, None)?;
        let surface_loader = surface::Instance::new(&entry, &instance);
        let (physical_device, queue_family_index) = Self::select_physical_device(&instance, &surface_loader, surface)?;
        let (device, queue) = Self::create_logical_device(&instance, physical_device, queue_family_index)?;

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

    unsafe fn create_instance(entry: &Entry, required_extensions: &[*const c_char]) -> Result<ash::Instance> {
        let app_name = CString::new("vulkan-pixel-shader")?;

        let layer_names: Vec<CString> = if cfg!(debug_assertions) {
            vec![CString::new("VK_LAYER_KHRONOS_validation")?]
        } else {
            vec![]
        };

        let layer_name_pointers: Vec<*const c_char> = layer_names
            .iter()
            .map(|name| name.as_ptr())
            .collect();

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

        Ok(entry.create_instance(&create_info, None)?)
    }

    unsafe fn select_physical_device(
        instance: &ash::Instance,
        surface_loader: &surface::Instance,
        surface: vk::SurfaceKHR,
    ) -> Result<(vk::PhysicalDevice, u32)> {
        let physical_devices = instance.enumerate_physical_devices()?;

        for device in physical_devices {
            let queue_families = instance.get_physical_device_queue_family_properties(device);

            for (index, queue_family) in queue_families.iter().enumerate() {
                let index = index as u32;

                if queue_family.queue_flags.contains(vk::QueueFlags::GRAPHICS) &&
                    surface_loader.get_physical_device_surface_support(device, index, surface)? {
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

        let device = instance.create_device(physical_device, &device_create_info, None)?;
        let queue = device.get_device_queue(queue_family_index, 0);

        Ok((device, queue))
    }
}

// Implement VulkanSwapchain
impl VulkanSwapchain {
    unsafe fn new(context: &VulkanContext, window: &Window) -> Result<Self> {
        let loader = swapchain::Device::new(&context.instance, &context.device);
        let surface_caps = context.surface_loader.get_physical_device_surface_capabilities(
            context.physical_device,
            context.surface,
        )?;
        let formats = context.surface_loader.get_physical_device_surface_formats(
            context.physical_device,
            context.surface,
        )?;

        let format = Self::choose_surface_format(&formats);
        let extent = Self::choose_extent(&surface_caps, window);
        let image_count = Self::choose_image_count(&surface_caps);

        let create_info = vk::SwapchainCreateInfoKHR {
            surface: context.surface,
            min_image_count: image_count,
            image_format: format,
            image_color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
            image_extent: extent,
            image_array_layers: 1,
            image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
            image_sharing_mode: vk::SharingMode::EXCLUSIVE,
            pre_transform: surface_caps.current_transform,
            composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
            present_mode: vk::PresentModeKHR::FIFO,
            clipped: vk::TRUE,
            ..Default::default()
        };

        let swapchain = loader.create_swapchain(&create_info, None)?;
        let images = loader.get_swapchain_images(swapchain)?;
        let views = Self::create_image_views(&context.device, &images, format)?;

        Ok(Self {
            loader,
            swapchain,
            extent,
            format,
            images,
            views,
        })
    }

    fn choose_surface_format(formats: &[vk::SurfaceFormatKHR]) -> vk::Format {
        formats
            .iter()
            .find(|f| f.format == vk::Format::B8G8R8A8_SRGB && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR)
            .map(|f| f.format)
            .unwrap_or(formats[0].format)
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
                device.create_image_view(&create_info, None)
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(Into::into)
    }

    unsafe fn cleanup(&mut self, device: &ash::Device) {
        for &view in &self.views {
            device.destroy_image_view(view, None);
        }
        self.loader.destroy_swapchain(self.swapchain, None);
    }
}

// Implement VulkanPipeline
impl VulkanPipeline {
    unsafe fn new(
        context: &VulkanContext,
        swapchain: &VulkanSwapchain,
        shader_config: &ShaderConfig,
    ) -> Result<Self> {
        let render_pass = Self::create_render_pass(&context.device, swapchain.format)?;
        let pipeline_layout = Self::create_pipeline_layout(&context.device)?;
        let pipeline = Self::create_graphics_pipeline(&context.device, render_pass, pipeline_layout, swapchain, shader_config)?;
        let framebuffers = Self::create_framebuffers(&context.device, render_pass, swapchain)?;

        Ok(Self {
            render_pass,
            pipeline_layout,
            pipeline,
            framebuffers,
        })
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
            ..Default::default()
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

        Ok(device.create_render_pass(&create_info, None)?)
    }

    unsafe fn create_pipeline_layout(device: &ash::Device) -> Result<vk::PipelineLayout> {
        let push_constant_range = vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            offset: 0,
            size: std::mem::size_of::<PushConstants>() as u32,
        };

        let create_info = vk::PipelineLayoutCreateInfo {
            push_constant_range_count: 1,
            p_push_constant_ranges: &push_constant_range,
            ..Default::default()
        };

        Ok(device.create_pipeline_layout(&create_info, None)?)
    }

    unsafe fn create_graphics_pipeline(
        device: &ash::Device,
        render_pass: vk::RenderPass,
        pipeline_layout: vk::PipelineLayout,
        swapchain: &VulkanSwapchain,
        shader_config: &ShaderConfig,
    ) -> Result<vk::Pipeline> {
        println!("Loading shader preset: {:?}", shader_config.preset);
        let shader_sources = ShaderSources::load_from_config(shader_config)?;

        println!("Compiling shaders...");
        let vert_code = Self::compile_shader(&shader_sources.vertex, shaderc::ShaderKind::Vertex)?;
        let frag_code = Self::compile_shader(&shader_sources.fragment, shaderc::ShaderKind::Fragment)?;

        let vert_module = Self::create_shader_module(device, &vert_code)?;
        let frag_module = Self::create_shader_module(device, &frag_code)?;

        let entry_name = CString::new("main")?;
        let shader_stages = [
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

        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default();
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo {
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            ..Default::default()
        };

        let viewport_state = vk::PipelineViewportStateCreateInfo {
            viewport_count: 1,
            scissor_count: 1,
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

        let pipelines = device
            .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
            .map_err(|e| e.1)?;

        // Clean up shader modules
        device.destroy_shader_module(vert_module, None);
        device.destroy_shader_module(frag_module, None);

        Ok(pipelines[0])
    }

    unsafe fn compile_shader(source: &str, kind: shaderc::ShaderKind) -> Result<Vec<u32>> {
        let compiler = shaderc::Compiler::new()
            .ok_or_else(|| anyhow!("Failed to create shader compiler"))?;

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
        Ok(device.create_shader_module(&create_info, None)?)
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
                device.create_framebuffer(&create_info, None)
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(Into::into)
    }

    unsafe fn create_pipeline(&mut self,
                              context: &VulkanContext,
                              swapchain: &VulkanSwapchain,
                              shader_config: &ShaderConfig
    ) -> Result<()> {
        self.pipeline = Self::create_graphics_pipeline(
            &context.device,
            self.render_pass,
            self.pipeline_layout,
            swapchain,
            shader_config
        )?;
        Ok(())
    }

    unsafe fn recreate_framebuffers(
        &mut self,
        device: &ash::Device,
        swapchain: &VulkanSwapchain,
    ) -> Result<()> {
        self.framebuffers = Self::create_framebuffers(device, self.render_pass, swapchain)?;
        Ok(())
    }

    unsafe fn cleanup_framebuffers(&mut self, device: &ash::Device) {
        for &framebuffer in &self.framebuffers {
            device.destroy_framebuffer(framebuffer, None);
        }
        self.framebuffers.clear();
    }

    unsafe fn cleanup_pipeline(&mut self, device: &ash::Device) {
        device.destroy_pipeline(self.pipeline, None);
    }
}

// Implement VulkanCommands
impl VulkanCommands {
    unsafe fn new(context: &VulkanContext) -> Result<Self> {
        let pool_info = vk::CommandPoolCreateInfo {
            flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            queue_family_index: context.queue_family_index,
            ..Default::default()
        };

        let pool = context.device.create_command_pool(&pool_info, None)?;

        let alloc_info = vk::CommandBufferAllocateInfo {
            command_pool: pool,
            level: vk::CommandBufferLevel::PRIMARY,
            command_buffer_count: 2,
            ..Default::default()
        };

        let buffers = context.device.allocate_command_buffers(&alloc_info)?;

        Ok(Self {
            pool,
            buffers,
            frame_index: RefCell::new(0),
        })
    }

    fn get_current_buffer(&self) -> vk::CommandBuffer {
        let mut frame_index = self.frame_index.borrow_mut();
        let cmd_buffer = self.buffers[*frame_index % 2];
        *frame_index += 1;
        cmd_buffer
    }
}

// Implement VulkanSync
impl VulkanSync {
    unsafe fn new(context: &VulkanContext) -> Result<Self> {
        let semaphore_info = vk::SemaphoreCreateInfo::default();
        let fence_info = vk::FenceCreateInfo {
            flags: vk::FenceCreateFlags::SIGNALED,
            ..Default::default()
        };

        Ok(Self {
            image_available: context.device.create_semaphore(&semaphore_info, None)?,
            render_finished: context.device.create_semaphore(&semaphore_info, None)?,
            in_flight: context.device.create_fence(&fence_info, None)?,
        })
    }
}

// Cleanup implementations
impl Drop for Gfx {
    fn drop(&mut self) {
        unsafe {
            let _ = self.context.device.device_wait_idle();

            // Clean up sync objects
            self.context.device.destroy_fence(self.sync.in_flight, None);
            self.context.device.destroy_semaphore(self.sync.render_finished, None);
            self.context.device.destroy_semaphore(self.sync.image_available, None);

            // Clean up commands
            self.context.device.free_command_buffers(self.commands.pool, &self.commands.buffers);
            self.context.device.destroy_command_pool(self.commands.pool, None);

            // Clean up pipeline
            self.pipeline.cleanup_framebuffers(&self.context.device);
            self.pipeline.cleanup_pipeline(&self.context.device);
            self.context.device.destroy_pipeline_layout(self.pipeline.pipeline_layout, None);
            self.context.device.destroy_render_pass(self.pipeline.render_pass, None);

            // Clean up swapchain
            self.swapchain.cleanup(&self.context.device);

            // Clean up context
            self.context.surface_loader.destroy_surface(self.context.surface, None);
            self.context.device.destroy_device(None);
            self.context.instance.destroy_instance(None);
        }
    }
}

// ==================== Optimized Input Handling ====================

pub struct InputManager {
    midi_state: Arc<Mutex<MidiState>>,
    _midi_connection: Option<midir::MidiInputConnection<()>>,
    audio_state: Arc<Mutex<AudioState>>,
    _audio_stream: Option<cpal::Stream>,
}

impl InputManager {
    pub fn new() -> Self {
        Self {
            midi_state: Arc::new(Mutex::new(MidiState::default())),
            _midi_connection: None,
            audio_state: Arc::new(Mutex::new(AudioState::new())),
            _audio_stream: None,
        }
    }

    // OPTIMIZATION #2: Single function to get all frame state with minimal lock time
    pub fn get_frame_state(&self) -> FrameState {
        // Get MIDI state (quick clone)
        let midi = self.midi_state.lock().unwrap().clone();

        // Get audio levels and perform analysis in one lock acquisition
        let audio_levels = {
            let mut audio_state = self.audio_state.lock().unwrap();
            audio_state.analyze_and_get_levels()
        };

        FrameState { midi, audio_levels }
    }

    pub fn setup_midi(&mut self, config: &MidiConfig) {
        if !config.enabled {
            println!("MIDI disabled in configuration");
            return;
        }

        match self.try_setup_midi(config) {
            Ok(connection) => {
                self._midi_connection = Some(connection);
                println!("MIDI input connected successfully!");
            }
            Err(e) => {
                eprintln!("MIDI setup failed: {}. Continuing without MIDI input.", e);
            }
        }
    }

    pub fn setup_audio(&mut self, config: &AudioConfig) {
        if !config.enabled {
            println!("Audio input disabled in configuration");
            return;
        }

        match self.try_setup_audio(config) {
            Ok(stream) => {
                if let Err(e) = stream.play() {
                    eprintln!("Failed to start audio stream: {}", e);
                    return;
                }
                self._audio_stream = Some(stream);
                println!("Audio input connected successfully!");
            }
            Err(e) => {
                eprintln!("Audio setup failed: {}", e);
            }
        }
    }

    fn try_setup_midi(&self, config: &MidiConfig) -> Result<midir::MidiInputConnection<()>, Box<dyn std::error::Error>> {
        let mut midi_in = MidiInput::new("Vulkan MIDI Visualizer")?;
        midi_in.ignore(Ignore::None);

        let ports = midi_in.ports();
        if ports.is_empty() {
            return Err("No MIDI input ports available".into());
        }

        let selected_port = self.select_midi_port(&midi_in, &ports, config)?;
        let port_name = midi_in.port_name(selected_port)?;
        println!("Connecting to MIDI port: {}", port_name);

        let midi_state = Arc::clone(&self.midi_state);
        let connection = midi_in.connect(selected_port, "vulkan-visualizer", move |_timestamp, message, _| {
            Self::handle_midi_message(&midi_state, message);
        }, ())?;

        Ok(connection)
    }

    fn select_midi_port<'a>(
        &self,
        midi_in: &MidiInput,
        ports: &'a [midir::MidiInputPort],
        config: &MidiConfig,
    ) -> Result<&'a midir::MidiInputPort, Box<dyn std::error::Error>> {
        if let Some(ref target_name) = config.port_name {
            ports
                .iter()
                .find(|port| {
                    midi_in.port_name(port)
                        .map_or(false, |name| name.contains(target_name))
                })
                .or_else(|| ports.first())
                .ok_or_else(|| "No suitable MIDI port found".into())
        } else {
            ports.first().ok_or_else(|| "No MIDI ports available".into())
        }
    }

    fn try_setup_audio(&self, config: &AudioConfig) -> Result<cpal::Stream, Box<dyn std::error::Error>> {
        let host = cpal::default_host();
        let device = self.select_audio_device(&host, config)?;
        let audio_config = self.configure_audio_device(&device, config)?;

        println!("Using audio device: {}", device.name().unwrap_or_else(|_| "Unknown".into()));
        println!("Audio config: {:?}Hz, {:?}ch", audio_config.sample_rate.0, audio_config.channels);

        let audio_state = Arc::clone(&self.audio_state);
        let channels = audio_config.channels as usize;

        let stream = device.build_input_stream(
            &audio_config,
            move |data: &[f32], _| {
                // OPTIMIZATION: Minimize lock time by doing conversion outside the lock
                if let Ok(mut state) = audio_state.try_lock() {
                    // Convert to mono and clone to avoid borrow conflicts
                    let mono_samples = state.convert_to_mono_optimized(data, channels).to_vec();
                    state.push_samples(&mono_samples, audio_config.sample_rate.0);
                }
            },
            move |err| {
                eprintln!("Audio input error: {}", err);
            },
            None,
        )?;

        Ok(stream)
    }

    fn select_audio_device(
        &self,
        host: &cpal::Host,
        config: &AudioConfig,
    ) -> Result<cpal::Device, Box<dyn std::error::Error>> {
        let device = if let Some(ref device_name) = config.device_name {
            host.input_devices()?
                .find(|device| {
                    device.name()
                        .map_or(false, |name| name.contains(device_name))
                })
                .or_else(|| host.default_input_device())
        } else {
            host.default_input_device()
        };

        device.ok_or_else(|| "No input audio device found".into())
    }

    fn configure_audio_device(
        &self,
        device: &cpal::Device,
        config: &AudioConfig,
    ) -> Result<cpal::StreamConfig, Box<dyn std::error::Error>> {
        let supported_configs = device.supported_input_configs()?.collect::<Vec<_>>();
        if supported_configs.is_empty() {
            return Err("No supported audio input configs".into());
        }

        let mut stream_config = supported_configs[0].with_max_sample_rate().config();

        // Try to find a better configuration
        for supported_config in &supported_configs {
            if supported_config.sample_format() == cpal::SampleFormat::F32 {
                if let Some(desired_rate) = config.sample_rate {
                    if supported_config.min_sample_rate().0 <= desired_rate &&
                        desired_rate <= supported_config.max_sample_rate().0 {
                        stream_config = supported_config
                            .clone()
                            .with_sample_rate(cpal::SampleRate(desired_rate))
                            .config();
                        break;
                    }
                } else {
                    stream_config = supported_config.with_max_sample_rate().config();
                    break;
                }
            }
        }

        Ok(stream_config)
    }

    fn handle_midi_message(midi_state: &Arc<Mutex<MidiState>>, message: &[u8]) {
        if message.is_empty() {
            return;
        }

        let Ok(mut state) = midi_state.lock() else { return };
        let status = message[0];

        match status & 0xF0 {
            0x80 => Self::handle_note_off(&mut state, message),
            0x90 => Self::handle_note_on(&mut state, message),
            0xB0 => Self::handle_control_change(&mut state, message),
            0xE0 => Self::handle_pitch_bend(&mut state, message),
            _ => {}
        }
    }

    fn handle_note_off(state: &mut MidiState, message: &[u8]) {
        if message.len() >= 3 {
            let note = message[1] as usize;
            if note < MAX_NOTES && state.notes[note] > 0.0 {
                state.note_count = state.note_count.saturating_sub(1);
                state.notes[note] = 0.0;
                println!("Note Off: {} (Count: {})", note, state.note_count);
            }
        }
    }

    fn handle_note_on(state: &mut MidiState, message: &[u8]) {
        if message.len() >= 3 {
            let note = message[1] as usize;
            let velocity = message[2];

            if note < MAX_NOTES {
                if velocity == 0 {
                    // Note off via velocity 0
                    if state.notes[note] > 0.0 {
                        state.note_count = state.note_count.saturating_sub(1);
                    }
                    state.notes[note] = 0.0;
                    println!("Note Off: {} (Count: {})", note, state.note_count);
                } else {
                    // Note on
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

    fn handle_control_change(state: &mut MidiState, message: &[u8]) {
        if message.len() >= 3 {
            let controller = message[1] as usize;
            let value = message[2];

            if controller < MAX_CONTROLLERS {
                state.controllers[controller] = value as f32 / 127.0;
                println!("CC{}: {}", controller, value);
            }
        }
    }

    fn handle_pitch_bend(state: &mut MidiState, message: &[u8]) {
        if message.len() >= 3 {
            let bend_value = (message[2] as u16) << 7 | (message[1] as u16);
            state.pitch_bend = (bend_value as f32 / 8192.0) - 1.0;
            println!("Pitch Bend: {:.3}", state.pitch_bend);
        }
    }
}

// ==================== Main Application ====================

pub struct App {
    window: Option<Window>,
    gfx: Option<Gfx>,
    start_time: Option<Instant>,
    mouse_pos: (f64, f64),
    mouse_pressed: bool,
    input_manager: InputManager,
    config: Config,
    is_fullscreen: bool,
    current_shader_index: usize,
    shader_presets: Vec<ShaderPreset>,
}

impl App {
    pub fn new(config: Config) -> Self {
        let shader_presets = vec![
            ShaderPreset::Torus,
            ShaderPreset::Terrain,
            ShaderPreset::Crystal,
        ];

        let current_shader_index = shader_presets
            .iter()
            .position(|p| *p == config.shader.preset)
            .unwrap_or(0);

        Self {
            window: None,
            gfx: None,
            start_time: None,
            mouse_pos: (0.0, 0.0),
            mouse_pressed: false,
            input_manager: InputManager::new(),
            is_fullscreen: config.window.fullscreen,
            current_shader_index,
            shader_presets,
            config,
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
        self.config.shader.preset = new_preset.clone();

        if let Some(gfx) = &mut self.gfx {
            println!("Switching to shader: {:?}", new_preset);
            if let Err(e) = unsafe { gfx.recreate_pipeline(&self.config.shader) } {
                eprintln!("Failed to switch shader: {}", e);
            }
        }
    }

    // OPTIMIZATION: Use frame state snapshot instead of multiple locks
    fn get_push_constants(&self, elapsed: f32) -> PushConstants {
        let frame_state = self.input_manager.get_frame_state();

        let note_velocity = if frame_state.midi.note_count > 0 {
            frame_state.midi.notes[frame_state.midi.last_note as usize]
        } else {
            0.0
        };

        // Blend MIDI and audio data
        let blended_velocity = note_velocity.max(frame_state.audio_levels.level_rms);
        let blended_pitch_bend = frame_state.midi.pitch_bend.max(frame_state.audio_levels.low * 2.0 - 1.0);
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
        }
    }

    fn print_controls(&self) {
        println!("Controls:");
        println!("  F11 - Toggle fullscreen");
        println!("  ESC - Exit (or exit fullscreen)");
        if self.config.shader.allow_runtime_switching {
            println!("  TAB - Cycle shaders");
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

        let window = event_loop.create_window(attributes).expect("Failed to create window");
        let gfx = unsafe {
            Gfx::new(&window, &self.config.shader).expect("Failed to initialize Vulkan")
        };

        self.window = Some(window);
        self.gfx = Some(gfx);
        self.start_time = Some(Instant::now());

        self.input_manager.setup_midi(&self.config.midi);
        self.input_manager.setup_audio(&self.config.audio);

        self.print_controls();
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _: winit::window::WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::Resized(new_size) => {
                if new_size.width > 0 && new_size.height > 0 {
                    println!("Window resized to {}x{}", new_size.width, new_size.height);
                    if let (Some(gfx), Some(window)) = (&mut self.gfx, &self.window) {
                        if let Err(e) = unsafe { gfx.recreate_swapchain(window) } {
                            eprintln!("Failed to recreate swapchain: {}", e);
                            event_loop.exit();
                        }
                    }
                }
            }

            WindowEvent::KeyboardInput {
                event: KeyEvent {
                    physical_key,
                    state: ElementState::Pressed,
                    ..
                },
                ..
            } => {
                match physical_key {
                    PhysicalKey::Code(KeyCode::F11) => self.toggle_fullscreen(),
                    PhysicalKey::Code(KeyCode::Escape) => {
                        if self.is_fullscreen {
                            self.toggle_fullscreen();
                        } else {
                            event_loop.exit();
                        }
                    }
                    PhysicalKey::Code(KeyCode::Tab) => self.cycle_shader(),
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
                if let (Some(start_time), Some(window)) = (&self.start_time, &self.window) {
                    let elapsed = start_time.elapsed().as_secs_f32();
                    let push_constants = self.get_push_constants(elapsed);

                    if let Some(gfx) = &mut self.gfx {
                        match unsafe { gfx.draw(&push_constants) } {
                            Ok(true) => {
                                // Swapchain needs recreation
                                if let Err(e) = unsafe { gfx.recreate_swapchain(window) } {
                                    eprintln!("Failed to recreate swapchain: {}", e);
                                    event_loop.exit();
                                }
                            }
                            Ok(false) => {
                                // Draw succeeded normally
                            }
                            Err(e) => {
                                eprintln!("Draw error: {}", e);
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

// ==================== Main Entry Point ====================

fn main() -> Result<()> {
    env_logger::init();

    let args = Args::parse();
    let mut config = load_or_create_config(&args.config)?;
    config.merge_with_args(&args);

    print_startup_info(&config);

    let event_loop = EventLoop::new().expect("Failed to create event loop");
    let frame_time = if config.graphics.vsync { FRAME_TIME_VSYNC } else { FRAME_TIME_NO_VSYNC };
    event_loop.set_control_flow(ControlFlow::WaitUntil(Instant::now() + frame_time));

    let mut app = App::new(config);
    event_loop.run_app(&mut app);

    Ok(())
}

fn load_or_create_config(config_path: &str) -> Result<Config> {
    if Path::new(config_path).exists() {
        match Config::load_from_file(config_path) {
            Ok(config) => {
                println!("Loaded configuration from: {}", config_path);
                Ok(config)
            }
            Err(e) => {
                eprintln!("Failed to load config file '{}': {}", config_path, e);
                println!("Using default configuration");
                Ok(Config::default())
            }
        }
    } else {
        println!("Config file '{}' not found, creating default config", config_path);
        let default_config = Config::default();
        if let Err(e) = default_config.save_to_file(config_path) {
            eprintln!("Failed to save default config: {}", e);
        } else {
            println!("Default configuration saved to: {}", config_path);
        }
        Ok(default_config)
    }
}

fn print_startup_info(config: &Config) {
    println!("Starting Vulkan MIDI Pixel Shader");
    println!("Window: {}x{} - {}",
             config.window.width,
             config.window.height,
             if config.window.fullscreen { "Fullscreen" } else { "Windowed" }
    );
    println!("Shader: {:?}", config.shader.preset);
    println!("MIDI: {}", if config.midi.enabled { "Enabled" } else { "Disabled" });
    println!("Audio: {}", if config.audio.enabled { "Enabled" } else { "Disabled" });
}