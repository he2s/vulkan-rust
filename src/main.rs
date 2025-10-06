// ============================================================================
// MODULAR ARCHITECTURE
// Refactored into clean module structure for better performance and maintainability
// ============================================================================

// Module declarations
mod audio;
mod beat_detection;
mod config;
mod gfx;
mod graphics;
mod input;
mod processing;
mod state;
mod utils;

// Core application imports
use crate::config::config::{
    Args, Config,
    load_or_create_config, print_startup_info,
};
use crate::gfx::Gfx;
use crate::graphics::PushConstants;
use crate::input::InputManager;
use crate::utils::DeviceLister;

// External crate imports
use anyhow::Result;
use clap::Parser;
use ash::{Entry, vk, khr::{surface, swapchain}};
use std::{
    collections::VecDeque,
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
    last_fps_log: Instant,
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

            last_fps_log: now,
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
        if !self.frame_count_since_log.is_multiple_of(30) {
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