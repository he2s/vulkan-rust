use vulkan_rust::config::config::{
    Args, Config,
    load_or_create_config, print_startup_info,
};
use vulkan_rust::gfx::Gfx;
use vulkan_rust::graphics::PushConstants;
use vulkan_rust::graphics::egui_overlay::EguiOverlay;
use vulkan_rust::input::InputManager;
use vulkan_rust::utils::DeviceLister;

// External crate imports
use anyhow::Result;
use clap::Parser;
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

const FRAME_TIME_VSYNC: Duration = Duration::from_millis(16);
const FRAME_TIME_NO_VSYNC: Duration = Duration::from_millis(1);


pub struct App {
    window: Option<Window>,
    gfx: Option<Gfx>,
    overlay: Option<EguiOverlay>,
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

    last_title_state: Option<(u32, bool)>,
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

            last_title_state: None,
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
            // Wait for all Vulkan operations to complete before changing window state
            if let Some(gfx) = &self.gfx
                && let Err(e) = unsafe { gfx.context.device.device_wait_idle() }
            {
                eprintln!("Warning: Failed to wait for device idle before fullscreen toggle: {e}");
            }

            self.is_fullscreen = !self.is_fullscreen;

            let fullscreen = if self.is_fullscreen {
                Some(Fullscreen::Borderless(window.current_monitor()))
            } else {
                None
            };

            window.set_fullscreen(fullscreen);

            // Force swapchain recreation after fullscreen toggle to handle surface changes
            if let Some(gfx) = &mut self.gfx {
                // Small delay to allow window manager to complete the transition
                std::thread::sleep(std::time::Duration::from_millis(50));

                if let Err(e) = unsafe { gfx.recreate_swapchain(window) } {
                    eprintln!("Failed to recreate swapchain after fullscreen toggle: {e}");
                } else {
                    println!(
                        "Toggled fullscreen: {} (swapchain recreated)",
                        if self.is_fullscreen { "ON" } else { "OFF" }
                    );
                }
            } else {
                println!(
                    "Toggled fullscreen: {}",
                    if self.is_fullscreen { "ON" } else { "OFF" }
                );
            }
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
        PushConstants::from_frame(
            &frame_state,
            elapsed,
            (self.mouse_pos.0 as u32, self.mouse_pos.1 as u32, self.mouse_pressed),
            (w, h),
        )
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
        if !self.frame_count_since_log.is_multiple_of(30) {
            return;
        }

        let Some(window) = &self.window else { return };

        let bpm = self.input_manager.get_frame_state().beat.bpm;
        let bucket = (bpm * 10.0).round() as u32;
        let state = (bucket, self.manual_tempo_mode);
        if self.last_title_state == Some(state) {
            return;
        }
        self.last_title_state = Some(state);

        let mode = if self.manual_tempo_mode { "Manual" } else { "Auto" };
        let title = format!("{} - {:.1} BPM ({})", self.config.window.title, bpm, mode);
        window.set_title(&title);
    }
}

impl Drop for App {
    fn drop(&mut self) {
        // egui depends on gfx; tear down first
        if let (Some(overlay), Some(gfx)) = (&mut self.overlay, &self.gfx) {
            unsafe { overlay.destroy(&gfx.context.device); }
        }
        // Drop gfx (and its Vulkan surface) while the window is still alive.
        // Otherwise field-drop order (window → gfx) would destroy the surface
        // after winit has invalidated the HWND, causing STATUS_ACCESS_VIOLATION.
        drop(self.gfx.take());
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

        let mut overlay = EguiOverlay::new(&window)
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

                let Some(start_time) = self.start_time else { return };
                let elapsed = start_time.elapsed().as_secs_f32();
                let (width, height) = self.cached_window_size;
                let push_constants = self.get_push_constants(elapsed, width.max(1), height.max(1));

                let egui_primitives = match (&self.window, &mut self.overlay) {
                    (Some(window), Some(overlay)) => {
                        let preset = &self.shader_presets[self.current_shader_index];
                        let (_, primitives) = overlay.run_ui(window, preset);
                        Some(primitives)
                    }
                    _ => None,
                };

                let Some(gfx) = &mut self.gfx else { return };
                match unsafe { gfx.draw(&push_constants, egui_primitives.as_ref(), self.overlay.as_mut()) } {
                    Ok(true) => {
                        if let Some(window) = &self.window
                            && let Err(e) = unsafe { gfx.recreate_swapchain(window) }
                        {
                            eprintln!("Failed to recreate swapchain: {e}");
                            event_loop.exit();
                        }
                    }
                    Ok(false) => {}
                    Err(e) => {
                        eprintln!("Draw error: {e}");
                        event_loop.exit();
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