// src/app.rs

use anyhow::Result;
use std::time::Instant;
use winit::{
    application::ApplicationHandler,
    event::{WindowEvent, ElementState, MouseButton, KeyEvent},
    event_loop::ActiveEventLoop,
    window::{Window, Fullscreen},
    keyboard::{PhysicalKey, KeyCode},
    dpi::PhysicalPosition,
};
use tracing::{info, debug, error};

use crate::config::{Config, ShaderPreset};
use crate::renderer::{Renderer, DrawResult};
use crate::input::{InputSystem, CombinedInputState};
use crate::utils::PushConstants;

pub struct Application {
    config: Config,
    window: Option<Window>,
    renderer: Option<Renderer>,
    input_system: Option<InputSystem>,
    runtime: tokio::runtime::Runtime,

    // Application state
    start_time: Instant,
    frame_count: u64,
    mouse_pos: PhysicalPosition<f64>,
    mouse_pressed: bool,
    is_fullscreen: bool,
    shader_presets: Vec<ShaderPreset>,
    current_shader_index: usize,
}

impl Application {
    pub fn new(config: Config) -> Result<Self> {
        let runtime = tokio::runtime::Runtime::new()?;

        let shader_presets = vec![
            ShaderPreset::Torus,
            ShaderPreset::Terrain,
            ShaderPreset::Crystal,
            ShaderPreset::Plasma,
        ];

        let current_shader_index = shader_presets
            .iter()
            .position(|p| *p == config.shader.preset)
            .unwrap_or(0);

        Ok(Self {
            is_fullscreen: config.window.fullscreen,
            config,
            window: None,
            renderer: None,
            input_system: None,
            runtime,
            start_time: Instant::now(),
            frame_count: 0,
            mouse_pos: PhysicalPosition::new(0.0, 0.0),
            mouse_pressed: false,
            shader_presets,
            current_shader_index,
        })
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
            info!("Toggled fullscreen: {}", self.is_fullscreen);
        }
    }

    fn cycle_shader(&mut self) {
        if !self.config.shader.hot_reload {
            debug!("Shader hot reload is disabled");
            return;
        }

        self.current_shader_index = (self.current_shader_index + 1) % self.shader_presets.len();
        let new_preset = self.shader_presets[self.current_shader_index].clone();

        info!("Switching to shader: {:?}", new_preset);

        if let Some(renderer) = &mut self.renderer {
            if let Err(e) = renderer.switch_shader_preset(new_preset.clone()) {
                error!("Failed to switch shader: {}", e);
            } else {
                self.config.shader.preset = new_preset;
            }
        }
    }

    fn handle_resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            return;
        }

        info!("Window resized to {}x{}", new_size.width, new_size.height);

        if let (Some(renderer), Some(window)) = (&mut self.renderer, &self.window) {
            if let Err(e) = renderer.resize(window) {
                error!("Failed to resize renderer: {}", e);
            }
        }
    }

    fn render_frame(&mut self) {
        if let (Some(renderer), Some(input_system)) = (&mut self.renderer, &self.input_system) {
            // Get input state
            let input_state = self.runtime.block_on(input_system.get_combined_state());

            // Prepare push constants
            let push_constants = self.prepare_push_constants(&input_state);

            // Draw frame
            match renderer.draw(&push_constants) {
                Ok(DrawResult::Success) => {
                    self.frame_count += 1;
                }
                Ok(DrawResult::Suboptimal) => {
                    debug!("Swapchain suboptimal");
                    if let Some(window) = &self.window {
                        if let Err(e) = renderer.resize(window) {
                            error!("Failed to resize renderer: {}", e);
                        }
                    }
                }
                Ok(DrawResult::NeedsRecreation) => {
                    info!("Swapchain needs recreation");
                    if let Some(window) = &self.window {
                        if let Err(e) = renderer.resize(window) {
                            error!("Failed to recreate swapchain: {}", e);
                        }
                    }
                }
                Err(e) => {
                    error!("Draw error: {}", e);
                }
            }
        }
    }

    fn prepare_push_constants(&self, input: &CombinedInputState) -> PushConstants {
        let elapsed = self.start_time.elapsed().as_secs_f32();

        PushConstants {
            time: elapsed,
            mouse_x: self.mouse_pos.x as f32,
            mouse_y: self.mouse_pos.y as f32,
            mouse_pressed: if self.mouse_pressed { 1.0 } else { 0.0 },
            resolution_x: self.window.as_ref()
                .map(|w| w.inner_size().width as f32)
                .unwrap_or(1280.0),
            resolution_y: self.window.as_ref()
                .map(|w| w.inner_size().height as f32)
                .unwrap_or(720.0),
            note_velocity: input.velocity,
            pitch_bend: input.pitch_bend,
            modulation: input.modulation,
            expression: input.expression,
            note_count: input.note_count as f32,
            last_note: input.last_note as f32,
        }
    }

    fn print_controls(&self) {
        info!("Controls:");
        info!("  F11     - Toggle fullscreen");
        info!("  ESC     - Exit (or exit fullscreen if fullscreen)");
        info!("  TAB     - Cycle shader presets");
        info!("  R       - Reload current shader");
        info!("  Space   - Toggle pause");
        info!("  1-4     - Select shader preset directly");
    }
}

impl ApplicationHandler for Application {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        info!("Application resumed");

        // Create window
        let fullscreen = if self.is_fullscreen {
            Some(Fullscreen::Borderless(None))
        } else {
            None
        };

        let window_attributes = Window::default_attributes()
            .with_title(&self.config.window.title)
            .with_inner_size(winit::dpi::LogicalSize::new(
                self.config.window.width,
                self.config.window.height,
            ))
            .with_resizable(self.config.window.resizable)
            .with_fullscreen(fullscreen);

        match event_loop.create_window(window_attributes) {
            Ok(window) => {
                // Create renderer
                match Renderer::new(&window, &self.config.graphics, &self.config.shader) {
                    Ok(renderer) => {
                        self.renderer = Some(renderer);
                        info!("Renderer created successfully");
                    }
                    Err(e) => {
                        error!("Failed to create renderer: {}", e);
                        event_loop.exit();
                        return;
                    }
                }

                self.window = Some(window);
            }
            Err(e) => {
                error!("Failed to create window: {}", e);
                event_loop.exit();
                return;
            }
        }

        // Create input system
        self.input_system = Some(InputSystem::new(&self.config.midi, &self.config.audio));

        // Reset timers
        self.start_time = Instant::now();
        self.frame_count = 0;

        self.print_controls();
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                info!("Close requested");
                event_loop.exit();
            }

            WindowEvent::Resized(physical_size) => {
                self.handle_resize(physical_size);
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
                    PhysicalKey::Code(KeyCode::KeyR) => {
                        if let Some(renderer) = &mut self.renderer {
                            info!("Reloading shaders");
                            if let Err(e) = renderer.reload_shaders() {
                                error!("Failed to reload shaders: {}", e);
                            }
                        }
                    }
                    PhysicalKey::Code(KeyCode::Digit1) => {
                        if let Some(renderer) = &mut self.renderer {
                            let _ = renderer.switch_shader_preset(ShaderPreset::Torus);
                        }
                    }
                    PhysicalKey::Code(KeyCode::Digit2) => {
                        if let Some(renderer) = &mut self.renderer {
                            let _ = renderer.switch_shader_preset(ShaderPreset::Terrain);
                        }
                    }
                    PhysicalKey::Code(KeyCode::Digit3) => {
                        if let Some(renderer) = &mut self.renderer {
                            let _ = renderer.switch_shader_preset(ShaderPreset::Crystal);
                        }
                    }
                    PhysicalKey::Code(KeyCode::Digit4) => {
                        if let Some(renderer) = &mut self.renderer {
                            let _ = renderer.switch_shader_preset(ShaderPreset::Plasma);
                        }
                    }
                    _ => {}
                }
            }

            WindowEvent::CursorMoved { position, .. } => {
                self.mouse_pos = position;
            }

            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => {
                self.mouse_pressed = state == ElementState::Pressed;
            }

            WindowEvent::RedrawRequested => {
                self.render_frame();

                // Request next frame
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }

            _ => {}
        }
    }

    fn suspended(&mut self, _event_loop: &ActiveEventLoop) {
        info!("Application suspended");
        self.renderer = None;
        self.window = None;
    }
}