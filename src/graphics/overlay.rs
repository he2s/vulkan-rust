use anyhow::Result;
use imgui::*;

pub struct OverlayRenderer {
    pub context: imgui::Context,
    pub show_menu: bool,
}

impl OverlayRenderer {
    pub fn new() -> Result<Self> {
        let mut context = imgui::Context::create();
        context.set_ini_filename(None); // Disable imgui.ini

        // Build font atlas (required before rendering)
        let fonts = context.fonts();
        fonts.build_rgba32_texture();

        // Configure ImGui style
        let style = context.style_mut();
        style.window_rounding = 8.0;
        style.frame_rounding = 4.0;
        style.grab_rounding = 4.0;

        Ok(Self {
            context,
            show_menu: false,
        })
    }

    pub fn prepare_frame(&mut self, delta_time: f32, display_size: [f32; 2]) {
        let io = self.context.io_mut();
        io.display_size = display_size;
        io.delta_time = delta_time;
    }

    pub fn build_ui(&mut self, push_constants: &crate::PushConstants) {
        let ui = self.context.new_frame();

        if self.show_menu {
            ui.window("Controls")
                .size([350.0, 500.0], Condition::FirstUseEver)
                .position([20.0, 20.0], Condition::FirstUseEver)
                .build(|| {
                    ui.text("Audio Visualizer Controls");
                    ui.separator();

                    ui.text(format!("Time: {:.2}s", push_constants.time));
                    ui.text(format!("Resolution: {}x{}", push_constants.render_w, push_constants.render_h));

                    ui.separator();
                    ui.text("Audio Input:");
                    ui.text(format!("Intensity: {:.2}", push_constants.audio_intensity));
                    ui.text(format!("Bass: {:.2}", push_constants.bass_level));
                    ui.text(format!("Mid: {:.2}", push_constants.mid_level));
                    ui.text(format!("High: {:.2}", push_constants.high_level));

                    ui.separator();
                    ui.text("Beat Detection:");
                    ui.text(format!("BPM: {:.1}", push_constants.bpm));
                    ui.text(format!("Time to next: {:.2}s", push_constants.time_to_next_beat));
                    ui.text(format!("Time since last: {:.2}s", push_constants.time_since_last_beat));
                    ui.text(format!("Beats per bar: {}", push_constants.beats_per_bar));

                    ui.separator();
                    ui.text("MIDI:");
                    ui.text(format!("Velocity: {:.2}", push_constants.note_velocity));
                    ui.text(format!("Pitch Bend: {:.2}", push_constants.pitch_bend));
                    ui.text(format!("CC1: {:.2}", push_constants.cc1));
                    ui.text(format!("CC74: {:.2}", push_constants.cc74));
                    ui.text(format!("Note Count: {}", push_constants.note_count));
                    ui.text(format!("Last Note: {}", push_constants.last_note));

                    ui.separator();
                    ui.text("OSC:");
                    ui.text(format!("Channel 1: {:.2}", push_constants.osc_ch1));
                    ui.text(format!("Channel 2: {:.2}", push_constants.osc_ch2));

                    ui.separator();
                    ui.text("Rendering:");
                    ui.text(format!("Max Points: {}", push_constants.max_points));
                    ui.text(format!("FFT Size: {}", push_constants.fft_size));

                    ui.separator();
                    ui.text("Keyboard Shortcuts:");
                    ui.text("H - Toggle this menu");
                    ui.text("Tab - Cycle shaders");
                    ui.text("F5 - Reload shaders");
                    ui.text("F11 - Toggle fullscreen");
                    ui.text("Space - Tap tempo");
                });
        }
    }

    #[allow(dead_code)]
    pub fn render_draw_data(&mut self) -> &imgui::DrawData {
        self.context.render()
    }

    pub fn toggle_menu(&mut self) {
        self.show_menu = !self.show_menu;
    }
}
