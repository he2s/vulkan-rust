use anyhow::Result;

pub struct OverlayRenderer {
    pub show_menu: bool,
}

impl OverlayRenderer {
    pub fn new() -> Result<Self> {
        Ok(Self {
            show_menu: false,
        })
    }

    pub fn prepare_frame(&mut self, _delta_time: f32, _display_size: [f32; 2]) {
        // No-op for console output
    }

    pub fn build_ui(&mut self, push_constants: &crate::PushConstants) {
        if self.show_menu {
            // Print info to console
            println!("\n========================================");
            println!("      Audio Visualizer Controls");
            println!("========================================");

            println!("\n[Time & Display]");
            println!("  Time: {:.2}s", push_constants.time);
            println!("  Resolution: {}x{}", push_constants.render_w, push_constants.render_h);

            println!("\n[Audio Input]");
            println!("  Intensity: {:.2}", push_constants.audio_intensity);
            println!("  Bass:      {:.2}", push_constants.bass_level);
            println!("  Mid:       {:.2}", push_constants.mid_level);
            println!("  High:      {:.2}", push_constants.high_level);

            println!("\n[Beat Detection]");
            println!("  BPM:            {:.1}", push_constants.bpm);
            println!("  Time to next:   {:.2}s", push_constants.time_to_next_beat);
            println!("  Time since last: {:.2}s", push_constants.time_since_last_beat);
            println!("  Beats per bar:  {}", push_constants.beats_per_bar);

            println!("\n[MIDI]");
            println!("  Velocity:   {:.2}", push_constants.note_velocity);
            println!("  Pitch Bend: {:.2}", push_constants.pitch_bend);
            println!("  CC1:        {:.2}", push_constants.cc1);
            println!("  CC74:       {:.2}", push_constants.cc74);
            println!("  Note Count: {}", push_constants.note_count);
            println!("  Last Note:  {}", push_constants.last_note);

            println!("\n[OSC]");
            println!("  Channel 1: {:.2}", push_constants.osc_ch1);
            println!("  Channel 2: {:.2}", push_constants.osc_ch2);

            println!("\n[Rendering]");
            println!("  Max Points: {}", push_constants.max_points);
            println!("  FFT Size:   {}", push_constants.fft_size);

            println!("\n[Keyboard Shortcuts]");
            println!("  H     - Toggle this info display");
            println!("  Tab   - Cycle shaders");
            println!("  F5    - Reload shaders");
            println!("  F11   - Toggle fullscreen");
            println!("  Space - Tap tempo");
            println!("========================================\n");

            // Only show once per toggle
            self.show_menu = false;
        }
    }

    pub fn toggle_menu(&mut self) {
        self.show_menu = !self.show_menu;
    }
}
