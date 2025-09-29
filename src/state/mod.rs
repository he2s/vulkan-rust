use crate::audio::{AudioLevels, BeatState};
use crate::input::midi::MidiStateSnapshot;
use crate::input::osc::OscStateSnapshot;

pub mod frame;
pub mod cache;

// Core frame state that gets passed to shaders
#[derive(Clone, Debug)]
pub struct FrameState {
    pub midi: MidiStateSnapshot,
    pub audio_levels: AudioLevels,
    pub osc: OscStateSnapshot,
    pub beat: BeatState,
}

impl Default for FrameState {
    fn default() -> Self {
        Self {
            midi: MidiStateSnapshot {
                notes: [0.0; 128],
                controllers: [0.5; 128],
                pitch_bend: 0.0,
                last_note: 0,
                note_count: 0,
            },
            audio_levels: AudioLevels::default(),
            osc: OscStateSnapshot {
                channel1: 0.0,
                channel2: 0.0,
            },
            beat: BeatState::default(),
        }
    }
}

// Application state management
pub struct AppState {
    pub current_frame: FrameState,
    pub frame_count: u64,
    pub start_time: std::time::Instant,
    pub last_frame_time: std::time::Instant,
    pub delta_time: f32,
}

impl AppState {
    pub fn new() -> Self {
        let now = std::time::Instant::now();
        Self {
            current_frame: FrameState::default(),
            frame_count: 0,
            start_time: now,
            last_frame_time: now,
            delta_time: 0.0,
        }
    }

    pub fn update_timing(&mut self) {
        let now = std::time::Instant::now();
        self.delta_time = (now - self.last_frame_time).as_secs_f32();
        self.last_frame_time = now;
        self.frame_count += 1;
    }

    pub fn get_time(&self) -> f32 {
        self.start_time.elapsed().as_secs_f32()
    }
}