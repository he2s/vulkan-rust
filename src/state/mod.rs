use crate::audio::{AudioLevels, BeatState};
use crate::input::midi::MidiStateSnapshot;
use crate::input::osc::OscStateSnapshot;

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
            midi: MidiStateSnapshot::default(),
            audio_levels: AudioLevels::default(),
            osc: OscStateSnapshot {
                channel1: 0.0,
                channel2: 0.0,
            },
            beat: BeatState::default(),
        }
    }
}
