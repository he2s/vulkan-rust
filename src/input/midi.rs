use serde::{Deserialize, Serialize};
use crate::{default_true, MAX_CONTROLLERS, MAX_NOTES};

#[derive(Deserialize, Serialize)]
pub struct MidiConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub auto_connect: bool,
    #[serde(default)]
    pub port_name: Option<String>,
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