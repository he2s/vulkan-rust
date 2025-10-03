pub mod audio;
pub mod manager;
pub mod midi;
pub mod osc;

// Re-export commonly used types
// pub use manager::InputManager; // Still in main.rs, needs to be extracted
pub use midi::{MidiConfig, MidiManager, MidiStateSnapshot};
pub use osc::{OscConfig, OscManager, OscStateSnapshot};
