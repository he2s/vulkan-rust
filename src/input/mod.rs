pub mod audio;
pub mod manager;
pub mod midi;
pub mod osc;

// Re-export commonly used types
pub use manager::InputManager;
pub use midi::{MidiConfig, MidiManager, MidiStateSnapshot};
pub use osc::{OscConfig, OscManager, OscStateSnapshot};
