// Core modules
pub mod audio;
pub mod beat_detection;
pub mod config;
pub mod graphics;
pub mod input;
pub mod processing;
pub mod state;
pub mod utils;
pub mod gfx;

// Re-export commonly used types for convenience
// Only re-export what's needed by external consumers of the library
pub use config::config::Config;
pub use graphics::PushConstants;
pub use state::FrameState;