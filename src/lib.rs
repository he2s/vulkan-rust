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
pub use config::config::{Args, Config, ShaderConfig, ShaderPreset, AudioConfig};
pub use graphics::{GeometryMode, Vertex, InstanceData, PushConstants};
pub use graphics::vulkan::{VulkanContext, VulkanSwapchain, VulkanBuffers, VulkanCommands, VulkanSync};
pub use input::{MidiConfig, MidiManager, OscConfig, OscManager};
pub use state::FrameState;
pub use audio::{AudioLevels, AudioState, BeatState};