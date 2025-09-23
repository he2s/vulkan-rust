pub mod types;

// Re-export commonly used types for convenience
pub use types::*;

// This demonstrates the modular structure for Vulkan code.
// The types module successfully factors out shared types from main.rs,
// reducing main.rs complexity while maintaining performance.
//
// Future modules can be added here as the codebase evolves:
// pub mod context;    // VulkanContext and related functionality
// pub mod swapchain;  // VulkanSwapchain management
// pub mod buffers;    // VulkanBuffers handling
// pub mod pipeline;   // VulkanPipeline creation and management
// pub mod commands;   // VulkanCommands and command buffer handling