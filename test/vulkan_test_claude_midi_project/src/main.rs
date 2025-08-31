// src/main.rs

mod config;
mod renderer;
mod input;
mod app;
mod utils;

use anyhow::Result;
use clap::Parser;
use tracing::{info, error};
use tracing_subscriber::{fmt, prelude::*, EnvFilter};
use winit::event_loop::{ControlFlow, EventLoop};
use std::path::Path;

use crate::config::{Config, Args};
use crate::app::Application;

fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(EnvFilter::from_default_env().add_directive("vulkan_visualizer=info".parse()?))
        .init();

    info!("Starting Vulkan MIDI Visualizer v{}", env!("CARGO_PKG_VERSION"));

    // Parse command line arguments
    let args = Args::parse();

    // Load configuration
    let config = load_config(&args)?;

    // Create event loop
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);

    // Create and run application
    let mut app = Application::new(config)?;

    // Run the event loop
    event_loop.run_app(&mut app)?;

    Ok(())
}

fn load_config(args: &Args) -> Result<Config> {
    let config_path = &args.config;

    let mut config = if Path::new(config_path).exists() {
        match Config::load_from_file(config_path) {
            Ok(cfg) => {
                info!("Loaded configuration from: {}", config_path);
                cfg
            }
            Err(e) => {
                error!("Failed to load config file '{}': {}", config_path, e);
                info!("Using default configuration");
                Config::default()
            }
        }
    } else {
        info!("Config file '{}' not found, creating default", config_path);
        let default_config = Config::default();

        if let Err(e) = default_config.save_to_file(config_path) {
            error!("Failed to save default config: {}", e);
        } else {
            info!("Default configuration saved to: {}", config_path);
        }

        default_config
    };

    // Merge with command line arguments
    config.merge_with_args(args);

    info!("Configuration loaded:");
    info!("  Window: {}x{} - {}",
        config.window.width,
        config.window.height,
        if config.window.fullscreen { "Fullscreen" } else { "Windowed" }
    );
    info!("  Shader: {:?}", config.shader.preset);
    info!("  MIDI: {}", if config.midi.enabled { "Enabled" } else { "Disabled" });
    info!("  Audio: {}", if config.audio.enabled { "Enabled" } else { "Disabled" });

    Ok(config)
}