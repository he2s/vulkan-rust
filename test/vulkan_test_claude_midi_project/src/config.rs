// src/config.rs

use anyhow::Result;
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::{fs, path::Path};

#[derive(Parser, Debug)]
#[command(name = "vulkan-visualizer")]
#[command(about = "A high-performance MIDI-reactive Vulkan visualizer")]
pub struct Args {
    /// Path to configuration file
    #[arg(short, long, default_value = "config.toml")]
    pub config: String,

    /// Start in fullscreen mode
    #[arg(short, long)]
    pub fullscreen: bool,

    /// Window width (ignored in fullscreen)
    #[arg(long)]
    pub width: Option<u32>,

    /// Window height (ignored in fullscreen)
    #[arg(long)]
    pub height: Option<u32>,

    /// Override shader preset
    #[arg(long)]
    pub shader: Option<String>,

    /// Enable debug/validation layers
    #[arg(long)]
    pub debug: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub window: WindowConfig,
    pub graphics: GraphicsConfig,
    pub shader: ShaderConfig,
    pub midi: MidiConfig,
    pub audio: AudioConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowConfig {
    #[serde(default = "default::window_width")]
    pub width: u32,
    #[serde(default = "default::window_height")]
    pub height: u32,
    #[serde(default = "default::window_title")]
    pub title: String,
    #[serde(default)]
    pub fullscreen: bool,
    #[serde(default = "default::true_val")]
    pub resizable: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphicsConfig {
    #[serde(default = "default::true_val")]
    pub vsync: bool,
    #[serde(default)]
    pub validation_layers: bool,
    #[serde(default = "default::msaa_samples")]
    pub msaa_samples: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShaderConfig {
    #[serde(default = "default::shader_preset")]
    pub preset: ShaderPreset,
    pub custom_vertex_path: Option<String>,
    pub custom_fragment_path: Option<String>,
    #[serde(default = "default::true_val")]
    pub hot_reload: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ShaderPreset {
    Torus,
    Terrain,
    Crystal,
    Plasma,
    Custom,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MidiConfig {
    #[serde(default = "default::true_val")]
    pub enabled: bool,
    #[serde(default = "default::true_val")]
    pub auto_connect: bool,
    pub port_name: Option<String>,
    #[serde(default = "default::midi_buffer_size")]
    pub buffer_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioConfig {
    #[serde(default)]
    pub enabled: bool,
    pub device_name: Option<String>,
    pub sample_rate: Option<u32>,
    #[serde(default = "default::fft_size")]
    pub fft_size: usize,
}

mod default {
    pub fn window_width() -> u32 { 1280 }
    pub fn window_height() -> u32 { 720 }
    pub fn window_title() -> String { "Vulkan MIDI Visualizer".to_string() }
    pub fn true_val() -> bool { true }
    pub fn msaa_samples() -> u32 { 1 }
    pub fn shader_preset() -> super::ShaderPreset { super::ShaderPreset::Torus }
    pub fn midi_buffer_size() -> usize { 1024 }
    pub fn fft_size() -> usize { 2048 }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            window: WindowConfig::default(),
            graphics: GraphicsConfig::default(),
            shader: ShaderConfig::default(),
            midi: MidiConfig::default(),
            audio: AudioConfig::default(),
        }
    }
}

impl Default for WindowConfig {
    fn default() -> Self {
        Self {
            width: default::window_width(),
            height: default::window_height(),
            title: default::window_title(),
            fullscreen: false,
            resizable: true,
        }
    }
}

impl Default for GraphicsConfig {
    fn default() -> Self {
        Self {
            vsync: true,
            validation_layers: cfg!(debug_assertions),
            msaa_samples: 1,
        }
    }
}

impl Default for ShaderConfig {
    fn default() -> Self {
        Self {
            preset: default::shader_preset(),
            custom_vertex_path: None,
            custom_fragment_path: None,
            hot_reload: true,
        }
    }
}

impl Default for MidiConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            auto_connect: true,
            port_name: None,
            buffer_size: default::midi_buffer_size(),
        }
    }
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            device_name: None,
            sample_rate: None,
            fft_size: default::fft_size(),
        }
    }
}

impl Config {
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = fs::read_to_string(path)?;
        Ok(toml::from_str(&content)?)
    }

    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let content = toml::to_string_pretty(self)?;
        fs::write(path, content)?;
        Ok(())
    }

    pub fn merge_with_args(&mut self, args: &Args) {
        if args.fullscreen {
            self.window.fullscreen = true;
        }

        if let Some(width) = args.width {
            self.window.width = width;
        }

        if let Some(height) = args.height {
            self.window.height = height;
        }

        if args.debug {
            self.graphics.validation_layers = true;
        }

        if let Some(ref shader) = args.shader {
            self.shader.preset = match shader.as_str() {
                "torus" => ShaderPreset::Torus,
                "terrain" => ShaderPreset::Terrain,
                "crystal" => ShaderPreset::Crystal,
                "plasma" => ShaderPreset::Plasma,
                "custom" => ShaderPreset::Custom,
                _ => self.shader.preset.clone(),
            };
        }
    }
}