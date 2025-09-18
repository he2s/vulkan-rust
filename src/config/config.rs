use crate::input::midi::MidiConfig;
use crate::input::osc::OscConfig;
use anyhow::Result;
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Parser)]
#[command(name = "vulkan-midi-visualizer")]
#[command(about = "A MIDI-reactive Vulkan pixel shader visualizer")]
pub struct Args {
    /// Path to configuration file
    #[arg(short, long, default_value = "config.toml")]
    pub config: String,

    /// List all available devices (MIDI, Audio, GPU) and exit
    #[arg(long)]
    pub list_devices: bool,

    /// Start in fullscreen mode
    #[arg(short, long)]
    pub fullscreen: bool,

    /// Window width (ignored in fullscreen)
    #[arg(long)]
    pub width: Option<u32>,

    /// Window height (ignored in fullscreen)
    #[arg(long)]
    pub height: Option<u32>,

    /// Window title
    #[arg(long)]
    pub title: Option<String>,

    /// Override shader from config
    #[arg(long)]
    pub shader: Option<String>,
}

#[derive(Deserialize, Serialize, Default)]
pub struct Config {
    #[serde(default)]
    pub window: WindowConfig,

    #[serde(default)]
    pub midi: MidiConfig,

    #[serde(default)]
    pub graphics: GraphicsConfig,

    #[serde(default)]
    pub audio: AudioConfig,

    #[serde(default)]
    pub shader: ShaderConfig,

    #[serde(default)]
    pub osc: OscConfig,
}

#[derive(Deserialize, Serialize)]
pub struct WindowConfig {
    #[serde(default = "default_width")]
    pub width: u32,

    #[serde(default = "default_height")]
    pub height: u32,

    #[serde(default = "default_title")]
    pub title: String,

    #[serde(default)]
    pub fullscreen: bool,

    #[serde(default = "default_true")]
    pub resizable: bool,
}

#[derive(Deserialize, Serialize)]
pub struct GraphicsConfig {
    #[serde(default = "default_true")]
    pub vsync: bool,

    #[serde(default = "default_validation_layers")]
    pub validation_layers: bool,
}

#[derive(Deserialize, Serialize)]
pub struct AudioConfig {
    #[serde(default)]
    pub enabled: bool,

    #[serde(default)]
    pub device_name: Option<String>,

    #[serde(default)]
    pub sample_rate: Option<u32>,
}

#[derive(Deserialize, Serialize, Clone)]
pub struct ShaderConfig {
    #[serde(default = "default_shader_preset")]
    pub preset: ShaderPreset,

    #[serde(default)]
    pub custom_vertex_path: Option<String>,

    #[serde(default)]
    pub custom_fragment_path: Option<String>,

    #[serde(default = "default_true")]
    pub allow_runtime_switching: bool,
}

#[derive(Deserialize, Serialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ShaderPreset {
    Torus,
    Terrain,
    Crystal,
    Custom,
    Stars,
}

const DEFAULT_WIDTH: u32 = 800;
const DEFAULT_HEIGHT: u32 = 600;
const DEFAULT_TITLE: &str = "Vulkan MIDI Pixel Shader";

const fn default_width() -> u32 {
    DEFAULT_WIDTH
}
const fn default_height() -> u32 {
    DEFAULT_HEIGHT
}
fn default_title() -> String {
    DEFAULT_TITLE.to_string()
}
const fn default_true() -> bool {
    true
}
const fn default_shader_preset() -> ShaderPreset {
    ShaderPreset::Torus
}
const fn default_validation_layers() -> bool {
    cfg!(debug_assertions)
}
const fn default_osc_port() -> u16 {
    8000
}

impl Default for WindowConfig {
    fn default() -> Self {
        Self {
            width: default_width(),
            height: default_height(),
            title: default_title(),
            fullscreen: false,
            resizable: default_true(),
        }
    }
}

impl Default for GraphicsConfig {
    fn default() -> Self {
        Self {
            vsync: default_true(),
            validation_layers: default_validation_layers(),
        }
    }
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            device_name: None,
            sample_rate: None,
        }
    }
}

impl Default for ShaderConfig {
    fn default() -> Self {
        Self {
            preset: default_shader_preset(),
            custom_vertex_path: None,
            custom_fragment_path: None,
            allow_runtime_switching: true,
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
        if let Some(ref title) = args.title {
            self.window.title = title.clone();
        }
    }
}

pub fn parse_shader_preset(shader_str: &str) -> ShaderPreset {
    match shader_str.to_lowercase().as_str() {
        "torus" => ShaderPreset::Torus,
        "terrain" => ShaderPreset::Terrain,
        "crystal" => ShaderPreset::Crystal,
        "custom" => ShaderPreset::Custom,
        "stars" => ShaderPreset::Stars,
        _ => {
            eprintln!("Unknown shader preset '{}', using default", shader_str);
            ShaderPreset::Torus
        }
    }
}

pub fn load_or_create_config(config_path: &str) -> Result<Config> {
    if Path::new(config_path).exists() {
        match Config::load_from_file(config_path) {
            Ok(config) => {
                println!("Loaded configuration from: {}", config_path);
                Ok(config)
            }
            Err(e) => {
                eprintln!("Failed to load config file '{}': {}", config_path, e);
                println!("Using default configuration");
                Ok(Config::default())
            }
        }
    } else {
        println!(
            "Config file '{}' not found, creating default config",
            config_path
        );
        let default_config = Config::default();
        if let Err(e) = default_config.save_to_file(config_path) {
            eprintln!("Failed to save default config: {}", e);
        } else {
            println!("Default configuration saved to: {}", config_path);
        }
        Ok(default_config)
    }
}

pub fn print_startup_info(config: &Config) {
    println!("Starting Vulkan MIDI Pixel Shader");
    println!(
        "Window: {}x{} - {}",
        config.window.width,
        config.window.height,
        if config.window.fullscreen {
            "Fullscreen"
        } else {
            "Windowed"
        }
    );
    println!("Shader: {:?}", config.shader.preset);
    println!(
        "MIDI: {}",
        if config.midi.enabled {
            "Enabled"
        } else {
            "Disabled"
        }
    );
    println!(
        "Audio: {}",
        if config.audio.enabled {
            "Enabled"
        } else {
            "Disabled"
        }
    );
    println!(
        "OSC: {}",
        if config.osc.enabled {
            format!("Enabled (port {})", config.osc.port)
        } else {
            "Disabled".to_string()
        }
    );
}
