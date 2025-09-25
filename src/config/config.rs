use crate::input::midi::MidiConfig;
use crate::input::osc::OscConfig;
use anyhow::Result;
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
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

    #[serde(default)]
    pub tap_tempo: TapTempoConfig,
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

#[derive(Deserialize, Serialize, Default)]
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
    #[serde(default = "default_active_preset")]
    pub active_preset: String,

    #[serde(default = "default_true")]
    pub allow_runtime_switching: bool,

    #[serde(default)]
    pub presets: HashMap<String, ShaderPreset>,
}

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct ShaderPreset {
    pub name: String,

    #[serde(default = "default_true")]
    pub enabled: bool,

    pub geometry_type: GeometryType,
    pub vertex: String,

    #[serde(default)]
    pub geometry: Option<String>,

    pub fragment: String,

    #[serde(default)]
    pub compute: Option<String>,
}

#[derive(Deserialize, Serialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum GeometryType {
    Fullscreen,
    Points,
    Vertices,
    Compute,
}

#[derive(Deserialize, Serialize)]
pub struct TapTempoConfig {
    /// Maximum number of recent taps to keep for BPM calculation
    #[serde(default = "default_tap_history_size")]
    pub max_tap_history: usize,

    /// Timeout in seconds after which old taps are discarded
    #[serde(default = "default_tap_timeout")]
    pub tap_timeout_seconds: u64,

    /// Minimum BPM value accepted
    #[serde(default = "default_min_bpm")]
    pub min_bpm: f32,

    /// Maximum BPM value accepted
    #[serde(default = "default_max_bpm")]
    pub max_bpm: f32,
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
fn default_active_preset() -> String {
    "torus".to_string()
}
const fn default_validation_layers() -> bool {
    cfg!(debug_assertions)
}
#[allow(dead_code)]
const fn default_osc_port() -> u16 {
    8000
}

const fn default_tap_history_size() -> usize {
    8
}
const fn default_tap_timeout() -> u64 {
    10
}
const fn default_min_bpm() -> f32 {
    40.0
}
const fn default_max_bpm() -> f32 {
    300.0
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


impl Default for ShaderConfig {
    fn default() -> Self {
        let mut presets = HashMap::new();

        presets.insert("torus".to_string(), ShaderPreset {
            name: "Torus Fullscreen".to_string(),
            enabled: true,
            geometry_type: GeometryType::Fullscreen,
            vertex: "fullscreen.vert".to_string(),
            geometry: None,
            fragment: "gradient.frag".to_string(),
            compute: None,
        });

        presets.insert("terrain".to_string(), ShaderPreset {
            name: "Terrain".to_string(),
            enabled: true,
            geometry_type: GeometryType::Vertices,
            vertex: "terrain.vert".to_string(),
            geometry: None,
            fragment: "terrain.frag".to_string(),
            compute: None,
        });

        presets.insert("crystal".to_string(), ShaderPreset {
            name: "Simple Crystal".to_string(),
            enabled: true,
            geometry_type: GeometryType::Points,
            vertex: "points.vert".to_string(),
            geometry: Some("simple.geom".to_string()),
            fragment: "simple_geometry.frag".to_string(),
            compute: None,
        });

        presets.insert("weirdcrystal".to_string(), ShaderPreset {
            name: "Complex Crystal".to_string(),
            enabled: true,
            geometry_type: GeometryType::Points,
            vertex: "points.vert".to_string(),
            geometry: Some("example.geom".to_string()),
            fragment: "weird_crystal.frag".to_string(),
            compute: None,
        });

        presets.insert("stars".to_string(), ShaderPreset {
            name: "Stars".to_string(),
            enabled: true,
            geometry_type: GeometryType::Vertices,
            vertex: "stars.vert".to_string(),
            geometry: None,
            fragment: "stars.frag".to_string(),
            compute: None,
        });

        presets.insert("computeparticles".to_string(), ShaderPreset {
            name: "Compute Particles".to_string(),
            enabled: true,
            geometry_type: GeometryType::Compute,
            vertex: "instanced_triangles.vert".to_string(),
            geometry: None,
            fragment: "instanced_triangles.frag".to_string(),
            compute: Some("point_generator.comp".to_string()),
        });

        Self {
            active_preset: default_active_preset(),
            allow_runtime_switching: true,
            presets,
        }
    }
}

impl Default for TapTempoConfig {
    fn default() -> Self {
        Self {
            max_tap_history: default_tap_history_size(),
            tap_timeout_seconds: default_tap_timeout(),
            min_bpm: default_min_bpm(),
            max_bpm: default_max_bpm(),
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

#[allow(dead_code)]
pub fn validate_shader_preset(shader_str: &str, config: &Config) -> bool {
    config.shader.presets.contains_key(shader_str)
}

pub fn load_or_create_config(config_path: &str) -> Result<Config> {
    if Path::new(config_path).exists() {
        match Config::load_from_file(config_path) {
            Ok(config) => {
                println!("Loaded configuration from: {config_path}");
                Ok(config)
            }
            Err(e) => {
                eprintln!("Failed to load config file '{config_path}': {e}");
                println!("Using default configuration");
                Ok(Config::default())
            }
        }
    } else {
        println!("Config file '{config_path}' not found, creating default config");
        let default_config = Config::default();
        if let Err(e) = default_config.save_to_file(config_path) {
            eprintln!("Failed to save default config: {e}");
        } else {
            println!("Default configuration saved to: {config_path}");
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
    println!("Active Shader: {}", config.shader.active_preset);
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
