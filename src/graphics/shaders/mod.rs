use anyhow::{Result, anyhow};
use std::fs;
use crate::config::config::{ShaderConfig, ShaderPreset, GeometryType};
use crate::graphics::GeometryMode;

pub mod compiler;
pub mod cache;

pub struct ShaderSources {
    pub vertex: String,
    pub geometry: Option<String>,
    pub fragment: String,
}

fn try_read_to_string<P: AsRef<std::path::Path>>(p: P) -> Option<String> {
    std::fs::read_to_string(p).ok()
}

impl ShaderSources {
    pub fn determine_geometry_mode(&self) -> GeometryMode {
        // Check for geometry shader first
        if self.geometry.is_some() {
            return GeometryMode::GeometryShader;
        }

        // Check vertex shader content to determine mode
        if self.vertex.contains("fullscreen.vert") ||
           self.vertex.contains("in_position") &&
           !self.vertex.contains("instance_") &&
           !self.vertex.contains("gl_InstanceIndex") {
            GeometryMode::Trivial
        } else if self.vertex.contains("instance_") ||
                  self.vertex.contains("gl_InstanceIndex") {
            GeometryMode::InstancedTriangles
        } else if self.vertex.contains("compute") ||
                  self.fragment.contains("compute") {
            GeometryMode::ComputeGenerated
        } else {
            GeometryMode::Trivial
        }
    }

    pub fn geometry_mode_from_preset(preset: &ShaderPreset) -> GeometryMode {
        match preset.geometry_type {
            GeometryType::Fullscreen => GeometryMode::Trivial,
            GeometryType::Points if preset.geometry.is_some() => GeometryMode::GeometryShader,
            GeometryType::Points => GeometryMode::Trivial,
            GeometryType::Vertices => GeometryMode::Trivial,
            GeometryType::Compute => GeometryMode::ComputeGenerated,
        }
    }

    pub fn load_preset(preset: &ShaderPreset) -> Result<Self> {
        println!("Loading preset: {}", preset.name);

        // Try to load from external directory first if set
        if let Ok(dir) = std::env::var("SHADER_PRESET_DIR") {
            println!("Loading from external directory: {}", dir);
            let shader_dir = std::path::Path::new(&dir).join("shaders");

            let vpath = shader_dir.join(&preset.vertex);
            let fpath = shader_dir.join(&preset.fragment);
            let gpath = preset.geometry.as_ref().map(|g| shader_dir.join(g));

            if let (Some(vs), Some(fs)) = (try_read_to_string(&vpath), try_read_to_string(&fpath)) {
                let geometry = if let Some(gpath) = gpath {
                    try_read_to_string(&gpath)
                } else {
                    None
                };

                return Ok(Self {
                    vertex: vs,
                    geometry,
                    fragment: fs,
                });
            }
        }

        // Fall back to embedded shaders
        println!("Loading embedded shaders for preset: {}", preset.name);

        let vertex = Self::load_embedded_shader(&preset.vertex)?;
        let fragment = Self::load_embedded_shader(&preset.fragment)?;
        let geometry = if let Some(ref geom_path) = preset.geometry {
            Some(Self::load_embedded_shader(geom_path)?)
        } else {
            None
        };

        Ok(Self {
            vertex,
            geometry,
            fragment,
        })
    }

    fn load_embedded_shader(filename: &str) -> Result<String> {
        use std::path::Path;

        let shader_path = Path::new("shaders").join(filename);
        match fs::read_to_string(&shader_path) {
            Ok(content) => Ok(content),
            Err(e) => Err(anyhow!("Failed to load shader '{}': {}", shader_path.display(), e)),
        }
    }

    pub fn load_from_files(vertex_path: &str, fragment_path: &str) -> Result<Self> {
        Ok(Self {
            vertex: fs::read_to_string(vertex_path)?,
            geometry: None,
            fragment: fs::read_to_string(fragment_path)?,
        })
    }

    pub fn load_from_files_with_geometry(vertex_path: &str, geometry_path: &str, fragment_path: &str) -> Result<Self> {
        Ok(Self {
            vertex: fs::read_to_string(vertex_path)?,
            geometry: Some(fs::read_to_string(geometry_path)?),
            fragment: fs::read_to_string(fragment_path)?,
        })
    }

    pub fn load_from_config(config: &ShaderConfig) -> Result<Self> {
        let preset = config.presets.get(&config.active_preset)
            .ok_or_else(|| anyhow!("Shader preset '{}' not found in config", config.active_preset))?;

        if !preset.enabled {
            return Err(anyhow!("Shader preset '{}' is disabled", config.active_preset));
        }

        Self::load_preset(preset)
    }
}