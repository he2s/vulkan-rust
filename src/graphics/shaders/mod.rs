use anyhow::{Result, anyhow};
// use std::collections::HashMap; // Unused
use crate::config::config::ShaderConfig;
// use crate::config::config::GeometryType; // Unused
use crate::graphics::GeometryMode;

pub mod compiler;
pub mod cache;

#[allow(dead_code)]
pub struct ShaderSources {
    pub vertex: String,
    pub geometry: Option<String>,
    pub fragment: String,
    pub compute: Option<String>,
}

#[allow(dead_code)]
impl ShaderSources {
    pub fn load_from_config(config: &ShaderConfig) -> Result<Self> {
        let preset = config.presets.get(&config.active_preset)
            .ok_or_else(|| anyhow!("Shader preset '{}' not found in config", config.active_preset))?;

        let vertex = std::fs::read_to_string(format!("shaders/{}", preset.vertex))
            .map_err(|e| anyhow!("Failed to read vertex shader '{}': {}", preset.vertex, e))?;

        let fragment = std::fs::read_to_string(format!("shaders/{}", preset.fragment))
            .map_err(|e| anyhow!("Failed to read fragment shader '{}': {}", preset.fragment, e))?;

        let geometry = preset.geometry.as_ref()
            .map(|path| {
                std::fs::read_to_string(format!("shaders/{}", path))
                    .map_err(|e| anyhow!("Failed to read geometry shader '{}': {}", path, e))
            })
            .transpose()?;

        let compute = if let Some(ref compute_path) = preset.compute {
            Some(std::fs::read_to_string(format!("shaders/{}", compute_path))
                .map_err(|e| anyhow!("Failed to read compute shader '{}': {}", compute_path, e))?)
        } else {
            None
        };

        Ok(Self {
            vertex,
            geometry,
            fragment,
            compute,
        })
    }

    pub fn determine_geometry_mode(&self) -> GeometryMode {
        if self.compute.is_some() {
            GeometryMode::ComputeGenerated
        } else if self.geometry.is_some() {
            GeometryMode::GeometryShader
        } else if self.vertex.contains("gl_InstanceIndex") {
            GeometryMode::InstancedTriangles
        } else {
            GeometryMode::Trivial
        }
    }
}