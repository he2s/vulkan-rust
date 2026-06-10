use anyhow::{Result, anyhow};
use std::cell::RefCell;
use std::fs;
use crate::config::config::{ShaderConfig, ShaderPreset, GeometryType};
use crate::graphics::GeometryMode;

thread_local! {
    static SHADER_COMPILER: RefCell<Option<shaderc::Compiler>> = const { RefCell::new(None) };
}

pub fn compile_to_spirv(source: &str, kind: shaderc::ShaderKind, filename: &str) -> Result<Vec<u32>> {
    SHADER_COMPILER.with(|cell| {
        let mut slot = cell.borrow_mut();
        let compiler = slot.get_or_insert_with(|| {
            shaderc::Compiler::new().expect("Failed to create shader compiler")
        });
        let result = compiler
            .compile_into_spirv(source, kind, filename, "main", None)
            .map_err(|e| anyhow!("Shader compilation failed: {}", e))?;
        Ok(result.as_binary().to_vec())
    })
}

pub struct ShaderSources {
    pub vertex: String,
    pub geometry: Option<String>,
    pub fragment: String,
}

fn try_read_to_string<P: AsRef<std::path::Path>>(p: P) -> Option<String> {
    std::fs::read_to_string(p).ok()
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

pub fn geometry_mode_from_config(config: &ShaderConfig) -> Result<GeometryMode> {
    let preset = config.presets.get(&config.active_preset)
        .ok_or_else(|| anyhow!("Active preset '{}' not found", config.active_preset))?;
    Ok(geometry_mode_from_preset(preset))
}

impl ShaderSources {

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

    pub fn load_from_config(config: &ShaderConfig) -> Result<Self> {
        let preset = config.presets.get(&config.active_preset)
            .ok_or_else(|| anyhow!("Shader preset '{}' not found in config", config.active_preset))?;

        if !preset.enabled {
            return Err(anyhow!("Shader preset '{}' is disabled", config.active_preset));
        }

        Self::load_preset(preset)
    }
}