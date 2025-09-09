use serde::{Deserialize, Serialize};
use crate::default_true;

#[derive(Deserialize, Serialize)]
pub struct MidiConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub auto_connect: bool,
    #[serde(default)]
    pub port_name: Option<String>,
}

impl Default for MidiConfig {
    fn default() -> Self {
        Self {
            enabled: default_true(),
            auto_connect: true,
            port_name: None,
        }
    }
}
