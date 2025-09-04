use std::{
    net::{UdpSocket, SocketAddr},
    sync::{Arc, Mutex},
    thread::{self, JoinHandle},
    time::Duration,
};
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

// OSC message parsing constants
const OSC_BUNDLE_TAG: &[u8] = b"#bundle\0";

/// Configuration for OSC input
#[derive(Deserialize, Serialize, Clone)]
pub struct OscConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_osc_port")]
    pub port: u16,
    #[serde(default = "default_osc_address")]
    pub bind_address: String,
    #[serde(default = "default_channel1_path")]
    pub channel1_path: String,
    #[serde(default = "default_channel2_path")]
    pub channel2_path: String,
    #[serde(default = "default_smoothing")]
    pub smoothing_factor: f32,
}

impl Default for OscConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            port: default_osc_port(),
            bind_address: default_osc_address(),
            channel1_path: default_channel1_path(),
            channel2_path: default_channel2_path(),
            smoothing_factor: default_smoothing(),
        }
    }
}

const fn default_osc_port() -> u16 { 7000 }
fn default_osc_address() -> String { "127.0.0.1".to_string() }
fn default_channel1_path() -> String { "/ch1".to_string() }
fn default_channel2_path() -> String { "/ch2".to_string() }
const fn default_smoothing() -> f32 { 0.1 }

/// OSC state containing two float channels
#[derive(Clone, Debug)]
pub struct OscState {
    pub channel1: f32,
    pub channel2: f32,
}

impl Default for OscState {
    fn default() -> Self {
        Self {
            channel1: 0.0,
            channel2: 0.0,
        }
    }
}

/// OSC message representation
#[derive(Debug)]
struct OscMessage {
    address: String,
    args: Vec<OscArgument>,
}

#[derive(Debug)]
enum OscArgument {
    Float(f32),
    Int(i32),
    String(String),
}

/// Main OSC manager struct
pub struct OscManager {
    state: Arc<Mutex<OscState>>,
    config: OscConfig,
    _server_thread: Option<JoinHandle<()>>,
    should_stop: Arc<Mutex<bool>>,
}

impl OscManager {
    pub fn new(config: OscConfig) -> Self {
        Self {
            state: Arc::new(Mutex::new(OscState::default())),
            config,
            _server_thread: None,
            should_stop: Arc::new(Mutex::new(false)),
        }
    }

    /// Start the OSC server
    pub fn start(&mut self) -> Result<()> {
        if !self.config.enabled {
            println!("OSC disabled in configuration");
            return Ok(());
        }

        let bind_addr = format!("{}:{}", self.config.bind_address, self.config.port);
        let socket_addr: SocketAddr = bind_addr.parse()
            .map_err(|e| anyhow!("Invalid OSC bind address '{}': {}", bind_addr, e))?;

        let socket = UdpSocket::bind(socket_addr)
            .map_err(|e| anyhow!("Failed to bind OSC socket to {}: {}", socket_addr, e))?;

        socket.set_read_timeout(Some(Duration::from_millis(100)))
            .map_err(|e| anyhow!("Failed to set socket timeout: {}", e))?;

        println!("OSC server listening on {}", socket_addr);
        println!("OSC channels: {} -> channel1, {} -> channel2",
                 self.config.channel1_path, self.config.channel2_path);

        let state = Arc::clone(&self.state);
        let config = self.config.clone();
        let should_stop = Arc::clone(&self.should_stop);

        let server_thread = thread::spawn(move || {
            Self::server_loop(socket, state, config, should_stop);
        });

        self._server_thread = Some(server_thread);
        Ok(())
    }

    /// Get current OSC state
    pub fn get_state(&self) -> OscState {
        self.state.lock().unwrap().clone()
    }

    /// Main server loop
    fn server_loop(
        socket: UdpSocket,
        state: Arc<Mutex<OscState>>,
        config: OscConfig,
        should_stop: Arc<Mutex<bool>>,
    ) {
        let mut buffer = [0u8; 1024];

        while !*should_stop.lock().unwrap() {
            match socket.recv(&mut buffer) {
                Ok(size) => {
                    if let Err(e) = Self::process_osc_packet(&buffer[..size], &state, &config) {
                        eprintln!("OSC processing error: {}", e);
                    }
                }
                Err(e) => {
                    // Timeout is expected, other errors are not
                    if e.kind() != std::io::ErrorKind::TimedOut {
                        eprintln!("OSC socket error: {}", e);
                        break;
                    }
                }
            }
        }

        println!("OSC server stopped");
    }

    /// Process incoming OSC packet
    fn process_osc_packet(
        data: &[u8],
        state: &Arc<Mutex<OscState>>,
        config: &OscConfig,
    ) -> Result<()> {
        if data.len() < 4 {
            return Err(anyhow!("OSC packet too short"));
        }

        // Check if it's a bundle
        if data.starts_with(OSC_BUNDLE_TAG) {
            Self::process_osc_bundle(data, state, config)
        } else {
            // Single message
            let message = Self::parse_osc_message(data)?;
            Self::handle_osc_message(&message, state, config);
            Ok(())
        }
    }

    /// Process OSC bundle (multiple messages)
    fn process_osc_bundle(
        data: &[u8],
        state: &Arc<Mutex<OscState>>,
        config: &OscConfig,
    ) -> Result<()> {
        if data.len() < 16 {
            return Err(anyhow!("OSC bundle too short"));
        }

        // Skip bundle tag and time tag (8 bytes each)
        let mut offset = 16;

        while offset < data.len() {
            if offset + 4 > data.len() {
                break;
            }

            let element_size = u32::from_be_bytes([
                data[offset], data[offset + 1], data[offset + 2], data[offset + 3]
            ]) as usize;

            offset += 4;

            if offset + element_size > data.len() {
                return Err(anyhow!("Invalid bundle element size"));
            }

            let message = Self::parse_osc_message(&data[offset..offset + element_size])?;
            Self::handle_osc_message(&message, state, config);

            offset += element_size;
        }

        Ok(())
    }

    /// Parse a single OSC message
    fn parse_osc_message(data: &[u8]) -> Result<OscMessage> {
        let mut offset = 0;

        // Parse address
        let address = Self::read_osc_string(data, &mut offset)?;

        // Parse type tag
        let type_tag = Self::read_osc_string(data, &mut offset)?;

        if !type_tag.starts_with(',') {
            return Err(anyhow!("Invalid OSC type tag: {}", type_tag));
        }

        // Parse arguments
        let mut args = Vec::new();
        for type_char in type_tag.chars().skip(1) {
            match type_char {
                'f' => {
                    if offset + 4 > data.len() {
                        return Err(anyhow!("Not enough data for float argument"));
                    }
                    let bytes = [data[offset], data[offset + 1], data[offset + 2], data[offset + 3]];
                    let value = f32::from_be_bytes(bytes);
                    args.push(OscArgument::Float(value));
                    offset += 4;
                }
                'i' => {
                    if offset + 4 > data.len() {
                        return Err(anyhow!("Not enough data for int argument"));
                    }
                    let bytes = [data[offset], data[offset + 1], data[offset + 2], data[offset + 3]];
                    let value = i32::from_be_bytes(bytes);
                    args.push(OscArgument::Int(value));
                    offset += 4;
                }
                's' => {
                    let value = Self::read_osc_string(data, &mut offset)?;
                    args.push(OscArgument::String(value));
                }
                _ => {
                    // Skip unknown argument types
                    eprintln!("Unsupported OSC argument type: {}", type_char);
                }
            }
        }

        Ok(OscMessage { address, args })
    }

    /// Read null-terminated string from OSC data with proper padding
    fn read_osc_string(data: &[u8], offset: &mut usize) -> Result<String> {
        if *offset >= data.len() {
            return Err(anyhow!("Offset beyond data length"));
        }

        let start = *offset;
        let mut end = start;

        // Find null terminator
        while end < data.len() && data[end] != 0 {
            end += 1;
        }

        if end >= data.len() {
            return Err(anyhow!("No null terminator found"));
        }

        let string = String::from_utf8(data[start..end].to_vec())
            .map_err(|e| anyhow!("Invalid UTF-8 in OSC string: {}", e))?;

        // OSC strings are padded to 4-byte boundaries
        *offset = ((end + 4) / 4) * 4;

        Ok(string)
    }

    /// Handle parsed OSC message
    fn handle_osc_message(
        message: &OscMessage,
        state: &Arc<Mutex<OscState>>,
        config: &OscConfig,
    ) {
        // Extract float value from message
        let value = match message.args.first() {
            Some(OscArgument::Float(f)) => *f,
            Some(OscArgument::Int(i)) => *i as f32,
            _ => {
                eprintln!("OSC message {} has no numeric argument", message.address);
                return;
            }
        };

        // Update appropriate channel
        if let Ok(mut state) = state.lock() {
            let smoothing = config.smoothing_factor;

            if message.address == config.channel1_path {
                state.channel1 = Self::smooth_value(state.channel1, value, smoothing);
                println!("OSC CH1: {:.3}", state.channel1);
            } else if message.address == config.channel2_path {
                state.channel2 = Self::smooth_value(state.channel2, value, smoothing);
                println!("OSC CH2: {:.3}", state.channel2);
            }
        }
    }

    /// Apply smoothing to value changes
    fn smooth_value(current: f32, target: f32, factor: f32) -> f32 {
        if factor <= 0.0 {
            target
        } else if factor >= 1.0 {
            current
        } else {
            current * (1.0 - factor) + target * factor
        }
    }

    /// Stop the OSC server
    pub fn stop(&mut self) {
        *self.should_stop.lock().unwrap() = true;
        if let Some(thread) = self._server_thread.take() {
            let _ = thread.join();
        }
    }
}

impl Drop for OscManager {
    fn drop(&mut self) {
        self.stop();
    }
}

// Test client functionality
#[cfg(test)]
mod tests {
    use super::*;
    use std::net::UdpSocket;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_osc_message_parsing() {
        // Test data for "/test" with float argument 1.5
        let test_data = [
            0x2F, 0x74, 0x65, 0x73, 0x74, 0x00, 0x00, 0x00, // "/test\0\0\0"
            0x2C, 0x66, 0x00, 0x00,                         // ",f\0\0"
            0x3F, 0xC0, 0x00, 0x00,                         // 1.5 as big-endian float
        ];

        let message = OscManager::parse_osc_message(&test_data).unwrap();
        assert_eq!(message.address, "/test");
        assert_eq!(message.args.len(), 1);

        if let OscArgument::Float(f) = &message.args[0] {
            assert_eq!(*f, 1.5);
        } else {
            panic!("Expected float argument");
        }
    }

    /// Helper function to create a simple OSC test client
    pub fn send_osc_message(target: &str, address: &str, value: f32) -> Result<()> {
        let socket = UdpSocket::bind("127.0.0.1:0")?;

        // Simple OSC message construction
        let mut message = Vec::new();

        // Address
        message.extend_from_slice(address.as_bytes());
        message.push(0); // null terminator
        while message.len() % 4 != 0 {
            message.push(0); // pad to 4-byte boundary
        }

        // Type tag
        message.extend_from_slice(b",f");
        while message.len() % 4 != 0 {
            message.push(0);
        }

        // Float argument
        message.extend_from_slice(&value.to_be_bytes());

        socket.send_to(&message, target)?;
        Ok(())
    }
}