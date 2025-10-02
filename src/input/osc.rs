use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::{
    net::{SocketAddr, UdpSocket},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU32, Ordering},
    },
    thread::{self, JoinHandle},
    time::Duration,
};

const OSC_BUNDLE_TAG: &[u8] = b"#bundle\0";
const MAX_PACKET_SIZE: usize = 1024; // Reduced from potential 65KB UDP max
const CHANNEL_COUNT: usize = 2;

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
    #[serde(default = "default_buffer_size")]
    pub socket_buffer_size: usize,
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
            socket_buffer_size: default_buffer_size(),
        }
    }
}

const fn default_osc_port() -> u16 {
    7000
}
fn default_osc_address() -> String {
    "0.0.0.0".to_string()
}
fn default_channel1_path() -> String {
    "/3/xy/x".to_string()
}
fn default_channel2_path() -> String {
    "/3/xy/y".to_string()
}
const fn default_smoothing() -> f32 {
    0.1
}
const fn default_buffer_size() -> usize {
    64 * 1024
} // 64KB socket buffer

#[derive(Debug)]
pub struct OscState {
    channel1: AtomicU32,
    channel2: AtomicU32,
}

impl Default for OscState {
    fn default() -> Self {
        Self {
            channel1: AtomicU32::new(0.0f32.to_bits()),
            channel2: AtomicU32::new(0.0f32.to_bits()),
        }
    }
}

impl OscState {
    #[inline(always)]
    pub fn get_channel1(&self) -> f32 {
        f32::from_bits(self.channel1.load(Ordering::Relaxed))
    }

    #[inline(always)]
    pub fn get_channel2(&self) -> f32 {
        f32::from_bits(self.channel2.load(Ordering::Relaxed))
    }

    #[inline(always)]
    pub fn set_channel1(&self, value: f32) {
        self.channel1.store(value.to_bits(), Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn set_channel2(&self, value: f32) {
        self.channel2.store(value.to_bits(), Ordering::Relaxed);
    }

    // For compatibility with existing code
    pub fn clone_values(&self) -> OscStateSnapshot {
        OscStateSnapshot {
            channel1: self.get_channel1(),
            channel2: self.get_channel2(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct OscStateSnapshot {
    pub channel1: f32,
    pub channel2: f32,
}

#[derive(Clone)]
struct ChannelInfo {
    address_bytes: Vec<u8>,
    channel_index: u8,
}

pub struct OscManager {
    state: Arc<OscState>,
    config: OscConfig,
    _server_thread: Option<JoinHandle<()>>,
    should_stop: Arc<AtomicBool>,
    // Pre-computed channel lookup for hot path
    channel_lookup: Vec<ChannelInfo>,
}

impl OscManager {
    pub fn new(config: OscConfig) -> Self {
        let mut channel_lookup = Vec::with_capacity(CHANNEL_COUNT);

        channel_lookup.push(ChannelInfo {
            address_bytes: config.channel1_path.as_bytes().to_vec(),
            channel_index: 0,
        });
        channel_lookup.push(ChannelInfo {
            address_bytes: config.channel2_path.as_bytes().to_vec(),
            channel_index: 1,
        });

        Self {
            state: Arc::new(OscState::default()),
            config,
            _server_thread: None,
            should_stop: Arc::new(AtomicBool::new(false)),
            channel_lookup,
        }
    }

    pub fn start(&mut self) -> Result<()> {
        if !self.config.enabled {
            println!("OSC disabled in configuration");
            return Ok(());
        }

        let bind_addr = format!("{}:{}", self.config.bind_address, self.config.port);
        let socket_addr: SocketAddr = bind_addr
            .parse()
            .map_err(|e| anyhow!("Invalid OSC bind address '{}': {}", bind_addr, e))?;

        let socket = UdpSocket::bind(socket_addr)
            .map_err(|e| anyhow!("Failed to bind OSC socket to {}: {}", socket_addr, e))?;

        if let Err(e) = socket.set_read_timeout(Some(Duration::from_millis(1))) {
            eprintln!("Warning: Failed to set socket timeout: {e}");
        }

        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            let fd = socket.as_raw_fd();
            unsafe {
                let buf_size = self.config.socket_buffer_size as libc::c_int;
                libc::setsockopt(
                    fd,
                    libc::SOL_SOCKET,
                    libc::SO_RCVBUF,
                    &buf_size as *const _ as *const libc::c_void,
                    std::mem::size_of::<libc::c_int>() as libc::socklen_t,
                );
            }
        }

        println!("OSC server listening on {socket_addr}");
        println!(
            "OSC channels: {} -> channel1, {} -> channel2",
            self.config.channel1_path, self.config.channel2_path
        );

        let state = Arc::clone(&self.state);
        let config = self.config.clone();
        let should_stop = Arc::clone(&self.should_stop);
        let channel_lookup = self.channel_lookup.clone();

        let server_thread = thread::spawn(move || {
            Self::optimized_server_loop(socket, state, config, should_stop, channel_lookup);
        });

        self._server_thread = Some(server_thread);
        Ok(())
    }

    pub fn get_state(&self) -> OscStateSnapshot {
        self.state.clone_values()
    }

    fn optimized_server_loop(
        socket: UdpSocket,
        state: Arc<OscState>,
        config: OscConfig,
        should_stop: Arc<AtomicBool>,
        channel_lookup: Vec<ChannelInfo>,
    ) {
        let mut buffer = [0u8; MAX_PACKET_SIZE];
        let mut parser = FastOscParser::new();

        let smoothing = config.smoothing_factor;
        let inv_smoothing = 1.0 - smoothing;

        while !should_stop.load(Ordering::Relaxed) {
            match socket.recv(&mut buffer) {
                Ok(size) => {
                    if size > 0 && !buffer[..8.min(size)].starts_with(OSC_BUNDLE_TAG) {
                        if let Some((channel_idx, value)) =
                            parser.fast_parse_single_message(&buffer[..size], &channel_lookup)
                        {
                            match channel_idx {
                                0 => {
                                    let current = state.get_channel1();
                                    let new_value = if smoothing <= 0.0 {
                                        value
                                    } else {
                                        current * smoothing + value * inv_smoothing
                                    };
                                    state.set_channel1(new_value);
                                }
                                1 => {
                                    let current = state.get_channel2();
                                    let new_value = if smoothing <= 0.0 {
                                        value
                                    } else {
                                        current * smoothing + value * inv_smoothing
                                    };
                                    state.set_channel2(new_value);
                                }
                                _ => {}
                            }
                        }
                    } else if size > 16 {
                        // Handle bundles with the slower path
                        let _ = Self::process_osc_bundle(
                            &buffer[..size],
                            &state,
                            &config,
                            &channel_lookup,
                        );
                    }
                }
                Err(e) => {
                    //println!("{:?}", e);
                    if e.kind() != std::io::ErrorKind::TimedOut
                        && e.kind() != std::io::ErrorKind::WouldBlock
                    {
                        eprintln!("OSC socket error: {e}");
                        break;
                    }
                }
            }
        }

        println!("OSC server stopped");
    }

    fn process_osc_bundle(
        data: &[u8],
        state: &Arc<OscState>,
        config: &OscConfig,
        channel_lookup: &[ChannelInfo],
    ) -> Result<()> {
        if data.len() < 16 {
            return Ok(());
        }

        let mut offset = 16;
        let mut parser = FastOscParser::new();

        while offset + 4 <= data.len() {
            let element_size = u32::from_be_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]) as usize;

            offset += 4;

            if offset + element_size > data.len() {
                break;
            }

            if let Some((channel_idx, value)) = parser
                .fast_parse_single_message(&data[offset..offset + element_size], channel_lookup)
            {
                Self::update_channel_atomic(state, channel_idx, value, config.smoothing_factor);
            }

            offset += element_size;
        }

        Ok(())
    }

    #[inline(always)]
    fn update_channel_atomic(state: &Arc<OscState>, channel_idx: u8, value: f32, smoothing: f32) {
        match channel_idx {
            0 => {
                let current = state.get_channel1();
                let new_value = if smoothing <= 0.0 {
                    value
                } else {
                    current * smoothing + value * (1.0 - smoothing)
                };
                state.set_channel1(new_value);
            }
            1 => {
                let current = state.get_channel2();
                let new_value = if smoothing <= 0.0 {
                    value
                } else {
                    current * smoothing + value * (1.0 - smoothing)
                };
                state.set_channel2(new_value);
            }
            _ => {}
        }
    }

    pub fn stop(&mut self) {
        self.should_stop.store(true, Ordering::Release);
        if let Some(thread) = self._server_thread.take() {
            let _ = thread.join();
        }
    }
}

struct FastOscParser {
    #[allow(dead_code)]
    workspace: Vec<u8>,
}

impl FastOscParser {
    fn new() -> Self {
        Self {
            workspace: Vec::with_capacity(256),
        }
    }

    fn fast_parse_single_message(
        &mut self,
        data: &[u8],
        channel_lookup: &[ChannelInfo],
    ) -> Option<(u8, f32)> {
        if data.len() < 8 {
            return None;
        }

        let addr_end = data.iter().position(|&b| b == 0)?;
        let address_bytes = &data[..addr_end];

        let channel_idx = channel_lookup
            .iter()
            .find(|info| info.address_bytes == address_bytes)?
            .channel_index;

        let mut offset = ((addr_end + 4) / 4) * 4;
        if offset >= data.len() {
            return None;
        }

        if offset + 4 > data.len() || data[offset] != b',' || data[offset + 1] != b'f' {
            return None;
        }

        offset = ((offset + 2 + 4) / 4) * 4;
        if offset + 4 > data.len() {
            return None;
        }

        let bytes = [
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ];
        let value = f32::from_be_bytes(bytes);

        Some((channel_idx, value))
    }
}

impl Drop for OscManager {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::UdpSocket;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_atomic_state_operations() {
        let state = OscState::default();

        state.set_channel1(1.5);
        state.set_channel2(2.5);

        assert_eq!(state.get_channel1(), 1.5);
        assert_eq!(state.get_channel2(), 2.5);
    }

    #[test]
    fn test_fast_osc_parser() {
        let mut parser = FastOscParser::new();

        let test_data = [
            0x2F, 0x74, 0x65, 0x73, 0x74, 0x00, 0x00, 0x00, // "/test\0\0\0"
            0x2C, 0x66, 0x00, 0x00, // ",f\0\0"
            0x3F, 0xC0, 0x00, 0x00, // 1.5 as big-endian float
        ];

        let channel_lookup = vec![ChannelInfo {
            address_bytes: b"/test".to_vec(),
            channel_index: 0,
        }];

        let result = parser.fast_parse_single_message(&test_data, &channel_lookup);
        assert_eq!(result, Some((0, 1.5)));
    }

    pub fn send_osc_message(target: &str, address: &str, value: f32) -> Result<()> {
        let socket = UdpSocket::bind("127.0.0.1:0")?;

        let mut message = Vec::new();

        message.extend_from_slice(address.as_bytes());
        message.push(0); // null terminator
        while message.len() % 4 != 0 {
            message.push(0); // pad to 4-byte boundary
        }

        message.extend_from_slice(b",f");
        while message.len() % 4 != 0 {
            message.push(0);
        }

        message.extend_from_slice(&value.to_be_bytes());

        socket.send_to(&message, target)?;
        Ok(())
    }
}
