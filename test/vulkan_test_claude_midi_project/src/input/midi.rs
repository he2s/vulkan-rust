// src/input/midi.rs

use anyhow::{anyhow, Result};
use crossbeam_channel::Sender;
use midir::{MidiInput as MidiInputPort, MidiInputConnection, Ignore};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, debug};

use crate::config::MidiConfig;
use super::InputEvent;

pub struct MidiInput {
    _connection: MidiInputConnection<()>,
}

impl MidiInput {
    pub fn new(
        config: &MidiConfig,
        state: Arc<RwLock<MidiState>>,
        event_sender: Sender<InputEvent>,
    ) -> Result<Self> {
        let mut midi_in = MidiInputPort::new("Vulkan Visualizer MIDI")?;
        midi_in.ignore(Ignore::None);

        let ports = midi_in.ports();
        if ports.is_empty() {
            return Err(anyhow!("No MIDI input ports available"));
        }

        // Select port
        let port = if let Some(ref name) = config.port_name {
            ports
                .iter()
                .find(|p| {
                    midi_in
                        .port_name(p)
                        .map(|n| n.contains(name))
                        .unwrap_or(false)
                })
                .ok_or_else(|| anyhow!("MIDI port '{}' not found", name))?
        } else if config.auto_connect {
            &ports[0]
        } else {
            return Err(anyhow!("No MIDI port specified and auto-connect disabled"));
        };

        let port_name = midi_in.port_name(port)?;
        info!("Connecting to MIDI port: {}", port_name);

        let connection = midi_in.connect(
            port,
            "vulkan-visualizer",
            move |_timestamp, message, _| {
                Self::handle_message(message, &state, &event_sender);
            },
            (),
        )?;

        Ok(Self {
            _connection: connection,
        })
    }

    fn handle_message(
        message: &[u8],
        state: &Arc<RwLock<MidiState>>,
        event_sender: &Sender<InputEvent>,
    ) {
        if message.len() < 2 {
            return;
        }

        let status = message[0] & 0xF0;
        let channel = message[0] & 0x0F;

        // Use tokio's block_in_place or spawn_blocking for async in sync context
        let state_clone = Arc::clone(state);
        let event = match status {
            0x80 | 0x90 => {
                // Note Off / Note On
                if message.len() >= 3 {
                    let note = message[1];
                    let velocity = message[2];

                    if velocity == 0 || status == 0x80 {
                        // Note Off
                        tokio::task::block_in_place(|| {
                            tokio::runtime::Handle::current().block_on(async {
                                let mut s = state_clone.write().await;
                                s.note_off(note);
                            })
                        });
                        Some(InputEvent::MidiNoteOff { note })
                    } else {
                        // Note On
                        tokio::task::block_in_place(|| {
                            tokio::runtime::Handle::current().block_on(async {
                                let mut s = state_clone.write().await;
                                s.note_on(note, velocity);
                            })
                        });
                        Some(InputEvent::MidiNoteOn { note, velocity })
                    }
                } else {
                    None
                }
            }
            0xB0 => {
                // Control Change
                if message.len() >= 3 {
                    let controller = message[1];
                    let value = message[2];

                    tokio::task::block_in_place(|| {
                        tokio::runtime::Handle::current().block_on(async {
                            let mut s = state_clone.write().await;
                            s.control_change(controller, value);
                        })
                    });

                    Some(InputEvent::MidiControlChange { controller, value })
                } else {
                    None
                }
            }
            0xE0 => {
                // Pitch Bend
                if message.len() >= 3 {
                    let lsb = message[1];
                    let msb = message[2];
                    let value = ((msb as u16) << 7) | (lsb as u16);
                    let normalized = (value as f32 / 8192.0) - 1.0;

                    tokio::task::block_in_place(|| {
                        tokio::runtime::Handle::current().block_on(async {
                            let mut s = state_clone.write().await;
                            s.pitch_bend = normalized;
                        })
                    });

                    Some(InputEvent::MidiPitchBend { value: normalized })
                } else {
                    None
                }
            }
            _ => {
                debug!("Unhandled MIDI message: {:?}", message);
                None
            }
        };

        if let Some(event) = event {
            let _ = event_sender.send(event);
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct MidiState {
    pub notes: [f32; 128],
    pub controllers: [f32; 128],
    pub pitch_bend: f32,
    pub last_note: u8,
    pub note_count: u32,
    pub channel_pressure: f32,
    pub program: u8,
}

impl MidiState {
    pub fn note_on(&mut self, note: u8, velocity: u8) {
        let idx = note as usize;
        if idx < 128 {
            if self.notes[idx] == 0.0 {
                self.note_count += 1;
            }
            self.notes[idx] = velocity as f32 / 127.0;
            self.last_note = note;

            debug!("Note ON: {} vel={} (total={})", note, velocity, self.note_count);
        }
    }

    pub fn note_off(&mut self, note: u8) {
        let idx = note as usize;
        if idx < 128 {
            if self.notes[idx] > 0.0 {
                self.note_count = self.note_count.saturating_sub(1);
            }
            self.notes[idx] = 0.0;

            debug!("Note OFF: {} (total={})", note, self.note_count);
        }
    }

    pub fn control_change(&mut self, controller: u8, value: u8) {
        let idx = controller as usize;
        if idx < 128 {
            self.controllers[idx] = value as f32 / 127.0;
            debug!("CC{}: {}", controller, value);
        }
    }

    pub fn current_velocity(&self) -> f32 {
        if self.note_count > 0 && self.last_note < 128 {
            self.notes[self.last_note as usize]
        } else {
            0.0
        }
    }

    pub fn reset(&mut self) {
        self.notes = [0.0; 128];
        self.controllers = [0.5; 128];
        self.pitch_bend = 0.0;
        self.note_count = 0;
    }
}