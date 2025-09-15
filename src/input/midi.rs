use serde::{Deserialize, Serialize};
use anyhow::Result;
use midir::{MidiInput, MidiInputConnection, Ignore};
use std::sync::{Arc, Mutex};

// Constants
const MAX_NOTES: usize = 128;
const MAX_CONTROLLERS: usize = 128;

#[derive(Deserialize, Serialize)]
pub struct MidiConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub auto_connect: bool,
    #[serde(default)]
    pub port_name: Option<String>,
}

impl Default for MidiConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            auto_connect: true,
            port_name: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct MidiState {
    pub notes: [f32; MAX_NOTES],
    pub controllers: [f32; MAX_CONTROLLERS],
    pub pitch_bend: f32,
    pub last_note: u8,
    pub note_count: u32,
}

impl Default for MidiState {
    fn default() -> Self {
        Self {
            notes: [0.0; MAX_NOTES],
            controllers: [0.5; MAX_CONTROLLERS],
            pitch_bend: 0.0,
            last_note: 60,
            note_count: 0,
        }
    }
}

pub struct MidiManager {
    state: Arc<Mutex<MidiState>>,
    connection: Option<MidiInputConnection<()>>,
}

impl MidiManager {
    pub fn new() -> Self {
        Self {
            state: Arc::new(Mutex::new(MidiState::default())),
            connection: None,
        }
    }

    pub fn get_state(&self) -> Arc<Mutex<MidiState>> {
        Arc::clone(&self.state)
    }

    pub fn get_state_snapshot(&self) -> MidiState {
        self.state.lock().unwrap().clone()
    }

    pub fn setup(&mut self, config: &MidiConfig) {
        if !config.enabled {
            println!("MIDI disabled in configuration");
            return;
        }

        match self.try_connect(config) {
            Ok(connection) => {
                self.connection = Some(connection);
                println!("MIDI input connected successfully!");
                //Ok(())
            }
            Err(e) => {
                eprintln!("MIDI setup failed: {}. Continuing without MIDI input.", e);
                //Err(e.into())
            }
        }
    }

    fn try_connect(&self, config: &MidiConfig) -> Result<MidiInputConnection<()>, Box<dyn std::error::Error>> {
        let mut midi_in = MidiInput::new("Vulkan MIDI Visualizer")?;
        midi_in.ignore(Ignore::None);

        let ports = midi_in.ports();
        if ports.is_empty() {
            return Err("No MIDI input ports available".into());
        }

        let selected_port = Self::select_port(&midi_in, &ports, config)?;
        let port_name = midi_in.port_name(selected_port)?;
        println!("Connecting to MIDI port: {}", port_name);

        let state = Arc::clone(&self.state);
        let connection = midi_in.connect(
            selected_port,
            "vulkan-visualizer",
            move |_timestamp, message, _| {
                Self::handle_message(&state, message);
            },
            (),
        )?;

        Ok(connection)
    }

    fn select_port<'a>(
        midi_in: &MidiInput,
        ports: &'a [midir::MidiInputPort],
        config: &MidiConfig,
    ) -> Result<&'a midir::MidiInputPort, Box<dyn std::error::Error>> {
        if let Some(ref target_name) = config.port_name {
            ports
                .iter()
                .find(|port| {
                    midi_in
                        .port_name(port)
                        .map_or(false, |name| name.contains(target_name))
                })
                .or_else(|| ports.first())
                .ok_or_else(|| "No suitable MIDI port found".into())
        } else {
            ports.first().ok_or_else(|| "No MIDI ports available".into())
        }
    }

    fn handle_message(state: &Arc<Mutex<MidiState>>, message: &[u8]) {
        if message.is_empty() {
            return;
        }

        let Ok(mut state) = state.lock() else {
            return;
        };

        let status = message[0];
        let channel = status & 0x0F;
        let message_type = status & 0xF0;

        match message_type {
            0x80 => Self::handle_note_off(&mut state, message),
            0x90 => Self::handle_note_on(&mut state, message),
            0xB0 => Self::handle_control_change(&mut state, message),
            0xE0 => Self::handle_pitch_bend(&mut state, message),
            _ => {}
        }
    }

    fn handle_note_off(state: &mut MidiState, message: &[u8]) {
        if message.len() >= 3 {
            let note = message[1] as usize;
            if note < MAX_NOTES && state.notes[note] > 0.0 {
                state.note_count = state.note_count.saturating_sub(1);
                state.notes[note] = 0.0;
                println!("Note Off: {} (Count: {})", note, state.note_count);
            }
        }
    }

    fn handle_note_on(state: &mut MidiState, message: &[u8]) {
        if message.len() >= 3 {
            let note = message[1] as usize;
            let velocity = message[2];

            if note < MAX_NOTES {
                if velocity == 0 {
                    // Note off via velocity 0
                    if state.notes[note] > 0.0 {
                        state.note_count = state.note_count.saturating_sub(1);
                    }
                    state.notes[note] = 0.0;
                    println!("Note Off: {} (Count: {})", note, state.note_count);
                } else {
                    // Note on
                    if state.notes[note] == 0.0 {
                        state.note_count += 1;
                    }
                    state.notes[note] = velocity as f32 / 127.0;
                    state.last_note = note as u8;
                    println!("Note On: {} Velocity: {} (Count: {})", note, velocity, state.note_count);
                }
            }
        }
    }

    fn handle_control_change(state: &mut MidiState, message: &[u8]) {
        if message.len() >= 3 {
            let controller = message[1] as usize;
            let value = message[2];

            if controller < MAX_CONTROLLERS {
                state.controllers[controller] = value as f32 / 127.0;
                println!("CC{}: {}", controller, value);
            }
        }
    }

    fn handle_pitch_bend(state: &mut MidiState, message: &[u8]) {
        if message.len() >= 3 {
            let bend_value = (message[2] as u16) << 7 | (message[1] as u16);
            state.pitch_bend = (bend_value as f32 / 8192.0) - 1.0;
            println!("Pitch Bend: {:.3}", state.pitch_bend);
        }
    }
}

impl Drop for MidiManager {
    fn drop(&mut self) {
        if let Some(connection) = self.connection.take() {
            connection.close();
        }
    }
}






/*

pub fn setup_midi(&mut self, config: &MidiConfig) {
    if !config.enabled {
        println!("MIDI disabled in configuration");
        return;
    }

    match self.try_setup_midi(config) {
        Ok(connection) => {
            self._midi_connection = Some(connection);
            println!("MIDI input connected successfully!");
            let pn = config.port_name.clone();
            println!("{}", pn.unwrap_or_else(|| "No port name specified".to_string()));
        }
        Err(e) => {
            eprintln!("MIDI setup failed: {}. Continuing without MIDI input.", e);
        }
    }
}

fn try_setup_midi(&self, config: &MidiConfig) -> Result<midir::MidiInputConnection<()>, Box<dyn std::error::Error>> {
    let mut midi_in = MidiInput::new("Vulkan MIDI Visualizer")?;
    midi_in.ignore(Ignore::None);

    let ports = midi_in.ports();
    if ports.is_empty() {
        return Err("No MIDI input ports available".into());
    }

    let selected_port = self.select_midi_port(&midi_in, &ports, config)?;
    let port_name = midi_in.port_name(selected_port)?;
    println!("Connecting to MIDI port: {}", port_name);

    let midi_state = Arc::clone(&self.midi_state);
    let connection = midi_in.connect(selected_port, "vulkan-visualizer", move |_timestamp, message, _| {
        Self::handle_midi_message(&midi_state, message);
    }, ())?;

    Ok(connection)
}

fn select_midi_port<'a>(
    &self,
    midi_in: &MidiInput,
    ports: &'a [midir::MidiInputPort],
    config: &MidiConfig,
) -> Result<&'a midir::MidiInputPort, Box<dyn std::error::Error>> {
    if let Some(ref target_name) = config.port_name {
        ports
            .iter()
            .find(|port| {
                midi_in.port_name(port)
                    .map_or(false, |name| name.contains(target_name))
            })
            .or_else(|| ports.first())
            .ok_or_else(|| "No suitable MIDI port found".into())
    } else {
        ports.first().ok_or_else(|| "No MIDI ports available".into())
    }
}
fn handle_midi_message(midi_state: &Arc<Mutex<MidiState>>, message: &[u8]) {
    if message.is_empty() {
        return;
    }

    let Ok(mut state) = midi_state.lock() else { return };
    let status = message[0];

    match status & 0xF0 {
        0x80 => Self::handle_note_off(&mut state, message),
        0x90 => Self::handle_note_on(&mut state, message),
        0xB0 => Self::handle_control_change(&mut state, message),
        0xE0 => Self::handle_pitch_bend(&mut state, message),
        _ => {}
    }
}

fn handle_note_off(state: &mut MidiState, message: &[u8]) {
    if message.len() >= 3 {
        let note = message[1] as usize;
        if note < crate::MAX_NOTES && state.notes[note] > 0.0 {
            state.note_count = state.note_count.saturating_sub(1);
            state.notes[note] = 0.0;
            println!("Note Off: {} (Count: {})", note, state.note_count);
        }
    }
}

fn handle_note_on(state: &mut MidiState, message: &[u8]) {
    if message.len() >= 3 {
        let note = message[1] as usize;
        let velocity = message[2];

        if note < crate::MAX_NOTES {
            if velocity == 0 {
                // Note off via velocity 0
                if state.notes[note] > 0.0 {
                    state.note_count = state.note_count.saturating_sub(1);
                }
                state.notes[note] = 0.0;
                println!("Note Off: {} (Count: {})", note, state.note_count);
            } else {
                // Note on
                if state.notes[note] == 0.0 {
                    state.note_count += 1;
                }
                state.notes[note] = velocity as f32 / 127.0;
                state.last_note = note as u8;
                println!("Note On: {} Velocity: {} (Count: {})", note, velocity, state.note_count);
            }
        }
    }
}

fn handle_control_change(state: &mut MidiState, message: &[u8]) {
    if message.len() >= 3 {
        let controller = message[1] as usize;
        let value = message[2];

        if controller < crate::MAX_CONTROLLERS {
            state.controllers[controller] = value as f32 / 127.0;
            println!("CC{}: {}", controller, value);
        }
    }
}

fn handle_pitch_bend(state: &mut MidiState, message: &[u8]) {
    if message.len() >= 3 {
        let bend_value = (message[2] as u16) << 7 | (message[1] as u16);
        state.pitch_bend = (bend_value as f32 / 8192.0) - 1.0;
        println!("Pitch Bend: {:.3}", state.pitch_bend);
    }
}

*/