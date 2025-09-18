use anyhow::Result;
use midir::{Ignore, MidiInput, MidiInputConnection};
use serde::{Deserialize, Serialize};
use std::sync::{
    Arc,
    atomic::{AtomicU8, AtomicU32, Ordering},
};

const MAX_NOTES: usize = 128;
const MAX_CONTROLLERS: usize = 128;

const INV_127: f32 = 1.0 / 127.0;
const INV_8192: f32 = 1.0 / 8192.0;

#[repr(align(64))]
#[derive(Debug)]
pub struct MidiState {
    notes: [AtomicU32; MAX_NOTES],
    last_note: AtomicU8,
    note_count: AtomicU32,

    _pad1: [u8; 52],
    controllers: [AtomicU32; MAX_CONTROLLERS],

    _pad2: [u8; 64],
    pitch_bend: AtomicU32,
}

impl Default for MidiState {
    fn default() -> Self {
        const ZERO: AtomicU32 = AtomicU32::new(0);
        const HALF: AtomicU32 = AtomicU32::new(0x3F000000); // 0.5f32.to_bits()

        Self {
            notes: [ZERO; MAX_NOTES],
            last_note: AtomicU8::new(60),
            note_count: AtomicU32::new(0),
            _pad1: [0; 52],
            controllers: [HALF; MAX_CONTROLLERS],
            _pad2: [0; 64],
            pitch_bend: AtomicU32::new(0),
        }
    }
}

impl MidiState {
    #[inline(always)]
    pub fn get_note(&self, index: usize) -> f32 {
        debug_assert!(index < MAX_NOTES);
        unsafe { f32::from_bits(self.notes.get_unchecked(index).load(Ordering::Relaxed)) }
    }

    #[inline(always)]
    pub fn set_note(&self, index: usize, value: f32) {
        debug_assert!(index < MAX_NOTES);
        unsafe {
            self.notes
                .get_unchecked(index)
                .store(value.to_bits(), Ordering::Relaxed);
        }
    }

    #[inline(always)]
    pub fn get_controller(&self, index: usize) -> f32 {
        debug_assert!(index < MAX_CONTROLLERS);
        unsafe {
            f32::from_bits(
                self.controllers
                    .get_unchecked(index)
                    .load(Ordering::Relaxed),
            )
        }
    }

    #[inline(always)]
    pub fn set_controller(&self, index: usize, value: f32) {
        debug_assert!(index < MAX_CONTROLLERS);
        unsafe {
            self.controllers
                .get_unchecked(index)
                .store(value.to_bits(), Ordering::Relaxed);
        }
    }

    #[inline(always)]
    pub fn get_pitch_bend(&self) -> f32 {
        f32::from_bits(self.pitch_bend.load(Ordering::Relaxed))
    }

    #[inline(always)]
    pub fn set_pitch_bend(&self, value: f32) {
        self.pitch_bend.store(value.to_bits(), Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn get_last_note(&self) -> u8 {
        self.last_note.load(Ordering::Relaxed)
    }

    #[inline(always)]
    pub fn set_last_note(&self, note: u8) {
        self.last_note.store(note, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn get_note_count(&self) -> u32 {
        self.note_count.load(Ordering::Relaxed)
    }

    #[inline(always)]
    pub fn increment_note_count(&self) {
        self.note_count.fetch_add(1, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn decrement_note_count(&self) {
        let _ = self
            .note_count
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |x| {
                if x > 0 { Some(x - 1) } else { Some(0) }
            });
    }

    #[cfg(target_arch = "x86_64")]
    pub fn get_active_notes(&self, out: &mut Vec<(u8, f32)>) {
        out.clear();

        let mut i = 0;
        while i < MAX_NOTES {
            let v0 = f32::from_bits(self.notes[i].load(Ordering::Relaxed));
            let v1 = f32::from_bits(self.notes[i + 1].load(Ordering::Relaxed));
            let v2 = f32::from_bits(self.notes[i + 2].load(Ordering::Relaxed));
            let v3 = f32::from_bits(self.notes[i + 3].load(Ordering::Relaxed));

            if v0 > 0.0 {
                out.push((i as u8, v0));
            }
            if v1 > 0.0 {
                out.push(((i + 1) as u8, v1));
            }
            if v2 > 0.0 {
                out.push(((i + 2) as u8, v2));
            }
            if v3 > 0.0 {
                out.push(((i + 3) as u8, v3));
            }

            i += 4;
        }
    }
}

pub struct MidiManager {
    state: Arc<MidiState>,
    connection: Option<MidiInputConnection<()>>,
}

impl MidiManager {
    pub fn new() -> Self {
        Self {
            state: Arc::new(MidiState::default()),
            connection: None,
        }
    }

    pub fn get_state(&self) -> Arc<MidiState> {
        Arc::clone(&self.state)
    }

    pub fn get_state_snapshot(&self) -> MidiStateSnapshot {
        self.state.snapshot()
    }

    pub fn setup(&mut self, config: &MidiConfig) {
        if !config.enabled {
            return;
        }

        if let Ok(connection) = self.try_connect(config) {
            self.connection = Some(connection);
        }
    }

    fn try_connect(
        &self,
        config: &MidiConfig,
    ) -> Result<MidiInputConnection<()>, Box<dyn std::error::Error>> {
        let mut midi_in = MidiInput::new("MIDI Visualizer")?;
        midi_in.ignore(Ignore::None);

        let ports = midi_in.ports();
        if ports.is_empty() {
            return Err("No MIDI ports".into());
        }

        let port = if let Some(name_sub) = &config.port_name {
            ports
                .iter()
                .find(|p| midi_in.port_name(p).map_or(false, |n| n.contains(name_sub)))
                .ok_or("Configured MIDI port not found")?
        } else {
            &ports[0]
        };

        let state = Arc::clone(&self.state);

        let connection = midi_in.connect(
            port,
            "visualizer",
            move |_, msg, _| {
                handle_message_optimized(&state, msg);
            },
            (),
        )?;

        Ok(connection)
    }
}

#[inline(always)]
fn handle_message_optimized(state: &MidiState, msg: &[u8]) {
    if msg.len() < 2 {
        return;
    }

    let status = unsafe { *msg.get_unchecked(0) };
    let msg_type = status & 0xF0;

    match msg_type {
        0x80 | 0x90 => {
            if msg.len() < 3 {
                return;
            }
            unsafe {
                let note = *msg.get_unchecked(1) as usize;
                let velocity = *msg.get_unchecked(2);

                let new_velocity = if msg_type == 0x90 && velocity != 0 {
                    state.set_last_note(note as u8);
                    velocity as f32 * INV_127
                } else {
                    0.0
                };

                let old_velocity = state.get_note(note);
                if old_velocity == 0.0 && new_velocity > 0.0 {
                    state.increment_note_count();
                } else if old_velocity > 0.0 && new_velocity == 0.0 {
                    state.decrement_note_count();
                }

                state.set_note(note, new_velocity);
            }
        }
        0xB0 => {
            if msg.len() < 3 {
                return;
            }
            unsafe {
                let controller = *msg.get_unchecked(1) as usize;
                let value = *msg.get_unchecked(2);
                state.set_controller(controller, value as f32 * INV_127);
            }
        }
        0xE0 => {
            if msg.len() < 3 {
                return;
            }
            unsafe {
                let lsb = *msg.get_unchecked(1) as u16;
                let msb = *msg.get_unchecked(2) as u16;
                let bend = (msb << 7) | lsb;
                state.set_pitch_bend((bend as f32 * INV_8192) - 1.0);
            }
        }
        _ => {}
    }
}

#[derive(Clone, Debug)]
pub struct MidiStateSnapshot {
    pub notes: [f32; MAX_NOTES],
    pub controllers: [f32; MAX_CONTROLLERS],
    pub pitch_bend: f32,
    pub last_note: u8,
    pub note_count: u32,
}

impl MidiState {
    pub fn snapshot(&self) -> MidiStateSnapshot {
        let mut notes = [0.0f32; MAX_NOTES];
        let mut controllers = [0.5f32; MAX_CONTROLLERS];

        for chunk in 0..(MAX_NOTES / 8) {
            let base = chunk * 8;
            for i in 0..8 {
                notes[base + i] = f32::from_bits(self.notes[base + i].load(Ordering::Relaxed));
            }
        }

        for chunk in 0..(MAX_CONTROLLERS / 8) {
            let base = chunk * 8;
            for i in 0..8 {
                controllers[base + i] =
                    f32::from_bits(self.controllers[base + i].load(Ordering::Relaxed));
            }
        }

        MidiStateSnapshot {
            notes,
            controllers,
            pitch_bend: self.get_pitch_bend(),
            last_note: self.get_last_note(),
            note_count: self.get_note_count(),
        }
    }

    pub fn clone_values(&self) -> MidiStateSnapshot {
        self.snapshot()
    }
}

impl Drop for MidiManager {
    fn drop(&mut self) {
        if let Some(conn) = self.connection.take() {
            conn.close();
        }
    }
}

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
