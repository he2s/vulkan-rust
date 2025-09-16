use serde::{Deserialize, Serialize};
use anyhow::Result;
use midir::{MidiInput, MidiInputConnection, Ignore};
use std::sync::{
    atomic::{AtomicU32, AtomicU8, Ordering},
    Arc,
};

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

/// Lock-free MIDI state using atomic operations
#[derive(Debug)]
pub struct MidiState {
    // Use atomic arrays for lock-free access
    notes: [AtomicU32; MAX_NOTES],           // Store f32 as u32 bits
    controllers: [AtomicU32; MAX_CONTROLLERS], // Store f32 as u32 bits
    pitch_bend: AtomicU32,                   // Store f32 as u32 bits
    last_note: AtomicU8,
    note_count: AtomicU32,
}

impl Default for MidiState {
    fn default() -> Self {
        // Initialize atomic arrays
        const ZERO_ATOMIC: AtomicU32 = AtomicU32::new(0);
        const HALF_ATOMIC: AtomicU32 = AtomicU32::new(0x3F000000); // 0.5f32.to_bits()

        Self {
            notes: [ZERO_ATOMIC; MAX_NOTES],
            controllers: [HALF_ATOMIC; MAX_CONTROLLERS],
            pitch_bend: AtomicU32::new(0), // 0.0f32.to_bits()
            last_note: AtomicU8::new(60),
            note_count: AtomicU32::new(0),
        }
    }
}

impl MidiState {
    #[inline(always)]
    pub fn get_note(&self, index: usize) -> f32 {
        if index < MAX_NOTES {
            f32::from_bits(self.notes[index].load(Ordering::Relaxed))
        } else {
            0.0
        }
    }

    #[inline(always)]
    pub fn set_note(&self, index: usize, value: f32) {
        if index < MAX_NOTES {
            self.notes[index].store(value.to_bits(), Ordering::Relaxed);
        }
    }

    #[inline(always)]
    pub fn get_controller(&self, index: usize) -> f32 {
        if index < MAX_CONTROLLERS {
            f32::from_bits(self.controllers[index].load(Ordering::Relaxed))
        } else {
            0.5 // Default CC value
        }
    }

    #[inline(always)]
    pub fn set_controller(&self, index: usize, value: f32) {
        if index < MAX_CONTROLLERS {
            self.controllers[index].store(value.to_bits(), Ordering::Relaxed);
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
        // Use fetch_update for saturating subtraction
        self.note_count.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |x| {
            Some(x.saturating_sub(1))
        }).ok();
    }

    // For compatibility with existing code - creates a snapshot
    pub fn clone_values(&self) -> MidiStateSnapshot {
        let mut notes = [0.0f32; MAX_NOTES];
        let mut controllers = [0.5f32; MAX_CONTROLLERS];

        // Batch read all atomic values
        for i in 0..MAX_NOTES {
            notes[i] = f32::from_bits(self.notes[i].load(Ordering::Relaxed));
        }
        for i in 0..MAX_CONTROLLERS {
            controllers[i] = f32::from_bits(self.controllers[i].load(Ordering::Relaxed));
        }

        MidiStateSnapshot {
            notes,
            controllers,
            pitch_bend: f32::from_bits(self.pitch_bend.load(Ordering::Relaxed)),
            last_note: self.last_note.load(Ordering::Relaxed),
            note_count: self.note_count.load(Ordering::Relaxed),
        }
    }
}

/// Snapshot for compatibility with existing code
#[derive(Clone, Debug)]
pub struct MidiStateSnapshot {
    pub notes: [f32; MAX_NOTES],
    pub controllers: [f32; MAX_CONTROLLERS],
    pub pitch_bend: f32,
    pub last_note: u8,
    pub note_count: u32,
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

    /// Get state snapshot for compatibility
    pub fn get_state_snapshot(&self) -> MidiStateSnapshot {
        self.state.clone_values()
    }

    /// Get direct access to atomic state (for high-performance access)
    pub fn get_state_atomic(&self) -> Arc<MidiState> {
        Arc::clone(&self.state)
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
            }
            Err(e) => {
                eprintln!("MIDI setup failed: {}. Continuing without MIDI input.", e);
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
                Self::handle_message_optimized(&state, message);
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

    /// OPTIMIZATION: Zero-lock message handling with inlined hot path
    #[inline(always)]
    fn handle_message_optimized(state: &Arc<MidiState>, message: &[u8]) {
        if message.is_empty() {
            return;
        }

        let status = message[0];
        let message_type = status & 0xF0;

        // OPTIMIZATION: Use match for better branch prediction
        match message_type {
            0x80 => Self::handle_note_off_atomic(state, message),
            0x90 => Self::handle_note_on_atomic(state, message),
            0xB0 => Self::handle_control_change_atomic(state, message),
            0xE0 => Self::handle_pitch_bend_atomic(state, message),
            _ => {} // Ignore other message types
        }
    }

    #[inline(always)]
    fn handle_note_off_atomic(state: &Arc<MidiState>, message: &[u8]) {
        if message.len() >= 3 {
            let note = message[1] as usize;
            if note < MAX_NOTES {
                let current_velocity = state.get_note(note);
                if current_velocity > 0.0 {
                    state.decrement_note_count();
                    state.set_note(note, 0.0);
                    println!("Note Off: {} (Count: {})", note, state.get_note_count());
                }
            }
        }
    }

    #[inline(always)]
    fn handle_note_on_atomic(state: &Arc<MidiState>, message: &[u8]) {
        if message.len() >= 3 {
            let note = message[1] as usize;
            let velocity = message[2];

            if note < MAX_NOTES {
                if velocity == 0 {
                    // Note off via velocity 0
                    let current_velocity = state.get_note(note);
                    if current_velocity > 0.0 {
                        state.decrement_note_count();
                    }
                    state.set_note(note, 0.0);
                    println!("Note Off: {} (Count: {})", note, state.get_note_count());
                } else {
                    // Note on
                    let current_velocity = state.get_note(note);
                    if current_velocity == 0.0 {
                        state.increment_note_count();
                    }
                    let normalized_velocity = velocity as f32 * (1.0 / 127.0); // Faster than division
                    state.set_note(note, normalized_velocity);
                    state.set_last_note(note as u8);
                    println!("Note On: {} Velocity: {} (Count: {})", note, velocity, state.get_note_count());
                }
            }
        }
    }

    #[inline(always)]
    fn handle_control_change_atomic(state: &Arc<MidiState>, message: &[u8]) {
        if message.len() >= 3 {
            let controller = message[1] as usize;
            let value = message[2];

            if controller < MAX_CONTROLLERS {
                let normalized_value = value as f32 * (1.0 / 127.0); // Faster than division
                state.set_controller(controller, normalized_value);
                println!("CC{}: {}", controller, value);
            }
        }
    }

    #[inline(always)]
    fn handle_pitch_bend_atomic(state: &Arc<MidiState>, message: &[u8]) {
        if message.len() >= 3 {
            let bend_value = ((message[2] as u16) << 7) | (message[1] as u16);
            let normalized_bend = (bend_value as f32 * (1.0 / 8192.0)) - 1.0; // Faster than division
            state.set_pitch_bend(normalized_bend);
            println!("Pitch Bend: {:.3}", normalized_bend);
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

// OPTIMIZATION: High-performance direct access helpers
impl MidiState {
    /// Get note velocity without bounds checking (unsafe but faster)
    #[inline(always)]
    pub unsafe fn get_note_unchecked(&self, index: usize) -> f32 {
        f32::from_bits(self.notes.get_unchecked(index).load(Ordering::Relaxed))
    }

    /// Get controller value without bounds checking (unsafe but faster)
    #[inline(always)]
    pub unsafe fn get_controller_unchecked(&self, index: usize) -> f32 {
        f32::from_bits(self.controllers.get_unchecked(index).load(Ordering::Relaxed))
    }

    /// Batch read multiple notes efficiently
    #[inline]
    pub fn get_notes_range(&self, start: usize, end: usize) -> Vec<f32> {
        let end = end.min(MAX_NOTES);
        let start = start.min(end);

        (start..end)
            .map(|i| f32::from_bits(self.notes[i].load(Ordering::Relaxed)))
            .collect()
    }

    /// Batch read multiple controllers efficiently
    #[inline]
    pub fn get_controllers_range(&self, start: usize, end: usize) -> Vec<f32> {
        let end = end.min(MAX_CONTROLLERS);
        let start = start.min(end);

        (start..end)
            .map(|i| f32::from_bits(self.controllers[i].load(Ordering::Relaxed)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atomic_midi_state() {
        let state = MidiState::default();

        // Test note operations
        state.set_note(60, 0.8);
        assert_eq!(state.get_note(60), 0.8);

        // Test controller operations
        state.set_controller(1, 0.75);
        assert_eq!(state.get_controller(1), 0.75);

        // Test pitch bend
        state.set_pitch_bend(-0.5);
        assert_eq!(state.get_pitch_bend(), -0.5);

        // Test note count
        state.increment_note_count();
        state.increment_note_count();
        assert_eq!(state.get_note_count(), 2);

        state.decrement_note_count();
        assert_eq!(state.get_note_count(), 1);
    }

    #[test]
    fn test_midi_state_snapshot() {
        let state = MidiState::default();

        state.set_note(60, 0.9);
        state.set_controller(7, 0.6);
        state.set_pitch_bend(0.3);
        state.set_last_note(72);
        state.increment_note_count();

        let snapshot = state.clone_values();

        assert_eq!(snapshot.notes[60], 0.9);
        assert_eq!(snapshot.controllers[7], 0.6);
        assert_eq!(snapshot.pitch_bend, 0.3);
        assert_eq!(snapshot.last_note, 72);
        assert_eq!(snapshot.note_count, 1);
    }

    #[test]
    fn test_concurrent_access() {
        use std::thread;

        let state = Arc::new(MidiState::default());
        let state_clone = Arc::clone(&state);

        // Simulate MIDI input thread
        let handle = thread::spawn(move || {
            for i in 0..100 {
                state_clone.set_note(60, i as f32 / 100.0);
                state_clone.increment_note_count();
            }
        });

        // Simulate main thread reading
        for _ in 0..100 {
            let _ = state.get_note(60);
            let _ = state.get_note_count();
        }

        handle.join().unwrap();
        assert_eq!(state.get_note_count(), 100);
    }
}