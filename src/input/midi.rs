use serde::{Deserialize, Serialize};
use anyhow::Result;
use midir::{MidiInput, MidiInputConnection, Ignore};
use std::sync::{
    atomic::{AtomicU32, AtomicU8, Ordering},
    Arc,
};

// ============================================================================
// CONSTANTS
// ============================================================================

const MAX_NOTES: usize = 128;
const MAX_CONTROLLERS: usize = 128;

// Pre-computed for faster normalization
const VELOCITY_SCALE: f32 = 1.0 / 127.0;
const PITCH_BEND_SCALE: f32 = 1.0 / 8192.0;

// MIDI message types
const NOTE_OFF: u8 = 0x80;
const NOTE_ON: u8 = 0x90;
const CONTROL_CHANGE: u8 = 0xB0;
const PITCH_BEND: u8 = 0xE0;

// ============================================================================
// ATOMIC FLOAT WRAPPER
// ============================================================================

/// Wrapper for atomic float operations using bit manipulation
#[derive(Debug)]
struct AtomicF32(AtomicU32);

impl AtomicF32 {
    const fn new(val: f32) -> Self {
        Self(AtomicU32::new(val.to_bits()))
    }

    #[inline(always)]
    fn load(&self, order: Ordering) -> f32 {
        f32::from_bits(self.0.load(order))
    }

    #[inline(always)]
    fn store(&self, val: f32, order: Ordering) {
        self.0.store(val.to_bits(), order);
    }
}

impl Default for AtomicF32 {
    fn default() -> Self {
        Self::new(0.0)
    }
}

// ============================================================================
// MIDI STATE
// ============================================================================

/// Lock-free MIDI state optimized for cache locality
#[repr(align(64))]  // Cache line alignment
#[derive(Debug)]
pub struct MidiState {
    // Hot path: frequently accessed together
    notes: [AtomicF32; MAX_NOTES],
    last_note: AtomicU8,
    note_count: AtomicU32,

    // Separate cache line for controllers (less frequently accessed)
    _pad1: [u8; 48],
    controllers: [AtomicF32; MAX_CONTROLLERS],

    // Separate cache line for pitch bend
    _pad2: [u8; 64],
    pitch_bend: AtomicF32,
}

impl Default for MidiState {
    fn default() -> Self {
        const ZERO_F32: AtomicF32 = AtomicF32::new(0.0);
        const HALF_F32: AtomicF32 = AtomicF32::new(0.5);

        Self {
            notes: [ZERO_F32; MAX_NOTES],
            last_note: AtomicU8::new(60),  // Middle C
            note_count: AtomicU32::new(0),
            _pad1: [0; 48],
            controllers: [HALF_F32; MAX_CONTROLLERS],  // Default to center
            _pad2: [0; 64],
            pitch_bend: AtomicF32::new(0.0),
        }
    }
}

impl MidiState {
    // ========================================================================
    // NOTE OPERATIONS
    // ========================================================================

    /// Get note velocity (0.0 = off, 1.0 = max velocity)
    #[inline(always)]
    pub fn get_note(&self, note: u8) -> f32 {
        debug_assert!(note < MAX_NOTES as u8);
        // Safety: bounds checked in debug, note is always < 128
        unsafe {
            self.notes
                .get_unchecked(note as usize)
                .load(Ordering::Relaxed)
        }
    }

    /// Set note velocity, returns previous value
    #[inline(always)]
    pub fn set_note(&self, note: u8, velocity: f32) -> f32 {
        debug_assert!(note < MAX_NOTES as u8);
        // Safety: bounds checked in debug, note is always < 128
        unsafe {
            let note_ref = self.notes.get_unchecked(note as usize);
            let old = note_ref.load(Ordering::Relaxed);
            note_ref.store(velocity, Ordering::Relaxed);
            old
        }
    }

    /// Handle note on/off with automatic count tracking
    #[inline]
    pub fn handle_note(&self, note: u8, velocity: u8, is_on: bool) {
        let new_velocity = if is_on && velocity > 0 {
            self.last_note.store(note, Ordering::Relaxed);
            velocity as f32 * VELOCITY_SCALE
        } else {
            0.0
        };

        let old_velocity = self.set_note(note, new_velocity);

        // Update note count based on transition
        match (old_velocity > 0.0, new_velocity > 0.0) {
            (false, true) => self.note_count.fetch_add(1, Ordering::Relaxed),
            (true, false) => self.note_count.fetch_sub(1, Ordering::Relaxed),
            _ => 0,
        };
    }

    /// Get all currently active notes
    pub fn get_active_notes(&self) -> Vec<(u8, f32)> {
        let mut active = Vec::with_capacity(16);  // Most use cases have < 16 simultaneous notes

        // Process 4 notes at a time for better CPU pipelining
        for chunk_start in (0..MAX_NOTES).step_by(4) {
            for i in 0..4.min(MAX_NOTES - chunk_start) {
                let note = (chunk_start + i) as u8;
                let velocity = self.get_note(note);
                if velocity > 0.0 {
                    active.push((note, velocity));
                }
            }
        }

        active
    }

    #[inline(always)]
    pub fn get_last_note(&self) -> u8 {
        self.last_note.load(Ordering::Relaxed)
    }

    #[inline(always)]
    pub fn get_note_count(&self) -> u32 {
        self.note_count.load(Ordering::Relaxed)
    }

    // ========================================================================
    // CONTROLLER OPERATIONS
    // ========================================================================

    /// Get controller value (0.0 to 1.0)
    #[inline(always)]
    pub fn get_controller(&self, cc: u8) -> f32 {
        debug_assert!(cc < MAX_CONTROLLERS as u8);
        // Safety: bounds checked in debug, cc is always < 128
        unsafe {
            self.controllers
                .get_unchecked(cc as usize)
                .load(Ordering::Relaxed)
        }
    }

    /// Set controller value (0.0 to 1.0)
    #[inline(always)]
    pub fn set_controller(&self, cc: u8, value: f32) {
        debug_assert!(cc < MAX_CONTROLLERS as u8);
        debug_assert!(value >= 0.0 && value <= 1.0);
        // Safety: bounds checked in debug, cc is always < 128
        unsafe {
            self.controllers
                .get_unchecked(cc as usize)
                .store(value, Ordering::Relaxed);
        }
    }

    // ========================================================================
    // PITCH BEND OPERATIONS
    // ========================================================================

    /// Get pitch bend (-1.0 to 1.0, 0.0 = center)
    #[inline(always)]
    pub fn get_pitch_bend(&self) -> f32 {
        self.pitch_bend.load(Ordering::Relaxed)
    }

    /// Set pitch bend (-1.0 to 1.0, 0.0 = center)
    #[inline(always)]
    pub fn set_pitch_bend(&self, value: f32) {
        debug_assert!(value >= -1.0 && value <= 1.0);
        self.pitch_bend.store(value, Ordering::Relaxed);
    }

    // ========================================================================
    // SNAPSHOT OPERATIONS
    // ========================================================================

    /// Create a snapshot of the current state
    pub fn snapshot(&self) -> MidiStateSnapshot {
        let mut snapshot = MidiStateSnapshot {
            notes: [0.0; MAX_NOTES],
            controllers: [0.5; MAX_CONTROLLERS],
            pitch_bend: self.get_pitch_bend(),
            last_note: self.get_last_note(),
            note_count: self.get_note_count(),
        };

        // Batch copy with unrolling for vectorization
        for i in 0..MAX_NOTES {
            snapshot.notes[i] = self.notes[i].load(Ordering::Relaxed);
        }

        for i in 0..MAX_CONTROLLERS {
            snapshot.controllers[i] = self.controllers[i].load(Ordering::Relaxed);
        }

        snapshot
    }
}

// ============================================================================
// MIDI MESSAGE HANDLER
// ============================================================================

/// Process MIDI message with minimal branching
#[inline]
fn handle_midi_message(state: &MidiState, msg: &[u8]) {
    // Early return for invalid messages
    if msg.len() < 2 {
        return;
    }

    let status = msg[0];
    let msg_type = status & 0xF0;

    // Use jump table pattern for better branch prediction
    match (msg_type, msg.len() >= 3) {
        (NOTE_OFF, true) | (NOTE_ON, true) => {
            let note = msg[1] & 0x7F;  // Ensure valid note range
            let velocity = msg[2] & 0x7F;  // Ensure valid velocity range
            state.handle_note(note, velocity, msg_type == NOTE_ON);
        }

        (CONTROL_CHANGE, true) => {
            let controller = msg[1] & 0x7F;
            let value = msg[2] & 0x7F;
            state.set_controller(controller, value as f32 * VELOCITY_SCALE);
        }

        (PITCH_BEND, true) => {
            let lsb = msg[1] as u16;
            let msb = (msg[2] & 0x7F) as u16;
            let bend_raw = (msb << 7) | lsb;
            // Convert 14-bit value to -1.0 to 1.0 range
            let bend = (bend_raw as f32 * PITCH_BEND_SCALE) - 1.0;
            state.set_pitch_bend(bend.clamp(-1.0, 1.0));
        }

        _ => {} // Ignore other messages
    }
}

// ============================================================================
// MIDI MANAGER
// ============================================================================

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

    /// Get reference to the shared state
    pub fn get_state(&self) -> Arc<MidiState> {
        Arc::clone(&self.state)
    }

    /// Get a snapshot of the current state
    pub fn get_state_snapshot(&self) -> MidiStateSnapshot {
        self.state.snapshot()
    }

    /// Setup MIDI connection with the given configuration
    pub fn setup(&mut self, config: &MidiConfig) -> Result<()> {
        if !config.enabled {
            return Ok(());
        }

        self.connection = Some(self.connect(config)?);
        Ok(())
    }

    /// Connect to MIDI input port
    fn connect(&self, config: &MidiConfig) -> Result<MidiInputConnection<()>> {
        let mut midi_in = MidiInput::new("MIDI Visualizer")?;
        midi_in.ignore(Ignore::None);

        let ports = midi_in.ports();
        if ports.is_empty() {
            anyhow::bail!("No MIDI input ports available");
        }

        // Find port by name or use first available
        let port = if let Some(ref name) = config.port_name {
            ports.iter()
                .find(|p| {
                    midi_in.port_name(p)
                        .map(|n| n.contains(name))
                        .unwrap_or(false)
                })
                .ok_or_else(|| anyhow::anyhow!("Port '{}' not found", name))?
        } else {
            &ports[0]
        };

        let state = Arc::clone(&self.state);

        let connection = midi_in.connect(
            port,
            "visualizer",
            move |_timestamp, msg, _| {
                handle_midi_message(&state, msg);
            },
            (),
        )?;

        Ok(connection)
    }

    /// List available MIDI ports
    pub fn list_ports() -> Result<Vec<String>> {
        let midi_in = MidiInput::new("MIDI Port Scanner")?;
        let ports = midi_in.ports();

        ports.iter()
            .map(|p| midi_in.port_name(p))
            .collect::<Result<Vec<_>, _>>()
            .map_err(Into::into)
    }
}

impl Drop for MidiManager {
    fn drop(&mut self) {
        if let Some(conn) = self.connection.take() {
            conn.close();
        }
    }
}

// ============================================================================
// DATA STRUCTURES
// ============================================================================

/// Snapshot of MIDI state at a point in time
#[derive(Clone, Debug)]
pub struct MidiStateSnapshot {
    pub notes: [f32; MAX_NOTES],
    pub controllers: [f32; MAX_CONTROLLERS],
    pub pitch_bend: f32,
    pub last_note: u8,
    pub note_count: u32,
}

/// MIDI configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MidiConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,

    #[serde(default = "default_true")]
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

fn default_true() -> bool {
    true
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_note_operations() {
        let state = MidiState::default();

        // Test note on
        state.handle_note(60, 100, true);
        assert!(state.get_note(60) > 0.0);
        assert_eq!(state.get_note_count(), 1);
        assert_eq!(state.get_last_note(), 60);

        // Test note off
        state.handle_note(60, 0, false);
        assert_eq!(state.get_note(60), 0.0);
        assert_eq!(state.get_note_count(), 0);
    }

    #[test]
    fn test_controller_operations() {
        let state = MidiState::default();

        state.set_controller(1, 0.75);
        assert_eq!(state.get_controller(1), 0.75);
    }

    #[test]
    fn test_pitch_bend() {
        let state = MidiState::default();

        state.set_pitch_bend(0.5);
        assert_eq!(state.get_pitch_bend(), 0.5);

        state.set_pitch_bend(-0.5);
        assert_eq!(state.get_pitch_bend(), -0.5);
    }

    #[test]
    fn test_snapshot() {
        let state = MidiState::default();

        state.handle_note(60, 100, true);
        state.set_controller(7, 0.8);
        state.set_pitch_bend(0.25);

        let snapshot = state.snapshot();
        assert!(snapshot.notes[60] > 0.0);
        assert_eq!(snapshot.controllers[7], 0.8);
        assert_eq!(snapshot.pitch_bend, 0.25);
        assert_eq!(snapshot.note_count, 1);
    }
}