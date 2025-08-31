// src/input/mod.rs

mod midi;
mod audio;

pub use midi::{MidiInput, MidiState};
pub use audio::{AudioInput, AudioState};

use crossbeam_channel::{Sender, Receiver};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::info;

/// Unified input system that manages all input sources
pub struct InputSystem {
    midi_state: Arc<RwLock<MidiState>>,
    audio_state: Arc<RwLock<AudioState>>,
    _midi_input: Option<MidiInput>,
    _audio_input: Option<AudioInput>,
    event_sender: Sender<InputEvent>,
    event_receiver: Receiver<InputEvent>,
}

#[derive(Debug, Clone)]
pub enum InputEvent {
    MidiNoteOn { note: u8, velocity: u8 },
    MidiNoteOff { note: u8 },
    MidiControlChange { controller: u8, value: u8 },
    MidiPitchBend { value: f32 },
    AudioLevel { rms: f32, peak: f32 },
    AudioSpectrum { low: f32, mid: f32, high: f32 },
}

impl InputSystem {
    pub fn new(midi_config: &crate::config::MidiConfig, audio_config: &crate::config::AudioConfig) -> Self {
        let (event_sender, event_receiver) = crossbeam_channel::unbounded();

        let midi_state = Arc::new(RwLock::new(MidiState::default()));
        let audio_state = Arc::new(RwLock::new(AudioState::default()));

        let mut system = Self {
            midi_state: Arc::clone(&midi_state),
            audio_state: Arc::clone(&audio_state),
            _midi_input: None,
            _audio_input: None,
            event_sender,
            event_receiver,
        };

        // Initialize MIDI if enabled
        if midi_config.enabled {
            match MidiInput::new(midi_config, Arc::clone(&midi_state), system.event_sender.clone()) {
                Ok(midi) => {
                    info!("MIDI input initialized");
                    system._midi_input = Some(midi);
                }
                Err(e) => {
                    tracing::error!("Failed to initialize MIDI: {}", e);
                }
            }
        }

        // Initialize audio if enabled
        if audio_config.enabled {
            match AudioInput::new(audio_config, Arc::clone(&audio_state), system.event_sender.clone()) {
                Ok(audio) => {
                    info!("Audio input initialized");
                    system._audio_input = Some(audio);
                }
                Err(e) => {
                    tracing::error!("Failed to initialize audio: {}", e);
                }
            }
        }

        system
    }

    pub async fn get_combined_state(&self) -> CombinedInputState {
        let midi = self.midi_state.read().await.clone();
        let audio = self.audio_state.read().await.clone();

        CombinedInputState {
            // Blend MIDI and audio values
            velocity: midi.current_velocity().max(audio.level_rms),
            pitch_bend: midi.pitch_bend.max(audio.spectrum_low * 2.0 - 1.0),
            modulation: midi.controllers[1].max(audio.spectrum_mid),
            expression: midi.controllers[11].max(audio.spectrum_high),
            note_count: midi.note_count,
            last_note: midi.last_note,

            // Raw states for detailed access
            midi,
            audio,
        }
    }

    pub fn poll_events(&self) -> Vec<InputEvent> {
        let mut events = Vec::new();
        while let Ok(event) = self.event_receiver.try_recv() {
            events.push(event);
        }
        events
    }
}

#[derive(Debug, Clone)]
pub struct CombinedInputState {
    pub velocity: f32,
    pub pitch_bend: f32,
    pub modulation: f32,
    pub expression: f32,
    pub note_count: u32,
    pub last_note: u8,
    pub midi: MidiState,
    pub audio: AudioState,
}