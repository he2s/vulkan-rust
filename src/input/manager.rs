use std::sync::{Arc, Mutex};
use anyhow::Result;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

use crate::config::config::AudioConfig;
use crate::input::midi::MidiConfig;
use crate::input::osc::OscConfig;
use crate::audio::{AudioLevels, AudioState, BeatState};
use crate::state::FrameState;
use super::midi::MidiManager;
use super::osc::OscManager;

pub struct InputManager {
    midi_manager: MidiManager,
    audio_state: Arc<Mutex<AudioState>>,
    cached_audio_levels: AudioLevels,
    cached_beat_state: BeatState,
    _audio_stream: Option<cpal::Stream>,
    osc_manager: OscManager,
}

impl Default for InputManager {
    fn default() -> Self {
        Self::new()
    }
}

impl InputManager {
    pub fn new() -> Self {
        Self {
            midi_manager: MidiManager::new(),
            audio_state: Arc::new(Mutex::new(AudioState::new())),
            cached_audio_levels: AudioLevels::default(),
            cached_beat_state: BeatState::default(),
            _audio_stream: None,
            osc_manager: OscManager::new(OscConfig::default()),
        }
    }

    pub fn setup_midi(&mut self, config: &MidiConfig) {
        if !config.enabled {
            println!("MIDI disabled in configuration");
            return;
        }

        self.midi_manager.setup(config);
    }

    pub fn setup_osc(&mut self, config: &OscConfig) {
        self.osc_manager = OscManager::new(config.clone());
        if let Err(e) = self.osc_manager.start() {
            eprintln!("OSC setup failed: {e}");
        }
    }

    pub fn get_frame_state(&mut self) -> FrameState {
        let midi = self.midi_manager.get_state_snapshot();
        let osc = self.osc_manager.get_state();

        // Try to update cached audio data - never block the render thread
        if let Ok(mut audio_state) = self.audio_state.try_lock() {
            self.cached_audio_levels = audio_state.analyze_and_get_levels();
            self.cached_beat_state = audio_state.get_simple_beat_state();
        }
        // If locked, use existing cached values - guarantees zero blocking

        FrameState {
            midi,
            audio_levels: self.cached_audio_levels,
            osc,
            beat: self.cached_beat_state,
        }
    }

    pub fn setup_audio(&mut self, config: &AudioConfig) {
        if !config.enabled {
            println!("Audio input disabled in configuration");
            return;
        }

        match self.try_setup_audio(config) {
            Ok(stream) => {
                if let Err(e) = stream.play() {
                    eprintln!("Failed to start audio stream: {e}");
                    return;
                }
                self._audio_stream = Some(stream);
                println!("Audio input connected successfully!");
            }
            Err(e) => {
                eprintln!("Audio setup failed: {e}");
            }
        }
    }

    fn try_setup_audio(
        &self,
        config: &AudioConfig,
    ) -> Result<cpal::Stream, Box<dyn std::error::Error>> {
        let host = cpal::default_host();
        let device = self.select_audio_device(&host, config)?;
        let audio_config = self.configure_audio_device(&device, config)?;

        println!(
            "Using audio device: {}",
            device.name().unwrap_or_else(|_| "Unknown".into())
        );
        println!(
            "Audio config: {:?}Hz, {:?}ch",
            audio_config.sample_rate.0, audio_config.channels
        );

        let audio_state = Arc::clone(&self.audio_state);
        let channels = audio_config.channels as usize;

        let stream = device.build_input_stream(
            &audio_config,
            move |data: &[f32], _| {
                if let Ok(mut state) = audio_state.try_lock() {
                    // Use optimized method that avoids borrow conflicts
                    state.push_audio_data(data, channels, audio_config.sample_rate.0);
                }
            },
            move |err| {
                eprintln!("Audio input error: {err}");
            },
            None,
        )?;

        Ok(stream)
    }

    fn select_audio_device(
        &self,
        host: &cpal::Host,
        config: &AudioConfig,
    ) -> Result<cpal::Device, Box<dyn std::error::Error>> {
        let device = if let Some(ref device_name) = config.device_name {
            host.input_devices()?
                .find(|device| {
                    device
                        .name()
                        .is_ok_and(|name| name.contains(device_name))
                })
                .or_else(|| host.default_input_device())
        } else {
            host.default_input_device()
        };

        device.ok_or_else(|| "No input audio device found".into())
    }

    fn configure_audio_device(
        &self,
        device: &cpal::Device,
        config: &AudioConfig,
    ) -> Result<cpal::StreamConfig, Box<dyn std::error::Error>> {
        let supported_configs = device.supported_input_configs()?.collect::<Vec<_>>();
        if supported_configs.is_empty() {
            return Err("No supported audio input configs".into());
        }

        let mut stream_config = supported_configs[0].with_max_sample_rate().config();

        for supported_config in &supported_configs {
            if supported_config.sample_format() == cpal::SampleFormat::F32 {
                if let Some(desired_rate) = config.sample_rate {
                    if supported_config.min_sample_rate().0 <= desired_rate
                        && desired_rate <= supported_config.max_sample_rate().0
                    {
                        stream_config = (*supported_config)
                            .with_sample_rate(cpal::SampleRate(desired_rate))
                            .config();
                        break;
                    }
                } else {
                    stream_config = supported_config.with_max_sample_rate().config();
                    break;
                }
            }
        }

        Ok(stream_config)
    }

    /// Set manual BPM override for beat detection
    pub fn set_manual_bpm(&mut self, bpm: f32) {
        if let Ok(mut audio_state) = self.audio_state.try_lock() {
            audio_state.set_manual_bpm(bpm);
        }
    }

    /// Check if audio system is in manual tempo mode
    pub fn is_manual_tempo_mode(&self) -> bool {
        if let Ok(audio_state) = self.audio_state.try_lock() {
            audio_state.is_manual_tempo_mode()
        } else {
            false
        }
    }
}