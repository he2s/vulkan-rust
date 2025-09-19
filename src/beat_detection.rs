// beat_detection.rs
use std::collections::VecDeque;
use std::f32::consts::PI;

/// Configuration for beat detection
#[derive(Clone, Debug)]
pub struct BeatDetectorConfig {
    /// Window size for onset detection (in samples)
    pub onset_window_size: usize,
    /// Threshold multiplier for peak detection
    pub onset_threshold: f32,
    /// Minimum time between beats (in seconds)
    pub min_beat_interval: f32,
    /// Maximum time between beats (in seconds)
    pub max_beat_interval: f32,
    /// Number of beats to average for tempo calculation
    pub tempo_averaging_window: usize,
    /// Smoothing factor for tempo changes (0.0 - 1.0)
    pub tempo_smoothing: f32,
    /// Default beats per bar if not detected
    pub default_beats_per_bar: u32,
}

impl Default for BeatDetectorConfig {
    fn default() -> Self {
        Self {
            onset_window_size: 2048,
            onset_threshold: 1.5,
            min_beat_interval: 0.25,  // 240 BPM max
            max_beat_interval: 1.5,   // 40 BPM min
            tempo_averaging_window: 8,
            tempo_smoothing: 0.85,
            default_beats_per_bar: 4,
        }
    }
}

/// Represents the current beat state
#[derive(Clone, Copy, Debug)]
pub struct BeatState {
    /// Detected BPM
    pub bpm: f32,
    /// Time until next beat (0.0 - 1.0, normalized to beat interval)
    pub time_to_next_beat: f32,
    /// Time since last beat (0.0 - 1.0, normalized to beat interval)
    pub time_since_last_beat: f32,
    /// Number of beats per bar (typically 4)
    pub beats_per_bar: u32,
    /// Current beat within the bar (0 to beats_per_bar - 1)
    pub current_beat_in_bar: u32,
    /// Total beat count since start
    pub total_beats: u64,
}

impl Default for BeatState {
    fn default() -> Self {
        Self {
            bpm: 120.0,
            time_to_next_beat: 0.5,
            time_since_last_beat: 0.5,
            beats_per_bar: 4,
            current_beat_in_bar: 0,
            total_beats: 0,
        }
    }
}

/// Main beat detection system
pub struct BeatDetector {
    config: BeatDetectorConfig,

    // Onset detection
    onset_buffer: VecDeque<f32>,
    spectral_flux_history: VecDeque<f32>,

    // Beat tracking
    beat_times: VecDeque<f64>,
    last_beat_time: f64,
    current_time: f64,

    // Tempo estimation
    tempo_estimates: VecDeque<f32>,
    current_bpm: f32,
    beat_interval: f32,

    // Time signature detection
    beat_strengths: VecDeque<f32>,
    beats_per_bar: u32,
    current_beat_in_bar: u32,
    total_beats: u64,

    // Sample rate tracking
    sample_rate: f32,
    samples_processed: u64,

    // Spectral analysis buffers
    fft_buffer: Vec<f32>,
    magnitude_spectrum: Vec<f32>,
    previous_spectrum: Vec<f32>,
}

impl BeatDetector {
    pub fn new(config: BeatDetectorConfig) -> Self {
        let fft_size = config.onset_window_size;

        Self {
            config,
            onset_buffer: VecDeque::with_capacity(fft_size),
            spectral_flux_history: VecDeque::with_capacity(100),
            beat_times: VecDeque::with_capacity(32),
            last_beat_time: 0.0,
            current_time: 0.0,
            tempo_estimates: VecDeque::with_capacity(16),
            current_bpm: 120.0,
            beat_interval: 0.5,
            beat_strengths: VecDeque::with_capacity(32),
            beats_per_bar: 4,
            current_beat_in_bar: 0,
            total_beats: 0,
            sample_rate: 48000.0,
            samples_processed: 0,
            fft_buffer: vec![0.0; fft_size],
            magnitude_spectrum: vec![0.0; fft_size / 2],
            previous_spectrum: vec![0.0; fft_size / 2],
        }
    }

    /// Process audio samples and detect beats
    pub fn process_samples(&mut self, samples: &[f32], sample_rate: f32) {
        self.sample_rate = sample_rate;

        for &sample in samples {
            self.onset_buffer.push_back(sample);
            if self.onset_buffer.len() > self.config.onset_window_size {
                self.onset_buffer.pop_front();
            }

            self.samples_processed += 1;

            // Process window when full
            if self.onset_buffer.len() == self.config.onset_window_size {
                if self.samples_processed % (self.config.onset_window_size as u64 / 4) == 0 {
                    self.process_window();
                }
            }
        }

        self.current_time = self.samples_processed as f64 / self.sample_rate as f64;
    }

    /// Process a window of audio for onset detection
    fn process_window(&mut self) {
        // Calculate spectral flux for onset detection
        let spectral_flux = self.calculate_spectral_flux();
        self.spectral_flux_history.push_back(spectral_flux);

        if self.spectral_flux_history.len() > 100 {
            self.spectral_flux_history.pop_front();
        }

        // Detect onset if flux exceeds adaptive threshold
        if self.is_onset(spectral_flux) {
            self.register_beat();
        }

        // Update tempo estimation
        self.update_tempo();
    }

    /// Calculate spectral flux (change in spectrum magnitude)
    fn calculate_spectral_flux(&mut self) -> f32 {
        // Copy onset buffer to FFT buffer with windowing
        for (i, &sample) in self.onset_buffer.iter().enumerate() {
            let window = 0.5 - 0.5 * (2.0 * PI * i as f32 / self.config.onset_window_size as f32).cos();
            self.fft_buffer[i] = sample * window;
        }

        // Simple DFT for magnitude spectrum (in production, use rustfft)
        for k in 0..self.magnitude_spectrum.len() {
            let mut real = 0.0;
            let mut imag = 0.0;
            let omega = -2.0 * PI * k as f32 / self.config.onset_window_size as f32;

            for (n, &sample) in self.fft_buffer.iter().enumerate() {
                let angle = omega * n as f32;
                real += sample * angle.cos();
                imag += sample * angle.sin();
            }

            self.magnitude_spectrum[k] = (real * real + imag * imag).sqrt();
        }

        // Calculate flux as sum of positive differences
        let mut flux = 0.0;
        for i in 0..self.magnitude_spectrum.len() {
            let diff = self.magnitude_spectrum[i] - self.previous_spectrum[i];
            if diff > 0.0 {
                flux += diff;
            }
        }

        // Update previous spectrum
        self.previous_spectrum.copy_from_slice(&self.magnitude_spectrum);

        flux
    }

    /// Check if current spectral flux indicates an onset
    fn is_onset(&self, flux: f32) -> bool {
        if self.spectral_flux_history.len() < 10 {
            return false;
        }

        // Calculate adaptive threshold based on recent history
        let mean: f32 = self.spectral_flux_history.iter().sum::<f32>()
            / self.spectral_flux_history.len() as f32;
        let threshold = mean * self.config.onset_threshold;

        // Check if this is a peak and above threshold
        let is_peak = flux > threshold;

        // Ensure minimum time between beats
        let time_since_last = self.current_time - self.last_beat_time;
        let respects_minimum = time_since_last >= self.config.min_beat_interval as f64;

        is_peak && respects_minimum
    }

    /// Register a detected beat
    fn register_beat(&mut self) {
        let beat_time = self.current_time;

        // Calculate beat strength for time signature detection
        let strength = if let Some(&last_flux) = self.spectral_flux_history.back() {
            last_flux
        } else {
            1.0
        };

        self.beat_strengths.push_back(strength);
        if self.beat_strengths.len() > 32 {
            self.beat_strengths.pop_front();
        }

        // Update beat timing
        if self.last_beat_time > 0.0 {
            let interval = beat_time - self.last_beat_time;

            // Only register if within valid tempo range
            if interval >= self.config.min_beat_interval as f64
                && interval <= self.config.max_beat_interval as f64 {

                self.beat_times.push_back(beat_time);
                if self.beat_times.len() > self.config.tempo_averaging_window {
                    self.beat_times.pop_front();
                }

                // Estimate tempo from this interval
                let instant_bpm = 60.0 / interval as f32;
                self.tempo_estimates.push_back(instant_bpm);
                if self.tempo_estimates.len() > self.config.tempo_averaging_window {
                    self.tempo_estimates.pop_front();
                }
            }
        }

        self.last_beat_time = beat_time;
        self.total_beats += 1;

        // Update beat position in bar
        self.current_beat_in_bar = (self.current_beat_in_bar + 1) % self.beats_per_bar;

        // Detect time signature periodically
        if self.total_beats % 16 == 0 {
            self.detect_time_signature();
        }
    }

    /// Update tempo estimation using recent beat intervals
    fn update_tempo(&mut self) {
        if self.tempo_estimates.len() < 2 {
            return;
        }

        // Calculate weighted average of recent tempo estimates
        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;

        for (i, &tempo) in self.tempo_estimates.iter().enumerate() {
            let weight = (i + 1) as f32; // More recent = higher weight
            weighted_sum += tempo * weight;
            weight_sum += weight;
        }

        let new_bpm = weighted_sum / weight_sum;

        // Apply smoothing
        self.current_bpm = self.config.tempo_smoothing * self.current_bpm
            + (1.0 - self.config.tempo_smoothing) * new_bpm;

        // Update beat interval
        self.beat_interval = 60.0 / self.current_bpm;
    }

    /// Detect time signature from beat strength patterns
    fn detect_time_signature(&mut self) {
        if self.beat_strengths.len() < 16 {
            return;
        }

        // Look for repeating patterns in beat strengths
        // This is simplified - real implementation would use autocorrelation
        let mut pattern_scores = vec![0.0f32; 8];

        for bar_length in 2..=8 {
            let mut score = 0.0;
            let num_comparisons = self.beat_strengths.len() / bar_length;

            for i in 0..num_comparisons.saturating_sub(1) {
                for beat in 0..bar_length {
                    let idx1 = i * bar_length + beat;
                    let idx2 = (i + 1) * bar_length + beat;

                    if idx2 < self.beat_strengths.len() {
                        let diff = (self.beat_strengths[idx1] - self.beat_strengths[idx2]).abs();
                        score += 1.0 / (1.0 + diff);
                    }
                }
            }

            pattern_scores[bar_length - 2] = score / num_comparisons as f32;
        }

        // Find best pattern (most common is 4/4)
        let mut best_pattern = 4;
        let mut best_score = pattern_scores[2]; // Index for 4 beats

        for (i, &score) in pattern_scores.iter().enumerate() {
            // Bias towards common time signatures
            let bias = match i + 2 {
                3 => 1.1,  // 3/4
                4 => 1.2,  // 4/4
                6 => 1.05, // 6/8
                _ => 1.0,
            };

            if score * bias > best_score {
                best_score = score * bias;
                best_pattern = i + 2;
            }
        }

        self.beats_per_bar = best_pattern as u32;
    }

    /// Get the current beat state for shader consumption
    pub fn get_state(&self) -> BeatState {
        let time_since_last = (self.current_time - self.last_beat_time) as f32;
        let normalized_time_since = (time_since_last / self.beat_interval).min(1.0);
        let normalized_time_to_next = 1.0 - normalized_time_since;

        BeatState {
            bpm: self.current_bpm,
            time_to_next_beat: normalized_time_to_next,
            time_since_last_beat: normalized_time_since,
            beats_per_bar: self.beats_per_bar,
            current_beat_in_bar: self.current_beat_in_bar,
            total_beats: self.total_beats,
        }
    }

    /// Reset the detector
    pub fn reset(&mut self) {
        self.beat_times.clear();
        self.tempo_estimates.clear();
        self.beat_strengths.clear();
        self.last_beat_time = 0.0;
        self.current_time = 0.0;
        self.samples_processed = 0;
        self.current_beat_in_bar = 0;
        self.total_beats = 0;
        self.current_bpm = 120.0;
        self.beat_interval = 0.5;
    }
}