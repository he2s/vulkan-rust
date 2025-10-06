// src/audio.rs - Unified audio processing module
use rustfft::{Fft, FftPlanner, num_complex::Complex32};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

const AUDIO_RING_CAPACITY: usize = 4096;
const FFT_SAMPLE_SIZE: usize = 1024;
const DEFAULT_SAMPLE_RATE: u32 = 48000;
const MAX_FFT_SIZE: usize = 2048;
const MIN_FFT_SIZE: usize = 256;

#[derive(Clone, Copy, Debug)]
pub struct BeatState {
    pub bpm: f32,
    pub time_to_next_beat: f32,
    pub time_since_last_beat: f32,
    pub beats_per_bar: u32,
    pub current_beat_in_bar: u32,
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

pub struct AudioState {
    ring: VecDeque<f32>,
    capacity: usize,
    last_sample_rate: u32,
    pub levels: AudioLevels,

    fft_planner: FftPlanner<f32>,
    fft_cache: HashMap<usize, Arc<dyn Fft<f32>>>,

    processing_buffer: Vec<Complex32>,
    windowing_buffer: Vec<f32>,

    beat_history: VecDeque<f64>,
    last_beat_time: f64,
    current_bpm: f32,
    beat_threshold: f32,
    last_energy: f32,
    samples_since_beat: u64,
    total_beats: u64,
    beats_per_bar: u32,
}

impl std::fmt::Debug for AudioState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AudioState")
            .field("ring_len", &self.ring.len())
            .field("capacity", &self.capacity)
            .field("last_sample_rate", &self.last_sample_rate)
            .field("levels", &self.levels)
            .field("fft_cache_size", &self.fft_cache.len())
            .field(
                "processing_buffer_capacity",
                &self.processing_buffer.capacity(),
            )
            .field(
                "windowing_buffer_capacity",
                &self.windowing_buffer.capacity(),
            )
            .finish()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct AudioLevels {
    pub level_rms: f32,
    pub low: f32,
    pub mid: f32,
    pub high: f32,
}

impl Default for AudioLevels {
    fn default() -> Self {
        Self {
            level_rms: 0.0,
            low: 0.0,
            mid: 0.0,
            high: 0.0,
        }
    }
}

impl AudioState {
    pub fn new() -> Self {
        Self {
            ring: VecDeque::with_capacity(AUDIO_RING_CAPACITY),
            capacity: AUDIO_RING_CAPACITY,
            last_sample_rate: DEFAULT_SAMPLE_RATE,
            levels: AudioLevels::default(),

            fft_planner: FftPlanner::<f32>::new(),
            fft_cache: HashMap::with_capacity(8),

            processing_buffer: Vec::with_capacity(MAX_FFT_SIZE),
            windowing_buffer: Vec::with_capacity(MAX_FFT_SIZE),

            beat_history: VecDeque::with_capacity(16),
            last_beat_time: 0.0,
            current_bpm: 120.0,
            beat_threshold: 0.0,
            last_energy: 0.0,
            samples_since_beat: 0,
            total_beats: 0,
            beats_per_bar: 4,
        }
    }

    pub fn push_samples(&mut self, samples: &[f32], sample_rate: u32) {
        self.last_sample_rate = sample_rate;
        for &sample in samples {
            if self.ring.len() == self.capacity {
                self.ring.pop_front();
            }
            self.ring.push_back(sample);
        }
    }

    fn detect_beat_from_spectrum(&mut self) {
        // Use existing frequency bands from analyze_frequency_bands
        let energy = self.levels.low + self.levels.mid * 0.5;

        // Simple onset detection
        let energy_diff = energy - self.last_energy;
        self.last_energy = energy;

        // Adaptive threshold
        if self.beat_history.len() > 4 {
            let avg_energy: f32 = self.beat_history.iter()
                .map(|&x| x as f32)
                .sum::<f32>() / self.beat_history.len() as f32;
            self.beat_threshold = avg_energy * 1.3;
        }

        // Check for beat
        let min_samples_between_beats = (self.last_sample_rate as f64 * 0.25) as u64; // 240 BPM max

        if energy_diff > self.beat_threshold
            && energy > 0.3
            && self.samples_since_beat > min_samples_between_beats {

            // Register beat
            let current_time = self.samples_since_beat as f64 / self.last_sample_rate as f64;

            if self.last_beat_time > 0.0 {
                let interval = current_time;
                let instant_bpm = 60.0 / interval as f32;

                // Smooth BPM update
                if instant_bpm > 40.0 && instant_bpm < 240.0 {
                    self.current_bpm = self.current_bpm * 0.8 + instant_bpm * 0.2;
                }
            }

            self.last_beat_time = current_time;
            self.samples_since_beat = 0;
            self.total_beats += 1;

            // Update history
            self.beat_history.push_back(energy as f64);
            if self.beat_history.len() > 16 {
                self.beat_history.pop_front();
            }
        }

        self.samples_since_beat += FFT_SAMPLE_SIZE as u64;
    }

    pub fn get_simple_beat_state(&self) -> BeatState {
        let beat_interval = 60.0 / self.current_bpm;
        let time_since = (self.samples_since_beat as f64 / self.last_sample_rate as f64).min(beat_interval as f64);
        let normalized_time_since = (time_since / beat_interval as f64).min(1.0) as f32;

        BeatState {
            bpm: self.current_bpm,
            time_to_next_beat: 1.0 - normalized_time_since,
            time_since_last_beat: normalized_time_since,
            beats_per_bar: self.beats_per_bar,
            current_beat_in_bar: (self.total_beats % self.beats_per_bar as u64) as u32,
            total_beats: self.total_beats,
        }
    }

    /// Override BPM with a manual value
    pub fn set_manual_bpm(&mut self, bpm: f32) {
        // Accept a wider range for manual input since validation is done at the UI level
        if (20.0..=999.0).contains(&bpm) {
            self.current_bpm = bpm;

            // Clear beat history to prevent interference with manual mode
            self.beat_history.clear();

            // Reset beat timing based on new BPM
            let beat_interval_samples = (self.last_sample_rate as f64 * 60.0 / bpm as f64) as u64;
            if self.samples_since_beat >= beat_interval_samples {
                // We've passed the expected beat time, reset timing
                self.samples_since_beat = 0;
                self.total_beats += 1;
            }
        }
    }

    /// Check if the audio system is in manual tempo mode (no automatic detection)
    pub fn is_manual_tempo_mode(&self) -> bool {
        self.beat_history.is_empty()
    }

    pub fn analyze_and_get_levels(&mut self) -> AudioLevels {
        if self.ring.is_empty() {
            self.levels = AudioLevels::default();
            return self.levels;
        }

        let take = FFT_SAMPLE_SIZE.min(self.ring.len());

        // Reuse windowing buffer to avoid allocation - truncate(0) is faster than clear()
        self.windowing_buffer.truncate(0);
        self.windowing_buffer
            .extend(self.ring.iter().rev().take(take));
        self.windowing_buffer.reverse();

        // Calculate RMS
        let rms = self.calculate_rms(&self.windowing_buffer);

        // Perform frequency analysis with cached FFT
        let (low, mid, high) = self.perform_cached_fft_analysis();

        // Apply smoothing
        self.apply_smoothing(rms, low, mid, high);

        // Run lightweight beat detection using the spectrum data we just calculated
        self.detect_beat_from_spectrum();

        self.levels
    }

    fn calculate_rms(&self, buffer: &[f32]) -> f32 {
        let sum_squares: f32 = buffer.iter().map(|x| x * x).sum();
        (sum_squares / buffer.len() as f32).sqrt()
    }

    // #[target_feature(enable = "avx2")]
    // unsafe fn calculate_rms_simd(&self, buffer: &[f32]) -> f32 {
    //     let len = buffer.len();
    //     let simd_len = len - (len % 8);
    //     let mut sum = _mm256_setzero_ps();
    //
    //     for i in (0..simd_len).step_by(8) {
    //         let values = _mm256_loadu_ps(buffer.as_ptr().add(i));
    //         let squared = _mm256_mul_ps(values, values);
    //         sum = _mm256_add_ps(sum, squared);
    //     }
    //
    //     // Sum the SIMD register elements
    //     let mut result = 0.0f32;
    //     let sum_array = std::mem::transmute::<__m256, [f32; 8]>(sum);
    //     for val in sum_array.iter() {
    //         result += val;
    //     }
    //
    //     // Handle remaining elements
    //     for i in simd_len..len {
    //         result += buffer[i] * buffer[i];
    //     }
    //
    //     (result / len as f32).sqrt()
    // }

    fn perform_cached_fft_analysis(&mut self) -> (f32, f32, f32) {
        let fft_len = self
            .windowing_buffer
            .len()
            .next_power_of_two().clamp(MIN_FFT_SIZE, MAX_FFT_SIZE);
        self.windowing_buffer.resize(fft_len, 0.0);

        Self::apply_hann_window(&mut self.windowing_buffer);

        let fft = Arc::clone(
            self.fft_cache
                .entry(fft_len)
                .or_insert_with(|| self.fft_planner.plan_fft_forward(fft_len))
        );

        // truncate(0) is faster than clear() as it preserves capacity
        self.processing_buffer.truncate(0);
        self.processing_buffer.extend(
            self.windowing_buffer
                .iter()
                .map(|&v| Complex32::new(v, 0.0)),
        );

        fft.process(&mut self.processing_buffer);

        self.analyze_frequency_bands(fft_len)
    }

    fn apply_hann_window(buffer: &mut [f32]) {
        let len = buffer.len();
        let denom = (len.saturating_sub(1)).max(1) as f32;
        for (i, sample) in buffer.iter_mut().enumerate() {
            let n = i as f32;
            let w = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * n / denom).cos());
            *sample *= w;
        }
    }

    fn analyze_frequency_bands(&self, fft_len: usize) -> (f32, f32, f32) {
        let sample_rate = self.last_sample_rate as f32;
        let bin_hz = sample_rate / fft_len as f32;
        let mut low = 0.0;
        let mut mid = 0.0;
        let mut high = 0.0;

        for (i, complex) in self.processing_buffer.iter().enumerate().take(fft_len / 2) {
            let frequency = i as f32 * bin_hz;
            let magnitude = complex.norm();

            if frequency < 20.0 {
                continue;
            }

            match frequency {
                f if f <= 250.0 => low += magnitude,
                f if f <= 2000.0 => mid += magnitude,
                f if f <= 8000.0 => high += magnitude,
                _ => {}
            }
        }

        (low, mid, high)
    }

    fn apply_smoothing(&mut self, rms: f32, low: f32, mid: f32, high: f32) {
        const SMOOTHING_FACTOR: f32 = 0.7;
        let normalize = |x: f32| (x / 1000.0).min(1.0);

        self.levels.level_rms =
            SMOOTHING_FACTOR * self.levels.level_rms + (1.0 - SMOOTHING_FACTOR) * rms.min(1.0);
        self.levels.low =
            SMOOTHING_FACTOR * self.levels.low + (1.0 - SMOOTHING_FACTOR) * normalize(low);
        self.levels.mid =
            SMOOTHING_FACTOR * self.levels.mid + (1.0 - SMOOTHING_FACTOR) * normalize(mid);
        self.levels.high =
            SMOOTHING_FACTOR * self.levels.high + (1.0 - SMOOTHING_FACTOR) * normalize(high);
    }


    pub fn push_audio_data(&mut self, data: &[f32], channels: usize, sample_rate: u32) {
        if channels == 1 {
            self.push_samples(data, sample_rate);
        } else {
            // Convert to mono and push directly to ring buffer to avoid borrow conflicts
            self.last_sample_rate = sample_rate;

            let samples_len = data.len() / channels;
            if samples_len == 0 {
                return;
            }

            // Convert and push directly without intermediate buffer
            for frame in data.chunks_exact(channels) {
                let sample = frame.iter().sum::<f32>() / channels as f32;

                // Push directly to ring buffer (inlined from push_samples)
                if self.ring.len() == self.capacity {
                    self.ring.pop_front();
                }
                self.ring.push_back(sample);
            }
        }
    }
}