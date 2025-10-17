"""
Enhanced feature extraction for Chinese traditional instruments.

Provides specialized algorithms for extracting culturally-relevant features
from Chinese instruments with emphasis on traditional musical characteristics.
"""

import numpy as np
import librosa
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from ..core import AudioFeatures, InstrumentType, get_logger


@dataclass
class ChineseInstrumentFeatures:
    """
    Extended features specifically for Chinese traditional instruments.
    
    Captures traditional musical elements like:
    - Sliding techniques (滑音)
    - Vibrato patterns (颤音) 
    - Tonal characteristics
    - Harmonic structures unique to Chinese music
    """
    # Traditional technique features
    sliding_detection: Optional[np.ndarray] = None      # Sliding note detection
    vibrato_analysis: Optional[Dict[str, float]] = None # Vibrato characteristics
    pitch_bending: Optional[np.ndarray] = None          # Pitch bend analysis
    
    # Tonal features for Chinese music
    pentatonic_adherence: Optional[float] = None        # Pentatonic scale usage
    microtonal_variations: Optional[np.ndarray] = None  # Quarter-tone variations
    
    # Instrument-specific features
    string_resonance: Optional[np.ndarray] = None       # String resonance patterns
    breath_control: Optional[np.ndarray] = None         # Wind instrument breath
    strike_characteristics: Optional[Dict[str, float]] = None  # Percussion attacks
    
    # Cultural musical elements
    ornament_density: Optional[float] = None            # Decorative note density
    rhythmic_complexity: Optional[float] = None         # Traditional rhythm patterns


class ChineseInstrumentAnalyzer:
    """
    Advanced analyzer for Chinese traditional instruments.
    
    Implements culturally-aware feature extraction that recognizes
    traditional playing techniques and musical characteristics.
    """
    
    def __init__(self):
        """Initialize Chinese instrument analyzer."""
        self.logger = get_logger()
        
        # Chinese music theory constants
        self.pentatonic_intervals = [0, 2, 4, 7, 9]  # Major pentatonic
        self.traditional_scales = {
            'gong': [0, 2, 4, 7, 9],      # 宫调
            'shang': [0, 2, 5, 7, 10],    # 商调
            'jue': [0, 3, 5, 8, 10],      # 角调
            'zhi': [0, 2, 5, 7, 9],       # 徵调
            'yu': [0, 3, 5, 8, 10]        # 羽调
        }
        
        # Instrument-specific parameters
        self.instrument_params = {
            InstrumentType.ERHU: {
                'fundamental_range': (196, 1568),    # G3 to G6
                'expected_vibrato_rate': (4, 8),     # Hz
                'sliding_sensitivity': 0.7
            },
            InstrumentType.PIPA: {
                'fundamental_range': (220, 2093),    # A3 to C7
                'attack_time_range': (0.01, 0.1),    # Fast attack
                'tremolo_detection': True
            },
            InstrumentType.GUZHENG: {
                'fundamental_range': (196, 2093),    # G3 to C7
                'resonance_decay': (2.0, 8.0),       # Long resonance
                'glissando_common': True
            },
            InstrumentType.DIZI: {
                'fundamental_range': (293, 2349),    # D4 to D7
                'breath_noise_expected': True,
                'trill_frequency': (8, 15)           # Hz
            }
        }
    
    def extract_chinese_features(self, 
                                audio_data: np.ndarray, 
                                sample_rate: int,
                                instrument_hint: Optional[InstrumentType] = None) -> ChineseInstrumentFeatures:
        """
        Extract comprehensive Chinese instrument features.
        
        Args:
            audio_data: Audio signal data
            sample_rate: Audio sample rate
            instrument_hint: Expected instrument type for optimization
            
        Returns:
            ChineseInstrumentFeatures with extracted cultural features
        """
        features = ChineseInstrumentFeatures()
        
        try:
            # Extract fundamental frequency for pitch analysis
            f0 = self._extract_fundamental_frequency(audio_data, sample_rate)
            
            # Analyze traditional techniques
            features.sliding_detection = self._detect_sliding_techniques(f0, sample_rate)
            features.vibrato_analysis = self._analyze_vibrato_patterns(f0, sample_rate)
            features.pitch_bending = self._analyze_pitch_bending(f0)
            
            # Analyze tonal characteristics
            features.pentatonic_adherence = self._analyze_pentatonic_usage(f0)
            features.microtonal_variations = self._detect_microtonal_variations(f0)
            
            # Instrument-specific analysis
            if instrument_hint:
                features = self._add_instrument_specific_features(
                    features, audio_data, sample_rate, instrument_hint
                )
            
            # Cultural musical elements
            features.ornament_density = self._calculate_ornament_density(f0)
            features.rhythmic_complexity = self._analyze_rhythmic_complexity(audio_data, sample_rate)
            
            self.logger.debug("Chinese instrument features extracted successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to extract Chinese features: {e}")
        
        return features
    
    def _extract_fundamental_frequency(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Extract fundamental frequency with high precision for Chinese instruments.
        
        Args:
            audio_data: Audio signal
            sample_rate: Sample rate
            
        Returns:
            F0 contour in Hz
        """
        # Use PYIN algorithm for better pitch tracking of traditional instruments
        f0 = librosa.pyin(audio_data, 
                         fmin=librosa.note_to_hz('C2'),  # 65 Hz
                         fmax=librosa.note_to_hz('C7'),  # 2093 Hz
                         sr=sample_rate,
                         hop_length=512)[0]
        return f0
    
    def _detect_sliding_techniques(self, f0: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Detect sliding techniques (滑音) common in Chinese instruments.
        
        Args:
            f0: Fundamental frequency contour
            sample_rate: Sample rate
            
        Returns:
            Array indicating sliding regions
        """
        if len(f0) < 10:
            return np.zeros_like(f0)
        
        # Calculate pitch derivatives to find sliding regions
        valid_f0 = f0[~np.isnan(f0)]
        if len(valid_f0) < 5:
            return np.zeros_like(f0)
        
        # Convert to cents for better sensitivity
        log_f0 = np.log2(valid_f0 + 1e-10) * 1200
        
        # Smooth derivative to reduce noise
        diff = np.gradient(log_f0)
        smoothed_diff = librosa.util.smooth(diff, length=5)
        
        # Detect significant pitch changes (> 50 cents)
        sliding_threshold = 50  # cents
        sliding_regions = np.abs(smoothed_diff) > sliding_threshold
        
        # Expand to original f0 length
        result = np.zeros_like(f0)
        result[~np.isnan(f0)] = sliding_regions
        
        return result.astype(float)
    
    def _analyze_vibrato_patterns(self, f0: np.ndarray, sample_rate: int) -> Dict[str, float]:
        """
        Analyze vibrato patterns specific to Chinese instruments.
        
        Args:
            f0: Fundamental frequency contour
            sample_rate: Sample rate
            
        Returns:
            Dictionary with vibrato characteristics
        """
        vibrato_analysis = {
            'rate_hz': 0.0,
            'extent_cents': 0.0,
            'regularity': 0.0,
            'presence': 0.0
        }
        
        # Remove NaN values
        valid_f0 = f0[~np.isnan(f0)]
        if len(valid_f0) < 100:  # Need sufficient data
            return vibrato_analysis
        
        try:
            # Convert to cents for analysis
            log_f0 = np.log2(valid_f0 + 1e-10) * 1200
            
            # Remove trend (long-term pitch changes)
            detrended = librosa.decompose.hpss(log_f0, margin=3.0)[1]
            
            # Find vibrato frequency using autocorrelation
            hop_length = 512
            time_per_frame = hop_length / sample_rate
            
            # Autocorrelation to find periodicity
            autocorr = np.correlate(detrended, detrended, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Find peaks in autocorrelation
            min_period = int(0.1 / time_per_frame)  # 10 Hz max
            max_period = int(2.0 / time_per_frame)  # 0.5 Hz min
            
            if max_period < len(autocorr):
                autocorr_region = autocorr[min_period:max_period]
                if len(autocorr_region) > 0:
                    peak_idx = np.argmax(autocorr_region) + min_period
                    vibrato_rate = 1.0 / (peak_idx * time_per_frame)
                    
                    if 2.0 <= vibrato_rate <= 15.0:  # Reasonable vibrato range
                        vibrato_analysis['rate_hz'] = vibrato_rate
                        vibrato_analysis['extent_cents'] = np.std(detrended)
                        vibrato_analysis['regularity'] = autocorr[peak_idx] / autocorr[0]
                        vibrato_analysis['presence'] = min(1.0, vibrato_analysis['extent_cents'] / 50.0)
        
        except Exception as e:
            self.logger.debug(f"Vibrato analysis failed: {e}")
        
        return vibrato_analysis
    
    def _analyze_pitch_bending(self, f0: np.ndarray) -> np.ndarray:
        """
        Analyze pitch bending characteristics.
        
        Args:
            f0: Fundamental frequency contour
            
        Returns:
            Array of pitch bend values in cents
        """
        if len(f0) < 3:
            return np.zeros_like(f0)
        
        valid_f0 = f0[~np.isnan(f0)]
        if len(valid_f0) < 3:
            return np.zeros_like(f0)
        
        # Convert to cents
        log_f0 = np.log2(valid_f0 + 1e-10) * 1200
        
        # Calculate local pitch deviation from expected notes
        # Quantize to nearest semitone
        semitones = np.round(log_f0 / 100) * 100
        pitch_bend = log_f0 - semitones
        
        # Map back to original f0 length
        result = np.zeros_like(f0)
        result[~np.isnan(f0)] = pitch_bend
        
        return result
    
    def _analyze_pentatonic_usage(self, f0: np.ndarray) -> float:
        """
        Analyze adherence to pentatonic scales.
        
        Args:
            f0: Fundamental frequency contour
            
        Returns:
            Pentatonic adherence score (0.0-1.0)
        """
        from .chinese_music_theory import ChineseMusicTheory
        
        theory = ChineseMusicTheory()
        
        # Convert frequencies to MIDI notes
        valid_f0 = f0[~np.isnan(f0)]
        if len(valid_f0) == 0:
            return 0.0
        
        midi_notes = librosa.hz_to_midi(valid_f0)
        return theory.analyze_pentatonic_adherence(midi_notes)
    
    def _detect_microtonal_variations(self, f0: np.ndarray) -> np.ndarray:
        """
        Detect microtonal variations (quarter-tones, etc.).
        
        Args:
            f0: Fundamental frequency contour
            
        Returns:
            Array of microtonal deviations in cents
        """
        if len(f0) < 3:
            return np.zeros_like(f0)
        
        valid_f0 = f0[~np.isnan(f0)]
        if len(valid_f0) < 3:
            return np.zeros_like(f0)
        
        # Convert to cents
        log_f0 = np.log2(valid_f0 + 1e-10) * 1200
        
        # Find deviations from 12-TET (12-tone equal temperament)
        semitone_grid = np.round(log_f0 / 100) * 100
        microtonal_deviations = log_f0 - semitone_grid
        
        # Map back to original length
        result = np.zeros_like(f0)
        result[~np.isnan(f0)] = microtonal_deviations
        
        return result
    
    def _add_instrument_specific_features(self, 
                                        features: ChineseInstrumentFeatures,
                                        audio_data: np.ndarray,
                                        sample_rate: int,
                                        instrument: InstrumentType) -> ChineseInstrumentFeatures:
        """
        Add instrument-specific features based on instrument type.
        
        Args:
            features: Current feature set
            audio_data: Audio signal
            sample_rate: Sample rate
            instrument: Instrument type
            
        Returns:
            Enhanced features with instrument-specific analysis
        """
        if instrument in [InstrumentType.PIPA, InstrumentType.GUZHENG, InstrumentType.GUQIN]:
            # Plucked string instruments
            features.string_resonance = self._analyze_string_resonance(audio_data, sample_rate)
            features.strike_characteristics = self._analyze_pluck_characteristics(audio_data, sample_rate)
            
        elif instrument in [InstrumentType.DIZI, InstrumentType.XIAO, InstrumentType.SUONA]:
            # Wind instruments
            features.breath_control = self._analyze_breath_control(audio_data, sample_rate)
            
        elif instrument == InstrumentType.ERHU:
            # Bowed string instrument
            features.string_resonance = self._analyze_bow_resonance(audio_data, sample_rate)
        
        return features
    
    def _analyze_string_resonance(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Analyze string resonance patterns for plucked instruments."""
        # Calculate spectral decay characteristics
        stft = librosa.stft(audio_data)
        magnitude = np.abs(stft)
        
        # Track energy decay in different frequency bands
        decay_analysis = []
        for freq_bin in range(0, magnitude.shape[0], 10):
            if freq_bin < magnitude.shape[0]:
                energy_contour = magnitude[freq_bin, :]
                if np.max(energy_contour) > 0:
                    # Normalize and calculate decay rate
                    normalized = energy_contour / np.max(energy_contour)
                    decay_rate = -np.polyfit(range(len(normalized)), np.log(normalized + 1e-10), 1)[0]
                    decay_analysis.append(decay_rate)
        
        return np.array(decay_analysis) if decay_analysis else np.array([0.0])
    
    def _analyze_breath_control(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Analyze breath control patterns for wind instruments."""
        # Extract amplitude envelope
        amplitude_envelope = librosa.onset.onset_strength(y=audio_data, sr=sample_rate)
        
        # Analyze breathing patterns (periodic amplitude modulation)
        # Typical breathing cycle: 2-6 seconds
        breath_freq_range = (1/6, 1/2)  # 0.17-0.5 Hz
        
        # Use autocorrelation to find breathing periodicity
        autocorr = np.correlate(amplitude_envelope, amplitude_envelope, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        return autocorr[:min(100, len(autocorr))]  # Return first 100 values
    
    def _analyze_pluck_characteristics(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, float]:
        """Analyze plucking attack characteristics."""
        # Onset detection
        onsets = librosa.onset.onset_detect(y=audio_data, sr=sample_rate, units='time')
        
        characteristics = {
            'attack_sharpness': 0.0,
            'decay_rate': 0.0,
            'harmonic_richness': 0.0
        }
        
        if len(onsets) > 0:
            # Analyze first few onsets
            for i, onset_time in enumerate(onsets[:5]):
                onset_sample = int(onset_time * sample_rate)
                if onset_sample + 1024 < len(audio_data):
                    attack_segment = audio_data[onset_sample:onset_sample + 1024]
                    
                    # Attack sharpness (rise time)
                    envelope = np.abs(attack_segment)
                    peak_idx = np.argmax(envelope)
                    if peak_idx > 0:
                        attack_time = peak_idx / sample_rate
                        characteristics['attack_sharpness'] += 1.0 / (attack_time + 1e-6)
            
            # Average across onsets
            if len(onsets) > 0:
                characteristics['attack_sharpness'] /= min(5, len(onsets))
        
        return characteristics
    
    def _analyze_bow_resonance(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Analyze bowing resonance for erhu and similar instruments."""
        # Extract spectral features that indicate bowing technique
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate)[0]
        
        # Bow resonance is characterized by specific spectral relationships
        bow_resonance = spectral_rolloff / (spectral_centroids + 1e-10)
        
        return bow_resonance
    
    def _calculate_ornament_density(self, f0: np.ndarray) -> float:
        """Calculate density of ornamental notes."""
        valid_f0 = f0[~np.isnan(f0)]
        if len(valid_f0) < 10:
            return 0.0
        
        # Convert to semitones
        semitones = librosa.hz_to_midi(valid_f0)
        
        # Count rapid pitch changes (ornaments)
        pitch_changes = np.abs(np.diff(semitones))
        ornament_threshold = 0.5  # semitones
        ornaments = np.sum(pitch_changes > ornament_threshold)
        
        return ornaments / len(valid_f0)
    
    def _analyze_rhythmic_complexity(self, audio_data: np.ndarray, sample_rate: int) -> float:
        """Analyze rhythmic complexity typical of Chinese music."""
        # Extract onset times
        onsets = librosa.onset.onset_detect(y=audio_data, sr=sample_rate, units='time')
        
        if len(onsets) < 4:
            return 0.0
        
        # Calculate inter-onset intervals
        intervals = np.diff(onsets)
        
        # Rhythmic complexity based on interval variation
        if len(intervals) > 1:
            complexity = np.std(intervals) / (np.mean(intervals) + 1e-10)
            return min(1.0, complexity)  # Normalize to 0-1
        
        return 0.0