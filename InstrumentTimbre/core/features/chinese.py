"""
Chinese Instrument Feature Analyzer - Migrated and Enhanced
Enhanced feature extraction specifically designed for Chinese traditional instruments
"""

import numpy as np
import librosa
from scipy import signal
from typing import Dict, Any, Optional, List, Tuple
import warnings

from .base import BaseFeatureExtractor
# from ...modules.utils.chinese_music_theory import ChineseMusicTheory

class ChineseInstrumentAnalyzer(BaseFeatureExtractor):
    """
    Enhanced Chinese instrument analyzer with cultural and acoustic features
    Migrated from legacy chinese_instrument_features.py with improvements
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Chinese instrument analyzer
        
        Args:
            config: Configuration dictionary with instrument-specific parameters
        """
        super().__init__(config)
        
        # Default configuration
        self.sample_rate = self.config.get('sample_rate', 22050)
        self.hop_length = self.config.get('hop_length', 512)
        self.n_fft = self.config.get('n_fft', 2048)
        self.n_mfcc = self.config.get('n_mfcc', 50)
        
        # Chinese instrument specific parameters
        self.instrument_ranges = self.config.get('instrument_ranges', {
            'erhu': {'f_min': 196, 'f_max': 1568},
            'pipa': {'f_min': 220, 'f_max': 2093},
            'guzheng': {'f_min': 196, 'f_max': 2093},
            'dizi': {'f_min': 587, 'f_max': 2349},
            'guqin': {'f_min': 82, 'f_max': 698}
        })
        
        # Initialize Chinese music theory helper
        # self.music_theory = ChineseMusicTheory()  # TODO: Implement when ready
        
        self.logger.info("ChineseInstrumentAnalyzer initialized")
    
    def extract_features(self, audio_data: np.ndarray, sample_rate: int, 
                        instrument_type: Optional[str] = None) -> Dict[str, np.ndarray]:
        """
        Extract comprehensive features for Chinese instruments
        
        Args:
            audio_data: Audio signal as numpy array
            sample_rate: Sample rate of the audio
            instrument_type: Optional instrument type for optimization
            
        Returns:
            Dictionary of extracted features
        """
        self.validate_input(audio_data, sample_rate)
        audio_data = self.preprocess_audio(audio_data)
        
        features = {}
        
        try:
            # Basic acoustic features
            features.update(self._extract_basic_features(audio_data, sample_rate))
            
            # Fundamental frequency and harmonics
            features.update(self._extract_f0_features(audio_data, sample_rate, instrument_type))
            
            # Timbre and spectral features
            features.update(self._extract_timbre_features(audio_data, sample_rate))
            
            # Chinese music specific features
            features.update(self._extract_chinese_features(audio_data, sample_rate, instrument_type))
            
            # Performance technique features
            features.update(self._extract_technique_features(audio_data, sample_rate, instrument_type))
            
            self.logger.debug(f"Extracted {len(features)} feature groups")
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            # Return empty features on failure
            features = {name: np.array([0.0]) for name in self.get_feature_names()}
            
        return features
    
    def _extract_basic_features(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, np.ndarray]:
        """Extract basic acoustic features"""
        features = {}
        
        # MFCC features
        mfccs = librosa.feature.mfcc(
            y=audio_data, 
            sr=sample_rate, 
            n_mfcc=self.n_mfcc,
            hop_length=self.hop_length,
            n_fft=self.n_fft
        )
        features['mfcc'] = np.mean(mfccs, axis=1)
        features['mfcc_delta'] = np.mean(librosa.feature.delta(mfccs), axis=1)
        features['mfcc_delta2'] = np.mean(librosa.feature.delta(mfccs, order=2), axis=1)
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(
            y=audio_data, sr=sample_rate, hop_length=self.hop_length
        )
        features['spectral_centroid'] = np.mean(spectral_centroids)
        
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio_data, sr=sample_rate, hop_length=self.hop_length
        )
        features['spectral_rolloff'] = np.mean(spectral_rolloff)
        
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=audio_data, sr=sample_rate, hop_length=self.hop_length
        )
        features['spectral_bandwidth'] = np.mean(spectral_bandwidth)
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio_data, hop_length=self.hop_length)
        features['zcr'] = np.mean(zcr)
        
        return features
    
    def _extract_f0_features(self, audio_data: np.ndarray, sample_rate: int, 
                           instrument_type: Optional[str] = None) -> Dict[str, np.ndarray]:
        """Extract fundamental frequency related features"""
        features = {}
        
        # Set F0 range based on instrument type
        if instrument_type and instrument_type in self.instrument_ranges:
            fmin = self.instrument_ranges[instrument_type]['f_min']
            fmax = self.instrument_ranges[instrument_type]['f_max']
        else:
            fmin = 80
            fmax = 2000
        
        # Extract F0 using PYIN algorithm (good for Chinese instruments)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio_data, 
            fmin=fmin, 
            fmax=fmax, 
            sr=sample_rate,
            hop_length=self.hop_length
        )
        
        # Remove NaN values
        f0_clean = f0[~np.isnan(f0)]
        
        if len(f0_clean) > 0:
            features['f0_mean'] = np.mean(f0_clean)
            features['f0_std'] = np.std(f0_clean)
            features['f0_median'] = np.median(f0_clean)
            features['f0_range'] = np.max(f0_clean) - np.min(f0_clean)
            
            # Vibrato analysis
            features.update(self._analyze_vibrato(f0_clean, sample_rate))
        else:
            features['f0_mean'] = 0.0
            features['f0_std'] = 0.0
            features['f0_median'] = 0.0
            features['f0_range'] = 0.0
            features['vibrato_rate'] = 0.0
            features['vibrato_extent'] = 0.0
        
        # Voicing features
        features['voicing_ratio'] = np.mean(voiced_probs)
        
        return features
    
    def _extract_timbre_features(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, np.ndarray]:
        """Extract timbre-related features"""
        features = {}
        
        # Chroma features (important for Chinese pentatonic scales)
        chroma = librosa.feature.chroma_cqt(
            y=audio_data, 
            sr=sample_rate, 
            hop_length=self.hop_length
        )
        features['chroma'] = np.mean(chroma, axis=1)
        
        # Tonnetz features (harmonic network)
        tonnetz = librosa.feature.tonnetz(
            y=audio_data, 
            sr=sample_rate
        )
        features['tonnetz'] = np.mean(tonnetz, axis=1)
        
        # Mel-scaled spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio_data, 
            sr=sample_rate, 
            hop_length=self.hop_length,
            n_mels=128
        )
        features['mel_spec'] = np.mean(mel_spec, axis=1)
        
        # Spectral contrast
        contrast = librosa.feature.spectral_contrast(
            y=audio_data, 
            sr=sample_rate, 
            hop_length=self.hop_length
        )
        features['spectral_contrast'] = np.mean(contrast, axis=1)
        
        return features
    
    def _extract_chinese_features(self, audio_data: np.ndarray, sample_rate: int,
                                instrument_type: Optional[str] = None) -> Dict[str, np.ndarray]:
        """Extract Chinese music specific features"""
        features = {}
        
        # Pentatonic scale alignment (simplified implementation)
        chroma = librosa.feature.chroma_cqt(y=audio_data, sr=sample_rate)
        # Simplified pentatonic alignment calculation
        pentatonic_notes = [0, 2, 4, 7, 9]  # Major pentatonic scale indices
        pentatonic_alignment = np.mean([chroma[note] for note in pentatonic_notes])
        features['pentatonic_alignment'] = np.mean(pentatonic_alignment)
        
        # Simplified traditional interval analysis
        features['interval_stability'] = np.std(chroma)
        features['tonal_clarity'] = np.max(chroma) - np.min(chroma)
        
        # Instrument-specific features
        if instrument_type:
            specific_features = self._extract_instrument_specific_features(
                audio_data, sample_rate, instrument_type
            )
            features.update(specific_features)
        
        return features
    
    def _extract_technique_features(self, audio_data: np.ndarray, sample_rate: int,
                                  instrument_type: Optional[str] = None) -> Dict[str, np.ndarray]:
        """Extract performance technique features"""
        features = {}
        
        # Attack and decay analysis
        onset_frames = librosa.onset.onset_detect(
            y=audio_data, 
            sr=sample_rate, 
            hop_length=self.hop_length
        )
        
        if len(onset_frames) > 1:
            onset_times = librosa.frames_to_time(onset_frames, sr=sample_rate)
            note_durations = np.diff(onset_times)
            features['avg_note_duration'] = np.mean(note_durations)
            features['note_duration_std'] = np.std(note_durations)
        else:
            features['avg_note_duration'] = 0.0
            features['note_duration_std'] = 0.0
        
        # Dynamic range
        rms = librosa.feature.rms(y=audio_data, hop_length=self.hop_length)
        features['dynamic_range'] = np.max(rms) - np.min(rms)
        features['rms_mean'] = np.mean(rms)
        features['rms_std'] = np.std(rms)
        
        return features
    
    def _analyze_vibrato(self, f0: np.ndarray, sample_rate: int) -> Dict[str, float]:
        """Analyze vibrato characteristics"""
        if len(f0) < 10:
            return {'vibrato_rate': 0.0, 'vibrato_extent': 0.0}
        
        # Smooth F0 to find vibrato
        from scipy.ndimage import gaussian_filter1d
        f0_smooth = gaussian_filter1d(f0, sigma=2)
        vibrato = f0 - f0_smooth
        
        # Find vibrato rate (frequency of oscillation)
        if len(vibrato) > 0:
            # Use FFT to find dominant frequency in vibrato
            fft = np.fft.fft(vibrato)
            freqs = np.fft.fftfreq(len(vibrato), d=self.hop_length/sample_rate)
            
            # Find peak in positive frequencies
            positive_freqs = freqs[:len(freqs)//2]
            positive_fft = np.abs(fft[:len(fft)//2])
            
            if len(positive_fft) > 0:
                peak_idx = np.argmax(positive_fft[1:]) + 1  # Skip DC component
                vibrato_rate = positive_freqs[peak_idx] if peak_idx < len(positive_freqs) else 0.0
            else:
                vibrato_rate = 0.0
            
            vibrato_extent = np.std(vibrato)
        else:
            vibrato_rate = 0.0
            vibrato_extent = 0.0
        
        return {
            'vibrato_rate': vibrato_rate,
            'vibrato_extent': vibrato_extent
        }
    
    def _extract_instrument_specific_features(self, audio_data: np.ndarray, 
                                            sample_rate: int, 
                                            instrument_type: str) -> Dict[str, np.ndarray]:
        """Extract instrument-specific features"""
        features = {}
        
        if instrument_type == 'erhu':
            # Erhu-specific features (bowing characteristics, slides)
            features.update(self._extract_erhu_features(audio_data, sample_rate))
        elif instrument_type == 'pipa':
            # Pipa-specific features (plucking, tremolo)
            features.update(self._extract_pipa_features(audio_data, sample_rate))
        elif instrument_type == 'guzheng':
            # Guzheng-specific features (plucking, bending)
            features.update(self._extract_guzheng_features(audio_data, sample_rate))
        
        return features
    
    def _extract_erhu_features(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, np.ndarray]:
        """Extract Erhu-specific features"""
        features = {}
        
        # Bowing pressure analysis (via amplitude envelope)
        envelope = np.abs(signal.hilbert(audio_data))
        envelope_smooth = signal.savgol_filter(envelope, window_length=51, polyorder=3)
        
        features['bowing_pressure_var'] = np.var(envelope_smooth)
        features['bowing_smoothness'] = -np.mean(np.abs(np.diff(envelope_smooth, 2)))
        
        return features
    
    def _extract_pipa_features(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, np.ndarray]:
        """Extract Pipa-specific features"""
        features = {}
        
        # Plucking attack analysis
        onset_strength = librosa.onset.onset_strength(
            y=audio_data, sr=sample_rate, hop_length=self.hop_length
        )
        features['attack_sharpness'] = np.max(onset_strength)
        features['attack_consistency'] = -np.std(onset_strength)
        
        return features
    
    def _extract_guzheng_features(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, np.ndarray]:
        """Extract Guzheng-specific features"""
        features = {}
        
        # String resonance analysis
        # Analyze decay characteristics
        envelope = np.abs(signal.hilbert(audio_data))
        
        # Find peaks for decay analysis
        peaks, _ = signal.find_peaks(envelope, height=np.max(envelope) * 0.1)
        
        if len(peaks) > 1:
            # Analyze decay between peaks
            decay_rates = []
            for i in range(len(peaks) - 1):
                start_idx = peaks[i]
                end_idx = peaks[i + 1]
                if end_idx - start_idx > 10:
                    segment = envelope[start_idx:end_idx]
                    if len(segment) > 0 and segment[0] > 0:
                        # Fit exponential decay
                        try:
                            log_segment = np.log(segment / segment[0])
                            decay_rate = -np.polyfit(range(len(log_segment)), log_segment, 1)[0]
                            decay_rates.append(decay_rate)
                        except:
                            pass
            
            if decay_rates:
                features['resonance_decay'] = np.mean(decay_rates)
            else:
                features['resonance_decay'] = 0.0
        else:
            features['resonance_decay'] = 0.0
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names this extractor produces"""
        return [
            # Basic features
            'mfcc', 'mfcc_delta', 'mfcc_delta2',
            'spectral_centroid', 'spectral_rolloff', 'spectral_bandwidth', 'zcr',
            
            # F0 features  
            'f0_mean', 'f0_std', 'f0_median', 'f0_range',
            'vibrato_rate', 'vibrato_extent', 'voicing_ratio',
            
            # Timbre features
            'chroma', 'tonnetz', 'mel_spec', 'spectral_contrast',
            
            # Chinese music features
            'pentatonic_alignment',
            
            # Technique features
            'avg_note_duration', 'note_duration_std',
            'dynamic_range', 'rms_mean', 'rms_std',
            
            # Instrument-specific features (conditional)
            'bowing_pressure_var', 'bowing_smoothness',  # Erhu
            'attack_sharpness', 'attack_consistency',     # Pipa
            'resonance_decay'                             # Guzheng
        ]