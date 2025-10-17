"""
Unified Feature Extractor for Analysis + Generation

This module extends the existing 34-dimensional feature set to 48 dimensions,
adding generation-specific features while maintaining full backward compatibility.
"""

import numpy as np
import librosa
from scipy import signal
from typing import Dict, Any, Optional, List, Tuple
import warnings

from .base import BaseFeatureExtractor
from .chinese import ChineseInstrumentAnalyzer


class UnifiedFeatureExtractor(BaseFeatureExtractor):
    """
    Unified feature extractor supporting both analysis and generation tasks
    
    Features:
    - Extends existing 34D features to 48D
    - Maintains 100% backward compatibility
    - Adds generation-specific features
    - Optimized for both pure music and vocal music
    - Supports minimum 8-second audio segments
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize unified feature extractor
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # Initialize existing Chinese instrument analyzer
        self.chinese_analyzer = ChineseInstrumentAnalyzer(config)
        
        # Generation-specific parameters
        self.min_segment_length = self.config.get('min_segment_length', 8.0)  # 8 seconds minimum
        self.enable_generation_features = self.config.get('enable_generation_features', True)
        self.segment_analysis = self.config.get('segment_analysis', True)
        
        # Feature dimensions
        self.legacy_feature_dim = 34  # Existing Chinese features
        self.generation_feature_dim = 14  # New generation features
        self.total_feature_dim = 48  # Combined features
        
        self.logger.info(f"UnifiedFeatureExtractor initialized: {self.total_feature_dim}D features")
        self.logger.info(f"Generation features: {'enabled' if self.enable_generation_features else 'disabled'}")
    
    def extract_features(self, audio_data: np.ndarray, sample_rate: int, 
                        feature_type: str = 'unified') -> Dict[str, np.ndarray]:
        """
        Extract unified features for analysis and generation
        
        Args:
            audio_data: Audio signal
            sample_rate: Sample rate
            feature_type: Type of features ('legacy', 'generation', 'unified')
            
        Returns:
            Dictionary containing feature vectors
        """
        # Validate audio length
        audio_length = len(audio_data) / sample_rate
        if audio_length < 3.0:
            raise ValueError(f"Audio too short ({audio_length:.1f}s), minimum 3 seconds required")
        
        # Apply confidence penalty for short segments
        confidence_penalty = self._calculate_confidence_penalty(audio_length)
        
        # Extract legacy features (34D) - fully compatible
        legacy_features = self._extract_legacy_features(audio_data, sample_rate)
        
        if feature_type == 'legacy':
            return {
                'features': legacy_features,
                'feature_names': self._get_legacy_feature_names(),
                'confidence_penalty': confidence_penalty,
                'audio_length': audio_length
            }
        
        # Extract generation features (14D) - new functionality
        generation_features = None
        if self.enable_generation_features and feature_type in ['generation', 'unified']:
            generation_features = self._extract_generation_features(audio_data, sample_rate)
        
        if feature_type == 'generation':
            return {
                'features': generation_features,
                'feature_names': self._get_generation_feature_names(),
                'confidence_penalty': confidence_penalty,
                'audio_length': audio_length
            }
        
        # Unified features (48D) - default mode
        if generation_features is not None:
            unified_features = np.concatenate([legacy_features, generation_features])
        else:
            # Fallback to legacy features padded with zeros
            unified_features = np.concatenate([legacy_features, np.zeros(self.generation_feature_dim)])
        
        return {
            'features': unified_features,
            'legacy_features': legacy_features,
            'generation_features': generation_features,
            'feature_names': self._get_unified_feature_names(),
            'confidence_penalty': confidence_penalty,
            'audio_length': audio_length,
            'feature_breakdown': {
                'timbre': legacy_features[:8],
                'pitch': legacy_features[8:14],
                'rhythm': legacy_features[14:22],
                'harmony': legacy_features[22:28],
                'chinese': legacy_features[28:34],
                'melody_structure': unified_features[34:38] if generation_features is not None else np.zeros(4),
                'chord_progression': unified_features[38:42] if generation_features is not None else np.zeros(4),
                'arrangement_hints': unified_features[42:45] if generation_features is not None else np.zeros(3),
                'style_indicators': unified_features[45:48] if generation_features is not None else np.zeros(3)
            }
        }
    
    def _extract_legacy_features(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Extract existing 34D Chinese instrument features"""
        try:
            # Use existing Chinese analyzer
            chinese_features = self.chinese_analyzer.extract_features(audio_data, sample_rate)
            return chinese_features['features']
        except Exception as e:
            self.logger.warning(f"Legacy feature extraction failed: {e}")
            # Return zero features as fallback
            return np.zeros(self.legacy_feature_dim)
    
    def _extract_generation_features(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Extract new 14D generation-specific features"""
        try:
            # Melody structure features (4D)
            melody_features = self._extract_melody_structure_features(audio_data, sample_rate)
            
            # Chord progression features (4D)
            harmony_features = self._extract_chord_progression_features(audio_data, sample_rate)
            
            # Arrangement hints (3D)
            arrangement_features = self._extract_arrangement_features(audio_data, sample_rate)
            
            # Style indicators (3D)
            style_features = self._extract_style_indicators(audio_data, sample_rate)
            
            return np.concatenate([
                melody_features,      # 4D
                harmony_features,     # 4D
                arrangement_features, # 3D
                style_features       # 3D
            ])  # Total: 14D
            
        except Exception as e:
            self.logger.warning(f"Generation feature extraction failed: {e}")
            return np.zeros(self.generation_feature_dim)
    
    def _extract_melody_structure_features(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Extract melody structure features for generation (4D)"""
        
        # Extract pitch track
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio_data, fmin=80, fmax=2000, sr=sample_rate,
            frame_length=2048, hop_length=512
        )
        
        # Remove unvoiced frames
        f0_voiced = f0[voiced_flag]
        
        if len(f0_voiced) < 10:  # Too few voiced frames
            return np.zeros(4)
        
        # Feature 1: Melodic range (normalized)
        pitch_range = np.log2(np.max(f0_voiced) / np.min(f0_voiced)) if len(f0_voiced) > 0 else 0
        pitch_range_norm = np.clip(pitch_range / 4.0, 0, 1)  # Normalize to [0,1]
        
        # Feature 2: Melodic contour complexity
        pitch_diff = np.diff(f0_voiced)
        direction_changes = np.sum(np.diff(np.sign(pitch_diff)) != 0)
        contour_complexity = direction_changes / len(pitch_diff) if len(pitch_diff) > 0 else 0
        
        # Feature 3: Melodic density (notes per second)
        # Detect note onsets
        onset_frames = librosa.onset.onset_detect(
            y=audio_data, sr=sample_rate, hop_length=512, units='frames'
        )
        note_density = len(onset_frames) / (len(audio_data) / sample_rate)
        note_density_norm = np.clip(note_density / 10.0, 0, 1)  # Normalize
        
        # Feature 4: Melodic stability (consistency of pitch)
        pitch_std = np.std(f0_voiced) if len(f0_voiced) > 0 else 0
        pitch_stability = 1.0 / (1.0 + pitch_std / 100.0)  # Inverse of variability
        
        return np.array([
            pitch_range_norm,
            contour_complexity,
            note_density_norm,
            pitch_stability
        ])
    
    def _extract_chord_progression_features(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Extract chord progression features for generation (4D)"""
        
        # Compute chromagram
        chroma = librosa.feature.chroma_cqt(
            y=audio_data, sr=sample_rate, hop_length=512, n_chroma=12
        )
        
        if chroma.shape[1] < 10:  # Too short
            return np.zeros(4)
        
        # Feature 1: Harmonic clarity (how clear are the chord progressions)
        chroma_mean = np.mean(chroma, axis=1)
        harmonic_clarity = np.max(chroma_mean) / (np.mean(chroma_mean) + 1e-8)
        harmonic_clarity_norm = np.clip((harmonic_clarity - 1.0) / 2.0, 0, 1)
        
        # Feature 2: Harmonic change rate
        chroma_diff = np.diff(chroma, axis=1)
        change_magnitude = np.mean(np.sum(np.abs(chroma_diff), axis=0))
        change_rate_norm = np.clip(change_magnitude * 2.0, 0, 1)
        
        # Feature 3: Tonal stability (consistency with major/minor keys)
        # Compute correlation with major/minor templates
        major_template = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
        minor_template = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])
        
        major_corr = np.corrcoef(chroma_mean, major_template)[0, 1]
        minor_corr = np.corrcoef(chroma_mean, minor_template)[0, 1]
        tonal_stability = max(abs(major_corr), abs(minor_corr)) if not np.isnan(major_corr) else 0
        
        # Feature 4: Harmonic complexity (number of active pitches)
        active_pitches = np.mean(np.sum(chroma > 0.1, axis=0))
        harmonic_complexity = active_pitches / 12.0  # Normalize by max possible pitches
        
        return np.array([
            harmonic_clarity_norm,
            change_rate_norm,
            tonal_stability,
            harmonic_complexity
        ])
    
    def _extract_arrangement_features(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Extract arrangement hints for generation (3D)"""
        
        # Compute spectral features
        stft = librosa.stft(audio_data, hop_length=512, n_fft=2048)
        magnitude = np.abs(stft)
        
        # Feature 1: Spectral density (how full the arrangement is)
        spectral_density = np.mean(np.sum(magnitude > 0.01, axis=0)) / magnitude.shape[0]
        
        # Feature 2: Dynamic range (contrast between loud and soft parts)
        rms = librosa.feature.rms(y=audio_data, hop_length=512)[0]
        if len(rms) > 0:
            dynamic_range = (np.percentile(rms, 95) - np.percentile(rms, 5)) / (np.max(rms) + 1e-8)
        else:
            dynamic_range = 0
        
        # Feature 3: Textural complexity (how many different elements are present)
        # Use spectral rolloff and centroid variation
        rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate, hop_length=512)[0]
        centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate, hop_length=512)[0]
        
        rolloff_var = np.std(rolloff) / (np.mean(rolloff) + 1e-8)
        centroid_var = np.std(centroid) / (np.mean(centroid) + 1e-8)
        textural_complexity = (rolloff_var + centroid_var) / 2.0
        textural_complexity_norm = np.clip(textural_complexity, 0, 1)
        
        return np.array([
            spectral_density,
            dynamic_range,
            textural_complexity_norm
        ])
    
    def _extract_style_indicators(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Extract style indicators for generation (3D)"""
        
        # Feature 1: Rhythmic regularity (how steady the beat is)
        tempo, beats = librosa.beat.beat_track(y=audio_data, sr=sample_rate, hop_length=512)
        if len(beats) > 2:
            beat_intervals = np.diff(beats)
            rhythmic_regularity = 1.0 / (1.0 + np.std(beat_intervals) / np.mean(beat_intervals))
        else:
            rhythmic_regularity = 0.5
        
        # Feature 2: Spectral traditionality (how traditional vs modern the timbre is)
        # Traditional instruments have more harmonics, modern sounds more inharmonic
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate)[0]
        
        traditionality = np.mean(spectral_rolloff / (spectral_centroid + 1e-8))
        traditionality_norm = np.clip((traditionality - 2.0) / 4.0, 0, 1)
        
        # Feature 3: Tempo category (slow/medium/fast indicator)
        tempo_category = 0.0  # slow
        if tempo > 80:
            tempo_category = 0.5  # medium
        if tempo > 120:
            tempo_category = 1.0  # fast
        
        return np.array([
            rhythmic_regularity,
            traditionality_norm,
            tempo_category
        ])
    
    def _calculate_confidence_penalty(self, audio_length: float) -> float:
        """Calculate confidence penalty based on audio length"""
        if audio_length >= self.min_segment_length:
            return 0.0  # No penalty for adequate length
        elif audio_length >= 5.0:
            return 0.1  # Small penalty for short but usable
        elif audio_length >= 3.0:
            return 0.4  # Large penalty for very short
        else:
            return 0.8  # Severe penalty for extremely short
    
    def _get_legacy_feature_names(self) -> List[str]:
        """Get names of legacy 34D features"""
        return [
            # Timbre features (8D)
            'spectral_centroid', 'spectral_rolloff', 'spectral_bandwidth', 'spectral_contrast',
            'mfcc_mean', 'zero_crossing_rate', 'rms_energy', 'spectral_flatness',
            
            # Pitch features (6D)
            'fundamental_freq', 'pitch_stability', 'vibrato_rate', 'vibrato_extent',
            'pitch_range', 'inharmonicity',
            
            # Rhythm features (8D)
            'tempo', 'beat_strength', 'rhythm_regularity', 'onset_density',
            'attack_time', 'decay_time', 'pulse_clarity', 'meter_strength',
            
            # Harmony features (6D)
            'harmonic_centroid', 'harmonic_rolloff', 'harmonic_spread', 'harmonic_deviation',
            'chord_clarity', 'tonality_strength',
            
            # Chinese-specific features (6D)
            'pentatonic_alignment', 'traditional_interval_ratio', 'ornament_density',
            'glissando_frequency', 'tremolo_intensity', 'cultural_mode_strength'
        ]
    
    def _get_generation_feature_names(self) -> List[str]:
        """Get names of generation 14D features"""
        return [
            # Melody structure (4D)
            'melodic_range', 'contour_complexity', 'note_density', 'pitch_stability',
            
            # Chord progression (4D)
            'harmonic_clarity', 'harmonic_change_rate', 'tonal_stability', 'harmonic_complexity',
            
            # Arrangement hints (3D)
            'spectral_density', 'dynamic_range', 'textural_complexity',
            
            # Style indicators (3D)
            'rhythmic_regularity', 'spectral_traditionality', 'tempo_category'
        ]
    
    def _get_unified_feature_names(self) -> List[str]:
        """Get names of unified 48D features"""
        return self._get_legacy_feature_names() + self._get_generation_feature_names()
    
    def extract_for_segment_analysis(self, audio_data: np.ndarray, sample_rate: int,
                                   segment_length: float = 15.0, 
                                   overlap: float = 5.0) -> List[Dict[str, Any]]:
        """
        Extract features for multiple segments - useful for long audio files
        
        Args:
            audio_data: Audio signal
            sample_rate: Sample rate
            segment_length: Length of each segment in seconds
            overlap: Overlap between segments in seconds
            
        Returns:
            List of feature dictionaries for each segment
        """
        if not self.segment_analysis:
            # Return single segment analysis
            return [self.extract_features(audio_data, sample_rate)]
        
        segments = []
        audio_length = len(audio_data) / sample_rate
        
        if audio_length <= segment_length:
            # Audio shorter than segment length, analyze as single segment
            return [self.extract_features(audio_data, sample_rate)]
        
        # Generate overlapping segments
        step_size = segment_length - overlap
        start_time = 0.0
        
        while start_time + self.min_segment_length <= audio_length:
            end_time = min(start_time + segment_length, audio_length)
            
            # Extract segment
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            segment_audio = audio_data[start_sample:end_sample]
            
            # Extract features for segment
            segment_features = self.extract_features(segment_audio, sample_rate)
            segment_features['start_time'] = start_time
            segment_features['end_time'] = end_time
            segment_features['segment_duration'] = end_time - start_time
            
            segments.append(segment_features)
            
            # Move to next segment
            start_time += step_size
            
            # Break if we've covered the audio
            if end_time >= audio_length:
                break
        
        return segments
    
    def get_feature_dim(self) -> int:
        """Get total feature dimension"""
        return self.total_feature_dim