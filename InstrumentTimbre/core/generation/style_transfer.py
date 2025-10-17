"""
Style Transfer Engine - Core Generation Component

This module implements the core algorithm for transferring musical styles
while preserving melody characteristics using the melody preservation engine.

Supports three initial styles:
1. Chinese Traditional (中国传统) - Traditional Chinese instruments and scales
2. Western Classical (西方古典) - Classical Western harmony and orchestration  
3. Modern Pop (现代流行) - Contemporary pop music characteristics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d

from .melody_preservation import MelodyPreservationEngine


class StyleTransferEngine:
    """
    Core engine for transferring musical styles while preserving melody
    
    This engine can transform music between different styles while maintaining
    the essential melodic characteristics identified by the MelodyPreservationEngine.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize style transfer engine
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize melody preservation engine
        melody_config = self.config.get('melody_preservation', {})
        self.melody_engine = MelodyPreservationEngine(melody_config)
        
        # Style parameters
        self.target_style = self.config.get('target_style', 'chinese_traditional')
        self.preservation_threshold = self.config.get('preservation_threshold', 0.7)
        
        # Audio processing parameters
        self.sample_rate = self.config.get('sample_rate', 22050)
        self.hop_length = self.config.get('hop_length', 512)
        self.frame_length = self.config.get('frame_length', 2048)
        
        # Style definitions
        self.style_definitions = self._initialize_style_definitions()
        
        self.logger.info("StyleTransferEngine initialized")
        self.logger.info(f"Target style: {self.target_style}")
        self.logger.info(f"Preservation threshold: {self.preservation_threshold}")
    
    def _initialize_style_definitions(self) -> Dict[str, Dict[str, Any]]:
        """Initialize style parameter definitions"""
        
        return {
            'chinese_traditional': {
                'name': '中国传统',
                'scale_type': 'pentatonic',
                'preferred_intervals': [0, 2, 4, 7, 9],  # Pentatonic scale intervals
                'harmonic_complexity': 'simple',
                'rhythm_patterns': ['traditional_chinese'],
                'instrument_timbres': ['erhu', 'guzheng', 'dizi', 'pipa'],
                'tempo_characteristics': {'preferred_range': (60, 120), 'rubato': True},
                'dynamic_range': 'moderate',
                'vibrato_style': 'chinese_traditional'
            },
            
            'western_classical': {
                'name': '西方古典',
                'scale_type': 'major_minor',
                'preferred_intervals': [0, 2, 4, 5, 7, 9, 11],  # Diatonic scale
                'harmonic_complexity': 'complex',
                'rhythm_patterns': ['classical_meters'],
                'instrument_timbres': ['violin', 'piano', 'clarinet', 'cello'],
                'tempo_characteristics': {'preferred_range': (80, 160), 'rubato': True},
                'dynamic_range': 'wide',
                'vibrato_style': 'western_classical'
            },
            
            'modern_pop': {
                'name': '现代流行',
                'scale_type': 'pop_modes',
                'preferred_intervals': [0, 2, 3, 5, 7, 9, 10],  # Pop-friendly intervals
                'harmonic_complexity': 'moderate',
                'rhythm_patterns': ['pop_beats'],
                'instrument_timbres': ['electric_guitar', 'synth', 'bass', 'drums'],
                'tempo_characteristics': {'preferred_range': (100, 140), 'rubato': False},
                'dynamic_range': 'compressed',
                'vibrato_style': 'modern'
            }
        }
    
    def transfer_style(self, 
                      audio_data: np.ndarray, 
                      target_style: str,
                      intensity: float = 0.8,
                      preserve_melody: bool = True) -> Dict[str, Any]:
        """
        Transfer audio to target style while preserving melody
        
        Args:
            audio_data: Input audio signal
            target_style: Target style ('chinese_traditional', 'western_classical', 'modern_pop')
            intensity: Style transfer intensity [0, 1]
            preserve_melody: Whether to enforce melody preservation
            
        Returns:
            Dictionary containing transfer results
        """
        try:
            # Step 1: Extract original melody DNA if preservation is required
            original_dna = None
            if preserve_melody:
                original_dna = self.melody_engine.extract_melody_dna(audio_data)
                self.logger.info("Original melody DNA extracted")
            
            # Step 2: Analyze current style characteristics
            current_style = self._analyze_current_style(audio_data)
            
            # Step 3: Apply style transformation
            transformed_audio = self._apply_style_transformation(
                audio_data, target_style, intensity, current_style
            )
            
            # Step 4: Validate melody preservation
            preservation_score = 1.0
            if preserve_melody and original_dna is not None:
                validation_result = self.melody_engine.validate_preservation(
                    audio_data, transformed_audio
                )
                preservation_score = validation_result['overall_similarity']
                
                # If preservation score is too low, reduce intensity and retry
                if preservation_score < self.preservation_threshold:
                    self.logger.warning(f"Melody preservation too low ({preservation_score:.3f}), reducing intensity")
                    reduced_intensity = intensity * 0.7
                    transformed_audio = self._apply_style_transformation(
                        audio_data, target_style, reduced_intensity, current_style
                    )
                    # Re-validate
                    validation_result = self.melody_engine.validate_preservation(
                        audio_data, transformed_audio
                    )
                    preservation_score = validation_result['overall_similarity']
            
            # Step 5: Apply final polishing
            final_audio = self._apply_style_polishing(transformed_audio, target_style)
            
            return {
                'transformed_audio': final_audio,
                'original_style': current_style,
                'target_style': target_style,
                'applied_intensity': intensity,
                'preservation_score': preservation_score,
                'transformation_successful': True,
                'original_dna': original_dna,
                'style_characteristics': self.style_definitions[target_style]
            }
            
        except Exception as e:
            self.logger.error(f"Style transfer failed: {e}")
            return {
                'transformed_audio': audio_data,  # Return original on failure
                'error': str(e),
                'transformation_successful': False,
                'preservation_score': 0.0
            }
    
    def _analyze_current_style(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Analyze the current style characteristics of the audio"""
        
        # Extract basic acoustic features
        tempo, beats = librosa.beat.beat_track(
            y=audio_data, sr=self.sample_rate, hop_length=self.hop_length
        )
        
        # Extract spectral features
        spectral_centroids = librosa.feature.spectral_centroid(
            y=audio_data, sr=self.sample_rate, hop_length=self.hop_length
        )[0]
        
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio_data, sr=self.sample_rate, hop_length=self.hop_length
        )[0]
        
        # Extract pitch information
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio_data, fmin=80, fmax=2000, sr=self.sample_rate
        )
        
        # Analyze harmonic content
        harmonic = librosa.effects.harmonic(audio_data)
        percussive = librosa.effects.percussive(audio_data)
        
        return {
            'tempo': tempo,
            'beat_times': librosa.frames_to_time(beats, sr=self.sample_rate, hop_length=self.hop_length),
            'spectral_centroid_mean': np.mean(spectral_centroids),
            'spectral_centroid_std': np.std(spectral_centroids),
            'spectral_rolloff_mean': np.mean(spectral_rolloff),
            'harmonic_strength': np.mean(np.abs(harmonic)) / (np.mean(np.abs(audio_data)) + 1e-8),
            'percussive_strength': np.mean(np.abs(percussive)) / (np.mean(np.abs(audio_data)) + 1e-8),
            'pitch_range': np.max(f0[voiced_flag]) - np.min(f0[voiced_flag]) if np.any(voiced_flag) else 0,
            'pitch_stability': np.std(f0[voiced_flag]) if np.any(voiced_flag) else 0
        }
    
    def _apply_style_transformation(self, 
                                  audio_data: np.ndarray,
                                  target_style: str,
                                  intensity: float,
                                  current_style: Dict[str, Any]) -> np.ndarray:
        """Apply the core style transformation"""
        
        if target_style not in self.style_definitions:
            raise ValueError(f"Unknown target style: {target_style}")
        
        style_params = self.style_definitions[target_style]
        transformed = audio_data.copy()
        
        # 1. Tempo adjustment
        transformed = self._adjust_tempo(transformed, style_params, current_style, intensity)
        
        # 2. Harmonic transformation
        transformed = self._transform_harmonics(transformed, style_params, intensity)
        
        # 3. Timbral transformation
        transformed = self._transform_timbre(transformed, style_params, intensity)
        
        # 4. Dynamic range adjustment
        transformed = self._adjust_dynamics(transformed, style_params, intensity)
        
        # 5. Style-specific effects
        transformed = self._apply_style_effects(transformed, target_style, intensity)
        
        return transformed
    
    def _adjust_tempo(self, 
                     audio_data: np.ndarray,
                     style_params: Dict[str, Any],
                     current_style: Dict[str, Any],
                     intensity: float) -> np.ndarray:
        """Adjust tempo to match target style"""
        
        current_tempo = current_style['tempo']
        target_range = style_params['tempo_characteristics']['preferred_range']
        target_tempo = np.mean(target_range)
        
        # Calculate tempo adjustment ratio
        tempo_ratio = target_tempo / current_tempo
        
        # Apply intensity scaling
        tempo_ratio = 1.0 + intensity * (tempo_ratio - 1.0)
        
        # Limit tempo change to reasonable range
        tempo_ratio = np.clip(tempo_ratio, 0.7, 1.5)
        
        # Apply tempo change using librosa
        if abs(tempo_ratio - 1.0) > 0.05:  # Only apply if significant change
            try:
                transformed = librosa.effects.time_stretch(audio_data, rate=tempo_ratio)
                
                # Ensure output length matches input length
                if len(transformed) != len(audio_data):
                    if len(transformed) > len(audio_data):
                        # Trim excess
                        transformed = transformed[:len(audio_data)]
                    else:
                        # Pad with zeros
                        padding = len(audio_data) - len(transformed)
                        transformed = np.pad(transformed, (0, padding), mode='constant')
                
                return transformed
            except:
                self.logger.warning("Tempo adjustment failed, returning original")
                return audio_data
        
        return audio_data
    
    def _transform_harmonics(self, 
                           audio_data: np.ndarray,
                           style_params: Dict[str, Any],
                           intensity: float) -> np.ndarray:
        """Transform harmonic content to match target style"""
        
        # Extract harmonic and percussive components
        harmonic = librosa.effects.harmonic(audio_data)
        percussive = librosa.effects.percussive(audio_data)
        
        # Adjust harmonic balance based on style
        harmonic_complexity = style_params['harmonic_complexity']
        
        if harmonic_complexity == 'simple':
            # Reduce harmonic complexity (e.g., for Chinese traditional)
            harmonic = harmonic * (1.0 - 0.3 * intensity)
        elif harmonic_complexity == 'complex':
            # Enhance harmonic content (e.g., for Western classical)
            harmonic = harmonic * (1.0 + 0.3 * intensity)
        
        # Recombine
        transformed = harmonic + percussive
        
        return transformed
    
    def _transform_timbre(self, 
                         audio_data: np.ndarray,
                         style_params: Dict[str, Any],
                         intensity: float) -> np.ndarray:
        """Transform timbral characteristics"""
        
        # Apply spectral filtering based on target instrument timbres
        # This is a simplified implementation
        
        # Compute STFT
        stft = librosa.stft(audio_data, hop_length=self.hop_length, n_fft=self.frame_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Apply timbral shaping based on style
        target_timbres = style_params['instrument_timbres']
        
        # Simple spectral shaping (frequency emphasis/de-emphasis)
        freq_bins = stft.shape[0]
        freq_axis = np.linspace(0, self.sample_rate/2, freq_bins)
        
        # Create style-specific spectral envelope
        spectral_envelope = np.ones(freq_bins)
        
        if 'erhu' in target_timbres or 'chinese' in str(target_timbres):
            # Emphasize mid frequencies for Chinese string instruments
            mid_freq_mask = (freq_axis > 200) & (freq_axis < 2000)
            spectral_envelope[mid_freq_mask] *= (1.0 + 0.2 * intensity)
            
        elif 'violin' in target_timbres or 'classical' in str(target_timbres):
            # Emphasize higher frequencies for Western classical
            high_freq_mask = freq_axis > 1000
            spectral_envelope[high_freq_mask] *= (1.0 + 0.15 * intensity)
            
        elif 'synth' in target_timbres or 'modern' in str(target_timbres):
            # Add some high-frequency content for modern sounds
            high_freq_mask = freq_axis > 2000
            spectral_envelope[high_freq_mask] *= (1.0 + 0.25 * intensity)
        
        # Apply spectral envelope
        shaped_magnitude = magnitude * spectral_envelope[:, np.newaxis]
        
        # Reconstruct audio
        shaped_stft = shaped_magnitude * np.exp(1j * phase)
        transformed = librosa.istft(shaped_stft, hop_length=self.hop_length)
        
        # Ensure output length matches input length
        if len(transformed) != len(audio_data):
            if len(transformed) > len(audio_data):
                # Trim excess
                transformed = transformed[:len(audio_data)]
            else:
                # Pad with zeros
                padding = len(audio_data) - len(transformed)
                transformed = np.pad(transformed, (0, padding), mode='constant')
        
        return transformed
    
    def _adjust_dynamics(self, 
                        audio_data: np.ndarray,
                        style_params: Dict[str, Any],
                        intensity: float) -> np.ndarray:
        """Adjust dynamic range to match target style"""
        
        dynamic_range = style_params['dynamic_range']
        
        if dynamic_range == 'compressed':
            # Apply compression for modern pop style
            threshold = 0.1
            ratio = 4.0
            
            # Simple compression
            audio_abs = np.abs(audio_data)
            over_threshold = audio_abs > threshold
            
            compressed = audio_data.copy()
            compressed[over_threshold] = (
                np.sign(audio_data[over_threshold]) * 
                (threshold + (audio_abs[over_threshold] - threshold) / ratio)
            )
            
            # Apply with intensity
            transformed = audio_data * (1.0 - intensity) + compressed * intensity
            
        elif dynamic_range == 'wide':
            # Enhance dynamic range for classical style
            # Apply gentle expansion
            expanded = np.sign(audio_data) * (np.abs(audio_data) ** (1.0 - 0.2 * intensity))
            transformed = audio_data * (1.0 - intensity) + expanded * intensity
            
        else:  # moderate
            transformed = audio_data
        
        return transformed
    
    def _apply_style_effects(self, 
                           audio_data: np.ndarray,
                           target_style: str,
                           intensity: float) -> np.ndarray:
        """Apply style-specific audio effects"""
        
        transformed = audio_data.copy()
        
        if target_style == 'chinese_traditional':
            # Add subtle reverb and traditional Chinese characteristics
            # Simple reverb simulation using convolution with impulse response
            reverb_length = int(0.1 * self.sample_rate)  # 100ms reverb
            impulse_response = np.exp(-np.linspace(0, 5, reverb_length)) * np.random.randn(reverb_length) * 0.1
            
            # Apply reverb with intensity control
            reverb_audio = np.convolve(transformed, impulse_response, mode='same')
            transformed = transformed * (1.0 - 0.3 * intensity) + reverb_audio * (0.3 * intensity)
            
        elif target_style == 'western_classical':
            # Add classical hall reverb
            reverb_length = int(0.3 * self.sample_rate)  # 300ms reverb
            impulse_response = np.exp(-np.linspace(0, 3, reverb_length)) * np.random.randn(reverb_length) * 0.05
            
            reverb_audio = np.convolve(transformed, impulse_response, mode='same')
            transformed = transformed * (1.0 - 0.4 * intensity) + reverb_audio * (0.4 * intensity)
            
        elif target_style == 'modern_pop':
            # Add modern effects (subtle chorus/delay)
            delay_samples = int(0.02 * self.sample_rate)  # 20ms delay
            delayed_audio = np.concatenate([np.zeros(delay_samples), transformed[:-delay_samples]])
            
            # Chorus effect
            chorus_audio = transformed * 0.7 + delayed_audio * 0.3
            transformed = transformed * (1.0 - 0.2 * intensity) + chorus_audio * (0.2 * intensity)
        
        return transformed
    
    def _apply_style_polishing(self, 
                             audio_data: np.ndarray,
                             target_style: str) -> np.ndarray:
        """Apply final polishing and normalization"""
        
        # Smooth any artifacts
        smoothed = gaussian_filter1d(audio_data, sigma=1.0)
        polished = audio_data * 0.9 + smoothed * 0.1
        
        # Normalize audio
        max_val = np.max(np.abs(polished))
        if max_val > 0:
            polished = polished / max_val * 0.8
        
        return polished
    
    def get_available_styles(self) -> List[str]:
        """Get list of available style transfer options"""
        return list(self.style_definitions.keys())
    
    def get_style_info(self, style_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific style"""
        if style_name in self.style_definitions:
            return self.style_definitions[style_name].copy()
        else:
            raise ValueError(f"Unknown style: {style_name}")