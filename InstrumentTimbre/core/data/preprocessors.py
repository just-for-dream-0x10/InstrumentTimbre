"""
Audio preprocessing utilities for InstrumentTimbre
"""

import numpy as np
import librosa
from typing import Dict, Any, Optional, Tuple
import logging

class AudioPreprocessor:
    """
    Base audio preprocessor
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize preprocessor
        
        Args:
            config: Preprocessing configuration
        """
        self.config = config or {}
        self.sample_rate = self.config.get('sample_rate', 22050)
        self.normalize = self.config.get('normalize', True)
        self.convert_to_mono = self.config.get('convert_to_mono', True)
        self.logger = logging.getLogger(__name__)
    
    def preprocess(self, audio_data: np.ndarray, 
                  original_sr: Optional[int] = None) -> np.ndarray:
        """
        Preprocess audio data
        
        Args:
            audio_data: Raw audio data
            original_sr: Original sample rate
            
        Returns:
            Preprocessed audio data
        """
        # Resample if needed
        if original_sr and original_sr != self.sample_rate:
            audio_data = librosa.resample(
                audio_data, orig_sr=original_sr, target_sr=self.sample_rate
            )
        
        # Convert to mono
        if self.convert_to_mono and audio_data.ndim > 1:
            audio_data = librosa.to_mono(audio_data)
        
        # Normalize
        if self.normalize:
            audio_data = self._normalize_audio(audio_data)
        
        return audio_data
    
    def _normalize_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Normalize audio data"""
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            return audio_data / max_val
        return audio_data

class ChineseInstrumentPreprocessor(AudioPreprocessor):
    """
    Preprocessor optimized for Chinese instruments
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Chinese instrument specific settings
        self.enhance_fundamentals = self.config.get('enhance_fundamentals', True)
        self.filter_noise = self.config.get('filter_noise', True)
        self.fundamental_range = self.config.get('fundamental_range', (80, 2000))
    
    def preprocess(self, audio_data: np.ndarray, 
                  original_sr: Optional[int] = None,
                  instrument_type: Optional[str] = None) -> np.ndarray:
        """
        Preprocess with Chinese instrument optimizations
        
        Args:
            audio_data: Raw audio data
            original_sr: Original sample rate
            instrument_type: Type of Chinese instrument
            
        Returns:
            Preprocessed audio data
        """
        # Base preprocessing
        audio_data = super().preprocess(audio_data, original_sr)
        
        # Chinese instrument specific processing
        if self.filter_noise:
            audio_data = self._apply_noise_filter(audio_data)
        
        if self.enhance_fundamentals:
            audio_data = self._enhance_fundamentals(audio_data, instrument_type)
        
        return audio_data
    
    def _apply_noise_filter(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply noise filtering optimized for Chinese instruments"""
        # Simple high-pass filter to remove low-frequency noise
        from scipy import signal
        
        # Design high-pass filter
        nyquist = self.sample_rate / 2
        high_cutoff = 60 / nyquist  # Remove below 60 Hz
        
        b, a = signal.butter(2, high_cutoff, btype='high')
        filtered_audio = signal.filtfilt(b, a, audio_data)
        
        return filtered_audio
    
    def _enhance_fundamentals(self, audio_data: np.ndarray, 
                            instrument_type: Optional[str] = None) -> np.ndarray:
        """Enhance fundamental frequencies for Chinese instruments"""
        # This is a placeholder for more sophisticated enhancement
        # In practice, this could involve harmonic enhancement, 
        # formant adjustment, etc.
        
        # For now, just apply a gentle band-pass filter in the fundamental range
        from scipy import signal
        
        nyquist = self.sample_rate / 2
        low_cutoff = self.fundamental_range[0] / nyquist
        high_cutoff = self.fundamental_range[1] / nyquist
        
        if high_cutoff < 1.0:  # Ensure valid cutoff
            b, a = signal.butter(2, [low_cutoff, high_cutoff], btype='band')
            enhanced_fundamentals = signal.filtfilt(b, a, audio_data)
            
            # Blend with original (gentle enhancement)
            alpha = 0.3  # Enhancement strength
            audio_data = (1 - alpha) * audio_data + alpha * enhanced_fundamentals
        
        return audio_data