"""
Base feature extractor interface
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np
import logging

class BaseFeatureExtractor(ABC):
    """
    Abstract base class for all feature extractors
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize feature extractor
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
    @abstractmethod
    def extract_features(self, audio_data: np.ndarray, sample_rate: int, 
                        **kwargs) -> Dict[str, np.ndarray]:
        """
        Extract features from audio data
        
        Args:
            audio_data: Audio signal as numpy array
            sample_rate: Sample rate of the audio
            **kwargs: Additional parameters
            
        Returns:
            Dictionary of extracted features
        """
        pass
    
    @abstractmethod
    def get_feature_names(self) -> list:
        """
        Get list of feature names that this extractor produces
        
        Returns:
            List of feature names
        """
        pass
    
    def validate_input(self, audio_data: np.ndarray, sample_rate: int) -> None:
        """
        Validate input audio data
        
        Args:
            audio_data: Audio signal to validate
            sample_rate: Sample rate to validate
            
        Raises:
            ValueError: If input is invalid
        """
        if not isinstance(audio_data, np.ndarray):
            raise ValueError("Audio data must be a numpy array")
            
        if audio_data.size == 0:
            raise ValueError("Audio data cannot be empty")
            
        if sample_rate <= 0:
            raise ValueError("Sample rate must be positive")
            
        if len(audio_data.shape) > 2:
            raise ValueError("Audio data must be 1D or 2D (mono or stereo)")
            
    def preprocess_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Preprocess audio data (normalize, convert to mono, etc.)
        
        Args:
            audio_data: Raw audio data
            
        Returns:
            Preprocessed audio data
        """
        # Convert to mono if stereo
        if len(audio_data.shape) == 2:
            audio_data = np.mean(audio_data, axis=1)
            
        # Normalize
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data))
            
        return audio_data