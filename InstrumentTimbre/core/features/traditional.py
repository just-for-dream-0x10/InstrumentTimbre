"""
Traditional Audio Features Extractor
Standard audio features for general audio analysis
"""

import numpy as np
import librosa
from typing import Dict, Any, Optional, List
from .base import BaseFeatureExtractor

class TraditionalAudioFeatures(BaseFeatureExtractor):
    """
    Traditional audio feature extractor for general audio analysis
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        self.sample_rate = self.config.get('sample_rate', 22050)
        self.hop_length = self.config.get('hop_length', 512)
        self.n_fft = self.config.get('n_fft', 2048)
        self.n_mfcc = self.config.get('n_mfcc', 13)
        
    def extract_features(self, audio_data: np.ndarray, sample_rate: int, 
                        **kwargs) -> Dict[str, np.ndarray]:
        """Extract traditional audio features"""
        self.validate_input(audio_data, sample_rate)
        audio_data = self.preprocess_audio(audio_data)
        
        features = {}
        
        # MFCC
        mfccs = librosa.feature.mfcc(
            y=audio_data, sr=sample_rate, n_mfcc=self.n_mfcc,
            hop_length=self.hop_length, n_fft=self.n_fft
        )
        features['mfcc_mean'] = np.mean(mfccs, axis=1)
        features['mfcc_std'] = np.std(mfccs, axis=1)
        
        # Spectral features
        features['spectral_centroid'] = np.mean(
            librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)
        )
        features['spectral_rolloff'] = np.mean(
            librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate)
        )
        features['spectral_bandwidth'] = np.mean(
            librosa.feature.spectral_bandwidth(y=audio_data, sr=sample_rate)
        )
        
        # Zero crossing rate
        features['zcr'] = np.mean(
            librosa.feature.zero_crossing_rate(audio_data)
        )
        
        # Chroma
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate)
        features['chroma_mean'] = np.mean(chroma, axis=1)
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get feature names"""
        names = []
        names.extend([f'mfcc_mean_{i}' for i in range(self.n_mfcc)])
        names.extend([f'mfcc_std_{i}' for i in range(self.n_mfcc)])
        names.extend(['spectral_centroid', 'spectral_rolloff', 'spectral_bandwidth', 'zcr'])
        names.extend([f'chroma_mean_{i}' for i in range(12)])
        return names