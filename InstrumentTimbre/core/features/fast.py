"""
Fast feature extractor for quick training and testing
"""

import numpy as np
import librosa
from typing import Dict, Any, Optional, List
from .base import BaseFeatureExtractor

class FastFeatureExtractor(BaseFeatureExtractor):
    """
    Fast feature extractor with minimal but effective features
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Minimal configuration for speed
        self.sample_rate = self.config.get('sample_rate', 22050)
        self.hop_length = self.config.get('hop_length', 2048)  # Large hop for speed
        self.n_fft = self.config.get('n_fft', 2048)
        self.n_mfcc = self.config.get('n_mfcc', 13)
        
    def extract_features(self, audio_data: np.ndarray, sample_rate: int, 
                        **kwargs) -> Dict[str, np.ndarray]:
        """Extract minimal but effective features for speed"""
        self.validate_input(audio_data, sample_rate)
        audio_data = self.preprocess_audio(audio_data)
        
        features = {}
        
        try:
            # 1. MFCC (most important)
            mfccs = librosa.feature.mfcc(
                y=audio_data, 
                sr=sample_rate, 
                n_mfcc=self.n_mfcc,
                hop_length=self.hop_length,
                n_fft=self.n_fft
            )
            features['mfcc_mean'] = np.mean(mfccs, axis=1)
            features['mfcc_std'] = np.std(mfccs, axis=1)
            
            # 2. Basic spectral features
            spectral_centroids = librosa.feature.spectral_centroid(
                y=audio_data, sr=sample_rate, hop_length=self.hop_length
            )
            features['spectral_centroid_mean'] = np.mean(spectral_centroids)
            features['spectral_centroid_std'] = np.std(spectral_centroids)
            
            # 3. Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio_data, hop_length=self.hop_length)
            features['zcr_mean'] = np.mean(zcr)
            features['zcr_std'] = np.std(zcr)
            
            # 4. RMS Energy
            rms = librosa.feature.rms(y=audio_data, hop_length=self.hop_length)
            features['rms_mean'] = np.mean(rms)
            features['rms_std'] = np.std(rms)
            
            # 5. Spectral rolloff
            rolloff = librosa.feature.spectral_rolloff(
                y=audio_data, sr=sample_rate, hop_length=self.hop_length
            )
            features['rolloff_mean'] = np.mean(rolloff)
            features['rolloff_std'] = np.std(rolloff)
            
        except Exception as e:
            self.logger.error(f"Fast feature extraction failed: {e}")
            # Return zeros on failure
            features = {
                'mfcc_mean': np.zeros(self.n_mfcc),
                'mfcc_std': np.zeros(self.n_mfcc),
                'spectral_centroid_mean': 0.0,
                'spectral_centroid_std': 0.0,
                'zcr_mean': 0.0,
                'zcr_std': 0.0,
                'rms_mean': 0.0,
                'rms_std': 0.0,
                'rolloff_mean': 0.0,
                'rolloff_std': 0.0
            }
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names"""
        names = []
        # MFCC features
        names.extend([f'mfcc_mean_{i}' for i in range(self.n_mfcc)])
        names.extend([f'mfcc_std_{i}' for i in range(self.n_mfcc)])
        
        # Spectral features
        names.extend([
            'spectral_centroid_mean', 'spectral_centroid_std',
            'zcr_mean', 'zcr_std',
            'rms_mean', 'rms_std',
            'rolloff_mean', 'rolloff_std'
        ])
        
        return names