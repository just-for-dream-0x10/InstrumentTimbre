"""
 - System
Emotion Analysis Engine - Core module for System

6，
"""

import numpy as np
import torch
import torch.nn as nn
import librosa
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class EmotionType(Enum):
    """"""
    HAPPY = "happy"
    SAD = "sad"
    CALM = "calm"
    EXCITED = "excited"
    MELANCHOLY = "melancholy"
    ANGRY = "angry"

@dataclass
class EmotionResult:
    """Analysis results"""
    primary_emotion: EmotionType
    emotion_scores: Dict[EmotionType, float]
    intensity: float
    confidence: float
    temporal_trajectory: Optional[np.ndarray] = None

class EmotionAnalysisEngine:
    """
    
    
    ：
    1. 6 (、、、、、)
    2.  (0-1)
    3. 
    4. 
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        self.feature_extractor = AudioFeatureExtractor()
        
    def _load_model(self, model_path: Optional[str]) -> nn.Module:
        """loadEmotion analysis"""
        if model_path:
            # load
            model = torch.load(model_path, map_location=self.device)
        else:
            # create
            model = EmotionClassifier()
        
        model.to(self.device)
        model.eval()
        return model
    
    def analyze(self, audio_data: np.ndarray, sr: int = 22050) -> EmotionResult:
        """
        
        
        Args:
            audio_data: 
            sr: 
            
        Returns:
            EmotionResult: 
        """
        # 1. Audio features
        features = self.feature_extractor.extract(audio_data, sr)
        
        # 2. 
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            emotion_logits = self.model(features_tensor)
            emotion_probs = torch.softmax(emotion_logits, dim=-1)
        
        # 3. 
        emotion_scores = self._parse_emotion_scores(emotion_probs)
        primary_emotion = max(emotion_scores, key=emotion_scores.get)
        intensity = self._calculate_intensity(emotion_probs)
        confidence = float(torch.max(emotion_probs))
        
        # 4. 
        temporal_trajectory = self._analyze_temporal_trajectory(audio_data, sr)
        
        return EmotionResult(
            primary_emotion=primary_emotion,
            emotion_scores=emotion_scores,
            intensity=intensity,
            confidence=confidence,
            temporal_trajectory=temporal_trajectory
        )
    
    def _parse_emotion_scores(self, emotion_probs: torch.Tensor) -> Dict[EmotionType, float]:
        """"""
        emotion_list = list(EmotionType)
        scores = {}
        probs = emotion_probs.cpu().numpy().flatten()
        
        for i, emotion in enumerate(emotion_list):
            scores[emotion] = float(probs[i])
            
        return scores
    
    def _calculate_intensity(self, emotion_probs: torch.Tensor) -> float:
        """Emotion intensity"""
        # Emotion intensity
        probs = emotion_probs.cpu().numpy().flatten()
        entropy = -np.sum(probs * np.log(probs + 1e-8))
        max_entropy = np.log(len(probs))
        
        #  = 1 -  ()
        intensity = 1.0 - (entropy / max_entropy)
        return float(intensity)
    
    def _analyze_temporal_trajectory(self, audio_data: np.ndarray, sr: int) -> np.ndarray:
        """"""
        # 
        segment_length = sr * 2  # 2
        hop_length = sr // 2     # 0.5
        
        trajectory = []
        for start in range(0, len(audio_data) - segment_length, hop_length):
            segment = audio_data[start:start + segment_length]
            
            # Emotion analysis ()
            features = self.feature_extractor.extract_basic(segment, sr)
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
                emotion_logits = self.model(features_tensor)
                emotion_probs = torch.softmax(emotion_logits, dim=-1)
                intensity = self._calculate_intensity(emotion_probs)
                trajectory.append(intensity)
        
        return np.array(trajectory)

    def generate_emotion_constraint(self, emotion_result: EmotionResult) -> Dict:
        """
        
        
        """
        constraint = {
            "target_emotion": emotion_result.primary_emotion.value,
            "intensity_range": [
                max(0.0, emotion_result.intensity - 0.1),
                min(1.0, emotion_result.intensity + 0.1)
            ],
            "preserve_emotions": [
                emotion.value for emotion, score in emotion_result.emotion_scores.items()
                if score > 0.2  # 0.2
            ],
            "avoid_emotions": [
                emotion.value for emotion, score in emotion_result.emotion_scores.items()
                if score < 0.05  # 0.05
            ],
            "temporal_stability": np.std(emotion_result.temporal_trajectory) < 0.3
        }
        
        return constraint


class AudioFeatureExtractor:
    """Audio features"""
    
    def extract(self, audio_data: np.ndarray, sr: int) -> np.ndarray:
        """Audio features"""
        features = []
        
        # 1. MFCC
        mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
        features.append(np.mean(mfcc, axis=1))
        features.append(np.std(mfcc, axis=1))
        
        # 2. 
        chroma = librosa.feature.chroma(y=audio_data, sr=sr)
        features.append(np.mean(chroma, axis=1))
        
        # 3. 
        contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sr)
        features.append(np.mean(contrast, axis=1))
        
        # 4. 
        zcr = librosa.feature.zero_crossing_rate(audio_data)
        features.append([np.mean(zcr)])
        
        # 5. 
        centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr)
        features.append([np.mean(centroid)])
        
        # 6. 
        rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)
        features.append([np.mean(rolloff)])
        
        return np.concatenate(features)
    
    def extract_basic(self, audio_data: np.ndarray, sr: int) -> np.ndarray:
        """ ()"""
        # 
        mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=8)
        chroma = librosa.feature.chroma(y=audio_data, sr=sr)
        
        features = np.concatenate([
            np.mean(mfcc, axis=1),
            np.mean(chroma, axis=1)
        ])
        
        return features


class EmotionClassifier(nn.Module):
    """"""
    
    def __init__(self, input_dim: int = 48, num_emotions: int = 6):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            
            nn.Linear(32, num_emotions)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


# 
if __name__ == "__main__":
    # createEmotion analysis
    emotion_engine = EmotionAnalysisEngine()
    
    # load
    audio_file = "test_audio.wav"  # 
    try:
        audio_data, sr = librosa.load(audio_file, sr=22050)
        
        # 
        result = emotion_engine.analyze(audio_data, sr)
        
        print(f": {result.primary_emotion.value}")
        print(f": {result.intensity:.3f}")
        print(f": {result.confidence:.3f}")
        print("\n:")
        for emotion, score in result.emotion_scores.items():
            print(f"  {emotion.value}: {score:.3f}")
        
        # generate
        constraint = emotion_engine.generate_emotion_constraint(result)
        print(f"\n: {constraint}")
        
    except Exception as e:
        print(f": {e}")
        
        # generate
        test_audio = np.random.randn(22050 * 3)  # 3
        result = emotion_engine.analyze(test_audio, 22050)
        print(f" - : {result.primary_emotion.value}")