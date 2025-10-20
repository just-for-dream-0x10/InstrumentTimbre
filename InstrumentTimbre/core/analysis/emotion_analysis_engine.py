"""
Emotion Analysis Engine - Week 5 Development Task

Core emotion analysis engine for Music AI system.
Implements 6-category emotion classification with intensity regression.
"""

import numpy as np
import torch
import torch.nn as nn
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import librosa
from scipy.stats import pearsonr

# Configure logging
logger = logging.getLogger(__name__)


class EmotionType(Enum):
    """6 core emotion categories"""
    HAPPY = "happy"
    SAD = "sad"
    CALM = "calm"
    EXCITED = "excited"
    MELANCHOLY = "melancholy"
    ANGRY = "angry"


@dataclass
class EmotionResult:
    """Emotion analysis result structure"""
    primary_emotion: EmotionType
    emotion_scores: Dict[str, float]  # All 6 emotion probabilities
    intensity: float  # 0-1 continuous intensity
    confidence: float  # Analysis confidence
    temporal_emotions: List[Dict[str, Any]]  # Time-series emotion changes
    constraints: List[str]  # Generated constraints for track operations


class EmotionClassifier(nn.Module):
    """6-category emotion classifier network"""
    
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256):
        super(EmotionClassifier, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        self.emotion_classifier = nn.Linear(hidden_dim // 2, 6)  # 6 emotions
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        emotion_logits = self.emotion_classifier(features)
        return emotion_logits


class EmotionIntensityRegressor(nn.Module):
    """Emotion intensity regression network (0-1 continuous output)"""
    
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256):
        super(EmotionIntensityRegressor, self).__init__()
        
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()  # Ensure 0-1 output
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.regressor(x)


class EmotionAnalysisEngine:
    """
    Core emotion analysis engine for Music AI system
    
    Implements:
    - 6-category emotion classification
    - Continuous emotion intensity regression
    - Temporal emotion change detection
    - Constraint generation for track operations
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize emotion analysis engine
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model parameters
        self.sample_rate = self.config.get('sample_rate', 22050)
        self.hop_length = self.config.get('hop_length', 512)
        self.n_mels = self.config.get('n_mels', 128)
        self.window_size = self.config.get('window_size', 3.0)  # 3 second windows
        
        # Initialize models
        self.emotion_classifier = EmotionClassifier(input_dim=self.n_mels)
        self.intensity_regressor = EmotionIntensityRegressor(input_dim=self.n_mels)
        
        # Move models to device
        self.emotion_classifier.to(self.device)
        self.intensity_regressor.to(self.device)
        
        # Load pretrained weights if available
        self._load_pretrained_models()
        
        logger.info("EmotionAnalysisEngine initialized on device: %s", self.device)
    
    def analyze_emotion(self, audio: np.ndarray) -> EmotionResult:
        """
        Analyze emotion in audio signal
        
        Args:
            audio: Input audio signal
            
        Returns:
            EmotionResult with complete emotion analysis
        """
        logger.info("Starting emotion analysis for audio: %d samples", len(audio))
        
        try:
            # Extract audio features
            features = self._extract_features(audio)
            
            # Classify emotions
            emotion_scores = self._classify_emotions(features)
            
            # Regress emotion intensity
            intensity = self._regress_intensity(features)
            
            # Detect temporal emotion changes
            temporal_emotions = self._detect_temporal_changes(audio)
            
            # Generate constraints for track operations
            constraints = self._generate_constraints(emotion_scores, intensity)
            
            # Determine primary emotion
            primary_emotion = EmotionType(max(emotion_scores, key=emotion_scores.get))
            
            # Calculate confidence
            confidence = self._calculate_confidence(emotion_scores, intensity)
            
            result = EmotionResult(
                primary_emotion=primary_emotion,
                emotion_scores=emotion_scores,
                intensity=intensity,
                confidence=confidence,
                temporal_emotions=temporal_emotions,
                constraints=constraints
            )
            
            logger.info("Emotion analysis complete: %s (%.3f intensity, %.3f confidence)",
                       primary_emotion.value, intensity, confidence)
            
            return result
            
        except Exception as e:
            logger.error("Emotion analysis failed: %s", e)
            # Return default result
            return self._create_default_result()
    
    def _extract_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract mel-spectrogram features from audio"""
        
        # Ensure audio is not empty
        if len(audio) == 0:
            return np.zeros((self.n_mels, 1))
        
        # Extract mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        
        # Convert to log scale
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to [-1, 1]
        log_mel = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-8)
        
        return log_mel
    
    def _classify_emotions(self, features: np.ndarray) -> Dict[str, float]:
        """Classify emotions using the trained classifier"""
        
        self.emotion_classifier.eval()
        
        with torch.no_grad():
            # Average pooling over time dimension
            if features.shape[1] > 1:
                feature_vector = np.mean(features, axis=1)
            else:
                feature_vector = features[:, 0]
            
            # Convert to tensor
            input_tensor = torch.FloatTensor(feature_vector).unsqueeze(0).to(self.device)
            
            # Forward pass
            logits = self.emotion_classifier(input_tensor)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
            
            # Map to emotion names
            emotions = [e.value for e in EmotionType]
            emotion_scores = dict(zip(emotions, probabilities))
            
            return emotion_scores
    
    def _regress_intensity(self, features: np.ndarray) -> float:
        """Regress emotion intensity using the trained regressor"""
        
        self.intensity_regressor.eval()
        
        with torch.no_grad():
            # Average pooling over time dimension
            if features.shape[1] > 1:
                feature_vector = np.mean(features, axis=1)
            else:
                feature_vector = features[:, 0]
            
            # Convert to tensor
            input_tensor = torch.FloatTensor(feature_vector).unsqueeze(0).to(self.device)
            
            # Forward pass
            intensity = self.intensity_regressor(input_tensor).cpu().numpy()[0, 0]
            
            return float(intensity)
    
    def _detect_temporal_changes(self, audio: np.ndarray) -> List[Dict[str, Any]]:
        """Detect emotion changes over time"""
        
        temporal_emotions = []
        
        # Calculate window parameters
        window_samples = int(self.window_size * self.sample_rate)
        hop_samples = window_samples // 2  # 50% overlap
        
        if len(audio) < window_samples:
            # Audio too short, analyze as single segment
            features = self._extract_features(audio)
            emotion_scores = self._classify_emotions(features)
            intensity = self._regress_intensity(features)
            
            temporal_emotions.append({
                'start_time': 0.0,
                'end_time': len(audio) / self.sample_rate,
                'emotion_scores': emotion_scores,
                'intensity': intensity
            })
        else:
            # Sliding window analysis
            for start_sample in range(0, len(audio) - window_samples + 1, hop_samples):
                end_sample = start_sample + window_samples
                window_audio = audio[start_sample:end_sample]
                
                # Analyze this window
                features = self._extract_features(window_audio)
                emotion_scores = self._classify_emotions(features)
                intensity = self._regress_intensity(features)
                
                temporal_emotions.append({
                    'start_time': start_sample / self.sample_rate,
                    'end_time': end_sample / self.sample_rate,
                    'emotion_scores': emotion_scores,
                    'intensity': intensity
                })
        
        return temporal_emotions
    
    def _generate_constraints(self, emotion_scores: Dict[str, float], intensity: float) -> List[str]:
        """Generate operational constraints based on emotion analysis"""
        
        constraints = []
        
        # Get primary emotion
        primary_emotion = max(emotion_scores, key=emotion_scores.get)
        
        # Emotion-specific constraints
        if primary_emotion == "happy":
            constraints.extend([
                "prefer_bright_instruments",
                "maintain_upbeat_tempo",
                "add_rhythmic_drive",
                "use_major_harmonies"
            ])
        elif primary_emotion == "sad":
            constraints.extend([
                "prefer_warm_instruments",
                "allow_slower_tempo",
                "add_expressive_dynamics",
                "use_minor_harmonies"
            ])
        elif primary_emotion == "calm":
            constraints.extend([
                "prefer_soft_instruments",
                "maintain_steady_tempo",
                "add_gentle_dynamics",
                "use_consonant_harmonies"
            ])
        elif primary_emotion == "excited":
            constraints.extend([
                "prefer_energetic_instruments",
                "allow_faster_tempo",
                "add_strong_accents",
                "use_driving_rhythms"
            ])
        elif primary_emotion == "melancholy":
            constraints.extend([
                "prefer_expressive_instruments",
                "allow_rubato_tempo",
                "add_subtle_dynamics",
                "use_complex_harmonies"
            ])
        elif primary_emotion == "angry":
            constraints.extend([
                "prefer_aggressive_instruments",
                "maintain_strong_tempo",
                "add_sharp_accents",
                "use_dissonant_elements"
            ])
        
        # Intensity-based constraints
        if intensity > 0.8:
            constraints.append("high_intensity_expression")
        elif intensity > 0.5:
            constraints.append("moderate_intensity_expression")
        else:
            constraints.append("subtle_intensity_expression")
        
        # Add general preservation constraint
        constraints.append(f"preserve_{primary_emotion}_character")
        
        return constraints
    
    def _calculate_confidence(self, emotion_scores: Dict[str, float], intensity: float) -> float:
        """Calculate analysis confidence score"""
        
        # Confidence based on emotion score distribution
        scores = list(emotion_scores.values())
        max_score = max(scores)
        
        # Higher confidence if one emotion is clearly dominant
        if max_score > 0.6:
            emotion_confidence = max_score
        elif max_score > 0.4:
            emotion_confidence = max_score * 0.8
        else:
            emotion_confidence = max_score * 0.6
        
        # Intensity confidence (higher confidence for extreme values)
        if intensity > 0.8 or intensity < 0.2:
            intensity_confidence = 0.9
        elif intensity > 0.6 or intensity < 0.4:
            intensity_confidence = 0.7
        else:
            intensity_confidence = 0.5
        
        # Combined confidence
        overall_confidence = (emotion_confidence * 0.7 + intensity_confidence * 0.3)
        
        return min(1.0, overall_confidence)
    
    def _create_default_result(self) -> EmotionResult:
        """Create default emotion result for error cases"""
        
        return EmotionResult(
            primary_emotion=EmotionType.CALM,
            emotion_scores={e.value: 1.0/6 for e in EmotionType},  # Uniform distribution
            intensity=0.5,
            confidence=0.1,
            temporal_emotions=[],
            constraints=["maintain_original_character"]
        )
    
    def _load_pretrained_models(self) -> None:
        """Load pretrained model weights if available"""
        
        try:
            # Try to load pretrained classifier
            classifier_path = self.config.get('classifier_model_path')
            if classifier_path:
                self.emotion_classifier.load_state_dict(torch.load(classifier_path, map_location=self.device))
                logger.info("Loaded pretrained emotion classifier")
            
            # Try to load pretrained regressor
            regressor_path = self.config.get('regressor_model_path')
            if regressor_path:
                self.intensity_regressor.load_state_dict(torch.load(regressor_path, map_location=self.device))
                logger.info("Loaded pretrained intensity regressor")
                
        except Exception as e:
            logger.warning("Could not load pretrained models: %s", e)
            logger.info("Using randomly initialized models")
    
    def analyze_multi_track_emotion(self, tracks: Dict[str, np.ndarray]) -> Dict[str, EmotionResult]:
        """
        Analyze emotion for multiple audio tracks
        
        Args:
            tracks: Dictionary mapping track names to audio arrays
            
        Returns:
            Dictionary mapping track names to emotion results
        """
        logger.info("Analyzing emotion for %d tracks", len(tracks))
        
        results = {}
        
        for track_name, audio in tracks.items():
            logger.info("Processing track: %s", track_name)
            results[track_name] = self.analyze_emotion(audio)
        
        # Also compute overall combined emotion
        if len(tracks) > 1:
            # Simple approach: mix all tracks and analyze
            combined_audio = np.zeros(max(len(audio) for audio in tracks.values()))
            for audio in tracks.values():
                if len(audio) <= len(combined_audio):
                    combined_audio[:len(audio)] += audio
                else:
                    combined_audio += audio[:len(combined_audio)]
            
            # Normalize
            combined_audio = combined_audio / len(tracks)
            results['_combined'] = self.analyze_emotion(combined_audio)
            
            logger.info("Combined emotion analysis complete")
        
        return results
    
    def get_emotion_statistics(self) -> Dict[str, Any]:
        """Get emotion analysis statistics and capabilities"""
        
        return {
            'supported_emotions': [e.value for e in EmotionType],
            'emotion_count': len(EmotionType),
            'feature_dimension': self.n_mels,
            'window_size_seconds': self.window_size,
            'sample_rate': self.sample_rate,
            'device': str(self.device),
            'models_loaded': {
                'classifier': hasattr(self.emotion_classifier, 'weight'),
                'regressor': hasattr(self.intensity_regressor, 'weight')
            }
        }