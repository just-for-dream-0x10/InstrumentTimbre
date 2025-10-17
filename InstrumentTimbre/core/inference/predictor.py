"""
Single-instance predictor for Chinese instrument classification
"""

import torch
import numpy as np
import librosa
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging

from ..models.base import BaseModel
from ..features.chinese import ChineseInstrumentAnalyzer

class InstrumentPredictor:
    """
    Single-instance predictor for Chinese instrument classification
    """
    
    def __init__(self, model_path: str, config: Optional[Dict[str, Any]] = None,
                 device: str = 'auto'):
        """
        Initialize predictor
        
        Args:
            model_path: Path to trained model checkpoint
            config: Configuration for feature extraction
            device: Device to run inference on
        """
        self.logger = logging.getLogger(__name__)
        
        # Setup device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load model
        self.model, self.checkpoint = self._load_model(model_path)
        self.class_names = self.checkpoint.get('class_names', [
            'erhu', 'pipa', 'guzheng', 'dizi', 'guqin'
        ])
        
        # Initialize feature extractor
        feature_config = config or self.checkpoint.get('feature_config', {})
        self.feature_extractor = ChineseInstrumentAnalyzer(feature_config)
        
        self.logger.info(f"Predictor initialized on {self.device}")
        self.logger.info(f"Model classes: {self.class_names}")
    
    def _load_model(self, model_path: str) -> Tuple[BaseModel, Dict]:
        """Load model from checkpoint"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Determine model class
        model_class_name = checkpoint.get('model_class', 'EnhancedCNNClassifier')
        model_config = checkpoint.get('model_config', {})
        
        if model_class_name == 'CNNClassifier':
            from ..models.cnn import CNNClassifier
            model = CNNClassifier(model_config)
        else:
            from ..models.cnn import EnhancedCNNClassifier
            model = EnhancedCNNClassifier(model_config)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        return model, checkpoint
    
    def predict_file(self, audio_path: str, top_k: int = 3, 
                    return_features: bool = False) -> Dict[str, Any]:
        """
        Predict instrument type from audio file
        
        Args:
            audio_path: Path to audio file
            top_k: Number of top predictions to return
            return_features: Whether to return extracted features
            
        Returns:
            Prediction results
        """
        # Load audio
        try:
            audio_data, sr = librosa.load(audio_path, sr=22050)
        except Exception as e:
            self.logger.error(f"Failed to load audio: {e}")
            raise
        
        return self.predict_audio(audio_data, sr, top_k, return_features)
    
    def predict_audio(self, audio_data: np.ndarray, sample_rate: int, 
                     top_k: int = 3, return_features: bool = False) -> Dict[str, Any]:
        """
        Predict instrument type from audio data
        
        Args:
            audio_data: Audio signal
            sample_rate: Sample rate of audio
            top_k: Number of top predictions to return
            return_features: Whether to return extracted features
            
        Returns:
            Prediction results
        """
        # Extract features
        features = self.feature_extractor.extract_features(audio_data, sample_rate)
        
        # Convert to tensor
        feature_vector = self._features_to_tensor(features)
        input_tensor = feature_vector.unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidences, predicted_indices = torch.topk(probabilities, top_k, dim=1)
        
        # Prepare results
        predictions = []
        for i in range(top_k):
            class_idx = predicted_indices[0][i].item()
            confidence = confidences[0][i].item()
            
            predictions.append({
                'class': self.class_names[class_idx] if class_idx < len(self.class_names) else f'class_{class_idx}',
                'confidence': float(confidence),
                'class_index': class_idx
            })
        
        results = {
            'predictions': predictions,
            'top_prediction': predictions[0],
            'probabilities': probabilities[0].cpu().numpy().tolist()
        }
        
        if return_features:
            results['features'] = features
            
        return results
    
    def _features_to_tensor(self, features: Dict[str, np.ndarray]) -> torch.Tensor:
        """Convert feature dictionary to tensor"""
        feature_list = []
        
        for key in sorted(features.keys()):
            feature = features[key]
            if isinstance(feature, np.ndarray):
                if feature.ndim == 0:  # Scalar
                    feature_list.append(feature.item())
                else:  # Array
                    feature_list.extend(feature.flatten())
            else:  # Scalar
                feature_list.append(float(feature))
        
        return torch.FloatTensor(feature_list)
    
    def predict_batch(self, audio_files: List[str], 
                     batch_size: int = 8) -> List[Dict[str, Any]]:
        """
        Predict for a batch of audio files
        
        Args:
            audio_files: List of audio file paths
            batch_size: Batch size for processing
            
        Returns:
            List of prediction results
        """
        results = []
        
        for i in range(0, len(audio_files), batch_size):
            batch_files = audio_files[i:i + batch_size]
            batch_results = []
            
            for file_path in batch_files:
                try:
                    result = self.predict_file(file_path)
                    result['file'] = file_path
                    batch_results.append(result)
                except Exception as e:
                    self.logger.warning(f"Failed to process {file_path}: {e}")
                    batch_results.append({
                        'file': file_path,
                        'error': str(e),
                        'predictions': []
                    })
            
            results.extend(batch_results)
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            'model_class': self.checkpoint.get('model_class'),
            'num_classes': len(self.class_names),
            'class_names': self.class_names,
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
            'device': str(self.device),
            'epoch_trained': self.checkpoint.get('epoch', 'unknown'),
            'metrics': self.checkpoint.get('metrics', {})
        }