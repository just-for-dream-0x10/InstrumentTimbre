#!/usr/bin/env python3
"""
Model Evaluation Tool for InstrumentTimbre
InstrumentTimbre 模型评估工具
"""

import argparse
import torch
import numpy as np
import librosa
import os
import json
import pickle
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from InstrumentTimbre.modules.utils.chinese_instrument_features import ChineseInstrumentAnalyzer
    from InstrumentTimbre.modules.core.models import InstrumentType
    ENHANCED_FEATURES_AVAILABLE = True
except ImportError:
    ENHANCED_FEATURES_AVAILABLE = False
    logger.warning("Enhanced Chinese features not available, using basic features")


class ModelEvaluator:
    """Enhanced model evaluation with Chinese instrument features"""
    
    def __init__(self, model_path, device='auto'):
        """Initialize evaluator"""
        self.model_path = model_path
        self.device = self._setup_device(device)
        self.model = None
        self.label_encoder = None
        self.feature_size = 50
        self.enhanced_features = True
        
        # Load model
        self._load_model()
        
        # Initialize feature extractor
        if ENHANCED_FEATURES_AVAILABLE:
            self.analyzer = ChineseInstrumentAnalyzer()
            logger.info("Using enhanced Chinese instrument features")
        else:
            self.analyzer = None
            logger.info("Using basic audio features")
    
    def _setup_device(self, device):
        """Setup computation device"""
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        
        return torch.device(device)
    
    def _load_model(self):
        """Load trained model"""
        logger.info(f"Loading model from: {self.model_path}")
        
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            # Extract model info
            self.label_encoder = checkpoint.get('label_encoder')
            self.feature_size = checkpoint.get('feature_size', 50)
            self.enhanced_features = checkpoint.get('enhanced_features', True)
            num_classes = checkpoint.get('num_classes', len(self.label_encoder.classes_))
            
            logger.info(f"Model info:")
            logger.info(f"  Classes: {checkpoint.get('class_names', 'Unknown')}")
            logger.info(f"  Feature size: {self.feature_size}")
            logger.info(f"  Enhanced features: {self.enhanced_features}")
            logger.info(f"  Device: {self.device}")
            
            # Recreate model architecture
            self.model = self._create_model_architecture(self.feature_size, num_classes)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("✅ Model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def _create_model_architecture(self, input_size, num_classes):
        """Recreate model architecture"""
        class EnhancedChineseClassifier(torch.nn.Module):
            def __init__(self, input_size, num_classes):
                super().__init__()
                self.network = torch.nn.Sequential(
                    torch.nn.Linear(input_size, 256),
                    torch.nn.BatchNorm1d(256),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.3),
                    
                    torch.nn.Linear(256, 128),
                    torch.nn.BatchNorm1d(128),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.3),
                    
                    torch.nn.Linear(128, 64),
                    torch.nn.BatchNorm1d(64),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.2),
                    
                    torch.nn.Linear(64, 32),
                    torch.nn.ReLU(),
                    
                    torch.nn.Linear(32, num_classes)
                )
            
            def forward(self, x):
                return self.network(x)
        
        return EnhancedChineseClassifier(input_size, num_classes)
    
    def extract_features(self, audio_file):
        """Extract features from audio file"""
        try:
            audio_data, sr = librosa.load(audio_file, sr=22050)
            
            if self.analyzer and ENHANCED_FEATURES_AVAILABLE:
                # Use enhanced Chinese features
                instrument_type = self._guess_instrument_type(audio_file)
                chinese_features = self.analyzer.extract_chinese_features(
                    audio_data, sr, instrument_type
                )
                return self._chinese_features_to_vector(chinese_features)
            else:
                # Use basic audio features
                return self._extract_basic_features(audio_data, sr)
                
        except Exception as e:
            logger.warning(f"Feature extraction failed for {audio_file}: {e}")
            return np.zeros(self.feature_size)
    
    def _guess_instrument_type(self, audio_file):
        """Guess instrument type from filename"""
        filename = Path(audio_file).stem.lower()
        if 'erhu' in filename:
            return InstrumentType.ERHU
        elif 'pipa' in filename:
            return InstrumentType.PIPA
        elif 'guzheng' in filename:
            return InstrumentType.GUZHENG
        elif 'dizi' in filename:
            return InstrumentType.DIZI
        elif 'guqin' in filename:
            return InstrumentType.GUQIN
        else:
            return InstrumentType.ERHU  # Default
    
    def _chinese_features_to_vector(self, chinese_features):
        """Convert Chinese features to vector (same as training)"""
        feature_vector = []
        
        # Basic features
        feature_vector.append(chinese_features.pentatonic_adherence or 0.0)
        feature_vector.append(chinese_features.ornament_density or 0.0)
        feature_vector.append(chinese_features.rhythmic_complexity or 0.0)
        
        # Vibrato features
        if chinese_features.vibrato_analysis:
            va = chinese_features.vibrato_analysis
            feature_vector.extend([
                va.get('rate_hz', 0.0),
                va.get('extent_cents', 0.0),
                va.get('regularity', 0.0),
                va.get('presence', 0.0)
            ])
        else:
            feature_vector.extend([0.0, 0.0, 0.0, 0.0])
        
        # Additional features (simplified)
        if hasattr(chinese_features, 'sliding_velocity') and chinese_features.sliding_velocity is not None:
            valid_velocities = chinese_features.sliding_velocity[~np.isnan(chinese_features.sliding_velocity)]
            if len(valid_velocities) > 0:
                feature_vector.extend([
                    np.mean(np.abs(valid_velocities)),
                    np.std(valid_velocities),
                    np.max(np.abs(valid_velocities))
                ])
            else:
                feature_vector.extend([0.0, 0.0, 0.0])
        else:
            feature_vector.extend([0.0, 0.0, 0.0])
        
        # Pad to target size
        while len(feature_vector) < self.feature_size:
            feature_vector.append(0.0)
            
        return np.array(feature_vector[:self.feature_size], dtype=np.float32)
    
    def _extract_basic_features(self, audio_data, sr):
        """Extract basic audio features as fallback"""
        try:
            # Basic spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)[0]
            zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
            
            # MFCC features
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
            
            # Combine features
            feature_vector = [
                np.mean(spectral_centroids),
                np.std(spectral_centroids),
                np.mean(spectral_rolloff),
                np.mean(zcr),
                np.std(zcr)
            ]
            
            # Add MFCC statistics
            for i in range(13):
                feature_vector.extend([np.mean(mfccs[i]), np.std(mfccs[i])])
            
            # Pad to target size
            while len(feature_vector) < self.feature_size:
                feature_vector.append(0.0)
                
            return np.array(feature_vector[:self.feature_size], dtype=np.float32)
            
        except Exception as e:
            logger.warning(f"Basic feature extraction failed: {e}")
            return np.zeros(self.feature_size, dtype=np.float32)
    
    def predict_single(self, audio_file):
        """Predict single audio file"""
        features = self.extract_features(audio_file)
        
        with torch.no_grad():
            feature_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            outputs = self.model(feature_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
            predicted_class_idx = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][predicted_class_idx].item()
            
            predicted_class = self.label_encoder.inverse_transform([predicted_class_idx])[0]
            
            return {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'probabilities': probabilities[0].cpu().numpy(),
                'class_names': self.label_encoder.classes_
            }
    
    def evaluate_dataset(self, test_dir, save_results=True, output_dir="evaluation_results"):
        """Evaluate model on test dataset"""
        logger.info(f"🔍 Evaluating model on dataset: {test_dir}")
        
        # Find test files
        audio_extensions = {'.wav', '.mp3', '.flac', '.m4a'}
        test_files = []
        true_labels = []
        
        for file_path in Path(test_dir).rglob('*'):
            if file_path.suffix.lower() in audio_extensions:
                test_files.append(str(file_path))
                # Extract label from parent directory or filename
                label = self._extract_label_from_path(file_path)
                true_labels.append(label)
        
        if not test_files:
            logger.error("❌ No test files found!")
            return None
        
        logger.info(f"📁 Found {len(test_files)} test files")
        logger.info(f"🏷️  Labels: {set(true_labels)}")
        
        # Make predictions
        predicted_labels = []
        prediction_details = []
        
        for i, audio_file in enumerate(test_files):
            try:
                result = self.predict_single(audio_file)
                predicted_labels.append(result['predicted_class'])
                prediction_details.append(result)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(test_files)} files")
                    
            except Exception as e:
                logger.warning(f"Prediction failed for {audio_file}: {e}")
                predicted_labels.append('unknown')
                prediction_details.append({
                    'predicted_class': 'unknown',
                    'confidence': 0.0,
                    'probabilities': np.zeros(len(self.label_encoder.classes_)),
                    'class_names': self.label_encoder.classes_
                })
        
        # Calculate metrics
        evaluation_results = self._calculate_metrics(
            true_labels, predicted_labels, test_files, prediction_details
        )
        
        # Save results
        if save_results:
            self._save_evaluation_results(evaluation_results, output_dir)
        
        return evaluation_results
    
    def _extract_label_from_path(self, file_path):
        """Extract label from file path"""
        # Try parent directory first
        label = file_path.parent.name.lower()
        
        # If generic folder name, try filename
        if label in ['wav', 'audio', 'data', 'test']:
            filename = file_path.stem.lower()
            if 'erhu' in filename:
                label = 'erhu'
            elif 'pipa' in filename:
                label = 'pipa'
            elif 'guzheng' in filename:
                label = 'guzheng'
            elif 'dizi' in filename:
                label = 'dizi'
            elif 'guqin' in filename:
                label = 'guqin'
            else:
                label = 'unknown'
        
        return label
    
    def _calculate_metrics(self, true_labels, predicted_labels, test_files, prediction_details):
        """Calculate evaluation metrics"""
        logger.info("📊 Calculating evaluation metrics...")
        
        # Overall accuracy
        accuracy = accuracy_score(true_labels, predicted_labels)
        
        # Classification report
        unique_labels = sorted(set(true_labels + predicted_labels))
        class_report = classification_report(
            true_labels, predicted_labels, 
            labels=unique_labels, 
            output_dict=True,
            zero_division=0
        )
        
        # Confusion matrix
        conf_matrix = confusion_matrix(
            true_labels, predicted_labels, 
            labels=unique_labels
        )
        
        # Per-file results
        file_results = []
        for i, (file_path, true_label, pred_label, details) in enumerate(
            zip(test_files, true_labels, predicted_labels, prediction_details)
        ):
            file_results.append({
                'file': os.path.basename(file_path),
                'true_label': true_label,
                'predicted_label': pred_label,
                'confidence': details['confidence'],
                'correct': true_label == pred_label
            })
        
        # Calculate confidence statistics
        confidences = [detail['confidence'] for detail in prediction_details]
        correct_predictions = [true == pred for true, pred in zip(true_labels, predicted_labels)]
        
        correct_confidences = [conf for conf, correct in zip(confidences, correct_predictions) if correct]
        incorrect_confidences = [conf for conf, correct in zip(confidences, correct_predictions) if not correct]
        
        results = {
            'model_path': self.model_path,
            'evaluation_date': datetime.now().isoformat(),
            'test_dataset': {
                'num_files': len(test_files),
                'unique_labels': unique_labels,
                'label_distribution': {label: true_labels.count(label) for label in unique_labels}
            },
            'metrics': {
                'accuracy': accuracy,
                'classification_report': class_report,
                'confusion_matrix': conf_matrix.tolist(),
                'confusion_matrix_labels': unique_labels
            },
            'confidence_analysis': {
                'mean_confidence': np.mean(confidences),
                'std_confidence': np.std(confidences),
                'mean_correct_confidence': np.mean(correct_confidences) if correct_confidences else 0.0,
                'mean_incorrect_confidence': np.mean(incorrect_confidences) if incorrect_confidences else 0.0
            },
            'file_results': file_results
        }
        
        # Print summary
        logger.info(f"📈 Evaluation Results:")
        logger.info(f"  Overall Accuracy: {accuracy:.3f}")
        logger.info(f"  Mean Confidence: {np.mean(confidences):.3f}")
        logger.info(f"  Correct Predictions: {sum(correct_predictions)}/{len(correct_predictions)}")
        
        return results
    
    def _save_evaluation_results(self, results, output_dir):
        """Save evaluation results"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save JSON report
        json_file = os.path.join(output_dir, f"evaluation_report_{timestamp}.json")
        with open(json_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = self._convert_numpy_to_list(results.copy())
            json.dump(json_results, f, indent=2)
        
        logger.info(f"💾 JSON report saved: {json_file}")
        
        # Save confusion matrix plot
        self._plot_confusion_matrix(results, output_dir, timestamp)
        
        # Save detailed CSV
        self._save_csv_results(results, output_dir, timestamp)
    
    def _convert_numpy_to_list(self, obj):
        """Convert numpy arrays to lists for JSON serialization"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_to_list(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_to_list(item) for item in obj]
        else:
            return obj
    
    def _plot_confusion_matrix(self, results, output_dir, timestamp):
        """Plot and save confusion matrix"""
        try:
            conf_matrix = np.array(results['metrics']['confusion_matrix'])
            labels = results['metrics']['confusion_matrix_labels']
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                       xticklabels=labels, yticklabels=labels)
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            
            plot_file = os.path.join(output_dir, f"confusion_matrix_{timestamp}.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"📊 Confusion matrix saved: {plot_file}")
            
        except Exception as e:
            logger.warning(f"Failed to save confusion matrix plot: {e}")
    
    def _save_csv_results(self, results, output_dir, timestamp):
        """Save detailed results to CSV"""
        try:
            import pandas as pd
            
            df = pd.DataFrame(results['file_results'])
            csv_file = os.path.join(output_dir, f"detailed_results_{timestamp}.csv")
            df.to_csv(csv_file, index=False)
            
            logger.info(f"📋 Detailed CSV saved: {csv_file}")
            
        except ImportError:
            logger.warning("Pandas not available, skipping CSV export")
        except Exception as e:
            logger.warning(f"Failed to save CSV: {e}")


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate InstrumentTimbre Model')
    
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--test-dir', required=True, help='Directory containing test audio files')
    parser.add_argument('--output-dir', default='evaluation_results', help='Output directory for results')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda', 'mps'], help='Computation device')
    parser.add_argument('--single-file', help='Evaluate single audio file')
    
    args = parser.parse_args()
    
    logger.info("🔍 InstrumentTimbre Model Evaluation")
    logger.info("=" * 50)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(args.model, args.device)
    
    if args.single_file:
        # Single file prediction
        logger.info(f"🎵 Predicting single file: {args.single_file}")
        result = evaluator.predict_single(args.single_file)
        
        print(f"\nPrediction Results:")
        print(f"  File: {os.path.basename(args.single_file)}")
        print(f"  Predicted Class: {result['predicted_class']}")
        print(f"  Confidence: {result['confidence']:.3f}")
        print(f"  All Probabilities:")
        for class_name, prob in zip(result['class_names'], result['probabilities']):
            print(f"    {class_name}: {prob:.3f}")
    
    else:
        # Dataset evaluation
        results = evaluator.evaluate_dataset(
            args.test_dir, 
            save_results=True, 
            output_dir=args.output_dir
        )
        
        if results:
            logger.info("🎉 Evaluation completed successfully!")
            logger.info(f"📁 Results saved in: {os.path.abspath(args.output_dir)}")
        else:
            logger.error("❌ Evaluation failed!")


if __name__ == "__main__":
    main()