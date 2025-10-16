#!/usr/bin/env python3
"""
Inference and Prediction Tool for InstrumentTimbre
InstrumentTimbre 推理预测工具
"""

import argparse
import torch
import numpy as np
import librosa
import os
import json
from pathlib import Path
import logging
from datetime import datetime

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


class InstrumentPredictor:
    """Chinese Instrument Predictor with Enhanced Features"""
    
    def __init__(self, model_path, device='auto'):
        """Initialize predictor"""
        self.model_path = model_path
        self.device = self._setup_device(device)
        self.model = None
        self.label_encoder = None
        self.feature_size = 50
        self.enhanced_features = True
        self.class_names = []
        
        # Load model
        self._load_model()
        
        # Initialize feature extractor
        if ENHANCED_FEATURES_AVAILABLE:
            self.analyzer = ChineseInstrumentAnalyzer()
            logger.info("✅ Using enhanced Chinese instrument features")
        else:
            self.analyzer = None
            logger.info("⚠️  Using basic audio features")
    
    def _setup_device(self, device):
        """Setup computation device"""
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
                logger.info(f"🔥 Using GPU: {torch.cuda.get_device_name(0)}")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
                logger.info("🍎 Using Apple Silicon MPS")
            else:
                device = 'cpu'
                logger.info("💻 Using CPU")
        
        return torch.device(device)
    
    def _load_model(self):
        """Load trained model"""
        logger.info(f"📥 Loading model from: {self.model_path}")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            # Extract model info
            self.label_encoder = checkpoint.get('label_encoder')
            self.feature_size = checkpoint.get('feature_size', 50)
            self.enhanced_features = checkpoint.get('enhanced_features', True)
            self.class_names = checkpoint.get('class_names', [])
            num_classes = checkpoint.get('num_classes', len(self.class_names))
            best_accuracy = checkpoint.get('best_accuracy', 'Unknown')
            
            logger.info(f"📊 Model Information:")
            logger.info(f"   Classes: {self.class_names}")
            logger.info(f"   Feature size: {self.feature_size}")
            logger.info(f"   Enhanced features: {self.enhanced_features}")
            logger.info(f"   Training accuracy: {best_accuracy}")
            logger.info(f"   Device: {self.device}")
            
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
        """Recreate model architecture (must match training)"""
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
        logger.info(f"🎵 Extracting features from: {os.path.basename(audio_file)}")
        
        try:
            # Load audio
            audio_data, sr = librosa.load(audio_file, sr=22050)
            duration = len(audio_data) / sr
            
            logger.info(f"📊 Audio: {duration:.1f}s, {sr}Hz")
            
            if self.analyzer and ENHANCED_FEATURES_AVAILABLE:
                # Use enhanced Chinese features
                instrument_type = self._guess_instrument_type(audio_file)
                chinese_features = self.analyzer.extract_chinese_features(
                    audio_data, sr, instrument_type
                )
                
                # Log feature details
                logger.info(f"🎼 Enhanced Features:")
                logger.info(f"   Pentatonic adherence: {chinese_features.pentatonic_adherence:.3f}")
                logger.info(f"   Ornament density: {chinese_features.ornament_density:.3f}")
                if chinese_features.vibrato_analysis:
                    va = chinese_features.vibrato_analysis
                    logger.info(f"   Vibrato rate: {va.get('rate_hz', 0):.1f} Hz")
                
                return self._chinese_features_to_vector(chinese_features)
            else:
                # Use basic audio features
                logger.info(f"📊 Using basic audio features")
                return self._extract_basic_features(audio_data, sr)
                
        except Exception as e:
            logger.error(f"❌ Feature extraction failed: {e}")
            raise
    
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
        
        # Enhanced sliding features
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
        
        # Portamento features
        if hasattr(chinese_features, 'portamento_detection') and chinese_features.portamento_detection:
            feature_vector.extend([
                chinese_features.portamento_detection.get('presence', 0.0),
                chinese_features.portamento_detection.get('smoothness', 0.0)
            ])
        else:
            feature_vector.extend([0.0, 0.0])
        
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
            logger.error(f"Basic feature extraction failed: {e}")
            return np.zeros(self.feature_size, dtype=np.float32)
    
    def predict(self, audio_file, return_features=False):
        """Make prediction on audio file"""
        logger.info(f"🔮 Making prediction for: {os.path.basename(audio_file)}")
        
        # Extract features
        features = self.extract_features(audio_file)
        
        # Make prediction
        with torch.no_grad():
            feature_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            outputs = self.model(feature_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
            # Get prediction
            predicted_class_idx = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][predicted_class_idx].item()
            predicted_class = self.label_encoder.inverse_transform([predicted_class_idx])[0]
            
            # Prepare result
            result = {
                'file': os.path.basename(audio_file),
                'predicted_instrument': predicted_class,
                'confidence': confidence,
                'prediction_time': datetime.now().isoformat(),
                'all_probabilities': {
                    class_name: prob for class_name, prob in 
                    zip(self.class_names, probabilities[0].cpu().numpy())
                }
            }
            
            if return_features:
                result['features'] = features.tolist()
            
            return result
    
    def predict_batch(self, audio_files, save_results=False, output_file=None):
        """Make predictions on multiple audio files"""
        logger.info(f"🎵 Processing {len(audio_files)} audio files")
        
        results = []
        
        for i, audio_file in enumerate(audio_files):
            try:
                result = self.predict(audio_file)
                results.append(result)
                
                logger.info(f"✅ {i+1}/{len(audio_files)}: {result['predicted_instrument']} "
                           f"({result['confidence']:.3f})")
                
            except Exception as e:
                logger.error(f"❌ Failed to process {audio_file}: {e}")
                results.append({
                    'file': os.path.basename(audio_file),
                    'predicted_instrument': 'error',
                    'confidence': 0.0,
                    'error': str(e)
                })
        
        # Save results if requested
        if save_results:
            if output_file is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_file = f"predictions_{timestamp}.json"
            
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"💾 Results saved to: {output_file}")
        
        return results
    
    def get_model_info(self):
        """Get model information"""
        return {
            'model_path': self.model_path,
            'device': str(self.device),
            'feature_size': self.feature_size,
            'enhanced_features': self.enhanced_features,
            'class_names': self.class_names,
            'num_classes': len(self.class_names)
        }


def find_audio_files(directory):
    """Find all audio files in directory"""
    audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
    audio_files = []
    
    for file_path in Path(directory).rglob('*'):
        if file_path.suffix.lower() in audio_extensions:
            audio_files.append(str(file_path))
    
    return sorted(audio_files)


def main():
    """Main prediction function"""
    parser = argparse.ArgumentParser(description='InstrumentTimbre Prediction Tool')
    
    parser.add_argument('--model', required=True, help='Path to trained model (.pt file)')
    parser.add_argument('--input', required=True, help='Input audio file or directory')
    parser.add_argument('--output', help='Output JSON file for results')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda', 'mps'], 
                       help='Computation device')
    parser.add_argument('--features', action='store_true', help='Include extracted features in output')
    parser.add_argument('--info', action='store_true', help='Show model information only')
    
    args = parser.parse_args()
    
    print("🎵 InstrumentTimbre Prediction Tool")
    print("=" * 50)
    
    # Initialize predictor
    try:
        predictor = InstrumentPredictor(args.model, args.device)
    except Exception as e:
        logger.error(f"Failed to initialize predictor: {e}")
        return
    
    # Show model info if requested
    if args.info:
        info = predictor.get_model_info()
        print("\n📊 Model Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        return
    
    # Determine input type
    if os.path.isfile(args.input):
        # Single file prediction
        logger.info(f"🎵 Single file prediction mode")
        
        try:
            result = predictor.predict(args.input, return_features=args.features)
            
            print(f"\n🔮 Prediction Results:")
            print(f"  File: {result['file']}")
            print(f"  Predicted Instrument: {result['predicted_instrument']}")
            print(f"  Confidence: {result['confidence']:.3f}")
            print(f"  Prediction Time: {result['prediction_time']}")
            
            print(f"\n📊 All Probabilities:")
            for instrument, prob in result['all_probabilities'].items():
                print(f"    {instrument}: {prob:.3f}")
            
            # Save result if output specified
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(result, f, indent=2)
                logger.info(f"💾 Result saved to: {args.output}")
                
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
    
    elif os.path.isdir(args.input):
        # Batch prediction mode
        logger.info(f"📁 Batch prediction mode")
        
        audio_files = find_audio_files(args.input)
        
        if not audio_files:
            logger.error(f"No audio files found in: {args.input}")
            return
        
        logger.info(f"Found {len(audio_files)} audio files")
        
        # Process batch
        results = predictor.predict_batch(
            audio_files, 
            save_results=True, 
            output_file=args.output
        )
        
        # Print summary
        print(f"\n📊 Batch Prediction Summary:")
        print(f"  Total files: {len(results)}")
        
        # Count predictions by instrument
        instrument_counts = {}
        total_confidence = 0
        successful_predictions = 0
        
        for result in results:
            if 'error' not in result:
                instrument = result['predicted_instrument']
                instrument_counts[instrument] = instrument_counts.get(instrument, 0) + 1
                total_confidence += result['confidence']
                successful_predictions += 1
        
        print(f"  Successful predictions: {successful_predictions}/{len(results)}")
        
        if successful_predictions > 0:
            print(f"  Average confidence: {total_confidence/successful_predictions:.3f}")
            print(f"  Instrument distribution:")
            for instrument, count in sorted(instrument_counts.items()):
                percentage = count / successful_predictions * 100
                print(f"    {instrument}: {count} ({percentage:.1f}%)")
    
    else:
        logger.error(f"Input path not found: {args.input}")


if __name__ == "__main__":
    main()