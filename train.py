#!/usr/bin/env python
"""
Enhanced Instrument Timbre Analysis and Conversion System - Training Script
增强版乐器音色分析训练脚本 - 支持中国传统乐器特征
"""

import os
import argparse
import logging
import torch
import sys
import numpy as np
import librosa
from pathlib import Path
import json
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.model import InstrumentTimbreModel
from utils.data import prepare_dataloader, prepare_chinese_instrument_dataloader

# Try to import enhanced Chinese features
try:
    from InstrumentTimbre.modules.utils.chinese_instrument_features import ChineseInstrumentAnalyzer
    from InstrumentTimbre.modules.core.models import InstrumentType
    ENHANCED_FEATURES_AVAILABLE = True
    print("✅ Enhanced Chinese instrument features available")
except ImportError as e:
    print(f"⚠️  Enhanced Chinese features not available: {e}")
    ENHANCED_FEATURES_AVAILABLE = False

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class EnhancedChineseInstrumentDataset(Dataset):
    """Enhanced dataset with Chinese instrument features"""
    
    def __init__(self, audio_files, labels, enhanced_features=True):
        self.audio_files = audio_files
        self.labels = labels
        self.enhanced_features = enhanced_features
        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(labels)
        
        if enhanced_features and ENHANCED_FEATURES_AVAILABLE:
            self.analyzer = ChineseInstrumentAnalyzer()
            logger.info("Using enhanced Chinese instrument features")
        else:
            self.analyzer = None
            logger.info("Using basic audio features")
            
        # Pre-extract features for efficiency
        logger.info("Pre-extracting features for training...")
        self.features = []
        self._extract_all_features()
        
    def _extract_all_features(self):
        """Pre-extract features from all audio files"""
        for i, audio_file in enumerate(self.audio_files):
            try:
                if self.analyzer and ENHANCED_FEATURES_AVAILABLE:
                    # Use enhanced Chinese features
                    audio_data, sr = librosa.load(audio_file, sr=22050)
                    chinese_features = self.analyzer.extract_chinese_features(
                        audio_data, sr, self._guess_instrument_type(audio_file)
                    )
                    feature_vector = self._chinese_features_to_vector(chinese_features)
                else:
                    # Use basic audio features
                    feature_vector = self._extract_basic_features(audio_file)
                
                self.features.append(feature_vector)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(self.audio_files)} files")
                    
            except Exception as e:
                logger.warning(f"Failed to process {audio_file}: {e}")
                # Use fallback features
                self.features.append(np.zeros(50))
    
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
        """Convert Chinese features to fixed-size vector"""
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
        
        # Sliding features
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
        
        # Enhanced features from previous analysis
        if hasattr(chinese_features, 'portamento_detection') and chinese_features.portamento_detection:
            feature_vector.extend([
                chinese_features.portamento_detection.get('presence', 0.0),
                chinese_features.portamento_detection.get('smoothness', 0.0)
            ])
        else:
            feature_vector.extend([0.0, 0.0])
        
        # Pad or truncate to fixed size (50 dimensions)
        target_size = 50
        if len(feature_vector) < target_size:
            feature_vector.extend([0.0] * (target_size - len(feature_vector)))
        else:
            feature_vector = feature_vector[:target_size]
            
        return np.array(feature_vector, dtype=np.float32)
    
    def _extract_basic_features(self, audio_file):
        """Extract basic audio features as fallback"""
        try:
            audio_data, sr = librosa.load(audio_file, sr=22050)
            
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
            
            # Pad to 50 dimensions
            while len(feature_vector) < 50:
                feature_vector.append(0.0)
                
            return np.array(feature_vector[:50], dtype=np.float32)
            
        except Exception as e:
            logger.warning(f"Basic feature extraction failed for {audio_file}: {e}")
            return np.zeros(50, dtype=np.float32)
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        feature_vector = self.features[idx]
        label = self.encoded_labels[idx]
        return torch.FloatTensor(feature_vector), torch.LongTensor([label])


def prepare_enhanced_training_data(data_dir, enhanced_features=True):
    """Prepare training data with enhanced Chinese features"""
    logger.info(f"Preparing training data from: {data_dir}")
    
    # Find audio files
    audio_extensions = {'.wav', '.mp3', '.flac', '.m4a'}
    audio_files = []
    labels = []
    
    for file_path in Path(data_dir).rglob('*'):
        if file_path.suffix.lower() in audio_extensions:
            audio_files.append(str(file_path))
            # Extract label from parent directory or filename
            label = file_path.parent.name.lower()
            if label in ['wav', 'audio', 'data']:  # Generic folder names
                # Try to extract from filename
                filename = file_path.stem.lower()
                if 'erhu' in filename:
                    label = 'erhu'
                elif 'pipa' in filename:
                    label = 'pipa'
                elif 'guzheng' in filename:
                    label = 'guzheng'
                elif 'dizi' in filename:
                    label = 'dizi'
                else:
                    label = 'unknown'
            labels.append(label)
    
    if not audio_files:
        # Use example files if no data directory found
        logger.warning("No audio files found in data directory, using example files")
        example_files = [
            "example/erhu1.wav",
            "example/erhu2.wav"
        ]
        audio_files = [f for f in example_files if os.path.exists(f)]
        labels = ['erhu'] * len(audio_files)
    
    logger.info(f"Found {len(audio_files)} audio files with {len(set(labels))} classes")
    logger.info(f"Classes: {set(labels)}")
    
    return audio_files, labels


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Instrument Timbre Analysis and Conversion System - Training Script"
    )

    # Dataset parameters
    parser.add_argument(
        "--data-dir", "--dataset-path", default="../wav", help="Training data directory"
    )
    parser.add_argument(
        "--model-path", default="./saved_models/model.pt", help="Model save path"
    )
    parser.add_argument(
        "--use-wav-files",
        action="store_true",
        help="Use WAV files directly as training data",
    )
    parser.add_argument(
        "--augment", action="store_true", help="Apply data augmentation"
    )

    # Training parameters
    parser.add_argument(
        "--epochs", type=int, default=30, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--lr", "--learning-rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument(
        "--patience", type=int, default=5, help="Early stopping patience"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of data loading threads"
    )

    # Model parameters
    parser.add_argument(
        "--chinese-instruments",
        action="store_true",
        help="Optimize for Chinese traditional instruments",
    )
    parser.add_argument(
        "--enhanced-features",
        action="store_true",
        default=True,
        help="Use enhanced Chinese instrument features",
    )
    parser.add_argument(
        "--feature-type",
        choices=["mel", "constant-q", "multi"],
        default="multi",
        help="Feature type (only applicable to Chinese instrument mode)",
    )
    parser.add_argument(
        "--pretrained", action="store_true", help="Use pretrained audio model"
    )

    # Other parameters
    parser.add_argument(
        "--cache-features", action="store_true", help="Enable feature caching"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--export-onnx", action="store_true", help="Export trained model to ONNX format"
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "mps", "auto"],
        default="auto",
        help="Computation device",
    )

    return parser.parse_args()


def train_enhanced_chinese_model(args):
    """Train model with enhanced Chinese instrument features"""
    logger.info("🎵 Starting Enhanced Chinese Instrument Training")
    logger.info("=" * 60)
    
    # Prepare enhanced training data
    audio_files, labels = prepare_enhanced_training_data(
        args.data_dir, 
        enhanced_features=args.enhanced_features
    )
    
    if len(audio_files) < 2:
        logger.error("❌ Need at least 2 audio files for training!")
        return None
    
    # Create enhanced dataset
    dataset = EnhancedChineseInstrumentDataset(
        audio_files, labels, enhanced_features=args.enhanced_features
    )
    
    # Create data loader
    dataloader = DataLoader(
        dataset, 
        batch_size=min(args.batch_size, len(dataset)),
        shuffle=True,
        num_workers=0  # Avoid multiprocessing issues with features
    )
    
    # Simple classifier for enhanced features
    class EnhancedChineseClassifier(torch.nn.Module):
        def __init__(self, input_size=50, num_classes=1):
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
    
    # Initialize model
    num_classes = len(set(labels))
    device = torch.device(args.device if args.device != "auto" else 
                         "cuda" if torch.cuda.is_available() else "cpu")
    
    model = EnhancedChineseClassifier(input_size=50, num_classes=num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    logger.info(f"🧠 Model initialized: {num_classes} classes, {sum(p.numel() for p in model.parameters())} parameters")
    logger.info(f"📊 Device: {device}")
    
    # Training loop
    logger.info("🚀 Starting enhanced training...")
    model.train()
    
    best_accuracy = 0.0
    patience_counter = 0
    
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        num_batches = 0
        
        for batch_idx, (features, labels_batch) in enumerate(dataloader):
            features = features.to(device)
            labels_batch = labels_batch.to(device).squeeze()
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels_batch)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            epoch_loss += loss.item()
            num_batches += 1
            
            # Accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels_batch.size(0)
            correct_predictions += (predicted == labels_batch).sum().item()
            
            if args.debug and batch_idx >= 5:  # Limit batches in debug mode
                break
        
        # Calculate metrics
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        logger.info(f"📈 Epoch {epoch + 1}/{args.epochs}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.3f}")
        
        # Early stopping
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"🛑 Early stopping at epoch {epoch + 1}")
                break
    
    # Save enhanced model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    enhanced_model_path = args.model_path.replace('.pt', f'_enhanced_{timestamp}.pt')
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'label_encoder': dataset.label_encoder,
        'feature_size': 50,
        'num_classes': num_classes,
        'class_names': dataset.label_encoder.classes_.tolist(),
        'enhanced_features': args.enhanced_features,
        'training_args': vars(args),
        'best_accuracy': best_accuracy,
        'timestamp': timestamp
    }, enhanced_model_path)
    
    logger.info(f"💾 Enhanced model saved to: {enhanced_model_path}")
    logger.info(f"🎯 Best accuracy: {best_accuracy:.3f}")
    
    # Test feature extraction on one sample
    if len(audio_files) > 0:
        logger.info("\n🔍 Testing Enhanced Feature Extraction:")
        test_file = audio_files[0]
        
        if dataset.analyzer and ENHANCED_FEATURES_AVAILABLE:
            try:
                audio_data, sr = librosa.load(test_file, sr=22050)
                features = dataset.analyzer.extract_chinese_features(
                    audio_data, sr, dataset._guess_instrument_type(test_file)
                )
                
                logger.info(f"📂 Test file: {Path(test_file).name}")
                logger.info(f"✓ Pentatonic adherence: {features.pentatonic_adherence:.3f}")
                logger.info(f"✓ Ornament density: {features.ornament_density:.3f}")
                logger.info(f"✓ Rhythmic complexity: {features.rhythmic_complexity:.3f}")
                
                if features.vibrato_analysis:
                    va = features.vibrato_analysis
                    logger.info(f"✓ Vibrato rate: {va.get('rate_hz', 0):.2f} Hz")
                    logger.info(f"✓ Vibrato extent: {va.get('extent_cents', 0):.1f} cents")
                    
            except Exception as e:
                logger.warning(f"Feature extraction test failed: {e}")
    
    logger.info("\n🎉 Enhanced training completed successfully!")
    return enhanced_model_path


def main():
    """Main function"""
    args = parse_args()

    # Ensure model save directory exists
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)

    logger.info(f"🎵 InstrumentTimbre Enhanced Training")
    logger.info(f"📁 Data directory: {args.data_dir}")
    logger.info(f"🎯 Model path: {args.model_path}")
    logger.info(f"⚡ Enhanced features: {args.enhanced_features}")
    logger.info(f"🇨🇳 Chinese instruments: {args.chinese_instruments}")

    # Use enhanced training if Chinese instruments mode is enabled
    if args.chinese_instruments or args.enhanced_features:
        model_path = train_enhanced_chinese_model(args)
        if model_path:
            logger.info(f"✅ Enhanced training completed: {model_path}")
        else:
            logger.error("❌ Enhanced training failed")
        return

    # Original training logic for non-Chinese instruments
    # Set device
    if args.device == "auto":
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    else:
        device = torch.device(args.device)

    logger.info(f"Using device: {device}")

    # Prepare data loader
    if args.chinese_instruments:
        logger.info("Using specialized data loader for Chinese traditional instruments")
        dataloader = prepare_chinese_instrument_dataloader(
            args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_wav_files=args.use_wav_files,
            augment=args.augment,
            debug=args.debug,
            feature_type=args.feature_type,
        )
    else:
        logger.info("Using standard data loader")
        dataloader = prepare_dataloader(
            args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_wav_files=args.use_wav_files,
            augment=args.augment,
            debug=args.debug,
        )

    # Initialize model
    model = InstrumentTimbreModel(
        use_pretrained=args.pretrained,
        chinese_instruments=args.chinese_instruments,
        feature_caching=args.cache_features,
        device=device,
    )

    # Train model
    logger.info("Starting model training...")
    model.train(
        dataloader=dataloader,
        epochs=args.epochs,
        learning_rate=args.lr,
        max_batches=100 if args.debug else None,
    )

    # Save model
    logger.info(f"Saving model to {args.model_path}")
    model.save_model(args.model_path)

    # Export ONNX (if needed)
    if args.export_onnx:
        onnx_path = args.model_path.replace(".pt", ".onnx")
        logger.info(f"Exporting ONNX model to {onnx_path}")
        model.export_to_onnx(onnx_path)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
