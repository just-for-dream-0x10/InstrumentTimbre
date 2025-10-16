#!/usr/bin/env python3
"""
Enhanced Chinese Instrument Training Script
使用增强特征的中国乐器训练脚本
"""

import os
import sys
import numpy as np
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our enhanced features
sys.path.append('.')
from InstrumentTimbre.modules.utils.chinese_instrument_features import ChineseInstrumentAnalyzer
from InstrumentTimbre.modules.core.models import InstrumentType

class EnhancedChineseDataset(Dataset):
    """Dataset with enhanced Chinese instrument features."""
    
    def __init__(self, audio_files, labels, analyzer):
        self.audio_files = audio_files
        self.labels = labels
        self.analyzer = analyzer
        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(labels)
        
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        try:
            # Load audio
            audio_data, sr = librosa.load(self.audio_files[idx], sr=22050)
            
            # Extract enhanced features
            features = self.analyzer.extract_chinese_features(
                audio_data, sr, InstrumentType.ERHU  # Assume ERHU for now
            )
            
            # Create feature vector from enhanced features
            feature_vector = self._create_feature_vector(features)
            
            return torch.FloatTensor(feature_vector), torch.LongTensor([self.encoded_labels[idx]])
            
        except Exception as e:
            logger.warning(f"Failed to process {self.audio_files[idx]}: {e}")
            # Return dummy features on error
            return torch.zeros(50), torch.LongTensor([0])
    
    def _create_feature_vector(self, features):
        """Create a fixed-size feature vector from extracted features."""
        vector = []
        
        # Basic features
        vector.append(features.pentatonic_adherence or 0.0)
        vector.append(features.ornament_density or 0.0)
        vector.append(features.rhythmic_complexity or 0.0)
        
        # Vibrato features
        if features.vibrato_analysis:
            vector.extend([
                features.vibrato_analysis.get('rate_hz', 0.0),
                features.vibrato_analysis.get('extent_cents', 0.0),
                features.vibrato_analysis.get('regularity', 0.0),
                features.vibrato_analysis.get('presence', 0.0)
            ])
        else:
            vector.extend([0.0, 0.0, 0.0, 0.0])
        
        # Sliding features statistics
        if features.sliding_detection is not None:
            vector.extend([
                np.mean(features.sliding_detection),
                np.std(features.sliding_detection),
                np.max(features.sliding_detection)
            ])
        else:
            vector.extend([0.0, 0.0, 0.0])
            
        # Enhanced sliding features
        if features.sliding_velocity is not None:
            vector.extend([
                np.mean(np.abs(features.sliding_velocity[~np.isnan(features.sliding_velocity)])) if np.any(~np.isnan(features.sliding_velocity)) else 0.0,
                np.std(features.sliding_velocity[~np.isnan(features.sliding_velocity)]) if np.any(~np.isnan(features.sliding_velocity)) else 0.0
            ])
        else:
            vector.extend([0.0, 0.0])
        
        # Portamento features
        if features.portamento_detection:
            vector.extend([
                features.portamento_detection.get('presence', 0.0),
                features.portamento_detection.get('smoothness', 0.0)
            ])
        else:
            vector.extend([0.0, 0.0])
            
        # Grace notes and trills
        if features.grace_note_detection is not None:
            vector.append(np.sum(features.grace_note_detection > 0.5))
        else:
            vector.append(0.0)
            
        if features.trill_detection:
            vector.extend([
                features.trill_detection.get('presence', 0.0),
                features.trill_detection.get('rate_hz', 0.0)
            ])
        else:
            vector.extend([0.0, 0.0])
        
        # Pad or truncate to fixed size
        target_size = 50
        if len(vector) < target_size:
            vector.extend([0.0] * (target_size - len(vector)))
        else:
            vector = vector[:target_size]
            
        return vector

class SimpleClassifier(nn.Module):
    """Simple neural network for testing enhanced features."""
    
    def __init__(self, input_size=50, num_classes=3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(), 
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)

def prepare_data():
    """Prepare training data."""
    audio_files = []
    labels = []
    
    # Use example files
    example_files = [
        ("example/erhu1.wav", "erhu"),
        ("example/erhu2.wav", "erhu")
    ]
    
    for file_path, label in example_files:
        if os.path.exists(file_path):
            audio_files.append(file_path)
            labels.append(label)
    
    return audio_files, labels

def train_enhanced_model():
    """Train model with enhanced features."""
    logger.info("🎵 Starting Enhanced Chinese Instrument Training")
    logger.info("=" * 60)
    
    # Initialize enhanced analyzer
    logger.info("📊 Initializing Enhanced Chinese Instrument Analyzer...")
    analyzer = ChineseInstrumentAnalyzer()
    
    # Prepare data
    logger.info("📁 Preparing training data...")
    audio_files, labels = prepare_data()
    
    if len(audio_files) == 0:
        logger.error("❌ No audio files found for training!")
        return
        
    logger.info(f"✅ Found {len(audio_files)} audio files")
    
    # Create dataset
    dataset = EnhancedChineseDataset(audio_files, labels, analyzer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Initialize model
    num_classes = len(set(labels))
    model = SimpleClassifier(input_size=50, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    logger.info(f"🧠 Model initialized with {num_classes} classes")
    
    # Training loop
    logger.info("🚀 Starting training...")
    model.train()
    
    for epoch in range(5):  # Small number for demo
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (features, labels_batch) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels_batch.squeeze())
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            logger.info(f"  Batch {batch_idx + 1}: Loss = {loss.item():.4f}")
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        logger.info(f"📈 Epoch {epoch + 1}: Average Loss = {avg_loss:.4f}")
    
    # Save enhanced model
    model_path = "saved_models/enhanced_chinese_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'label_encoder': dataset.label_encoder,
        'feature_size': 50
    }, model_path)
    
    logger.info(f"💾 Enhanced model saved to {model_path}")
    
    # Test feature extraction
    logger.info("\n🔍 Testing Enhanced Feature Extraction:")
    logger.info("-" * 40)
    
    test_file = "example/erhu1.wav"
    if os.path.exists(test_file):
        audio_data, sr = librosa.load(test_file, sr=22050)
        features = analyzer.extract_chinese_features(audio_data, sr, InstrumentType.ERHU)
        
        print(f"✓ Pentatonic adherence: {features.pentatonic_adherence:.3f}")
        print(f"✓ Ornament density: {features.ornament_density:.3f}")
        print(f"✓ Rhythmic complexity: {features.rhythmic_complexity:.3f}")
        
        if features.vibrato_analysis:
            print(f"✓ Vibrato rate: {features.vibrato_analysis.get('rate_hz', 0):.2f} Hz")
            print(f"✓ Vibrato extent: {features.vibrato_analysis.get('extent_cents', 0):.1f} cents")
        
        if features.sliding_velocity is not None:
            valid_velocities = features.sliding_velocity[~np.isnan(features.sliding_velocity)]
            if len(valid_velocities) > 0:
                print(f"✓ Max sliding velocity: {np.max(np.abs(valid_velocities)):.1f} cents/frame")
        
        if features.portamento_detection:
            print(f"✓ Portamento presence: {features.portamento_detection.get('presence', 0):.3f}")
        
        if features.grace_note_detection is not None:
            grace_count = np.sum(features.grace_note_detection > 0.5)
            print(f"✓ Grace notes detected: {grace_count}")
    
    logger.info("\n🎉 Enhanced training completed successfully!")
    return model_path

if __name__ == "__main__":
    train_enhanced_model()