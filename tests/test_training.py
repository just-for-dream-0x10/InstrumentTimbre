#!/usr/bin/env python3
"""
Test suite for training functionality
训练功能测试套件
"""

import pytest
import torch
import numpy as np
import tempfile
import os
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestTrainingComponents:
    """Test training-related components"""
    
    def test_pytorch_availability(self):
        """Test PyTorch installation"""
        assert torch.__version__ is not None
        print(f"PyTorch version: {torch.__version__}")
    
    def test_device_detection(self):
        """Test device detection logic"""
        # Test CPU
        cpu_device = torch.device('cpu')
        assert cpu_device.type == 'cpu'
        
        # Test CUDA if available
        if torch.cuda.is_available():
            cuda_device = torch.device('cuda')
            assert cuda_device.type == 'cuda'
            print(f"CUDA device available: {torch.cuda.get_device_name(0)}")
        
        # Test MPS if available (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            mps_device = torch.device('mps')
            assert mps_device.type == 'mps'
            print("MPS device available")
    
    def test_simple_model_creation(self):
        """Test creating a simple model"""
        class TestModel(torch.nn.Module):
            def __init__(self, input_size=50, num_classes=5):
                super().__init__()
                self.linear = torch.nn.Linear(input_size, num_classes)
            
            def forward(self, x):
                return self.linear(x)
        
        model = TestModel()
        assert model is not None
        
        # Test forward pass
        test_input = torch.randn(1, 50)
        output = model(test_input)
        assert output.shape == (1, 5)
    
    def test_model_training_loop(self):
        """Test basic training loop components"""
        # Create simple model and data
        model = torch.nn.Linear(10, 2)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Create synthetic data
        X = torch.randn(20, 10)
        y = torch.randint(0, 2, (20,))
        
        # Test one training step
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        assert loss.item() >= 0  # Loss should be non-negative
        print(f"Test loss: {loss.item():.4f}")
    
    def test_model_save_load(self):
        """Test model saving and loading"""
        # Create model
        model = torch.nn.Linear(10, 2)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp_file:
            torch.save(model.state_dict(), tmp_file.name)
            
            # Load model
            loaded_state = torch.load(tmp_file.name, map_location='cpu')
            
            # Create new model and load state
            new_model = torch.nn.Linear(10, 2)
            new_model.load_state_dict(loaded_state)
            
            # Test that models produce same output
            test_input = torch.randn(1, 10)
            original_output = model(test_input)
            loaded_output = new_model(test_input)
            
            assert torch.allclose(original_output, loaded_output)
            
            # Cleanup
            os.unlink(tmp_file.name)


class TestEnhancedTrainingData:
    """Test enhanced training data handling"""
    
    def test_feature_vector_creation(self):
        """Test feature vector creation from Chinese features"""
        # Mock Chinese features
        mock_features = {
            'pentatonic_adherence': 0.75,
            'ornament_density': 0.25,
            'rhythmic_complexity': 0.5,
            'sliding_presence': 0.3,
            'vibrato_rate': 5.0,
            'f0_mean': 440.0,
            'f0_std': 20.0
        }
        
        # Convert to fixed-size vector (simplified version)
        vector = []
        vector.append(mock_features.get('pentatonic_adherence', 0.0))
        vector.append(mock_features.get('ornament_density', 0.0))
        vector.append(mock_features.get('rhythmic_complexity', 0.0))
        vector.append(mock_features.get('sliding_presence', 0.0))
        vector.append(mock_features.get('vibrato_rate', 0.0) / 15.0)  # Normalize
        
        # Pad to target size
        target_size = 50
        while len(vector) < target_size:
            vector.append(0.0)
        
        vector = np.array(vector[:target_size])
        
        assert len(vector) == target_size
        assert vector[0] == 0.75  # pentatonic_adherence
        assert vector[1] == 0.25  # ornament_density
        assert abs(vector[4] - (5.0/15.0)) < 1e-6  # normalized vibrato_rate
    
    def test_label_encoding(self):
        """Test label encoding for Chinese instruments"""
        from sklearn.preprocessing import LabelEncoder
        
        labels = ['erhu', 'pipa', 'guzheng', 'erhu', 'dizi', 'pipa']
        
        encoder = LabelEncoder()
        encoded = encoder.fit_transform(labels)
        
        assert len(encoded) == len(labels)
        assert len(set(encoded)) == 4  # 4 unique instruments
        
        # Test inverse transform
        decoded = encoder.inverse_transform(encoded)
        assert list(decoded) == labels
    
    def test_data_augmentation_concepts(self):
        """Test data augmentation concepts"""
        # Generate sample audio
        sr = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        
        # Test time stretching concept (without actual librosa.effects)
        # Just test the concept by resampling
        stretch_factor = 1.1
        new_length = int(len(audio) / stretch_factor)
        stretched = np.interp(
            np.linspace(0, len(audio)-1, new_length),
            np.arange(len(audio)),
            audio
        )
        
        assert len(stretched) < len(audio)  # Should be shorter
        
        # Test pitch shifting concept
        pitch_shift_semitones = 2
        pitch_factor = 2 ** (pitch_shift_semitones / 12.0)
        shifted_freq = 440 * pitch_factor
        pitched = 0.5 * np.sin(2 * np.pi * shifted_freq * t)
        
        assert not np.array_equal(audio, pitched)  # Should be different
    
    def test_batch_creation(self):
        """Test batch creation for training"""
        # Mock dataset
        features = [
            torch.randn(50) for _ in range(10)
        ]
        labels = [
            torch.tensor([i % 3]) for i in range(10)
        ]
        
        # Create batches
        batch_size = 4
        batches = []
        
        for i in range(0, len(features), batch_size):
            batch_features = torch.stack(features[i:i+batch_size])
            batch_labels = torch.stack(labels[i:i+batch_size])
            batches.append((batch_features, batch_labels))
        
        assert len(batches) == 3  # 10 samples, batch_size 4 -> 3 batches
        assert batches[0][0].shape == (4, 50)  # First batch
        assert batches[2][0].shape == (2, 50)  # Last batch (partial)


class TestModelArchitecture:
    """Test model architecture components"""
    
    def test_enhanced_classifier_architecture(self):
        """Test enhanced classifier architecture"""
        class EnhancedChineseClassifier(torch.nn.Module):
            def __init__(self, input_size=50, num_classes=5):
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
        
        model = EnhancedChineseClassifier()
        
        # Test forward pass
        test_input = torch.randn(8, 50)  # Batch of 8
        output = model(test_input)
        
        assert output.shape == (8, 5)
        
        # Test parameter count
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
        assert total_params > 50000  # Should have reasonable number of parameters
    
    def test_loss_functions(self):
        """Test different loss functions"""
        # Cross-entropy for classification
        ce_loss = torch.nn.CrossEntropyLoss()
        
        # Test data
        predictions = torch.randn(10, 5)  # 10 samples, 5 classes
        targets = torch.randint(0, 5, (10,))
        
        loss = ce_loss(predictions, targets)
        assert loss.item() >= 0
        
        # Test that loss decreases with better predictions
        # Create "perfect" predictions
        perfect_predictions = torch.zeros(10, 5)
        perfect_predictions[range(10), targets] = 10.0  # High confidence for correct class
        
        perfect_loss = ce_loss(perfect_predictions, targets)
        assert perfect_loss < loss  # Should be lower
    
    def test_optimizer_setup(self):
        """Test optimizer configuration"""
        model = torch.nn.Linear(50, 5)
        
        # Test Adam optimizer
        adam_opt = torch.optim.Adam(model.parameters(), lr=0.001)
        assert adam_opt.param_groups[0]['lr'] == 0.001
        
        # Test SGD optimizer
        sgd_opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        assert sgd_opt.param_groups[0]['momentum'] == 0.9
        
        # Test learning rate scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(adam_opt, step_size=10, gamma=0.1)
        initial_lr = adam_opt.param_groups[0]['lr']
        
        # Simulate 10 steps
        for _ in range(10):
            scheduler.step()
        
        new_lr = adam_opt.param_groups[0]['lr']
        assert new_lr == initial_lr * 0.1  # Should be reduced


if __name__ == "__main__":
    pytest.main([__file__, "-v"])