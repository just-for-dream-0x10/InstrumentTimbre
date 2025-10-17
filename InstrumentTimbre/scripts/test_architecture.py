#!/usr/bin/env python3
"""
Test script for the new InstrumentTimbre architecture
This script validates that all new modules can be imported and basic functionality works
"""

import sys
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all new modules can be imported"""
    print("🧪 Testing module imports...")
    
    try:
        # Core modules
        from InstrumentTimbre.core.features.base import BaseFeatureExtractor
        from InstrumentTimbre.core.features.chinese import ChineseInstrumentAnalyzer
        from InstrumentTimbre.core.models.base import BaseModel
        from InstrumentTimbre.core.models.cnn import CNNClassifier, EnhancedCNNClassifier
        from InstrumentTimbre.core.training.trainer import Trainer
        from InstrumentTimbre.core.training.metrics import MetricsCalculator
        from InstrumentTimbre.core.data.datasets import AudioDataset, ChineseInstrumentDataset
        from InstrumentTimbre.core.data.loaders import get_dataloader, create_train_val_loaders
        from InstrumentTimbre.core.inference.predictor import InstrumentPredictor
        from InstrumentTimbre.core.visualization.audio_viz import AudioVisualizer
        
        # Service modules
        from InstrumentTimbre.modules.core.config import load_config
        from InstrumentTimbre.modules.core.logger import setup_logging
        
        print("✅ All core modules imported successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_feature_extraction():
    """Test feature extraction"""
    print("\n🧪 Testing feature extraction...")
    
    try:
        import numpy as np
        from InstrumentTimbre.core.features.chinese import ChineseInstrumentAnalyzer
        
        # Create dummy audio data
        sample_rate = 22050
        duration = 2.0  # 2 seconds
        audio_data = np.random.randn(int(sample_rate * duration)) * 0.1
        
        # Initialize analyzer
        analyzer = ChineseInstrumentAnalyzer()
        
        # Extract features
        features = analyzer.extract_features(audio_data, sample_rate, instrument_type='erhu')
        
        print(f"✅ Extracted {len(features)} feature groups")
        print(f"   Feature keys: {list(features.keys())[:5]}...")  # Show first 5
        
        return True
        
    except Exception as e:
        print(f"❌ Feature extraction test failed: {e}")
        return False

def test_model_creation():
    """Test model creation"""
    print("\n🧪 Testing model creation...")
    
    try:
        from InstrumentTimbre.core.models.cnn import CNNClassifier, EnhancedCNNClassifier
        
        config = {
            'input_dim': 128,
            'num_classes': 5,
            'hidden_dims': [64, 128, 64],
            'fc_dims': [256, 128]
        }
        
        # Test CNN model
        cnn_model = CNNClassifier(config)
        model_info = cnn_model.get_model_info()
        print(f"✅ CNN Model created: {model_info['total_parameters']} parameters")
        
        # Test Enhanced CNN model
        enhanced_model = EnhancedCNNClassifier(config)
        enhanced_info = enhanced_model.get_model_info()
        print(f"✅ Enhanced CNN Model created: {enhanced_info['total_parameters']} parameters")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation test failed: {e}")
        return False

def test_config_loading():
    """Test configuration loading"""
    print("\n🧪 Testing configuration loading...")
    
    try:
        from InstrumentTimbre.modules.core.config import load_config
        
        # Test loading default config
        config_path = project_root / "config" / "default_config.yaml"
        if config_path.exists():
            config = load_config(str(config_path))
            print(f"✅ Loaded config with {len(config)} sections")
            print(f"   Sections: {list(config.keys())}")
            return True
        else:
            print("⚠️  Default config file not found, but module works")
            return True
            
    except Exception as e:
        print(f"❌ Config loading test failed: {e}")
        return False

def test_cli_commands():
    """Test CLI command imports"""
    print("\n🧪 Testing CLI commands...")
    
    try:
        from InstrumentTimbre.cli.commands import train, predict, evaluate
        print("✅ CLI commands imported successfully")
        return True
        
    except ImportError as e:
        print(f"❌ CLI command import failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 InstrumentTimbre Architecture Test Suite")
    print("=" * 50)
    
    tests = [
        ("Module Imports", test_imports),
        ("Feature Extraction", test_feature_extraction),
        ("Model Creation", test_model_creation),
        ("Config Loading", test_config_loading),
        ("CLI Commands", test_cli_commands)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! Architecture is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the implementation.")
        return 1

if __name__ == '__main__':
    sys.exit(main())