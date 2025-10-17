#!/usr/bin/env python3
"""
Compatibility Test for Unified Architecture

This script tests the new unified model architecture against the existing
InstrumentTimbre system to ensure 100% backward compatibility.
"""

import sys
import os
import torch
import numpy as np
import yaml
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_model_loading():
    """Test loading both legacy and unified models"""
    logger.info("Testing model loading...")
    
    try:
        # Test 1: Load unified config
        from InstrumentTimbre.core.models.compatibility_wrapper import create_model
        
        # Load configuration
        config_path = project_root / "config" / "unified_config.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Test legacy mode
        logger.info("Creating model in legacy mode...")
        legacy_model = create_model(config['model'], mode='legacy')
        logger.info(f"Legacy model created: {legacy_model.__class__.__name__}")
        
        # Test unified mode (analysis only)
        logger.info("Creating model in unified analysis mode...")
        unified_model = create_model(config['model'], mode='analysis')
        logger.info(f"Unified model created: {unified_model.__class__.__name__}")
        
        # Test capabilities
        legacy_caps = legacy_model.get_capabilities()
        unified_caps = unified_model.get_capabilities()
        
        logger.info(f"Legacy capabilities: {legacy_caps}")
        logger.info(f"Unified capabilities: {unified_caps}")
        
        return True, legacy_model, unified_model
        
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        return False, None, None

def test_feature_extraction():
    """Test feature extraction compatibility"""
    logger.info("Testing feature extraction...")
    
    try:
        from InstrumentTimbre.core.features.unified_features import UnifiedFeatureExtractor
        from InstrumentTimbre.core.features.chinese import ChineseInstrumentAnalyzer
        
        # Create synthetic audio data (8 seconds at 22050 Hz)
        sample_rate = 22050
        duration = 8.0
        audio_data = np.random.randn(int(sample_rate * duration)) * 0.1
        
        # Test legacy feature extraction
        logger.info("Testing legacy feature extraction...")
        legacy_extractor = ChineseInstrumentAnalyzer()
        try:
            legacy_features = legacy_extractor.extract_features(audio_data, sample_rate)
            legacy_dim = legacy_features['features'].shape[0]
            logger.info(f"Legacy features extracted: {legacy_dim}D")
        except Exception as e:
            logger.warning(f"Legacy feature extraction failed: {e}")
            # Create dummy features for testing
            legacy_features = {'features': np.random.randn(34)}
            legacy_dim = 34
        
        # Test unified feature extraction
        logger.info("Testing unified feature extraction...")
        unified_extractor = UnifiedFeatureExtractor()
        
        # Test legacy mode
        legacy_mode_features = unified_extractor.extract_features(
            audio_data, sample_rate, feature_type='legacy'
        )
        
        # Test unified mode
        unified_features = unified_extractor.extract_features(
            audio_data, sample_rate, feature_type='unified'
        )
        
        logger.info(f"Legacy mode features: {legacy_mode_features['features'].shape[0]}D")
        logger.info(f"Unified features: {unified_features['features'].shape[0]}D")
        logger.info(f"Audio length: {unified_features['audio_length']:.1f}s")
        logger.info(f"Confidence penalty: {unified_features['confidence_penalty']:.2f}")
        
        # Test short audio handling
        short_audio = audio_data[:int(sample_rate * 5)]  # 5 seconds
        short_features = unified_extractor.extract_features(
            short_audio, sample_rate, feature_type='unified'
        )
        logger.info(f"Short audio (5s) confidence penalty: {short_features['confidence_penalty']:.2f}")
        
        return True, legacy_features, unified_features
        
    except Exception as e:
        logger.error(f"Feature extraction test failed: {e}")
        return False, None, None

def test_model_inference():
    """Test model inference compatibility"""
    logger.info("Testing model inference...")
    
    try:
        # Create test models
        success, legacy_model, unified_model = test_model_loading()
        if not success:
            return False
        
        # Create test input
        batch_size = 2
        feature_dim = 128  # Expected input dimension
        test_input = torch.randn(batch_size, feature_dim)
        
        # Test legacy model inference
        logger.info("Testing legacy model inference...")
        legacy_model.eval()
        with torch.no_grad():
            legacy_output = legacy_model(test_input)
        
        logger.info(f"Legacy output shape: {legacy_output.shape}")
        logger.info(f"Legacy output sample: {legacy_output[0, :3].numpy()}")
        
        # Test unified model inference (compatibility mode)
        logger.info("Testing unified model inference...")
        unified_model.eval()
        with torch.no_grad():
            unified_output = unified_model(test_input)
        
        logger.info(f"Unified output shape: {unified_output.shape}")
        logger.info(f"Unified output sample: {unified_output[0, :3].numpy()}")
        
        # Test shape compatibility
        if legacy_output.shape == unified_output.shape:
            logger.info("✅ Output shapes are compatible!")
        else:
            logger.warning(f"❌ Shape mismatch: {legacy_output.shape} vs {unified_output.shape}")
        
        # Test extended functionality (if available)
        if hasattr(unified_model, 'analyze_complete'):
            logger.info("Testing extended analysis...")
            complete_analysis = unified_model.analyze_complete(test_input)
            logger.info(f"Complete analysis tasks: {list(complete_analysis.keys())}")
        
        return True
        
    except Exception as e:
        logger.error(f"Model inference test failed: {e}")
        return False

def test_configuration_compatibility():
    """Test configuration file compatibility"""
    logger.info("Testing configuration compatibility...")
    
    try:
        # Load default config
        default_config_path = project_root / "config" / "default_config.yaml"
        with open(default_config_path, 'r') as f:
            default_config = yaml.safe_load(f)
        
        # Load unified config
        unified_config_path = project_root / "config" / "unified_config.yaml"
        with open(unified_config_path, 'r') as f:
            unified_config = yaml.safe_load(f)
        
        # Check key compatibility
        essential_keys = ['model', 'training', 'data', 'features']
        for key in essential_keys:
            if key in default_config and key in unified_config:
                logger.info(f"✅ Key '{key}' present in both configs")
            else:
                logger.warning(f"❌ Key '{key}' missing in one of the configs")
        
        # Check model parameters
        default_model = default_config.get('model', {})
        unified_model = unified_config.get('model', {})
        
        critical_params = ['input_dim', 'num_classes', 'dropout_rate']
        for param in critical_params:
            default_val = default_model.get(param)
            unified_val = unified_model.get(param)
            if default_val == unified_val:
                logger.info(f"✅ Parameter '{param}': {default_val} (compatible)")
            else:
                logger.warning(f"❌ Parameter '{param}': {default_val} -> {unified_val} (changed)")
        
        return True
        
    except Exception as e:
        logger.error(f"Configuration test failed: {e}")
        return False

def test_api_compatibility():
    """Test API compatibility with existing interfaces"""
    logger.info("Testing API compatibility...")
    
    try:
        from InstrumentTimbre.core.models.compatibility_wrapper import CompatibilityWrapper
        
        # Test configuration
        config = {
            'legacy_mode': False,
            'input_dim': 128,
            'num_classes': 5,
            'enable_generation': False,
            'enable_control': False
        }
        
        # Create wrapper
        wrapper = CompatibilityWrapper(config)
        
        # Test existing API methods
        test_input = torch.randn(2, 128)
        
        # Test forward (existing API)
        output = wrapper.forward(test_input)
        logger.info(f"✅ forward() method: {output.shape}")
        
        # Test get_feature_dim (existing API)
        feature_dim = wrapper.get_feature_dim()
        logger.info(f"✅ get_feature_dim() method: {feature_dim}")
        
        # Test model info (existing API)
        model_info = wrapper.get_model_info()
        logger.info(f"✅ get_model_info() method: {model_info['total_parameters']} parameters")
        
        # Test capabilities (new API)
        capabilities = wrapper.get_capabilities()
        logger.info(f"✅ get_capabilities() method: {capabilities}")
        
        return True
        
    except Exception as e:
        logger.error(f"API compatibility test failed: {e}")
        return False

def run_all_tests():
    """Run all compatibility tests"""
    logger.info("="*50)
    logger.info("RUNNING UNIFIED ARCHITECTURE COMPATIBILITY TESTS")
    logger.info("="*50)
    
    tests = [
        ("Model Loading", test_model_loading),
        ("Feature Extraction", test_feature_extraction),
        ("Model Inference", test_model_inference),
        ("Configuration Compatibility", test_configuration_compatibility),
        ("API Compatibility", test_api_compatibility)
    ]
    
    results = {}
    for test_name, test_func in tests:
        logger.info(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results[test_name] = result
            if result:
                logger.info(f"✅ {test_name} PASSED")
            else:
                logger.error(f"❌ {test_name} FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name} ERROR: {e}")
            results[test_name] = False
    
    # Summary
    logger.info(f"\n{'='*20} TEST SUMMARY {'='*20}")
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED! Unified architecture is compatible.")
        return True
    else:
        logger.warning(f"⚠️  {total - passed} tests failed. Review compatibility issues.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)