#!/usr/bin/env python3
"""
Test suite for Chinese instrument features
中国乐器特征测试套件
"""

import pytest
import numpy as np
import librosa
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from InstrumentTimbre.modules.utils.chinese_instrument_features import ChineseInstrumentAnalyzer
    from InstrumentTimbre.modules.core.models import InstrumentType
    ENHANCED_FEATURES_AVAILABLE = True
except ImportError:
    ENHANCED_FEATURES_AVAILABLE = False


class TestChineseInstrumentFeatures:
    """Test Chinese instrument feature extraction"""
    
    @pytest.fixture
    def sample_audio(self):
        """Generate sample audio for testing"""
        # Generate a simple sine wave with vibrato
        sr = 22050
        duration = 2.0  # 2 seconds
        t = np.linspace(0, duration, int(sr * duration))
        
        # Base frequency (A4 = 440 Hz)
        base_freq = 440.0
        
        # Add vibrato (5 Hz, 20 cents depth)
        vibrato_rate = 5.0
        vibrato_depth = 20.0 / 1200.0  # Convert cents to octaves
        vibrato = vibrato_depth * np.sin(2 * np.pi * vibrato_rate * t)
        
        # Generate audio with vibrato
        freq = base_freq * (2 ** vibrato)
        audio = 0.5 * np.sin(2 * np.pi * freq * t)
        
        # Add some sliding (frequency modulation)
        slide = 100 * t / duration  # 100 Hz slide over duration
        audio += 0.2 * np.sin(2 * np.pi * (base_freq + slide) * t)
        
        return audio, sr
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance"""
        if not ENHANCED_FEATURES_AVAILABLE:
            pytest.skip("Enhanced Chinese features not available")
        return ChineseInstrumentAnalyzer()
    
    def test_analyzer_initialization(self, analyzer):
        """Test analyzer initialization"""
        assert analyzer is not None
        assert hasattr(analyzer, 'extract_chinese_features')
        assert hasattr(analyzer, 'instrument_params')
    
    def test_feature_extraction_with_sample_audio(self, analyzer, sample_audio):
        """Test feature extraction with synthetic audio"""
        audio_data, sr = sample_audio
        
        features = analyzer.extract_chinese_features(
            audio_data, sr, InstrumentType.ERHU
        )
        
        # Check that features object is created
        assert features is not None
        
        # Check basic features
        assert hasattr(features, 'pentatonic_adherence')
        assert hasattr(features, 'ornament_density')
        assert hasattr(features, 'rhythmic_complexity')
        
        # Check values are in reasonable ranges
        if features.pentatonic_adherence is not None:
            assert 0.0 <= features.pentatonic_adherence <= 1.0
        
        if features.ornament_density is not None:
            assert features.ornament_density >= 0.0
    
    def test_vibrato_detection(self, analyzer, sample_audio):
        """Test vibrato detection with known vibrato signal"""
        audio_data, sr = sample_audio
        
        features = analyzer.extract_chinese_features(
            audio_data, sr, InstrumentType.ERHU
        )
        
        # Should detect vibrato in synthetic signal
        if features.vibrato_analysis:
            vibrato_rate = features.vibrato_analysis.get('rate_hz', 0)
            # Should detect vibrato around 5 Hz (within reasonable tolerance)
            assert 3.0 <= vibrato_rate <= 7.0, f"Expected vibrato ~5Hz, got {vibrato_rate}Hz"
    
    def test_sliding_detection(self, analyzer):
        """Test sliding detection with known sliding signal"""
        # Generate audio with clear sliding
        sr = 22050
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))
        
        # Frequency that slides from 440 to 880 Hz
        freq = 440 + (440 * t / duration)
        audio = 0.5 * np.sin(2 * np.pi * freq * np.cumsum(np.ones_like(t)) / sr)
        
        features = analyzer.extract_chinese_features(
            audio, sr, InstrumentType.ERHU
        )
        
        # Should detect sliding presence
        if hasattr(features, 'sliding_detection') and features.sliding_detection is not None:
            sliding_presence = np.mean(features.sliding_detection)
            assert sliding_presence > 0.1, f"Expected sliding presence > 0.1, got {sliding_presence}"
    
    def test_different_instrument_types(self, analyzer, sample_audio):
        """Test feature extraction with different instrument types"""
        audio_data, sr = sample_audio
        
        instruments = [InstrumentType.ERHU, InstrumentType.PIPA, InstrumentType.GUZHENG]
        
        for instrument in instruments:
            features = analyzer.extract_chinese_features(
                audio_data, sr, instrument
            )
            
            assert features is not None
            # Features should be consistent regardless of instrument hint
            assert hasattr(features, 'pentatonic_adherence')
    
    @pytest.mark.skipif(not os.path.exists("example/erhu1.wav"), 
                       reason="Test audio file not available")
    def test_real_audio_file(self, analyzer):
        """Test with real audio file if available"""
        audio_file = "example/erhu1.wav"
        
        audio_data, sr = librosa.load(audio_file, sr=22050)
        features = analyzer.extract_chinese_features(
            audio_data, sr, InstrumentType.ERHU
        )
        
        assert features is not None
        
        # Real erhu should have some pentatonic characteristics
        if features.pentatonic_adherence is not None:
            assert features.pentatonic_adherence > 0.0
        
        print(f"Real audio features:")
        print(f"  Pentatonic adherence: {features.pentatonic_adherence}")
        print(f"  Ornament density: {features.ornament_density}")
        print(f"  Rhythmic complexity: {features.rhythmic_complexity}")
    
    def test_empty_audio(self, analyzer):
        """Test with empty audio"""
        empty_audio = np.zeros(1000)
        sr = 22050
        
        features = analyzer.extract_chinese_features(
            empty_audio, sr, InstrumentType.ERHU
        )
        
        # Should handle empty audio gracefully
        assert features is not None
    
    def test_short_audio(self, analyzer):
        """Test with very short audio"""
        short_audio = np.random.randn(100)  # Very short
        sr = 22050
        
        features = analyzer.extract_chinese_features(
            short_audio, sr, InstrumentType.ERHU
        )
        
        # Should handle short audio gracefully
        assert features is not None
    
    def test_feature_consistency(self, analyzer, sample_audio):
        """Test that feature extraction is consistent"""
        audio_data, sr = sample_audio
        
        # Extract features twice
        features1 = analyzer.extract_chinese_features(
            audio_data, sr, InstrumentType.ERHU
        )
        features2 = analyzer.extract_chinese_features(
            audio_data, sr, InstrumentType.ERHU
        )
        
        # Results should be identical
        assert features1.pentatonic_adherence == features2.pentatonic_adherence
        assert features1.ornament_density == features2.ornament_density


class TestChineseInstrumentParameters:
    """Test instrument-specific parameters"""
    
    @pytest.fixture
    def analyzer(self):
        if not ENHANCED_FEATURES_AVAILABLE:
            pytest.skip("Enhanced Chinese features not available")
        return ChineseInstrumentAnalyzer()
    
    def test_instrument_parameters_exist(self, analyzer):
        """Test that all instruments have parameters"""
        expected_instruments = [
            InstrumentType.ERHU,
            InstrumentType.PIPA,
            InstrumentType.GUZHENG,
            InstrumentType.DIZI
        ]
        
        for instrument in expected_instruments:
            assert instrument in analyzer.instrument_params
            params = analyzer.instrument_params[instrument]
            assert 'fundamental_range' in params
    
    def test_parameter_ranges(self, analyzer):
        """Test parameter value ranges"""
        for instrument, params in analyzer.instrument_params.items():
            freq_range = params['fundamental_range']
            assert len(freq_range) == 2
            assert freq_range[0] < freq_range[1]  # Min < Max
            assert freq_range[0] > 50  # Reasonable minimum
            assert freq_range[1] < 5000  # Reasonable maximum


@pytest.mark.skipif(not ENHANCED_FEATURES_AVAILABLE, 
                   reason="Enhanced Chinese features not available")
class TestFeatureVisualization:
    """Test feature visualization components"""
    
    def test_visualization_import(self):
        """Test that visualization module can be imported"""
        try:
            from example.enhanced_chinese_visualization import EnhancedChineseInstrumentVisualizer
            visualizer = EnhancedChineseInstrumentVisualizer()
            assert visualizer is not None
        except ImportError as e:
            pytest.skip(f"Visualization module not available: {e}")
    
    @pytest.mark.skipif(not os.path.exists("example/erhu1.wav"), 
                       reason="Test audio file not available")
    def test_visualization_creation(self):
        """Test visualization creation"""
        try:
            from example.enhanced_chinese_visualization import EnhancedChineseInstrumentVisualizer
            
            visualizer = EnhancedChineseInstrumentVisualizer()
            
            # Test feature extraction (not full visualization to avoid GUI issues in tests)
            audio_data, sr = librosa.load("example/erhu1.wav", sr=22050)
            features, f0 = visualizer.extract_chinese_features(audio_data, sr)
            
            assert features is not None
            assert isinstance(features, dict)
            assert 'pentatonic_adherence' in features
            
        except ImportError as e:
            pytest.skip(f"Visualization module not available: {e}")


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])