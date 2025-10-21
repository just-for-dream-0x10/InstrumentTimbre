"""
Test the simplified configuration system
"""

import pytest
import numpy as np

from InstrumentTimbre.config import Config, TrackInfo, fast_config, traditional_config
from InstrumentTimbre.core.professional_audio import ProfessionalAudioEngine


def test_simple_config_creation():
    """Test creating simple configurations"""
    # Default config
    config = Config()
    assert config.sample_rate == 22050
    assert config.quality.value == "normal"
    assert config.style.value == "auto"
    
    # Fast config
    fast = fast_config()
    assert fast.quality.value == "fast"
    assert not fast.enable_reverb
    
    # Traditional config
    trad = traditional_config()
    assert trad.style.value == "traditional"
    assert trad.reverb == 0.4


def test_track_info():
    """Test simple track info"""
    track = TrackInfo(
        instrument="erhu",
        role="lead",
        importance=0.9
    )
    assert track.instrument == "erhu"
    assert track.role == "lead"
    assert track.importance == 0.9


def test_professional_audio_engine_with_simple_config():
    """Test professional audio engine with simple config"""
    config = traditional_config()
    engine = ProfessionalAudioEngine(simple_config=config)
    
    # Verify engine was initialized with simple config
    assert engine.simple_config is not None
    assert engine.simple_config.style.value == "traditional"


def test_simple_processing():
    """Test simple processing interface"""
    # Create test data
    sample_rate = 22050
    duration = 0.5
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    tracks = {
        "erhu": 0.3 * np.sin(2 * np.pi * 440 * t),
        "guzheng": 0.2 * np.sin(2 * np.pi * 293.66 * t)
    }
    
    track_info = {
        "erhu": TrackInfo(instrument="erhu", role="lead", importance=0.9),
        "guzheng": TrackInfo(instrument="guzheng", role="harmony", importance=0.7)
    }
    
    # Test with simple config
    config = traditional_config()
    engine = ProfessionalAudioEngine(simple_config=config)
    
    # Process using simple interface
    result, metadata = engine.process_simple(tracks, track_info)
    
    # Verify results
    assert result is not None
    assert result.ndim == 2  # Stereo output
    assert isinstance(metadata, dict)


def test_config_validation():
    """Test configuration validation"""
    from InstrumentTimbre.config import validate_config
    
    # Valid config
    valid_config = Config()
    assert validate_config(valid_config) == True
    
    # Invalid config (reverb out of range)
    invalid_config = Config(reverb=2.0)  # Should be 0.0-1.0
    assert validate_config(invalid_config) == False


def test_auto_recommendation():
    """Test automatic config recommendation"""
    from InstrumentTimbre.config import recommend_config
    
    # Traditional instruments
    traditional_instruments = ["erhu", "guzheng"]
    config = recommend_config(traditional_instruments)
    assert config.style.value == "traditional"
    
    # Many instruments (should use fast)
    many_instruments = ["piano", "guitar", "bass", "drums", "violin"]
    config = recommend_config(many_instruments)
    assert config.quality.value == "fast"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])