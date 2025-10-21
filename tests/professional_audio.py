"""
Test cases for System Professional Audio Processing System.

This module tests the intelligent mixing engine, dynamic range optimizer,
spatial positioning, intelligent EQ balancer, effects processor, and audio quality enhancer.
"""

import pytest
import numpy as np
from typing import Dict, Any

from InstrumentTimbre.core.professional_audio import (
    ProfessionalAudioEngine,
    ProcessingConfig,
    AudioTrackInfo,
    IntelligentMixingEngine,
    DynamicRangeOptimizer,
    SpatialPositioningAlgorithm,
    IntelligentEQBalancer,
    EffectsProcessor,
    AudioQualityEnhancer
)


class TestProfessionalAudioEngine:
    """Test the main professional audio engine orchestrator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = ProcessingConfig(
            sample_rate=22050,
            target_lufs=-16.0,
            max_peak_level=-1.0
        )
        self.engine = ProfessionalAudioEngine(self.config)
        
        # Create test audio tracks
        self.test_tracks = self._create_test_tracks()
        self.track_info = self._create_track_info()
    
    def _create_test_tracks(self) -> Dict[str, np.ndarray]:
        """Create synthetic test audio tracks."""
        duration = 2.0  # 2 seconds
        sample_rate = 22050
        num_samples = int(duration * sample_rate)
        
        tracks = {}
        
        # Erhu track - melodic with vibrato
        t = np.linspace(0, duration, num_samples)
        erhu_freq = 440.0  # A4
        vibrato = 1.0 + 0.05 * np.sin(2 * np.pi * 5.0 * t)  # 5Hz vibrato
        erhu_signal = 0.5 * np.sin(2 * np.pi * erhu_freq * vibrato * t)
        erhu_signal *= np.exp(-t * 0.3)  # Natural decay
        tracks["erhu_01"] = erhu_signal.astype(np.float32)
        
        # Guzheng track - plucked string with harmonics
        guzheng_freq = 293.66  # D4
        guzheng_signal = (
            0.6 * np.sin(2 * np.pi * guzheng_freq * t) +
            0.3 * np.sin(2 * np.pi * guzheng_freq * 2 * t) +
            0.1 * np.sin(2 * np.pi * guzheng_freq * 3 * t)
        )
        guzheng_signal *= np.exp(-t * 1.5)  # Plucked decay
        tracks["guzheng_01"] = guzheng_signal.astype(np.float32)
        
        # Dizi track - flute-like with breath noise
        dizi_freq = 587.33  # D5
        breath_noise = 0.02 * np.random.normal(0, 1, num_samples)
        dizi_signal = 0.4 * np.sin(2 * np.pi * dizi_freq * t) + breath_noise
        dizi_signal *= (np.sin(np.pi * t / duration) ** 2)  # Smooth envelope
        tracks["dizi_01"] = dizi_signal.astype(np.float32)
        
        return tracks
    
    def _create_track_info(self) -> Dict[str, AudioTrackInfo]:
        """Create track information metadata."""
        return {
            "erhu_01": AudioTrackInfo(
                track_id="erhu_01",
                instrument_type="erhu",
                role="lead",
                importance_weight=0.9,
                original_level=0.5,
                frequency_range=(196, 2637),
                dynamic_range=8.0,
                emotional_content={"energy": 0.7, "valence": 0.6}
            ),
            "guzheng_01": AudioTrackInfo(
                track_id="guzheng_01",
                instrument_type="guzheng",
                role="harmony",
                importance_weight=0.7,
                original_level=0.6,
                frequency_range=(75, 4186),
                dynamic_range=12.0,
                emotional_content={"energy": 0.5, "valence": 0.8}
            ),
            "dizi_01": AudioTrackInfo(
                track_id="dizi_01",
                instrument_type="dizi",
                role="melody",
                importance_weight=0.6,
                original_level=0.4,
                frequency_range=(294, 2349),
                dynamic_range=6.0,
                emotional_content={"energy": 0.4, "valence": 0.7}
            )
        }
    
    def test_engine_initialization(self):
        """Test proper initialization of the audio engine."""
        assert self.engine.config.sample_rate == 22050
        assert self.engine.config.target_lufs == -16.0
        assert not self.engine.is_processing
        assert self.engine.current_session_id is None
        
        # Check that all processors are initialized
        assert hasattr(self.engine, 'mixing_engine')
        assert hasattr(self.engine, 'dynamic_optimizer')
        assert hasattr(self.engine, 'spatial_processor')
        assert hasattr(self.engine, 'eq_balancer')
        assert hasattr(self.engine, 'effects_processor')
        assert hasattr(self.engine, 'quality_enhancer')
    
    def test_multitrack_processing_pipeline(self):
        """Test the complete multitrack processing pipeline."""
        musical_analysis = {
            "tempo": 120.0,
            "key": "D_major",
            "emotional_analysis": {"energy": 0.6, "valence": 0.7}
        }
        
        processed_audio, metadata = self.engine.process_multitrack_audio(
            self.test_tracks, self.track_info, musical_analysis
        )
        
        # Verify output
        assert isinstance(processed_audio, np.ndarray)
        assert processed_audio.ndim == 2  # Should be stereo
        assert processed_audio.shape[1] == 2  # Left and right channels
        
        # Verify metadata structure
        assert "processing_chain" in metadata
        assert "config" in metadata
        assert "stage_metadata" in metadata
        
        expected_stages = [
            "validation", "mixing", "eq_balancing", 
            "spatial_positioning", "effects", "dynamics", "quality_enhancement"
        ]
        assert metadata["processing_chain"] == expected_stages
    
    def test_processing_status_tracking(self):
        """Test processing status tracking functionality."""
        initial_status = self.engine.get_processing_status()
        assert not initial_status["is_processing"]
        
        # Processing status should be updated during processing
        # (This is tested implicitly in the pipeline test)
    
    def test_configuration_update(self):
        """Test configuration update functionality."""
        new_config = ProcessingConfig(
            sample_rate=44100,
            target_lufs=-14.0,
            max_peak_level=-0.5
        )
        
        self.engine.update_configuration(new_config)
        assert self.engine.config.sample_rate == 44100
        assert self.engine.config.target_lufs == -14.0


class TestIntelligentMixingEngine:
    """Test the intelligent mixing engine."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = ProcessingConfig()
        self.mixing_engine = IntelligentMixingEngine(self.config)
    
    def test_instrument_profile_initialization(self):
        """Test that instrument profiles are properly initialized."""
        # Check Chinese instruments
        assert "erhu" in self.mixing_engine.instrument_profiles
        assert "guzheng" in self.mixing_engine.instrument_profiles
        assert "pipa" in self.mixing_engine.instrument_profiles
        assert "dizi" in self.mixing_engine.instrument_profiles
        assert "guqin" in self.mixing_engine.instrument_profiles
        
        # Check Western instruments
        assert "piano" in self.mixing_engine.instrument_profiles
        assert "guitar" in self.mixing_engine.instrument_profiles
        assert "violin" in self.mixing_engine.instrument_profiles
        
        # Verify profile structure
        erhu_profile = self.mixing_engine.instrument_profiles["erhu"]
        assert hasattr(erhu_profile, 'default_level')
        assert hasattr(erhu_profile, 'frequency_emphasis')
        assert hasattr(erhu_profile, 'priority_weight')
    
    def test_mixing_strategy_determination(self):
        """Test mixing strategy determination based on instruments."""
        # Test Chinese traditional strategy
        chinese_track_info = {
            "track1": {"instrument_type": "erhu"},
            "track2": {"instrument_type": "guzheng"},
            "track3": {"instrument_type": "pipa"}
        }
        
        strategy = self.mixing_engine._determine_mixing_strategy(
            chinese_track_info, None
        )
        assert strategy.value == "chinese_traditional"


class TestDynamicRangeOptimizer:
    """Test the dynamic range optimizer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = ProcessingConfig()
        self.optimizer = DynamicRangeOptimizer(self.config)
    
    def test_compression_profile_initialization(self):
        """Test compression profile initialization."""
        # Check Chinese instruments have proper compression profiles
        assert "erhu" in self.optimizer.compression_profiles
        assert "guzheng" in self.optimizer.compression_profiles
        
        # Verify profile structure
        erhu_profile = self.optimizer.compression_profiles["erhu"]
        assert hasattr(erhu_profile, 'threshold')
        assert hasattr(erhu_profile, 'ratio')
        assert hasattr(erhu_profile, 'attack')
        assert hasattr(erhu_profile, 'release')
        
        # Check that Chinese instruments have appropriate settings
        assert erhu_profile.compression_type.value == "tube"
        assert erhu_profile.ratio == 2.2  # Gentle compression for erhu
    
    def test_dynamic_analysis(self):
        """Test dynamic characteristic analysis."""
        # Create test audio with known characteristics
        sample_rate = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # High dynamic range signal
        dynamic_signal = np.sin(2 * np.pi * 440 * t) * (0.1 + 0.8 * np.sin(2 * np.pi * 0.5 * t))
        
        tracks = {"test_track": dynamic_signal.astype(np.float32)}
        track_info = {"test_track": {"instrument_type": "erhu"}}
        
        analysis = self.optimizer._analyze_dynamic_characteristics(tracks, track_info)
        
        assert "test_track" in analysis
        characteristics = analysis["test_track"]
        assert "rms_level" in characteristics
        assert "peak_level" in characteristics
        assert "crest_factor" in characteristics
        assert "dynamic_range" in characteristics
        assert "needs_compression" in characteristics


class TestSpatialPositioningAlgorithm:
    """Test the spatial positioning algorithm."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = ProcessingConfig()
        self.spatial_processor = SpatialPositioningAlgorithm(self.config)
    
    def test_spatial_profile_initialization(self):
        """Test spatial profile initialization."""
        # Check Chinese instruments have appropriate spatial profiles
        assert "erhu" in self.spatial_processor.spatial_profiles
        assert "guzheng" in self.spatial_processor.spatial_profiles
        
        # Verify guzheng has wide stereo positioning (typical for this instrument)
        guzheng_profile = self.spatial_processor.spatial_profiles["guzheng"]
        assert guzheng_profile.width_preference >= 0.8  # Wide positioning
        assert guzheng_profile.default_position.azimuth == 0  # Center position
    
    def test_positioning_strategy_determination(self):
        """Test positioning strategy determination."""
        chinese_track_info = {
            "erhu": {"instrument_type": "erhu"},
            "guzheng": {"instrument_type": "guzheng"},
            "dizi": {"instrument_type": "dizi"}
        }
        
        strategy = self.spatial_processor._determine_positioning_strategy(
            chinese_track_info, None
        )
        assert strategy.value == "chinese_traditional"


class TestIntelligentEQBalancer:
    """Test the intelligent EQ balancer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = ProcessingConfig()
        self.eq_balancer = IntelligentEQBalancer(self.config)
    
    def test_eq_profile_initialization(self):
        """Test EQ profile initialization."""
        # Check Chinese instruments have appropriate EQ profiles
        assert "erhu" in self.eq_balancer.eq_profiles
        assert "dizi" in self.eq_balancer.eq_profiles
        
        # Verify erhu EQ profile characteristics
        erhu_profile = self.eq_balancer.eq_profiles["erhu"]
        assert erhu_profile.frequency_range == (196, 2637)  # Erhu frequency range
        assert len(erhu_profile.boost_frequencies) > 0
        assert len(erhu_profile.cut_frequencies) > 0
    
    def test_frequency_conflict_detection(self):
        """Test frequency conflict detection between instruments."""
        # Create mock frequency analysis with overlapping content
        frequency_analysis = {
            "track1": {
                "mid": 0.6,  # High energy in mid frequencies
                "high_mid": 0.3
            },
            "track2": {
                "mid": 0.5,  # Overlapping mid frequency content
                "high_mid": 0.4
            }
        }
        
        track_info = {
            "track1": {"instrument_type": "erhu"},
            "track2": {"instrument_type": "violin"}
        }
        
        conflicts = self.eq_balancer._detect_frequency_conflicts(
            frequency_analysis, track_info
        )
        
        # Should detect conflict in mid frequencies
        assert len(conflicts) > 0
        conflict_bands = [conflict[2] for conflict in conflicts]
        assert "mid" in conflict_bands


class TestEffectsProcessor:
    """Test the effects processor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = ProcessingConfig()
        self.effects_processor = EffectsProcessor(self.config)
    
    def test_effect_profile_initialization(self):
        """Test effect profile initialization."""
        # Check Chinese instruments have appropriate effect profiles
        assert "erhu" in self.effects_processor.effect_profiles
        assert "guzheng" in self.effects_processor.effect_profiles
        
        # Verify guzheng has hall reverb (typical for this instrument)
        guzheng_profile = self.effects_processor.effect_profiles["guzheng"]
        assert guzheng_profile.reverb_settings is not None
        assert guzheng_profile.reverb_settings.reverb_type.value == "hall"
        assert guzheng_profile.reverb_settings.room_size >= 0.8  # Large hall
    
    def test_musical_context_analysis(self):
        """Test musical context analysis for effect decisions."""
        musical_analysis = {
            "tempo": 90.0,
            "emotional_analysis": {"energy": 0.3, "valence": 0.4}  # Slow, sad music
        }
        
        track_info = {
            "erhu": {"instrument_type": "erhu"}
        }
        
        context = self.effects_processor._analyze_effect_context(musical_analysis, track_info)
        
        # Should increase reverb for sad, low-energy music
        assert context["reverb_amount_modifier"] > 1.0
        assert context["musical_style"] == "chinese_traditional"


class TestAudioQualityEnhancer:
    """Test the audio quality enhancer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = ProcessingConfig()
        self.quality_enhancer = AudioQualityEnhancer(self.config)
    
    def test_quality_enhancement_pipeline(self):
        """Test the complete quality enhancement pipeline."""
        # Create test audio
        sample_rate = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        test_audio = 0.3 * np.sin(2 * np.pi * 440 * t)  # Quiet sine wave
        
        enhanced_audio, metadata = self.quality_enhancer.enhance_audio_quality(
            test_audio, self.config
        )
        
        # Verify output
        assert enhanced_audio.ndim == 2  # Should be stereo
        assert enhanced_audio.shape[1] == 2
        
        # Verify enhancement metadata
        assert "original_rms_db" in metadata
        assert "enhanced_rms_db" in metadata
        assert "enhancement_settings" in metadata
        assert "quality_improvements" in metadata
        
        # Should have improved loudness
        loudness_improvement = metadata["quality_improvements"]["loudness_adjustment_db"]
        assert abs(loudness_improvement) > 0  # Some adjustment should be made
    
    def test_enhancement_statistics(self):
        """Test enhancement statistics tracking."""
        # Process some audio to generate history
        test_audio = 0.1 * np.random.normal(0, 1, 1000)
        self.quality_enhancer.enhance_audio_quality(test_audio, self.config)
        
        stats = self.quality_enhancer.get_enhancement_statistics()
        assert stats["total_processed"] == 1
        assert "average_loudness_improvement" in stats
        assert "last_enhancement" in stats


# Integration test
def test_System_integration():
    """Integration test for the complete System professional audio system."""
    config = ProcessingConfig(sample_rate=22050, target_lufs=-16.0)
    engine = ProfessionalAudioEngine(config)
    
    # Create a simple Chinese traditional music scenario
    duration = 1.0
    sample_rate = 22050
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    tracks = {
        "erhu": 0.4 * np.sin(2 * np.pi * 440 * t),  # Erhu melody
        "guzheng": 0.3 * np.sin(2 * np.pi * 293.66 * t)  # Guzheng harmony
    }
    
    track_info = {
        "erhu": AudioTrackInfo(
            track_id="erhu",
            instrument_type="erhu",
            role="lead",
            importance_weight=0.9,
            original_level=0.4,
            frequency_range=(196, 2637),
            dynamic_range=8.0,
            emotional_content={"energy": 0.6, "valence": 0.7}
        ),
        "guzheng": AudioTrackInfo(
            track_id="guzheng",
            instrument_type="guzheng",
            role="harmony",
            importance_weight=0.7,
            original_level=0.3,
            frequency_range=(75, 4186),
            dynamic_range=10.0,
            emotional_content={"energy": 0.5, "valence": 0.8}
        )
    }
    
    musical_analysis = {
        "tempo": 100.0,
        "key": "D_major",
        "emotional_analysis": {"energy": 0.6, "valence": 0.7},
        "musical_structure": {
            "sections": ["intro", "verse", "chorus"],
            "current_section": "verse"
        }
    }
    
    # Process the complete pipeline
    final_audio, metadata = engine.process_multitrack_audio(
        tracks, track_info, musical_analysis
    )
    
    # Verify the complete processing worked
    assert final_audio is not None
    assert final_audio.ndim == 2  # Stereo output
    assert len(metadata["processing_chain"]) == 7  # All processing stages
    
    # Verify Chinese traditional style was detected and applied
    stage_metadata = metadata["stage_metadata"]
    if "mixing" in stage_metadata:
        assert stage_metadata["mixing"]["mixing_strategy"] == "chinese_traditional"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])