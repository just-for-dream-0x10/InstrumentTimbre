"""

"""

import pytest
import numpy as np
from InstrumentTimbre.core.operations.data_structures import (
    EmotionConstraints, MusicConstraints, TrackData, ConflictReport,
    OperationRequest, OperationResult, GenerationConfig,
    EmotionType, TrackRole, OperationType, ConflictType,
    create_empty_track, merge_constraints
)


class TestEmotionConstraints:
    """"""
    
    def test_valid_emotion_constraints(self):
        """"""
        constraints = EmotionConstraints(
            primary_emotion=EmotionType.HAPPY,
            intensity=0.8,
            tempo_range=(120, 140)
        )
        
        assert constraints.primary_emotion == EmotionType.HAPPY
        assert constraints.intensity == 0.8
        assert constraints.tempo_range == (120, 140)
    
    def test_invalid_intensity(self):
        """Emotion intensity"""
        with pytest.raises(ValueError, match="0-1"):
            EmotionConstraints(
                primary_emotion=EmotionType.HAPPY,
                intensity=1.5  # 
            )
    
    def test_invalid_tempo_range(self):
        """tempo"""
        with pytest.raises(ValueError, match="tempo"):
            EmotionConstraints(
                primary_emotion=EmotionType.HAPPY,
                intensity=0.8,
                tempo_range=(140, 120)  # 
            )


class TestMusicConstraints:
    """"""
    
    def test_valid_music_constraints(self):
        """"""
        constraints = MusicConstraints(
            key="C_major",
            time_signature="4/4",
            tempo=120
        )
        
        assert constraints.key == "C_major"
        assert constraints.time_signature == "4/4"
        assert constraints.tempo == 120
    
    def test_invalid_tempo(self):
        """tempo"""
        with pytest.raises(ValueError, match="tempo"):
            MusicConstraints(
                key="C_major",
                time_signature="4/4",
                tempo=300  # 
            )


class TestTrackData:
    """"""
    
    def test_valid_track_data(self):
        """"""
        audio_data = np.random.random(22050)  # 1
        
        track = TrackData(
            track_id="test_track",
            instrument="violin",
            role=TrackRole.MELODY,
            audio_data=audio_data,
            duration=1.0
        )
        
        assert track.is_valid()
        assert track.instrument == "violin"
        assert track.role == TrackRole.MELODY
    
    def test_invalid_track_data(self):
        """"""
        track = TrackData(
            track_id="test_track",
            instrument="violin",
            role=TrackRole.MELODY,
            # MIDI
            duration=0.0
        )
        
        assert not track.is_valid()
    
    def test_track_with_midi_data(self):
        """MIDI"""
        midi_data = {"notes": [60, 64, 67], "durations": [0.5, 0.5, 1.0]}
        
        track = TrackData(
            track_id="test_track",
            instrument="piano",
            role=TrackRole.HARMONY,
            midi_data=midi_data,
            duration=2.0
        )
        
        assert track.is_valid()
        assert track.midi_data == midi_data


class TestOperationRequest:
    """Operation"""
    
    def test_valid_operation_request(self):
        """Operation"""
        request = OperationRequest(
            operation_type=OperationType.GENERATE,
            target_instrument="violin",
            target_role=TrackRole.HARMONY,
            intensity=0.7
        )
        
        assert request.validate()
        assert request.operation_type == OperationType.GENERATE
        assert request.target_instrument == "violin"
        assert request.target_role == TrackRole.HARMONY
    
    def test_invalid_intensity(self):
        """Operation"""
        request = OperationRequest(
            operation_type=OperationType.GENERATE,
            target_instrument="violin",
            target_role=TrackRole.HARMONY,
            intensity=1.5  # 
        )
        
        assert not request.validate()
    
    def test_invalid_complexity_level(self):
        """"""
        request = OperationRequest(
            operation_type=OperationType.GENERATE,
            target_instrument="violin",
            target_role=TrackRole.HARMONY,
            complexity_level="invalid"  # 
        )
        
        assert not request.validate()


class TestOperationResult:
    """Operation"""
    
    def test_successful_result(self):
        """successOperation"""
        track = create_empty_track("test", "violin", TrackRole.HARMONY)
        track.duration = 1.0
        track.audio_data = np.random.random(22050)
        
        result = OperationResult(
            success=True,
            generated_track=track,
            quality_score=0.9,
            emotion_consistency=0.8,
            harmonic_correctness=0.95
        )
        
        assert result.success
        assert result.get_overall_score() > 0.8
        assert not result.has_critical_conflicts()
    
    def test_result_with_conflicts(self):
        """Operation"""
        track = create_empty_track("test", "violin", TrackRole.HARMONY)
        
        conflict = ConflictReport(
            conflict_type=ConflictType.HARMONIC,
            severity=0.8,
            description="",
            location=(0.0, 1.0),
            affected_tracks=["track1", "track2"]
        )
        
        result = OperationResult(
            success=True,
            generated_track=track,
            quality_score=0.9,
            conflicts=[conflict]
        )
        
        assert result.has_critical_conflicts()
        # 
        assert result.get_overall_score() < 0.9


class TestGenerationConfig:
    """generate"""
    
    def test_valid_generation_config(self):
        """generate"""
        config = GenerationConfig(
            emotion_weight=0.4,
            harmony_weight=0.3,
            rhythm_weight=0.2,
            style_weight=0.1
        )
        
        assert config.validate()
    
    def test_invalid_weight_sum(self):
        """1"""
        config = GenerationConfig(
            emotion_weight=0.5,
            harmony_weight=0.3,
            rhythm_weight=0.2,
            style_weight=0.2  # 1.2
        )
        
        assert not config.validate()


class TestUtilityFunctions:
    """"""
    
    def test_create_empty_track(self):
        """create"""
        track = create_empty_track("test_id", "piano", TrackRole.MELODY)
        
        assert track.track_id == "test_id"
        assert track.instrument == "piano"
        assert track.role == TrackRole.MELODY
        assert not track.is_valid()  # 
    
    def test_merge_constraints(self):
        """"""
        emotion_constraints = EmotionConstraints(
            primary_emotion=EmotionType.HAPPY,
            intensity=0.8,
            tempo_range=(120, 140)
        )
        
        music_constraints = MusicConstraints(
            key="C_major",
            time_signature="4/4",
            tempo=130
        )
        
        merged = merge_constraints(emotion_constraints, music_constraints)
        
        assert merged['emotion'] == emotion_constraints
        assert merged['music'] == music_constraints
        assert merged['tempo'] == 130  # 
        assert merged['key'] == "C_major"
        assert merged['intensity'] == 0.8


if __name__ == "__main__":
    pytest.main([__file__])