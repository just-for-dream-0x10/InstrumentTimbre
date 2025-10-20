"""
测试音轨操作相关的数据结构
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
    """测试情感约束数据结构"""
    
    def test_valid_emotion_constraints(self):
        """测试有效的情感约束"""
        constraints = EmotionConstraints(
            primary_emotion=EmotionType.HAPPY,
            intensity=0.8,
            tempo_range=(120, 140)
        )
        
        assert constraints.primary_emotion == EmotionType.HAPPY
        assert constraints.intensity == 0.8
        assert constraints.tempo_range == (120, 140)
    
    def test_invalid_intensity(self):
        """测试无效的情感强度"""
        with pytest.raises(ValueError, match="情感强度必须在0-1之间"):
            EmotionConstraints(
                primary_emotion=EmotionType.HAPPY,
                intensity=1.5  # 无效值
            )
    
    def test_invalid_tempo_range(self):
        """测试无效的tempo范围"""
        with pytest.raises(ValueError, match="tempo范围无效"):
            EmotionConstraints(
                primary_emotion=EmotionType.HAPPY,
                intensity=0.8,
                tempo_range=(140, 120)  # 起始值大于结束值
            )


class TestMusicConstraints:
    """测试音乐约束数据结构"""
    
    def test_valid_music_constraints(self):
        """测试有效的音乐约束"""
        constraints = MusicConstraints(
            key="C_major",
            time_signature="4/4",
            tempo=120
        )
        
        assert constraints.key == "C_major"
        assert constraints.time_signature == "4/4"
        assert constraints.tempo == 120
    
    def test_invalid_tempo(self):
        """测试无效的tempo"""
        with pytest.raises(ValueError, match="tempo超出合理范围"):
            MusicConstraints(
                key="C_major",
                time_signature="4/4",
                tempo=300  # 超出范围
            )


class TestTrackData:
    """测试音轨数据结构"""
    
    def test_valid_track_data(self):
        """测试有效的音轨数据"""
        audio_data = np.random.random(22050)  # 1秒音频
        
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
        """测试无效的音轨数据"""
        track = TrackData(
            track_id="test_track",
            instrument="violin",
            role=TrackRole.MELODY,
            # 没有音频数据和MIDI数据
            duration=0.0
        )
        
        assert not track.is_valid()
    
    def test_track_with_midi_data(self):
        """测试包含MIDI数据的音轨"""
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
    """测试操作请求数据结构"""
    
    def test_valid_operation_request(self):
        """测试有效的操作请求"""
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
        """测试无效的操作强度"""
        request = OperationRequest(
            operation_type=OperationType.GENERATE,
            target_instrument="violin",
            target_role=TrackRole.HARMONY,
            intensity=1.5  # 无效值
        )
        
        assert not request.validate()
    
    def test_invalid_complexity_level(self):
        """测试无效的复杂度级别"""
        request = OperationRequest(
            operation_type=OperationType.GENERATE,
            target_instrument="violin",
            target_role=TrackRole.HARMONY,
            complexity_level="invalid"  # 无效值
        )
        
        assert not request.validate()


class TestOperationResult:
    """测试操作结果数据结构"""
    
    def test_successful_result(self):
        """测试成功的操作结果"""
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
        """测试有冲突的操作结果"""
        track = create_empty_track("test", "violin", TrackRole.HARMONY)
        
        conflict = ConflictReport(
            conflict_type=ConflictType.HARMONIC,
            severity=0.8,
            description="和声冲突",
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
        # 冲突会降低综合评分
        assert result.get_overall_score() < 0.9


class TestGenerationConfig:
    """测试生成配置数据结构"""
    
    def test_valid_generation_config(self):
        """测试有效的生成配置"""
        config = GenerationConfig(
            emotion_weight=0.4,
            harmony_weight=0.3,
            rhythm_weight=0.2,
            style_weight=0.1
        )
        
        assert config.validate()
    
    def test_invalid_weight_sum(self):
        """测试权重和不为1的配置"""
        config = GenerationConfig(
            emotion_weight=0.5,
            harmony_weight=0.3,
            rhythm_weight=0.2,
            style_weight=0.2  # 总和为1.2
        )
        
        assert not config.validate()


class TestUtilityFunctions:
    """测试工具函数"""
    
    def test_create_empty_track(self):
        """测试创建空白音轨"""
        track = create_empty_track("test_id", "piano", TrackRole.MELODY)
        
        assert track.track_id == "test_id"
        assert track.instrument == "piano"
        assert track.role == TrackRole.MELODY
        assert not track.is_valid()  # 没有音频数据
    
    def test_merge_constraints(self):
        """测试合并约束"""
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
        assert merged['tempo'] == 130  # 取较小值
        assert merged['key'] == "C_major"
        assert merged['intensity'] == 0.8


if __name__ == "__main__":
    pytest.main([__file__])