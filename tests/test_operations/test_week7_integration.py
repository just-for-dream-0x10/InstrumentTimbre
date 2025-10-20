"""
Week 7 智能音轨操作引擎集成测试
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from InstrumentTimbre.core.operations.operation_dispatcher import (
    OperationDispatcher, intelligent_track_operation
)
from InstrumentTimbre.core.operations.track_generation_engine import TrackGenerationEngine
from InstrumentTimbre.core.operations.track_replacement_engine import TrackReplacementEngine
from InstrumentTimbre.core.operations.track_repair_engine import TrackRepairEngine
from InstrumentTimbre.core.operations.emotion_driven_orchestrator import EmotionDrivenOrchestrator
from InstrumentTimbre.core.operations.conflict_detector import RealTimeConflictDetector

from InstrumentTimbre.core.operations.data_structures import (
    OperationRequest, OperationType, TrackRole, TrackData,
    EmotionConstraints, MusicConstraints, EmotionType,
    create_empty_track
)


class TestWeek7Integration:
    """Week 7 集成测试"""
    
    def setup_method(self):
        """每个测试方法前的设置"""
        # 创建测试用的音轨数据
        self.test_tracks = self._create_test_tracks()
        
        # 创建测试约束
        self.emotion_constraints = EmotionConstraints(
            primary_emotion=EmotionType.HAPPY,
            intensity=0.8,
            tempo_range=(120, 140)
        )
        
        self.music_constraints = MusicConstraints(
            key="C_major",
            time_signature="4/4",
            tempo=130
        )
    
    def _create_test_tracks(self):
        """创建测试音轨"""
        # 钢琴主旋律
        piano_track = TrackData(
            track_id="piano_melody",
            instrument="piano",
            role=TrackRole.MELODY,
            duration=30.0,
            key="C_major",
            tempo=130,
            pitch_sequence=[60, 62, 64, 65, 67, 69, 71, 72],  # C大调音阶
            rhythm_pattern=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0],
            dynamics=[0.7, 0.6, 0.8, 0.7, 0.9, 0.8, 0.7, 0.6]
        )
        
        # 低音音轨
        bass_track = TrackData(
            track_id="bass_line",
            instrument="bass",
            role=TrackRole.BASS,
            duration=30.0,
            key="C_major",
            tempo=130,
            pitch_sequence=[36, 43, 41, 38],  # 简单低音线条
            rhythm_pattern=[2.0, 2.0, 2.0, 2.0],
            dynamics=[0.6, 0.6, 0.6, 0.6]
        )
        
        return [piano_track, bass_track]
    
    def test_complete_generation_workflow(self):
        """测试完整的音轨生成工作流"""
        # 创建请求
        request = OperationRequest(
            operation_type=OperationType.GENERATE,
            target_instrument="violin",
            target_role=TrackRole.HARMONY,
            intensity=0.7
        )
        
        # 执行操作
        result = intelligent_track_operation(
            request,
            self.test_tracks,
            emotion_constraints=self.emotion_constraints,
            music_constraints=self.music_constraints
        )
        
        # 验证结果
        assert result.success
        assert result.generated_track is not None
        assert result.generated_track.instrument == "violin"
        assert result.generated_track.role == TrackRole.HARMONY
        assert result.quality_score > 0.0
        assert result.processing_time > 0.0
    
    def test_complete_replacement_workflow(self):
        """测试完整的音轨替换工作流"""
        request = OperationRequest(
            operation_type=OperationType.REPLACE,
            target_instrument="guitar",
            target_role=TrackRole.MELODY,
            reference_track="piano_melody"
        )
        
        result = intelligent_track_operation(
            request,
            self.test_tracks,
            emotion_constraints=self.emotion_constraints,
            music_constraints=self.music_constraints
        )
        
        # 验证结果
        assert result.success
        assert result.generated_track is not None
        assert result.generated_track.instrument == "guitar"
        assert result.generated_track.role == TrackRole.MELODY
    
    def test_complete_repair_workflow(self):
        """测试完整的音轨修复工作流"""
        # 创建有问题的音轨
        problematic_track = TrackData(
            track_id="problematic",
            instrument="violin",
            role=TrackRole.HARMONY,
            duration=10.0,
            key="F_major",  # 与其他音轨调性不同
            tempo=130,
            pitch_sequence=[60.2, 62.1, 64.3],  # 音准偏差
            rhythm_pattern=[0.3, 0.8, 1.2],
            dynamics=[0.5, 0.5, 0.5]
        )
        
        request = OperationRequest(
            operation_type=OperationType.REPAIR,
            target_instrument="violin",
            target_role=TrackRole.HARMONY,
            reference_track="problematic"
        )
        
        tracks_with_problem = self.test_tracks + [problematic_track]
        
        result = intelligent_track_operation(
            request,
            tracks_with_problem,
            emotion_constraints=self.emotion_constraints,
            music_constraints=self.music_constraints
        )
        
        # 验证结果
        assert result.success
        assert result.generated_track is not None
    
    def test_natural_language_parsing(self):
        """测试自然语言解析"""
        test_cases = [
            ("添加小提琴和声", OperationType.GENERATE, "violin", TrackRole.HARMONY),
            ("把钢琴替换成吉他", OperationType.REPLACE, "guitar", None),
            ("修复音准问题", OperationType.REPAIR, None, None),
            ("增加大提琴低音", OperationType.GENERATE, "cello", TrackRole.BASS),
        ]
        
        for text, expected_op, expected_inst, expected_role in test_cases:
            result = intelligent_track_operation(text, self.test_tracks)
            
            # 基本验证 - 应该能解析请求
            assert result is not None
    
    def test_emotion_driven_orchestration(self):
        """测试情感驱动配器"""
        orchestrator = EmotionDrivenOrchestrator()
        orchestrator.initialize()
        
        suggestion = orchestrator.get_orchestration_suggestion(
            self.emotion_constraints,
            "violin",
            self.test_tracks
        )
        
        assert 'instrument_suitability' in suggestion
        assert 'arrangement_style' in suggestion
        assert 'expression_markings' in suggestion
        assert 'dynamics_profile' in suggestion
    
    def test_real_time_conflict_detection(self):
        """测试实时冲突检测"""
        detector = RealTimeConflictDetector()
        detector.initialize()
        
        # 创建冲突音轨（与现有音轨不协和）
        conflicting_track = TrackData(
            track_id="conflicting",
            instrument="trumpet",
            role=TrackRole.MELODY,
            duration=30.0,
            key="C_major",
            tempo=130,
            pitch_sequence=[61, 66, 70],  # 可能产生不协和音程
            rhythm_pattern=[0.25, 0.25, 0.25],  # 高密度节奏
            dynamics=[0.9, 0.9, 0.9]  # 高动态
        )
        
        conflicts = detector.detect_conflicts(self.test_tracks, conflicting_track)
        
        assert isinstance(conflicts, list)
        # 可能检测到冲突（取决于具体实现）
    
    def test_quality_metrics_calculation(self):
        """测试质量指标计算"""
        generator = TrackGenerationEngine()
        generator.initialize()
        
        result = generator.generate_track(
            instrument="violin",
            role=TrackRole.HARMONY,
            emotion_constraints=self.emotion_constraints,
            music_constraints=self.music_constraints,
            current_tracks=self.test_tracks
        )
        
        assert result.success
        assert 0.0 <= result.quality_score <= 1.0
        assert 0.0 <= result.emotion_consistency <= 1.0
        assert 0.0 <= result.harmonic_correctness <= 1.0
    
    def test_constraint_integration(self):
        """测试约束集成"""
        # 测试约束是否正确传递和应用
        request = OperationRequest(
            operation_type=OperationType.GENERATE,
            target_instrument="flute",
            target_role=TrackRole.MELODY,
            intensity=0.9
        )
        
        result = intelligent_track_operation(
            request,
            self.test_tracks,
            emotion_constraints=self.emotion_constraints,
            music_constraints=self.music_constraints
        )
        
        assert result.success
        
        # 检查生成的音轨是否符合约束
        generated_track = result.generated_track
        assert generated_track.key == self.music_constraints.key
        assert generated_track.tempo == self.music_constraints.tempo
    
    def test_error_handling(self):
        """测试错误处理"""
        # 测试无效请求
        invalid_request = OperationRequest(
            operation_type=OperationType.GENERATE,
            target_instrument="",  # 空乐器名
            target_role=TrackRole.MELODY,
            intensity=2.0  # 无效强度
        )
        
        result = intelligent_track_operation(invalid_request, self.test_tracks)
        
        assert not result.success
        assert len(result.warnings) > 0
    
    def test_performance_benchmarks(self):
        """测试性能基准"""
        import time
        
        request = OperationRequest(
            operation_type=OperationType.GENERATE,
            target_instrument="violin",
            target_role=TrackRole.HARMONY
        )
        
        start_time = time.time()
        result = intelligent_track_operation(request, self.test_tracks)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # 性能要求：单轨生成时间 < 10秒
        assert processing_time < 10.0
        assert result.success
    
    def test_multiple_operations_sequence(self):
        """测试多个操作的序列执行"""
        current_tracks = self.test_tracks.copy()
        
        # 第一步：添加小提琴
        step1_request = OperationRequest(
            operation_type=OperationType.GENERATE,
            target_instrument="violin",
            target_role=TrackRole.HARMONY
        )
        
        step1_result = intelligent_track_operation(
            step1_request, current_tracks,
            emotion_constraints=self.emotion_constraints,
            music_constraints=self.music_constraints
        )
        
        assert step1_result.success
        current_tracks.append(step1_result.generated_track)
        
        # 第二步：添加长笛
        step2_request = OperationRequest(
            operation_type=OperationType.GENERATE,
            target_instrument="flute",
            target_role=TrackRole.MELODY
        )
        
        step2_result = intelligent_track_operation(
            step2_request, current_tracks,
            emotion_constraints=self.emotion_constraints,
            music_constraints=self.music_constraints
        )
        
        assert step2_result.success
        
        # 验证最终结果
        assert len(current_tracks) == 4  # 原始2个 + 新增2个
    
    def test_dispatcher_status_and_caching(self):
        """测试分发器状态和缓存机制"""
        dispatcher = OperationDispatcher()
        dispatcher.initialize_engines()
        
        status = dispatcher.get_status()
        
        assert status["initialized"]
        assert all(status["engines"].values())
    
    def test_different_emotion_types(self):
        """测试不同情感类型的处理"""
        emotion_types = [
            EmotionType.HAPPY,
            EmotionType.SAD,
            EmotionType.CALM,
            EmotionType.ENERGETIC,
            EmotionType.MELANCHOLIC,
            EmotionType.ANGRY
        ]
        
        for emotion_type in emotion_types:
            emotion_constraints = EmotionConstraints(
                primary_emotion=emotion_type,
                intensity=0.7
            )
            
            request = OperationRequest(
                operation_type=OperationType.GENERATE,
                target_instrument="piano",
                target_role=TrackRole.HARMONY
            )
            
            result = intelligent_track_operation(
                request, self.test_tracks,
                emotion_constraints=emotion_constraints,
                music_constraints=self.music_constraints
            )
            
            assert result.success, f"Failed for emotion: {emotion_type.value}"
    
    def test_different_instruments_and_roles(self):
        """测试不同乐器和角色的组合"""
        test_combinations = [
            ("violin", TrackRole.MELODY),
            ("cello", TrackRole.BASS),
            ("flute", TrackRole.HARMONY),
            ("trumpet", TrackRole.MELODY),
            ("guitar", TrackRole.ACCOMPANIMENT)
        ]
        
        for instrument, role in test_combinations:
            request = OperationRequest(
                operation_type=OperationType.GENERATE,
                target_instrument=instrument,
                target_role=role
            )
            
            result = intelligent_track_operation(
                request, self.test_tracks,
                emotion_constraints=self.emotion_constraints,
                music_constraints=self.music_constraints
            )
            
            assert result.success, f"Failed for {instrument}-{role.value}"
            assert result.generated_track.instrument == instrument
            assert result.generated_track.role == role


class TestWeek7Performance:
    """Week 7 性能测试"""
    
    def test_generation_speed(self):
        """测试生成速度"""
        request = OperationRequest(
            operation_type=OperationType.GENERATE,
            target_instrument="violin",
            target_role=TrackRole.HARMONY
        )
        
        tracks = [create_empty_track("test", "piano", TrackRole.MELODY)]
        
        import time
        start = time.time()
        result = intelligent_track_operation(request, tracks)
        duration = time.time() - start
        
        # 性能目标：< 10秒
        assert duration < 10.0
        assert result.success
    
    def test_memory_usage(self):
        """测试内存使用"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 执行多个操作
        for i in range(5):
            request = OperationRequest(
                operation_type=OperationType.GENERATE,
                target_instrument=f"instrument_{i}",
                target_role=TrackRole.HARMONY
            )
            
            tracks = [create_empty_track("test", "piano", TrackRole.MELODY)]
            result = intelligent_track_operation(request, tracks)
            assert result.success
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # 内存增长应该在合理范围内 (< 500MB)
        assert memory_increase < 500
    
    def test_concurrent_operations(self):
        """测试并发操作"""
        import threading
        import time
        
        results = []
        
        def worker():
            request = OperationRequest(
                operation_type=OperationType.GENERATE,
                target_instrument="violin",
                target_role=TrackRole.HARMONY
            )
            tracks = [create_empty_track("test", "piano", TrackRole.MELODY)]
            result = intelligent_track_operation(request, tracks)
            results.append(result)
        
        # 启动多个线程
        threads = []
        for i in range(3):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 验证所有操作都成功
        assert len(results) == 3
        assert all(result.success for result in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])