"""
音轨修复引擎 - 不协和音检测和自动修正
"""

import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
from .base_engine import BaseRepairEngine
from .data_structures import (
    TrackData, OperationResult, ConflictReport, ConflictType
)

logger = logging.getLogger(__name__)


class TrackRepairEngine(BaseRepairEngine):
    """智能音轨修复引擎"""
    
    def __init__(self):
        super().__init__("TrackRepairEngine")
        self.dissonance_detector = DissonanceDetector()
        self.pitch_corrector = PitchCorrector()
        self.rhythm_stabilizer = RhythmStabilizer()
        self.harmony_corrector = HarmonyCorrector()
        
    def initialize(self) -> bool:
        """初始化修复引擎"""
        try:
            self.dissonance_detector.load_detection_rules()
            self.pitch_corrector.load_correction_models()
            self.rhythm_stabilizer.load_rhythm_models()
            self.harmony_corrector.load_harmony_rules()
            
            self.is_initialized = True
            logger.info("音轨修复引擎初始化成功")
            return True
            
        except Exception as e:
            logger.error(f"音轨修复引擎初始化失败: {e}")
            return False
    
    def repair_track(self, problematic_track: TrackData,
                    other_tracks: Optional[List[TrackData]] = None,
                    constraints: Optional[Dict] = None) -> OperationResult:
        """
        修复音轨中的问题
        
        Args:
            problematic_track: 有问题的音轨
            other_tracks: 其他相关音轨
            constraints: 修复约束
            
        Returns:
            操作结果
        """
        if not self.is_initialized:
            self.initialize()
        
        # 验证输入
        errors = self.validate_repair_inputs(
            problematic_track=problematic_track,
            other_tracks=other_tracks,
            constraints=constraints
        )
        
        if errors:
            return self.create_failure_result(f"输入验证失败: {', '.join(errors)}")
        
        self.log_operation("repair_track", 
                          track_id=problematic_track.track_id,
                          instrument=problematic_track.instrument)
        
        try:
            # 检测问题
            issues = self._detect_issues(problematic_track, other_tracks or [])
            
            if not issues:
                return self.create_success_result(
                    generated_track=problematic_track,
                    metadata={'message': '未发现需要修复的问题'}
                )
            
            # 创建修复后的音轨副本
            repaired_track = self._create_track_copy(problematic_track)
            
            # 执行修复
            repair_results = []
            
            for issue in issues:
                repair_result = self._repair_issue(
                    repaired_track, issue, other_tracks or [], constraints
                )
                repair_results.append(repair_result)
            
            # 验证修复结果
            validation_result = self._validate_repair_result(
                problematic_track, repaired_track, issues
            )
            
            # 计算修复质量
            quality_metrics = self._calculate_repair_quality(
                problematic_track, repaired_track, repair_results
            )
            
            return self.create_success_result(
                generated_track=repaired_track,
                quality_metrics=quality_metrics,
                metadata={
                    'original_issues': len(issues),
                    'repaired_issues': sum(1 for r in repair_results if r.success),
                    'repair_details': repair_results,
                    'validation_result': validation_result.__dict__
                }
            )
            
        except Exception as e:
            logger.error(f"音轨修复失败: {e}")
            return self.create_failure_result(f"修复过程出错: {str(e)}")
    
    def process(self, *args, **kwargs) -> OperationResult:
        """实现基类的抽象方法"""
        return self.repair_track(*args, **kwargs)
    
    def _detect_issues(self, track: TrackData, other_tracks: List[TrackData]) -> List['TrackIssue']:
        """检测音轨中的问题"""
        issues = []
        
        # 检测不协和音问题
        dissonance_issues = self.dissonance_detector.detect_dissonance(track, other_tracks)
        issues.extend(dissonance_issues)
        
        # 检测音准问题
        pitch_issues = self.pitch_corrector.detect_pitch_issues(track)
        issues.extend(pitch_issues)
        
        # 检测节拍问题
        rhythm_issues = self.rhythm_stabilizer.detect_rhythm_issues(track)
        issues.extend(rhythm_issues)
        
        # 检测和声问题
        harmony_issues = self.harmony_corrector.detect_harmony_issues(track, other_tracks)
        issues.extend(harmony_issues)
        
        logger.info(f"检测到 {len(issues)} 个问题")
        return issues
    
    def _repair_issue(self, track: TrackData, issue: 'TrackIssue',
                     other_tracks: List[TrackData], constraints: Optional[Dict]) -> 'RepairResult':
        """修复单个问题"""
        try:
            if issue.issue_type == 'dissonance':
                return self.dissonance_detector.repair_dissonance(track, issue, other_tracks)
            elif issue.issue_type == 'pitch_drift':
                return self.pitch_corrector.repair_pitch_drift(track, issue)
            elif issue.issue_type == 'rhythm_instability':
                return self.rhythm_stabilizer.repair_rhythm_instability(track, issue)
            elif issue.issue_type == 'harmony_conflict':
                return self.harmony_corrector.repair_harmony_conflict(track, issue, other_tracks)
            else:
                return RepairResult(
                    success=False,
                    issue_type=issue.issue_type,
                    error_message=f"未知问题类型: {issue.issue_type}"
                )
                
        except Exception as e:
            logger.error(f"修复问题失败: {e}")
            return RepairResult(
                success=False,
                issue_type=issue.issue_type,
                error_message=str(e)
            )
    
    def _create_track_copy(self, original: TrackData) -> TrackData:
        """创建音轨副本"""
        return TrackData(
            track_id=f"repaired_{original.track_id}",
            instrument=original.instrument,
            role=original.role,
            audio_data=original.audio_data.copy() if original.audio_data is not None else None,
            midi_data=original.midi_data.copy() if original.midi_data else {},
            pitch_sequence=original.pitch_sequence.copy() if original.pitch_sequence else [],
            rhythm_pattern=original.rhythm_pattern.copy() if original.rhythm_pattern else [],
            dynamics=original.dynamics.copy() if original.dynamics else [],
            duration=original.duration,
            sample_rate=original.sample_rate,
            key=original.key,
            tempo=original.tempo
        )
    
    def _validate_repair_result(self, original: TrackData, repaired: TrackData,
                              issues: List['TrackIssue']) -> 'RepairValidation':
        """验证修复结果"""
        validation_errors = []
        improvement_score = 0.0
        
        # 检查基本属性是否保持
        if repaired.duration != original.duration:
            validation_errors.append("音轨时长发生变化")
        
        if repaired.key != original.key:
            validation_errors.append("调性发生变化")
        
        # 检查修复效果
        remaining_issues = self._detect_issues(repaired, [])
        original_issue_count = len(issues)
        remaining_issue_count = len(remaining_issues)
        
        if original_issue_count > 0:
            improvement_score = (original_issue_count - remaining_issue_count) / original_issue_count
        
        return RepairValidation(
            is_valid=len(validation_errors) == 0,
            improvement_score=improvement_score,
            original_issues=original_issue_count,
            remaining_issues=remaining_issue_count,
            validation_errors=validation_errors
        )
    
    def _calculate_repair_quality(self, original: TrackData, repaired: TrackData,
                                repair_results: List['RepairResult']) -> Dict[str, float]:
        """计算修复质量指标"""
        metrics = {}
        
        # 修复成功率
        successful_repairs = sum(1 for r in repair_results if r.success)
        total_repairs = len(repair_results)
        metrics['repair_success_rate'] = successful_repairs / total_repairs if total_repairs > 0 else 1.0
        
        # 音轨完整性保持度
        metrics['integrity_preservation'] = self._calculate_integrity_preservation(original, repaired)
        
        # 音乐质量改善度
        metrics['quality_improvement'] = self._calculate_quality_improvement(original, repaired)
        
        # 整体修复质量
        metrics['quality_score'] = (
            metrics['repair_success_rate'] * 0.4 +
            metrics['integrity_preservation'] * 0.3 +
            metrics['quality_improvement'] * 0.3
        )
        
        # 为了兼容性，添加其他指标的默认值
        metrics['emotion_consistency'] = 0.8
        metrics['harmonic_correctness'] = 0.8
        
        return metrics
    
    def _calculate_integrity_preservation(self, original: TrackData, repaired: TrackData) -> float:
        """计算完整性保持度"""
        score = 0.0
        
        # 时长保持
        if abs(repaired.duration - original.duration) < 0.1:
            score += 0.3
        
        # 调性保持
        if repaired.key == original.key:
            score += 0.3
        
        # 音轨角色保持
        if repaired.role == original.role:
            score += 0.2
        
        # 整体结构相似性
        if original.pitch_sequence and repaired.pitch_sequence:
            similarity = self._calculate_sequence_similarity(
                original.pitch_sequence, repaired.pitch_sequence
            )
            score += 0.2 * similarity
        
        return min(score, 1.0)
    
    def _calculate_quality_improvement(self, original: TrackData, repaired: TrackData) -> float:
        """计算质量改善度"""
        # 简化的质量改善评估
        # 实际应用中会使用更复杂的音乐质量评估算法
        
        improvement_indicators = 0
        total_indicators = 0
        
        # 检查音高稳定性改善
        if original.pitch_sequence and repaired.pitch_sequence:
            original_variance = np.var(np.diff(original.pitch_sequence))
            repaired_variance = np.var(np.diff(repaired.pitch_sequence))
            
            if repaired_variance < original_variance:
                improvement_indicators += 1
            total_indicators += 1
        
        # 检查节奏稳定性改善
        if original.rhythm_pattern and repaired.rhythm_pattern:
            original_rhythm_var = np.var(original.rhythm_pattern)
            repaired_rhythm_var = np.var(repaired.rhythm_pattern)
            
            if repaired_rhythm_var < original_rhythm_var:
                improvement_indicators += 1
            total_indicators += 1
        
        # 检查动态平滑性改善
        if original.dynamics and repaired.dynamics:
            original_dynamic_changes = np.sum(np.abs(np.diff(original.dynamics)))
            repaired_dynamic_changes = np.sum(np.abs(np.diff(repaired.dynamics)))
            
            if repaired_dynamic_changes < original_dynamic_changes:
                improvement_indicators += 1
            total_indicators += 1
        
        return improvement_indicators / total_indicators if total_indicators > 0 else 0.0
    
    def _calculate_sequence_similarity(self, seq1: List[float], seq2: List[float]) -> float:
        """计算序列相似性"""
        if not seq1 or not seq2:
            return 0.0
        
        min_len = min(len(seq1), len(seq2))
        if min_len == 0:
            return 0.0
        
        # 计算对应位置的相似性
        similarities = []
        for i in range(min_len):
            diff = abs(seq1[i] - seq2[i])
            similarity = max(0.0, 1.0 - diff / 12.0)  # 12半音为完全不同
            similarities.append(similarity)
        
        return np.mean(similarities)


class DissonanceDetector:
    """不协和音检测器"""
    
    def __init__(self):
        self.detection_rules = {}
        
    def load_detection_rules(self):
        """加载检测规则"""
        # 不协和音程定义
        self.dissonant_intervals = {1, 6, 10, 11}  # 小二度、三全音、小七度、大七度
        self.harsh_threshold = 0.3  # 不协和音比例阈值
        
        logger.info("不协和音检测规则加载完成")
    
    def detect_dissonance(self, track: TrackData, other_tracks: List[TrackData]) -> List['TrackIssue']:
        """检测不协和音"""
        issues = []
        
        if not track.pitch_sequence:
            return issues
        
        for other_track in other_tracks:
            if not other_track.pitch_sequence:
                continue
            
            dissonance_locations = self._find_dissonant_intervals(
                track.pitch_sequence, other_track.pitch_sequence
            )
            
            if dissonance_locations:
                issue = TrackIssue(
                    issue_type='dissonance',
                    severity=len(dissonance_locations) / len(track.pitch_sequence),
                    description=f"与{other_track.instrument}存在{len(dissonance_locations)}处不协和音",
                    locations=dissonance_locations,
                    affected_elements=dissonance_locations
                )
                issues.append(issue)
        
        return issues
    
    def _find_dissonant_intervals(self, pitches1: List[float], pitches2: List[float]) -> List[int]:
        """找到不协和音程的位置"""
        dissonant_positions = []
        min_len = min(len(pitches1), len(pitches2))
        
        for i in range(min_len):
            interval = abs(pitches1[i] - pitches2[i]) % 12
            if interval in self.dissonant_intervals:
                dissonant_positions.append(i)
        
        return dissonant_positions
    
    def repair_dissonance(self, track: TrackData, issue: 'TrackIssue',
                         other_tracks: List[TrackData]) -> 'RepairResult':
        """修复不协和音"""
        try:
            if not track.pitch_sequence:
                return RepairResult(
                    success=False,
                    issue_type='dissonance',
                    error_message="无音高序列可修复"
                )
            
            modifications_made = 0
            
            for position in issue.locations:
                if position < len(track.pitch_sequence):
                    # 寻找最近的协和音程
                    original_pitch = track.pitch_sequence[position]
                    corrected_pitch = self._find_consonant_alternative(
                        original_pitch, other_tracks, position
                    )
                    
                    if corrected_pitch != original_pitch:
                        track.pitch_sequence[position] = corrected_pitch
                        modifications_made += 1
            
            return RepairResult(
                success=modifications_made > 0,
                issue_type='dissonance',
                modifications_count=modifications_made,
                description=f"修正了{modifications_made}个不协和音"
            )
            
        except Exception as e:
            return RepairResult(
                success=False,
                issue_type='dissonance',
                error_message=str(e)
            )
    
    def _find_consonant_alternative(self, original_pitch: float,
                                  other_tracks: List[TrackData], position: int) -> float:
        """寻找协和音程的替代音高"""
        consonant_intervals = {0, 3, 4, 7, 8, 9, 12}  # 协和音程
        
        best_alternative = original_pitch
        min_distance = float('inf')
        
        # 检查周围的半音
        for semitone_offset in range(-6, 7):
            candidate_pitch = original_pitch + semitone_offset
            
            # 检查与其他音轨的协和性
            is_consonant = True
            total_distance = abs(semitone_offset)
            
            for other_track in other_tracks:
                if (other_track.pitch_sequence and 
                    position < len(other_track.pitch_sequence)):
                    
                    other_pitch = other_track.pitch_sequence[position]
                    interval = abs(candidate_pitch - other_pitch) % 12
                    
                    if interval not in consonant_intervals:
                        is_consonant = False
                        break
            
            # 选择距离最近的协和音
            if is_consonant and total_distance < min_distance:
                best_alternative = candidate_pitch
                min_distance = total_distance
        
        return best_alternative


class PitchCorrector:
    """音准校正器"""
    
    def __init__(self):
        self.correction_models = {}
        
    def load_correction_models(self):
        """加载校正模型"""
        # 音准偏差阈值（半音的百分比）
        self.pitch_deviation_threshold = 0.1  # 10%的半音
        
        logger.info("音准校正模型加载完成")
    
    def detect_pitch_issues(self, track: TrackData) -> List['TrackIssue']:
        """检测音准问题"""
        issues = []
        
        if not track.pitch_sequence:
            return issues
        
        # 检测音高漂移
        drift_positions = self._detect_pitch_drift(track.pitch_sequence)
        
        if drift_positions:
            issue = TrackIssue(
                issue_type='pitch_drift',
                severity=len(drift_positions) / len(track.pitch_sequence),
                description=f"检测到{len(drift_positions)}处音准偏差",
                locations=drift_positions,
                affected_elements=drift_positions
            )
            issues.append(issue)
        
        return issues
    
    def _detect_pitch_drift(self, pitches: List[float]) -> List[int]:
        """检测音高漂移"""
        drift_positions = []
        
        for i, pitch in enumerate(pitches):
            # 检查是否偏离最近的半音
            nearest_semitone = round(pitch)
            deviation = abs(pitch - nearest_semitone)
            
            if deviation > self.pitch_deviation_threshold:
                drift_positions.append(i)
        
        return drift_positions
    
    def repair_pitch_drift(self, track: TrackData, issue: 'TrackIssue') -> 'RepairResult':
        """修复音准漂移"""
        try:
            if not track.pitch_sequence:
                return RepairResult(
                    success=False,
                    issue_type='pitch_drift',
                    error_message="无音高序列可修复"
                )
            
            modifications_made = 0
            
            for position in issue.locations:
                if position < len(track.pitch_sequence):
                    original_pitch = track.pitch_sequence[position]
                    corrected_pitch = round(original_pitch)  # 校正到最近的半音
                    
                    track.pitch_sequence[position] = corrected_pitch
                    modifications_made += 1
            
            return RepairResult(
                success=modifications_made > 0,
                issue_type='pitch_drift',
                modifications_count=modifications_made,
                description=f"校正了{modifications_made}个音准偏差"
            )
            
        except Exception as e:
            return RepairResult(
                success=False,
                issue_type='pitch_drift',
                error_message=str(e)
            )


class RhythmStabilizer:
    """节奏稳定器"""
    
    def __init__(self):
        self.rhythm_models = {}
        
    def load_rhythm_models(self):
        """加载节奏模型"""
        # 节奏稳定性阈值
        self.rhythm_variance_threshold = 0.5
        
        logger.info("节奏稳定器模型加载完成")
    
    def detect_rhythm_issues(self, track: TrackData) -> List['TrackIssue']:
        """检测节奏问题"""
        issues = []
        
        if not track.rhythm_pattern:
            return issues
        
        # 检测节奏不稳定性
        if self._is_rhythm_unstable(track.rhythm_pattern):
            issue = TrackIssue(
                issue_type='rhythm_instability',
                severity=self._calculate_rhythm_instability_severity(track.rhythm_pattern),
                description="节奏不稳定",
                locations=list(range(len(track.rhythm_pattern))),
                affected_elements=track.rhythm_pattern
            )
            issues.append(issue)
        
        return issues
    
    def _is_rhythm_unstable(self, rhythm_pattern: List[float]) -> bool:
        """检查节奏是否不稳定"""
        if len(rhythm_pattern) < 2:
            return False
        
        variance = np.var(rhythm_pattern)
        return variance > self.rhythm_variance_threshold
    
    def _calculate_rhythm_instability_severity(self, rhythm_pattern: List[float]) -> float:
        """计算节奏不稳定性严重程度"""
        variance = np.var(rhythm_pattern)
        severity = min(variance / self.rhythm_variance_threshold, 1.0)
        return severity
    
    def repair_rhythm_instability(self, track: TrackData, issue: 'TrackIssue') -> 'RepairResult':
        """修复节奏不稳定性"""
        try:
            if not track.rhythm_pattern:
                return RepairResult(
                    success=False,
                    issue_type='rhythm_instability',
                    error_message="无节奏模式可修复"
                )
            
            # 平滑节奏变化
            original_pattern = track.rhythm_pattern.copy()
            smoothed_pattern = self._smooth_rhythm_pattern(track.rhythm_pattern)
            
            track.rhythm_pattern = smoothed_pattern
            
            # 计算改善程度
            original_variance = np.var(original_pattern)
            new_variance = np.var(smoothed_pattern)
            improvement = (original_variance - new_variance) / original_variance
            
            return RepairResult(
                success=improvement > 0.1,  # 至少10%的改善
                issue_type='rhythm_instability',
                improvement_score=improvement,
                description=f"节奏稳定性改善{improvement:.1%}"
            )
            
        except Exception as e:
            return RepairResult(
                success=False,
                issue_type='rhythm_instability',
                error_message=str(e)
            )
    
    def _smooth_rhythm_pattern(self, pattern: List[float]) -> List[float]:
        """平滑节奏模式"""
        if len(pattern) < 3:
            return pattern
        
        smoothed = [pattern[0]]  # 保持第一个值
        
        # 使用简单的移动平均
        for i in range(1, len(pattern) - 1):
            smoothed_value = (pattern[i-1] + pattern[i] + pattern[i+1]) / 3
            smoothed.append(smoothed_value)
        
        smoothed.append(pattern[-1])  # 保持最后一个值
        
        return smoothed


class HarmonyCorrector:
    """和声校正器"""
    
    def __init__(self):
        self.harmony_rules = {}
        
    def load_harmony_rules(self):
        """加载和声规则"""
        # 和声进行规则
        self.valid_progressions = {
            'C_major': ['C', 'F', 'G', 'Am', 'Dm', 'Em'],
            'G_major': ['G', 'C', 'D', 'Em', 'Am', 'Bm'],
            # 可以扩展更多调性
        }
        
        logger.info("和声校正规则加载完成")
    
    def detect_harmony_issues(self, track: TrackData, other_tracks: List[TrackData]) -> List['TrackIssue']:
        """检测和声问题"""
        issues = []
        
        # 简化实现：检查调性一致性
        for other_track in other_tracks:
            if track.key != other_track.key:
                issue = TrackIssue(
                    issue_type='harmony_conflict',
                    severity=0.8,
                    description=f"调性冲突: {track.key} vs {other_track.key}",
                    locations=[0],  # 整个音轨
                    affected_elements=[track.key, other_track.key]
                )
                issues.append(issue)
                break  # 只报告一次调性冲突
        
        return issues
    
    def repair_harmony_conflict(self, track: TrackData, issue: 'TrackIssue',
                               other_tracks: List[TrackData]) -> 'RepairResult':
        """修复和声冲突"""
        try:
            if issue.issue_type != 'harmony_conflict':
                return RepairResult(
                    success=False,
                    issue_type='harmony_conflict',
                    error_message="问题类型不匹配"
                )
            
            # 简化处理：将音轨调性统一到主要调性
            if other_tracks:
                # 使用第一个音轨的调性作为目标
                target_key = other_tracks[0].key
                original_key = track.key
                
                # 计算移调量
                transpose_amount = self._calculate_transpose_amount(original_key, target_key)
                
                # 应用移调
                if track.pitch_sequence and transpose_amount != 0:
                    track.pitch_sequence = [p + transpose_amount for p in track.pitch_sequence]
                    track.key = target_key
                    
                    return RepairResult(
                        success=True,
                        issue_type='harmony_conflict',
                        description=f"从{original_key}移调到{target_key}",
                        transpose_amount=transpose_amount
                    )
            
            return RepairResult(
                success=False,
                issue_type='harmony_conflict',
                error_message="无法确定目标调性"
            )
            
        except Exception as e:
            return RepairResult(
                success=False,
                issue_type='harmony_conflict',
                error_message=str(e)
            )
    
    def _calculate_transpose_amount(self, from_key: str, to_key: str) -> int:
        """计算移调量（半音数）"""
        # 简化的调性转换表
        key_to_semitone = {
            'C_major': 0, 'Db_major': 1, 'D_major': 2, 'Eb_major': 3,
            'E_major': 4, 'F_major': 5, 'Gb_major': 6, 'G_major': 7,
            'Ab_major': 8, 'A_major': 9, 'Bb_major': 10, 'B_major': 11
        }
        
        from_semitone = key_to_semitone.get(from_key, 0)
        to_semitone = key_to_semitone.get(to_key, 0)
        
        return to_semitone - from_semitone


# 辅助数据类
class TrackIssue:
    """音轨问题"""
    
    def __init__(self, issue_type: str, severity: float, description: str,
                 locations: List[int], affected_elements: List):
        self.issue_type = issue_type
        self.severity = severity
        self.description = description
        self.locations = locations
        self.affected_elements = affected_elements


class RepairResult:
    """修复结果"""
    
    def __init__(self, success: bool, issue_type: str, **kwargs):
        self.success = success
        self.issue_type = issue_type
        
        # 可选属性
        self.error_message = kwargs.get('error_message', '')
        self.modifications_count = kwargs.get('modifications_count', 0)
        self.improvement_score = kwargs.get('improvement_score', 0.0)
        self.description = kwargs.get('description', '')
        self.transpose_amount = kwargs.get('transpose_amount', 0)


class RepairValidation:
    """修复验证结果"""
    
    def __init__(self, is_valid: bool, improvement_score: float,
                 original_issues: int, remaining_issues: int,
                 validation_errors: List[str]):
        self.is_valid = is_valid
        self.improvement_score = improvement_score
        self.original_issues = original_issues
        self.remaining_issues = remaining_issues
        self.validation_errors = validation_errors