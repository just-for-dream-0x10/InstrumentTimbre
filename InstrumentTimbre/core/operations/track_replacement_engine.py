"""
音轨替换引擎 - 保持功能性的乐器替换算法
"""

import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
from .base_engine import BaseReplacementEngine
from .data_structures import (
    TrackData, OperationResult, EmotionConstraints, 
    MusicConstraints, TrackRole
)

logger = logging.getLogger(__name__)


class TrackReplacementEngine(BaseReplacementEngine):
    """智能音轨替换引擎"""
    
    def __init__(self):
        super().__init__("TrackReplacementEngine")
        self.instrument_mapper = InstrumentMapper()
        self.function_analyzer = FunctionAnalyzer()
        self.style_transferer = StyleTransferer()
        
    def initialize(self) -> bool:
        """初始化替换引擎"""
        try:
            self.instrument_mapper.load_mapping_rules()
            self.function_analyzer.load_analysis_models()
            self.style_transferer.load_transfer_models()
            
            self.is_initialized = True
            logger.info("音轨替换引擎初始化成功")
            return True
            
        except Exception as e:
            logger.error(f"音轨替换引擎初始化失败: {e}")
            return False
    
    def replace_track(self, original_track: TrackData, target_instrument: str,
                     emotion_constraints: Optional[EmotionConstraints] = None,
                     music_constraints: Optional[MusicConstraints] = None,
                     preserve_function: bool = True) -> OperationResult:
        """
        替换音轨，保持原有功能
        
        Args:
            original_track: 原始音轨
            target_instrument: 目标乐器
            emotion_constraints: 情感约束
            music_constraints: 音乐约束
            preserve_function: 是否保持原有功能
            
        Returns:
            操作结果
        """
        if not self.is_initialized:
            self.initialize()
        
        # 验证输入
        errors = self.validate_replacement_inputs(
            original_track=original_track,
            target_instrument=target_instrument,
            emotion_constraints=emotion_constraints,
            music_constraints=music_constraints
        )
        
        if errors:
            return self.create_failure_result(f"输入验证失败: {', '.join(errors)}")
        
        self.log_operation("replace_track", 
                          original_instrument=original_track.instrument,
                          target_instrument=target_instrument,
                          preserve_function=preserve_function)
        
        try:
            # 分析原音轨功能
            function_analysis = self.function_analyzer.analyze_function(original_track)
            
            # 检查乐器兼容性
            compatibility = self.instrument_mapper.check_compatibility(
                original_track.instrument, target_instrument, function_analysis.role
            )
            
            if not compatibility.is_compatible:
                return self.create_failure_result(
                    f"乐器不兼容: {compatibility.reason}"
                )
            
            # 执行风格迁移
            replaced_track = self.style_transferer.transfer_style(
                original_track=original_track,
                target_instrument=target_instrument,
                function_analysis=function_analysis,
                constraints={
                    'emotion': emotion_constraints,
                    'music': music_constraints,
                    'preserve_function': preserve_function
                }
            )
            
            # 验证替换结果
            validation_result = self._validate_replacement(
                original_track, replaced_track, function_analysis
            )
            
            # 计算质量指标
            quality_metrics = self._calculate_replacement_quality(
                original_track, replaced_track, emotion_constraints, music_constraints
            )
            
            return self.create_success_result(
                generated_track=replaced_track,
                quality_metrics=quality_metrics,
                metadata={
                    'original_instrument': original_track.instrument,
                    'function_analysis': function_analysis.__dict__,
                    'compatibility_score': compatibility.score,
                    'validation_result': validation_result.__dict__
                }
            )
            
        except Exception as e:
            logger.error(f"音轨替换失败: {e}")
            return self.create_failure_result(f"替换过程出错: {str(e)}")
    
    def process(self, *args, **kwargs) -> OperationResult:
        """实现基类的抽象方法"""
        return self.replace_track(*args, **kwargs)
    
    def _validate_replacement(self, original: TrackData, replaced: TrackData,
                             function_analysis: 'FunctionAnalysis') -> 'ReplacementValidation':
        """验证替换结果"""
        errors = []
        score = 1.0
        
        # 检查时长保持
        duration_diff = abs(replaced.duration - original.duration)
        if duration_diff > original.duration * 0.1:  # 允许10%差异
            errors.append("时长变化过大")
            score -= 0.2
        
        # 检查角色保持
        if replaced.role != original.role:
            errors.append("音轨角色发生变化")
            score -= 0.3
        
        # 检查调性保持
        if replaced.key != original.key:
            errors.append("调性发生变化")
            score -= 0.2
        
        return ReplacementValidation(
            is_valid=len(errors) == 0,
            score=max(0.0, score),
            errors=errors
        )
    
    def _calculate_replacement_quality(self, original: TrackData, replaced: TrackData,
                                     emotion_constraints: Optional[EmotionConstraints],
                                     music_constraints: Optional[MusicConstraints]) -> Dict[str, float]:
        """计算替换质量指标"""
        metrics = {}
        
        # 功能保持度
        metrics['function_preservation'] = self._calculate_function_preservation(original, replaced)
        
        # 风格一致性
        metrics['style_consistency'] = self._calculate_style_consistency(original, replaced)
        
        # 整体质量
        metrics['quality_score'] = (metrics['function_preservation'] + metrics['style_consistency']) / 2
        
        # 情感一致性（如果有约束）
        if emotion_constraints:
            metrics['emotion_consistency'] = self._calculate_emotion_consistency_for_replacement(
                replaced, emotion_constraints
            )
        else:
            metrics['emotion_consistency'] = 0.8
        
        # 和声正确性（如果有约束）
        if music_constraints:
            metrics['harmonic_correctness'] = self._calculate_harmonic_correctness_for_replacement(
                replaced, music_constraints
            )
        else:
            metrics['harmonic_correctness'] = 0.8
        
        return metrics
    
    def _calculate_function_preservation(self, original: TrackData, replaced: TrackData) -> float:
        """计算功能保持度"""
        score = 0.0
        
        # 角色一致性
        if original.role == replaced.role:
            score += 0.4
        
        # 音域相似性
        if original.pitch_sequence and replaced.pitch_sequence:
            orig_range = max(original.pitch_sequence) - min(original.pitch_sequence)
            repl_range = max(replaced.pitch_sequence) - min(replaced.pitch_sequence)
            range_similarity = 1.0 - abs(orig_range - repl_range) / max(orig_range, repl_range, 1)
            score += 0.3 * range_similarity
        else:
            score += 0.2  # 默认部分分数
        
        # 节奏相似性
        if original.rhythm_pattern and replaced.rhythm_pattern:
            rhythm_similarity = self._calculate_rhythm_similarity(
                original.rhythm_pattern, replaced.rhythm_pattern
            )
            score += 0.3 * rhythm_similarity
        else:
            score += 0.2
        
        return min(score, 1.0)
    
    def _calculate_style_consistency(self, original: TrackData, replaced: TrackData) -> float:
        """计算风格一致性"""
        score = 0.0
        
        # 动态表情相似性
        if original.dynamics and replaced.dynamics:
            orig_avg_dynamic = np.mean(original.dynamics)
            repl_avg_dynamic = np.mean(replaced.dynamics)
            dynamic_similarity = 1.0 - abs(orig_avg_dynamic - repl_avg_dynamic)
            score += 0.5 * dynamic_similarity
        else:
            score += 0.3
        
        # 调性和节拍一致性
        if original.key == replaced.key:
            score += 0.3
        
        if original.tempo and replaced.tempo:
            tempo_similarity = 1.0 - abs(original.tempo - replaced.tempo) / 120.0
            score += 0.2 * max(0, tempo_similarity)
        else:
            score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_rhythm_similarity(self, rhythm1: List[float], rhythm2: List[float]) -> float:
        """计算节奏相似性"""
        # 简化的节奏相似性计算
        min_len = min(len(rhythm1), len(rhythm2))
        if min_len == 0:
            return 0.0
        
        # 比较前min_len个节奏值
        similarities = []
        for i in range(min_len):
            r1, r2 = rhythm1[i], rhythm2[i]
            similarity = 1.0 - abs(r1 - r2) / max(r1, r2, 0.1)
            similarities.append(max(0.0, similarity))
        
        return np.mean(similarities)
    
    def _calculate_emotion_consistency_for_replacement(self, track: TrackData,
                                                     emotion_constraints: EmotionConstraints) -> float:
        """为替换音轨计算情感一致性"""
        # 重用生成引擎的方法
        from .track_generation_engine import TrackGenerationEngine
        temp_engine = TrackGenerationEngine()
        return temp_engine._calculate_emotion_consistency(track, emotion_constraints)
    
    def _calculate_harmonic_correctness_for_replacement(self, track: TrackData,
                                                      music_constraints: MusicConstraints) -> float:
        """为替换音轨计算和声正确性"""
        # 重用生成引擎的方法
        from .track_generation_engine import TrackGenerationEngine
        temp_engine = TrackGenerationEngine()
        return temp_engine._calculate_harmonic_correctness(track, music_constraints)


class InstrumentMapper:
    """乐器映射器 - 管理乐器间的功能对应关系"""
    
    def __init__(self):
        self.mapping_rules = {}
        self.compatibility_matrix = {}
    
    def load_mapping_rules(self):
        """加载乐器映射规则"""
        # 乐器家族分类
        self.instrument_families = {
            'strings': ['violin', 'viola', 'cello', 'bass', 'guitar'],
            'woodwinds': ['flute', 'oboe', 'clarinet', 'bassoon', 'saxophone'],
            'brass': ['trumpet', 'horn', 'trombone', 'tuba'],
            'percussion': ['drums', 'timpani', 'xylophone'],
            'keyboard': ['piano', 'organ', 'harpsichord']
        }
        
        # 角色兼容性矩阵
        self.role_compatibility = {
            TrackRole.MELODY: {
                'preferred': ['violin', 'flute', 'trumpet', 'piano'],
                'acceptable': ['cello', 'clarinet', 'guitar'],
                'incompatible': ['drums', 'bass', 'tuba']
            },
            TrackRole.HARMONY: {
                'preferred': ['piano', 'guitar', 'viola'],
                'acceptable': ['violin', 'cello', 'horn'],
                'incompatible': ['drums', 'bass']
            },
            TrackRole.BASS: {
                'preferred': ['bass', 'cello', 'tuba', 'bassoon'],
                'acceptable': ['trombone', 'piano'],
                'incompatible': ['violin', 'flute', 'trumpet']
            }
        }
    
    def check_compatibility(self, original_instrument: str, target_instrument: str,
                          role: TrackRole) -> 'CompatibilityResult':
        """检查乐器替换的兼容性"""
        # 获取角色兼容性
        role_compat = self.role_compatibility.get(role, {})
        
        if target_instrument in role_compat.get('incompatible', []):
            return CompatibilityResult(
                is_compatible=False,
                score=0.0,
                reason=f"{target_instrument}不适合{role.value}角色"
            )
        
        # 计算兼容性分数
        if target_instrument in role_compat.get('preferred', []):
            score = 0.9
        elif target_instrument in role_compat.get('acceptable', []):
            score = 0.6
        else:
            score = 0.3  # 未知乐器，给予基础分数
        
        # 考虑乐器家族相似性
        family_bonus = self._calculate_family_similarity(original_instrument, target_instrument)
        score = min(1.0, score + family_bonus * 0.1)
        
        return CompatibilityResult(
            is_compatible=score > 0.3,
            score=score,
            reason="兼容" if score > 0.3 else "兼容性较差"
        )
    
    def _calculate_family_similarity(self, instrument1: str, instrument2: str) -> float:
        """计算乐器家族相似性"""
        family1 = self._get_instrument_family(instrument1)
        family2 = self._get_instrument_family(instrument2)
        
        if family1 == family2:
            return 1.0
        elif family1 and family2:
            # 某些家族间有相似性
            similar_families = [
                ('strings', 'woodwinds'),
                ('woodwinds', 'brass')
            ]
            for fam1, fam2 in similar_families:
                if (family1, family2) == (fam1, fam2) or (family1, family2) == (fam2, fam1):
                    return 0.5
        
        return 0.0
    
    def _get_instrument_family(self, instrument: str) -> Optional[str]:
        """获取乐器所属家族"""
        for family, instruments in self.instrument_families.items():
            if instrument.lower() in instruments:
                return family
        return None


class FunctionAnalyzer:
    """功能分析器 - 识别原音轨在整体中的作用"""
    
    def __init__(self):
        self.analysis_models = {}
    
    def load_analysis_models(self):
        """加载分析模型"""
        # 在实际实现中会加载训练好的模型
        logger.info("功能分析模型加载完成")
    
    def analyze_function(self, track: TrackData) -> 'FunctionAnalysis':
        """分析音轨功能"""
        analysis = FunctionAnalysis()
        
        # 分析音轨角色
        analysis.role = track.role
        
        # 分析重要性
        analysis.importance = self._calculate_importance(track)
        
        # 分析音域特征
        if track.pitch_sequence:
            analysis.pitch_range = (min(track.pitch_sequence), max(track.pitch_sequence))
            analysis.average_pitch = np.mean(track.pitch_sequence)
        
        # 分析节奏特征
        if track.rhythm_pattern:
            analysis.rhythm_complexity = self._calculate_rhythm_complexity(track.rhythm_pattern)
        
        # 分析动态特征
        if track.dynamics:
            analysis.dynamic_range = (min(track.dynamics), max(track.dynamics))
            analysis.average_dynamic = np.mean(track.dynamics)
        
        return analysis
    
    def _calculate_importance(self, track: TrackData) -> float:
        """计算音轨重要性"""
        importance = 0.5  # 基础重要性
        
        # 基于角色的重要性
        role_importance = {
            TrackRole.MELODY: 0.9,
            TrackRole.BASS: 0.8,
            TrackRole.HARMONY: 0.7,
            TrackRole.RHYTHM: 0.6,
            TrackRole.ACCOMPANIMENT: 0.5
        }
        importance = role_importance.get(track.role, 0.5)
        
        # 基于音域范围的调整
        if track.pitch_sequence:
            pitch_range = max(track.pitch_sequence) - min(track.pitch_sequence)
            if pitch_range > 24:  # 超过两个八度
                importance += 0.1
        
        return min(1.0, importance)
    
    def _calculate_rhythm_complexity(self, rhythm_pattern: List[float]) -> float:
        """计算节奏复杂度"""
        if not rhythm_pattern:
            return 0.0
        
        # 计算节奏值的变化程度
        unique_values = len(set(rhythm_pattern))
        total_values = len(rhythm_pattern)
        
        complexity = unique_values / total_values
        
        # 考虑节奏变化的频率
        changes = sum(1 for i in range(1, len(rhythm_pattern)) 
                     if rhythm_pattern[i] != rhythm_pattern[i-1])
        change_rate = changes / len(rhythm_pattern)
        
        return (complexity + change_rate) / 2


class StyleTransferer:
    """风格迁移器 - 保持表现力的乐器转换"""
    
    def __init__(self):
        self.transfer_models = {}
    
    def load_transfer_models(self):
        """加载风格迁移模型"""
        logger.info("风格迁移模型加载完成")
    
    def transfer_style(self, original_track: TrackData, target_instrument: str,
                      function_analysis: 'FunctionAnalysis', constraints: Dict) -> TrackData:
        """执行风格迁移"""
        # 创建新的音轨对象
        new_track = TrackData(
            track_id=f"replaced_{target_instrument}_{original_track.role.value}",
            instrument=target_instrument,
            role=original_track.role,
            duration=original_track.duration,
            key=original_track.key,
            tempo=original_track.tempo
        )
        
        # 迁移音高序列
        new_track.pitch_sequence = self._transfer_pitches(
            original_track.pitch_sequence, target_instrument, function_analysis
        )
        
        # 迁移节奏模式
        new_track.rhythm_pattern = self._transfer_rhythms(
            original_track.rhythm_pattern, target_instrument, function_analysis
        )
        
        # 迁移动态表情
        new_track.dynamics = self._transfer_dynamics(
            original_track.dynamics, target_instrument, function_analysis
        )
        
        # 生成MIDI数据
        if new_track.pitch_sequence and new_track.rhythm_pattern:
            new_track.midi_data = {
                'notes': new_track.pitch_sequence,
                'durations': new_track.rhythm_pattern,
                'velocities': [int(d * 127) for d in new_track.dynamics] if new_track.dynamics else []
            }
        
        return new_track
    
    def _transfer_pitches(self, original_pitches: List[float], target_instrument: str,
                         function_analysis: 'FunctionAnalysis') -> List[float]:
        """迁移音高序列"""
        if not original_pitches:
            return []
        
        # 获取目标乐器的音域
        target_range = self._get_instrument_range(target_instrument)
        
        # 计算移调量
        original_center = np.mean(original_pitches)
        target_center = (target_range[0] + target_range[1]) / 2
        transpose = target_center - original_center
        
        # 应用移调
        transposed = [pitch + transpose for pitch in original_pitches]
        
        # 确保在乐器音域内
        clamped = [max(target_range[0], min(target_range[1], pitch)) for pitch in transposed]
        
        return clamped
    
    def _transfer_rhythms(self, original_rhythms: List[float], target_instrument: str,
                         function_analysis: 'FunctionAnalysis') -> List[float]:
        """迁移节奏模式"""
        if not original_rhythms:
            return []
        
        # 根据目标乐器调整节奏
        adjustment_factor = self._get_rhythm_adjustment_factor(target_instrument)
        adjusted = [rhythm * adjustment_factor for rhythm in original_rhythms]
        
        return adjusted
    
    def _transfer_dynamics(self, original_dynamics: List[float], target_instrument: str,
                          function_analysis: 'FunctionAnalysis') -> List[float]:
        """迁移动态表情"""
        if not original_dynamics:
            return [0.5] * len(function_analysis.__dict__.get('pitch_sequence', []))
        
        # 根据目标乐器的表达能力调整动态
        expression_range = self._get_instrument_expression_range(target_instrument)
        
        # 重新映射到目标乐器的表达范围
        original_min, original_max = min(original_dynamics), max(original_dynamics)
        original_range = original_max - original_min
        
        if original_range > 0:
            normalized = [(d - original_min) / original_range for d in original_dynamics]
            remapped = [expression_range[0] + n * (expression_range[1] - expression_range[0]) 
                       for n in normalized]
        else:
            remapped = [np.mean(expression_range)] * len(original_dynamics)
        
        return remapped
    
    def _get_instrument_range(self, instrument: str) -> Tuple[float, float]:
        """获取乐器音域"""
        ranges = {
            'violin': (196, 3520),
            'cello': (65, 1046),
            'piano': (27, 4186),
            'flute': (262, 2093),
            'trumpet': (165, 988),
            'guitar': (82, 1318)
        }
        return ranges.get(instrument.lower(), (80, 1000))
    
    def _get_rhythm_adjustment_factor(self, instrument: str) -> float:
        """获取节奏调整因子"""
        # 某些乐器更适合快速音符，某些更适合持续音
        factors = {
            'piano': 1.0,      # 标准
            'violin': 0.9,     # 稍快
            'flute': 0.8,      # 更快
            'cello': 1.2,      # 稍慢
            'brass': 1.1       # 稍慢
        }
        return factors.get(instrument.lower(), 1.0)
    
    def _get_instrument_expression_range(self, instrument: str) -> Tuple[float, float]:
        """获取乐器表达范围"""
        ranges = {
            'violin': (0.2, 0.9),
            'piano': (0.1, 1.0),
            'flute': (0.3, 0.8),
            'trumpet': (0.4, 0.9),
            'cello': (0.2, 0.8)
        }
        return ranges.get(instrument.lower(), (0.3, 0.7))


# 辅助数据类
class CompatibilityResult:
    """兼容性检查结果"""
    
    def __init__(self, is_compatible: bool, score: float, reason: str):
        self.is_compatible = is_compatible
        self.score = score
        self.reason = reason


class FunctionAnalysis:
    """功能分析结果"""
    
    def __init__(self):
        self.role = None
        self.importance = 0.5
        self.pitch_range = None
        self.average_pitch = 0.0
        self.rhythm_complexity = 0.0
        self.dynamic_range = None
        self.average_dynamic = 0.5


class ReplacementValidation:
    """替换验证结果"""
    
    def __init__(self, is_valid: bool, score: float, errors: List[str]):
        self.is_valid = is_valid
        self.score = score
        self.errors = errors