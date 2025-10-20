"""
情感驱动配器 - 根据情感特征选择合适乐器和编排
"""

import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
from .data_structures import (
    TrackData, EmotionConstraints, EmotionType, TrackRole
)

logger = logging.getLogger(__name__)


class EmotionDrivenOrchestrator:
    """情感驱动配器"""
    
    def __init__(self):
        self.emotion_instrument_map = EmotionInstrumentMap()
        self.arrangement_rules = ArrangementRules()
        self.expression_controller = ExpressionController()
        self.is_initialized = False
        
    def initialize(self) -> bool:
        """初始化配器"""
        try:
            self.emotion_instrument_map.load_mappings()
            self.arrangement_rules.load_rules()
            self.expression_controller.load_models()
            
            self.is_initialized = True
            logger.info("情感驱动配器初始化成功")
            return True
            
        except Exception as e:
            logger.error(f"情感驱动配器初始化失败: {e}")
            return False
    
    def get_orchestration_suggestion(self, emotion_constraints: EmotionConstraints,
                                   target_instrument: str,
                                   current_tracks: List[TrackData]) -> Dict:
        """
        获取配器建议
        
        Args:
            emotion_constraints: 情感约束
            target_instrument: 目标乐器
            current_tracks: 当前音轨
            
        Returns:
            配器建议
        """
        if not self.is_initialized:
            self.initialize()
        
        suggestion = {
            'instrument_suitability': self._evaluate_instrument_suitability(
                emotion_constraints, target_instrument
            ),
            'arrangement_style': self._suggest_arrangement_style(
                emotion_constraints, current_tracks
            ),
            'expression_markings': self._suggest_expression_markings(
                emotion_constraints, target_instrument
            ),
            'dynamics_profile': self._create_dynamics_profile(
                emotion_constraints, target_instrument
            ),
            'articulation_suggestions': self._suggest_articulations(
                emotion_constraints, target_instrument
            )
        }
        
        logger.info(f"生成配器建议: {target_instrument} for {emotion_constraints.primary_emotion.value}")
        return suggestion
    
    def orchestrate_track(self, base_track: TrackData, 
                         emotion_constraints: EmotionConstraints,
                         current_tracks: List[TrackData]) -> TrackData:
        """
        对音轨进行情感驱动的编配
        
        Args:
            base_track: 基础音轨
            emotion_constraints: 情感约束
            current_tracks: 当前音轨列表
            
        Returns:
            编配后的音轨
        """
        orchestrated_track = TrackData(
            track_id=f"orchestrated_{base_track.track_id}",
            instrument=base_track.instrument,
            role=base_track.role,
            duration=base_track.duration,
            key=base_track.key,
            tempo=base_track.tempo
        )
        
        # 应用情感驱动的修改
        orchestrated_track.pitch_sequence = self._adjust_pitches_for_emotion(
            base_track.pitch_sequence, emotion_constraints, base_track.instrument
        )
        
        orchestrated_track.rhythm_pattern = self._adjust_rhythms_for_emotion(
            base_track.rhythm_pattern, emotion_constraints, base_track.instrument
        )
        
        orchestrated_track.dynamics = self._create_emotional_dynamics(
            base_track, emotion_constraints
        )
        
        # 生成MIDI数据
        orchestrated_track.midi_data = self._create_emotional_midi(
            orchestrated_track, emotion_constraints
        )
        
        return orchestrated_track
    
    def _evaluate_instrument_suitability(self, emotion_constraints: EmotionConstraints,
                                       instrument: str) -> Dict:
        """评估乐器对情感的适配度"""
        emotion_type = emotion_constraints.primary_emotion
        intensity = emotion_constraints.intensity
        
        # 获取乐器的情感适配度
        suitability = self.emotion_instrument_map.get_suitability(
            emotion_type, instrument
        )
        
        # 根据强度调整适配度
        intensity_factor = self._calculate_intensity_factor(intensity, instrument)
        
        final_score = suitability * intensity_factor
        
        return {
            'base_suitability': suitability,
            'intensity_factor': intensity_factor,
            'final_score': final_score,
            'recommendation': self._get_suitability_recommendation(final_score)
        }
    
    def _suggest_arrangement_style(self, emotion_constraints: EmotionConstraints,
                                 current_tracks: List[TrackData]) -> Dict:
        """建议编排风格"""
        emotion_type = emotion_constraints.primary_emotion
        
        # 分析当前编配
        current_analysis = self._analyze_current_arrangement(current_tracks)
        
        # 根据情感获取编排建议
        style_suggestions = self.arrangement_rules.get_style_for_emotion(
            emotion_type, current_analysis
        )
        
        return {
            'suggested_style': style_suggestions['style'],
            'texture_density': style_suggestions['texture_density'],
            'harmonic_complexity': style_suggestions['harmonic_complexity'],
            'rhythmic_activity': style_suggestions['rhythmic_activity'],
            'spatial_arrangement': style_suggestions['spatial_arrangement']
        }
    
    def _suggest_expression_markings(self, emotion_constraints: EmotionConstraints,
                                   instrument: str) -> List[str]:
        """建议表情记号"""
        emotion_type = emotion_constraints.primary_emotion
        intensity = emotion_constraints.intensity
        
        markings = self.expression_controller.get_expression_markings(
            emotion_type, intensity, instrument
        )
        
        return markings
    
    def _create_dynamics_profile(self, emotion_constraints: EmotionConstraints,
                               instrument: str) -> Dict:
        """创建动态轮廓"""
        emotion_type = emotion_constraints.primary_emotion
        intensity = emotion_constraints.intensity
        
        # 基于情感的动态范围
        base_range = self._get_emotion_dynamic_range(emotion_type)
        
        # 根据强度调整
        adjusted_range = (
            base_range[0] * intensity,
            base_range[1] * intensity
        )
        
        # 乐器特定调整
        instrument_factor = self._get_instrument_dynamic_factor(instrument)
        final_range = (
            adjusted_range[0] * instrument_factor,
            adjusted_range[1] * instrument_factor
        )
        
        return {
            'base_range': base_range,
            'adjusted_range': adjusted_range,
            'final_range': final_range,
            'dynamic_curve': self._create_dynamic_curve(emotion_type, intensity)
        }
    
    def _suggest_articulations(self, emotion_constraints: EmotionConstraints,
                             instrument: str) -> List[str]:
        """建议演奏技法"""
        emotion_type = emotion_constraints.primary_emotion
        
        # 情感对应的演奏技法
        emotion_articulations = {
            EmotionType.HAPPY: ['staccato', 'leggiero', 'vivace'],
            EmotionType.SAD: ['legato', 'dolce', 'espressivo'],
            EmotionType.CALM: ['legato', 'dolce', 'tranquillo'],
            EmotionType.ENERGETIC: ['marcato', 'accent', 'forte'],
            EmotionType.MELANCHOLIC: ['legato', 'sostenuto', 'espressivo'],
            EmotionType.ANGRY: ['marcato', 'sforzando', 'agitato']
        }
        
        base_articulations = emotion_articulations.get(emotion_type, ['legato'])
        
        # 根据乐器特性过滤和调整
        instrument_specific = self._filter_articulations_for_instrument(
            base_articulations, instrument
        )
        
        return instrument_specific
    
    def _adjust_pitches_for_emotion(self, original_pitches: List[float],
                                  emotion_constraints: EmotionConstraints,
                                  instrument: str) -> List[float]:
        """根据情感调整音高"""
        if not original_pitches:
            return []
        
        emotion_type = emotion_constraints.primary_emotion
        intensity = emotion_constraints.intensity
        
        adjusted_pitches = original_pitches.copy()
        
        # 根据情感类型调整音高
        if emotion_type == EmotionType.HAPPY:
            # 快乐：倾向于更高的音域
            adjustment = 2 * intensity  # 上移半音
            adjusted_pitches = [p + adjustment for p in adjusted_pitches]
            
        elif emotion_type == EmotionType.SAD:
            # 悲伤：倾向于更低的音域
            adjustment = -2 * intensity  # 下移半音
            adjusted_pitches = [p + adjustment for p in adjusted_pitches]
            
        elif emotion_type == EmotionType.ENERGETIC:
            # 激动：增加音高变化
            for i in range(1, len(adjusted_pitches)):
                if np.random.random() < intensity * 0.3:
                    direction = np.random.choice([-1, 1])
                    adjusted_pitches[i] += direction * 1  # 小幅度变化
        
        # 确保在乐器音域内
        instrument_range = self._get_instrument_range(instrument)
        adjusted_pitches = [
            max(instrument_range[0], min(instrument_range[1], p))
            for p in adjusted_pitches
        ]
        
        return adjusted_pitches
    
    def _adjust_rhythms_for_emotion(self, original_rhythms: List[float],
                                  emotion_constraints: EmotionConstraints,
                                  instrument: str) -> List[float]:
        """根据情感调整节奏"""
        if not original_rhythms:
            return []
        
        emotion_type = emotion_constraints.primary_emotion
        intensity = emotion_constraints.intensity
        
        adjusted_rhythms = original_rhythms.copy()
        
        # 根据情感类型调整节奏
        if emotion_type == EmotionType.ENERGETIC:
            # 激动：缩短音符时值，增加密度
            factor = 1.0 - (intensity * 0.3)
            adjusted_rhythms = [r * factor for r in adjusted_rhythms]
            
        elif emotion_type == EmotionType.CALM:
            # 平静：延长音符时值
            factor = 1.0 + (intensity * 0.5)
            adjusted_rhythms = [r * factor for r in adjusted_rhythms]
            
        elif emotion_type == EmotionType.ANGRY:
            # 愤怒：不规则节奏变化
            for i in range(len(adjusted_rhythms)):
                if np.random.random() < intensity * 0.2:
                    adjusted_rhythms[i] *= np.random.uniform(0.7, 1.3)
        
        return adjusted_rhythms
    
    def _create_emotional_dynamics(self, base_track: TrackData,
                                 emotion_constraints: EmotionConstraints) -> List[float]:
        """创建情感化的动态"""
        duration_samples = int(base_track.duration * 10)  # 每秒10个采样点
        
        emotion_type = emotion_constraints.primary_emotion
        intensity = emotion_constraints.intensity
        
        # 获取情感的基础动态范围
        dynamic_range = self._get_emotion_dynamic_range(emotion_type)
        base_dynamic = (dynamic_range[0] + dynamic_range[1]) / 2 * intensity
        
        dynamics = []
        
        for i in range(duration_samples):
            # 基础动态
            current_dynamic = base_dynamic
            
            # 根据情感类型添加变化
            if emotion_type == EmotionType.HAPPY:
                # 快乐：轻微的上下波动
                variation = 0.1 * np.sin(i * 0.1) * intensity
                current_dynamic += variation
                
            elif emotion_type == EmotionType.SAD:
                # 悲伤：逐渐减弱
                fade_factor = 1.0 - (i / duration_samples) * 0.3 * intensity
                current_dynamic *= fade_factor
                
            elif emotion_type == EmotionType.ENERGETIC:
                # 激动：较大的动态变化
                variation = 0.2 * np.random.uniform(-1, 1) * intensity
                current_dynamic += variation
                
            elif emotion_type == EmotionType.CALM:
                # 平静：稳定的动态
                variation = 0.05 * np.sin(i * 0.05) * intensity
                current_dynamic += variation
            
            # 限制在合理范围内
            current_dynamic = max(0.1, min(1.0, current_dynamic))
            dynamics.append(current_dynamic)
        
        return dynamics
    
    def _create_emotional_midi(self, track: TrackData,
                             emotion_constraints: EmotionConstraints) -> Dict:
        """创建情感化的MIDI数据"""
        if not track.pitch_sequence or not track.rhythm_pattern:
            return {}
        
        midi_data = {
            'notes': track.pitch_sequence,
            'durations': track.rhythm_pattern,
            'velocities': []
        }
        
        # 将动态转换为MIDI力度
        if track.dynamics:
            velocities = [int(d * 127) for d in track.dynamics[:len(track.pitch_sequence)]]
            midi_data['velocities'] = velocities
        else:
            # 使用默认力度
            default_velocity = int(emotion_constraints.intensity * 100 + 27)
            midi_data['velocities'] = [default_velocity] * len(track.pitch_sequence)
        
        return midi_data
    
    def _get_emotion_dynamic_range(self, emotion_type: EmotionType) -> Tuple[float, float]:
        """获取情感对应的动态范围"""
        ranges = {
            EmotionType.HAPPY: (0.6, 0.9),
            EmotionType.SAD: (0.2, 0.5),
            EmotionType.CALM: (0.3, 0.6),
            EmotionType.ENERGETIC: (0.7, 1.0),
            EmotionType.MELANCHOLIC: (0.2, 0.6),
            EmotionType.ANGRY: (0.8, 1.0)
        }
        return ranges.get(emotion_type, (0.4, 0.7))
    
    def _get_instrument_range(self, instrument: str) -> Tuple[float, float]:
        """获取乐器音域"""
        ranges = {
            'violin': (196, 3520),
            'cello': (65, 1046),
            'piano': (27, 4186),
            'flute': (262, 2093),
            'trumpet': (165, 988)
        }
        return ranges.get(instrument.lower(), (80, 1000))
    
    def _calculate_intensity_factor(self, intensity: float, instrument: str) -> float:
        """计算强度因子"""
        # 某些乐器更适合高强度表达
        instrument_intensity_multiplier = {
            'violin': 1.2,
            'trumpet': 1.3,
            'piano': 1.0,
            'flute': 0.8,
            'cello': 1.1
        }
        
        multiplier = instrument_intensity_multiplier.get(instrument.lower(), 1.0)
        return min(1.0, intensity * multiplier)
    
    def _get_suitability_recommendation(self, score: float) -> str:
        """获取适配度建议"""
        if score >= 0.8:
            return "非常适合"
        elif score >= 0.6:
            return "适合"
        elif score >= 0.4:
            return "一般"
        else:
            return "不太适合"
    
    def _analyze_current_arrangement(self, tracks: List[TrackData]) -> Dict:
        """分析当前编配"""
        if not tracks:
            return {'track_count': 0}
        
        return {
            'track_count': len(tracks),
            'instruments': [track.instrument for track in tracks],
            'roles': [track.role.value for track in tracks],
            'average_tempo': np.mean([track.tempo for track in tracks if track.tempo]),
            'texture_density': len(tracks) / 8.0  # 归一化密度
        }
    
    def _create_dynamic_curve(self, emotion_type: EmotionType, intensity: float) -> str:
        """创建动态曲线类型"""
        curves = {
            EmotionType.HAPPY: "波浪型",
            EmotionType.SAD: "递减型", 
            EmotionType.CALM: "平稳型",
            EmotionType.ENERGETIC: "波动型",
            EmotionType.MELANCHOLIC: "缓慢递减型",
            EmotionType.ANGRY: "突变型"
        }
        return curves.get(emotion_type, "平稳型")
    
    def _get_instrument_dynamic_factor(self, instrument: str) -> float:
        """获取乐器动态因子"""
        factors = {
            'violin': 1.0,
            'piano': 1.2,
            'flute': 0.8,
            'trumpet': 1.1,
            'cello': 0.9
        }
        return factors.get(instrument.lower(), 1.0)
    
    def _filter_articulations_for_instrument(self, articulations: List[str],
                                           instrument: str) -> List[str]:
        """为特定乐器过滤演奏技法"""
        # 乐器特定的演奏技法
        instrument_articulations = {
            'violin': ['legato', 'staccato', 'pizzicato', 'vibrato', 'sul_ponticello'],
            'piano': ['legato', 'staccato', 'pedal', 'marcato'],
            'flute': ['legato', 'staccato', 'flutter', 'breath_control'],
            'trumpet': ['legato', 'staccato', 'muted', 'forte'],
            'cello': ['legato', 'pizzicato', 'sul_ponticello', 'vibrato']
        }
        
        available = instrument_articulations.get(instrument.lower(), articulations)
        
        # 过滤：只保留该乐器支持的技法
        filtered = [art for art in articulations if art in available]
        
        return filtered if filtered else ['legato']  # 默认连奏


class EmotionInstrumentMap:
    """情感-乐器映射表"""
    
    def __init__(self):
        self.mappings = {}
        
    def load_mappings(self):
        """加载情感-乐器映射"""
        self.mappings = {
            EmotionType.HAPPY: {
                'violin': 0.9,
                'flute': 0.8,
                'piano': 0.7,
                'trumpet': 0.8,
                'guitar': 0.7,
                'cello': 0.5
            },
            EmotionType.SAD: {
                'cello': 0.9,
                'violin': 0.8,
                'piano': 0.8,
                'oboe': 0.9,
                'guitar': 0.6,
                'trumpet': 0.3
            },
            EmotionType.CALM: {
                'flute': 0.9,
                'piano': 0.8,
                'guitar': 0.8,
                'violin': 0.7,
                'cello': 0.7,
                'harp': 0.9
            },
            EmotionType.ENERGETIC: {
                'trumpet': 0.9,
                'violin': 0.8,
                'piano': 0.7,
                'drums': 0.9,
                'electric_guitar': 0.8,
                'saxophone': 0.8
            },
            EmotionType.MELANCHOLIC: {
                'cello': 0.9,
                'violin': 0.8,
                'piano': 0.9,
                'oboe': 0.8,
                'guitar': 0.7,
                'flute': 0.6
            },
            EmotionType.ANGRY: {
                'trumpet': 0.8,
                'drums': 0.9,
                'electric_guitar': 0.9,
                'trombone': 0.8,
                'violin': 0.6,
                'piano': 0.5
            }
        }
        
        logger.info("情感-乐器映射加载完成")
    
    def get_suitability(self, emotion_type: EmotionType, instrument: str) -> float:
        """获取乐器对情感的适配度"""
        emotion_map = self.mappings.get(emotion_type, {})
        return emotion_map.get(instrument.lower(), 0.5)  # 默认中等适配度


class ArrangementRules:
    """编排规则引擎"""
    
    def __init__(self):
        self.rules = {}
        
    def load_rules(self):
        """加载编排规则"""
        self.rules = {
            EmotionType.HAPPY: {
                'style': 'bright_and_lively',
                'texture_density': 'medium_to_high',
                'harmonic_complexity': 'simple_to_medium',
                'rhythmic_activity': 'active',
                'spatial_arrangement': 'wide_spread'
            },
            EmotionType.SAD: {
                'style': 'intimate_and_warm',
                'texture_density': 'low_to_medium',
                'harmonic_complexity': 'simple',
                'rhythmic_activity': 'gentle',
                'spatial_arrangement': 'centered'
            },
            EmotionType.CALM: {
                'style': 'peaceful_and_flowing',
                'texture_density': 'low',
                'harmonic_complexity': 'simple',
                'rhythmic_activity': 'gentle',
                'spatial_arrangement': 'balanced'
            },
            EmotionType.ENERGETIC: {
                'style': 'powerful_and_driving',
                'texture_density': 'high',
                'harmonic_complexity': 'medium_to_high',
                'rhythmic_activity': 'very_active',
                'spatial_arrangement': 'full_range'
            }
        }
        
        logger.info("编排规则加载完成")
    
    def get_style_for_emotion(self, emotion_type: EmotionType,
                            current_analysis: Dict) -> Dict:
        """获取情感对应的编排风格"""
        base_style = self.rules.get(emotion_type, self.rules[EmotionType.CALM])
        
        # 根据当前分析调整
        adjusted_style = base_style.copy()
        
        # 如果当前已经有很多音轨，降低密度建议
        current_density = current_analysis.get('texture_density', 0)
        if current_density > 0.7:
            adjusted_style['texture_density'] = 'low'
        
        return adjusted_style


class ExpressionController:
    """表情控制器"""
    
    def __init__(self):
        self.expression_models = {}
        
    def load_models(self):
        """加载表情模型"""
        self.expression_markings = {
            EmotionType.HAPPY: {
                'low': ['dolce', 'leggiero'],
                'medium': ['vivace', 'allegro'],
                'high': ['brillante', 'giocoso']
            },
            EmotionType.SAD: {
                'low': ['dolce', 'piano'],
                'medium': ['espressivo', 'dolente'],
                'high': ['lamentoso', 'pianissimo']
            },
            EmotionType.CALM: {
                'low': ['dolce', 'tranquillo'],
                'medium': ['cantabile', 'sostenuto'],
                'high': ['sereno', 'placido']
            },
            EmotionType.ENERGETIC: {
                'low': ['marcato', 'forte'],
                'medium': ['energico', 'vivace'],
                'high': ['fortissimo', 'con_fuoco']
            }
        }
        
        logger.info("表情控制模型加载完成")
    
    def get_expression_markings(self, emotion_type: EmotionType,
                              intensity: float, instrument: str) -> List[str]:
        """获取表情记号"""
        # 根据强度确定级别
        if intensity < 0.3:
            level = 'low'
        elif intensity < 0.7:
            level = 'medium'
        else:
            level = 'high'
        
        emotion_markings = self.expression_markings.get(emotion_type, {})
        markings = emotion_markings.get(level, ['espressivo'])
        
        return markings