"""
音轨操作相关的核心数据结构定义
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union, Tuple
from enum import Enum
import numpy as np


class OperationType(Enum):
    """操作类型枚举"""
    GENERATE = "generate"
    REPLACE = "replace"
    REPAIR = "repair"
    MODIFY = "modify"


class TrackRole(Enum):
    """音轨角色枚举"""
    MELODY = "melody"           # 主旋律
    HARMONY = "harmony"         # 和声
    BASS = "bass"              # 低音
    RHYTHM = "rhythm"          # 节奏
    ACCOMPANIMENT = "accompaniment"  # 伴奏
    LEAD = "lead"              # 主音
    PAD = "pad"                # 垫音
    PERCUSSION = "percussion"   # 打击乐


class EmotionType(Enum):
    """情感类型枚举"""
    HAPPY = "happy"
    SAD = "sad"
    CALM = "calm"
    ENERGETIC = "energetic"
    MELANCHOLIC = "melancholic"
    ANGRY = "angry"


class ConflictType(Enum):
    """冲突类型枚举"""
    HARMONIC = "harmonic"      # 和声冲突
    RHYTHMIC = "rhythmic"      # 节奏冲突
    STYLISTIC = "stylistic"    # 风格冲突
    DYNAMIC = "dynamic"        # 动态冲突


@dataclass
class EmotionConstraints:
    """情感约束数据结构"""
    primary_emotion: EmotionType
    intensity: float  # 0-1
    secondary_emotions: Dict[EmotionType, float] = field(default_factory=dict)
    
    # 约束规则
    tempo_range: Tuple[int, int] = (60, 120)  # BPM范围
    instrument_preferences: List[str] = field(default_factory=list)
    harmonic_preferences: List[str] = field(default_factory=list)
    dynamic_range: Tuple[float, float] = (0.3, 0.8)  # 动态范围
    
    def __post_init__(self):
        """验证约束有效性"""
        if not 0 <= self.intensity <= 1:
            raise ValueError("情感强度必须在0-1之间")
        if self.tempo_range[0] >= self.tempo_range[1]:
            raise ValueError("tempo范围无效")


@dataclass
class MusicConstraints:
    """音乐约束数据结构"""
    key: str  # 调性，如 "C_major"
    time_signature: str  # 拍号，如 "4/4"
    tempo: int  # 速度 BPM
    
    # 结构约束
    track_roles: Dict[str, TrackRole] = field(default_factory=dict)
    importance_weights: Dict[str, float] = field(default_factory=dict)
    
    # 和声约束
    chord_progressions: List[str] = field(default_factory=list)
    forbidden_intervals: List[str] = field(default_factory=list)
    
    # 节奏约束
    rhythm_patterns: List[str] = field(default_factory=list)
    syncopation_level: float = 0.0  # 切分音程度
    
    def __post_init__(self):
        """验证约束有效性"""
        if self.tempo < 40 or self.tempo > 200:
            raise ValueError("tempo超出合理范围")


@dataclass
class TrackData:
    """音轨数据结构"""
    track_id: str
    instrument: str
    role: TrackRole
    
    # 音频数据
    audio_data: Optional[np.ndarray] = None
    midi_data: Optional[Dict] = None
    
    # 分析结果
    pitch_sequence: List[float] = field(default_factory=list)
    rhythm_pattern: List[float] = field(default_factory=list)
    dynamics: List[float] = field(default_factory=list)
    
    # 元数据
    duration: float = 0.0  # 秒
    sample_rate: int = 22050
    key: str = "C_major"
    tempo: int = 120
    
    def is_valid(self) -> bool:
        """检查音轨数据是否有效"""
        if self.audio_data is None and self.midi_data is None:
            return False
        if self.duration <= 0:
            return False
        return True


@dataclass
class ConflictReport:
    """冲突报告数据结构"""
    conflict_type: ConflictType
    severity: float  # 0-1，严重程度
    description: str
    location: Tuple[float, float]  # 时间位置 (start, end)
    affected_tracks: List[str]
    
    # 解决建议
    suggested_fixes: List[str] = field(default_factory=list)
    auto_fixable: bool = False
    
    def __str__(self) -> str:
        return f"{self.conflict_type.value}冲突 (严重程度: {self.severity:.2f}): {self.description}"


@dataclass
class OperationRequest:
    """操作请求数据结构"""
    operation_type: OperationType
    target_instrument: str
    target_role: TrackRole
    
    # 约束
    emotion_constraints: Optional[EmotionConstraints] = None
    music_constraints: Optional[MusicConstraints] = None
    
    # 操作参数
    intensity: float = 0.7  # 操作强度
    preserve_original: bool = True  # 是否保留原始特征
    reference_track: Optional[str] = None  # 参考音轨ID
    
    # 用户偏好
    style_preference: str = "auto"  # 风格偏好
    complexity_level: str = "medium"  # 复杂度水平
    
    def validate(self) -> bool:
        """验证请求是否有效"""
        if not 0 <= self.intensity <= 1:
            return False
        if self.complexity_level not in ["simple", "medium", "complex"]:
            return False
        return True


@dataclass
class OperationResult:
    """操作结果数据结构"""
    success: bool
    generated_track: Optional[TrackData] = None
    
    # 质量指标
    quality_score: float = 0.0  # 整体质量评分
    emotion_consistency: float = 0.0  # 情感一致性
    harmonic_correctness: float = 0.0  # 和声正确性
    
    # 冲突检测结果
    conflicts: List[ConflictReport] = field(default_factory=list)
    
    # 处理信息
    processing_time: float = 0.0  # 处理时间(秒)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    
    def has_critical_conflicts(self) -> bool:
        """是否有严重冲突"""
        return any(conflict.severity > 0.7 for conflict in self.conflicts)
    
    def get_overall_score(self) -> float:
        """获取综合评分"""
        if not self.success:
            return 0.0
        
        base_score = (self.quality_score + self.emotion_consistency + self.harmonic_correctness) / 3
        conflict_penalty = sum(conflict.severity for conflict in self.conflicts) * 0.1
        
        return max(0.0, base_score - conflict_penalty)


@dataclass
class GenerationConfig:
    """生成配置数据结构"""
    # 模型参数
    model_name: str = "default"
    temperature: float = 0.8  # 生成随机性
    max_length: int = 512  # 最大生成长度
    
    # 约束权重
    emotion_weight: float = 0.4
    harmony_weight: float = 0.3
    rhythm_weight: float = 0.2
    style_weight: float = 0.1
    
    # 生成策略
    use_beam_search: bool = True
    beam_size: int = 5
    repetition_penalty: float = 1.1
    
    # 质量控制
    min_quality_threshold: float = 0.7
    max_generation_attempts: int = 3
    
    def validate(self) -> bool:
        """验证配置有效性"""
        weights = [self.emotion_weight, self.harmony_weight, 
                  self.rhythm_weight, self.style_weight]
        if abs(sum(weights) - 1.0) > 0.01:
            return False
        return all(0 <= w <= 1 for w in weights)


# 工具函数
def create_empty_track(track_id: str, instrument: str, role: TrackRole) -> TrackData:
    """创建空白音轨"""
    return TrackData(
        track_id=track_id,
        instrument=instrument,
        role=role
    )


def merge_constraints(emotion_constraints: EmotionConstraints, 
                     music_constraints: MusicConstraints) -> Dict:
    """合并情感和音乐约束"""
    merged = {
        'emotion': emotion_constraints,
        'music': music_constraints,
        'tempo': min(emotion_constraints.tempo_range[1], music_constraints.tempo),
        'key': music_constraints.key,
        'intensity': emotion_constraints.intensity
    }
    return merged