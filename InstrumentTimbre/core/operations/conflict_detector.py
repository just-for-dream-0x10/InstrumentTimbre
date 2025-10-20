"""
实时冲突检测器 - 操作过程中的音乐冲突监测机制
"""

import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
from .base_engine import BaseConflictDetector
from .data_structures import (
    TrackData, ConflictReport, ConflictType
)

logger = logging.getLogger(__name__)


class RealTimeConflictDetector(BaseConflictDetector):
    """实时冲突检测器"""
    
    def __init__(self):
        super().__init__("RealTimeConflictDetector")
        self.harmony_checker = HarmonyChecker()
        self.rhythm_checker = RhythmChecker()
        self.style_checker = StyleChecker()
        
    def initialize(self) -> bool:
        """初始化冲突检测器"""
        try:
            self.harmony_checker.load_rules()
            self.rhythm_checker.load_patterns()
            self.style_checker.load_models()
            
            self.is_initialized = True
            logger.info("实时冲突检测器初始化成功")
            return True
            
        except Exception as e:
            logger.error(f"冲突检测器初始化失败: {e}")
            return False
    
    def detect_conflicts(self, existing_tracks: List[TrackData],
                        new_track: TrackData) -> List[ConflictReport]:
        """
        检测新音轨与现有音轨的冲突
        
        Args:
            existing_tracks: 现有音轨列表
            new_track: 新音轨
            
        Returns:
            冲突报告列表
        """
        if not self.is_initialized:
            self.initialize()
        
        # 验证输入
        errors = self.validate_conflict_detection_inputs(existing_tracks, new_track)
        if errors:
            logger.warning(f"冲突检测输入验证失败: {errors}")
            return []
        
        conflicts = []
        
        # 检测各种类型的冲突
        conflicts.extend(self._detect_harmonic_conflicts(existing_tracks, new_track))
        conflicts.extend(self._detect_rhythmic_conflicts(existing_tracks, new_track))
        conflicts.extend(self._detect_stylistic_conflicts(existing_tracks, new_track))
        conflicts.extend(self._detect_dynamic_conflicts(existing_tracks, new_track))
        
        # 按严重程度排序
        conflicts.sort(key=lambda x: x.severity, reverse=True)
        
        logger.info(f"检测到 {len(conflicts)} 个冲突")
        return conflicts
    
    def _detect_harmonic_conflicts(self, existing_tracks: List[TrackData],
                                  new_track: TrackData) -> List[ConflictReport]:
        """检测和声冲突"""
        conflicts = []
        
        for existing_track in existing_tracks:
            harmony_conflicts = self.harmony_checker.check_harmony_conflict(
                existing_track, new_track
            )
            conflicts.extend(harmony_conflicts)
        
        return conflicts
    
    def _detect_rhythmic_conflicts(self, existing_tracks: List[TrackData],
                                  new_track: TrackData) -> List[ConflictReport]:
        """检测节奏冲突"""
        conflicts = []
        
        for existing_track in existing_tracks:
            rhythm_conflicts = self.rhythm_checker.check_rhythm_conflict(
                existing_track, new_track
            )
            conflicts.extend(rhythm_conflicts)
        
        return conflicts
    
    def _detect_stylistic_conflicts(self, existing_tracks: List[TrackData],
                                   new_track: TrackData) -> List[ConflictReport]:
        """检测风格冲突"""
        conflicts = []
        
        # 检查整体风格一致性
        style_conflicts = self.style_checker.check_style_consistency(
            existing_tracks, new_track
        )
        conflicts.extend(style_conflicts)
        
        return conflicts
    
    def _detect_dynamic_conflicts(self, existing_tracks: List[TrackData],
                                 new_track: TrackData) -> List[ConflictReport]:
        """检测动态冲突"""
        conflicts = []
        
        # 检查动态范围冲突
        for existing_track in existing_tracks:
            if existing_track.dynamics and new_track.dynamics:
                dynamic_conflict = self._check_dynamic_range_conflict(
                    existing_track, new_track
                )
                if dynamic_conflict:
                    conflicts.append(dynamic_conflict)
        
        return conflicts
    
    def _check_dynamic_range_conflict(self, track1: TrackData, track2: TrackData) -> Optional[ConflictReport]:
        """检查动态范围冲突"""
        if not track1.dynamics or not track2.dynamics:
            return None
        
        # 计算平均动态
        avg_dynamic1 = np.mean(track1.dynamics)
        avg_dynamic2 = np.mean(track2.dynamics)
        
        # 如果两个音轨的动态范围差异过大
        dynamic_diff = abs(avg_dynamic1 - avg_dynamic2)
        
        if dynamic_diff > 0.4:  # 阈值
            severity = min(dynamic_diff, 1.0)
            return ConflictReport(
                conflict_type=ConflictType.DYNAMIC,
                severity=severity,
                description=f"动态范围冲突: {track1.instrument} ({avg_dynamic1:.2f}) vs {track2.instrument} ({avg_dynamic2:.2f})",
                location=(0.0, min(track1.duration, track2.duration)),
                affected_tracks=[track1.track_id, track2.track_id],
                suggested_fixes=["调整动态平衡", "应用动态压缩"],
                auto_fixable=True
            )
        
        return None


class HarmonyChecker:
    """和声检查器"""
    
    def __init__(self):
        self.dissonance_rules = {}
        self.interval_rules = {}
        
    def load_rules(self):
        """加载和声规则"""
        # 不协和音程定义
        self.dissonant_intervals = {
            1,    # 小二度
            6,    # 三全音
            10,   # 小七度
            11    # 大七度
        }
        
        # 协和音程定义
        self.consonant_intervals = {
            0,    # 同度
            3,    # 小三度
            4,    # 大三度
            7,    # 纯五度
            8,    # 小六度
            9,    # 大六度
            12    # 八度
        }
        
        logger.info("和声规则加载完成")
    
    def check_harmony_conflict(self, track1: TrackData, track2: TrackData) -> List[ConflictReport]:
        """检查两个音轨间的和声冲突"""
        conflicts = []
        
        if not track1.pitch_sequence or not track2.pitch_sequence:
            return conflicts
        
        # 检查同时发声的音符间的音程关系
        conflicts.extend(self._check_interval_conflicts(track1, track2))
        
        # 检查调性冲突
        key_conflict = self._check_key_conflict(track1, track2)
        if key_conflict:
            conflicts.append(key_conflict)
        
        return conflicts
    
    def _check_interval_conflicts(self, track1: TrackData, track2: TrackData) -> List[ConflictReport]:
        """检查音程冲突"""
        conflicts = []
        
        # 简化处理：比较音高序列的对应位置
        min_len = min(len(track1.pitch_sequence), len(track2.pitch_sequence))
        
        dissonant_count = 0
        total_intervals = 0
        
        for i in range(min_len):
            pitch1 = track1.pitch_sequence[i]
            pitch2 = track2.pitch_sequence[i]
            
            # 计算音程（半音数）
            interval = abs(pitch1 - pitch2) % 12
            total_intervals += 1
            
            if interval in self.dissonant_intervals:
                dissonant_count += 1
        
        # 如果不协和音程比例过高
        if total_intervals > 0:
            dissonance_ratio = dissonant_count / total_intervals
            
            if dissonance_ratio > 0.3:  # 阈值：30%
                severity = min(dissonance_ratio, 1.0)
                conflicts.append(ConflictReport(
                    conflict_type=ConflictType.HARMONIC,
                    severity=severity,
                    description=f"不协和音程过多: {dissonant_count}/{total_intervals} ({dissonance_ratio:.1%})",
                    location=(0.0, min(track1.duration, track2.duration)),
                    affected_tracks=[track1.track_id, track2.track_id],
                    suggested_fixes=["调整音高", "错开发声时间", "改变音轨角色"],
                    auto_fixable=True
                ))
        
        return conflicts
    
    def _check_key_conflict(self, track1: TrackData, track2: TrackData) -> Optional[ConflictReport]:
        """检查调性冲突"""
        if track1.key and track2.key and track1.key != track2.key:
            # 检查调性是否相关
            if not self._are_keys_related(track1.key, track2.key):
                return ConflictReport(
                    conflict_type=ConflictType.HARMONIC,
                    severity=0.8,
                    description=f"调性冲突: {track1.key} vs {track2.key}",
                    location=(0.0, min(track1.duration, track2.duration)),
                    affected_tracks=[track1.track_id, track2.track_id],
                    suggested_fixes=["统一调性", "使用调性转换"],
                    auto_fixable=False
                )
        
        return None
    
    def _are_keys_related(self, key1: str, key2: str) -> bool:
        """检查两个调性是否相关"""
        # 简化的调性关系检查
        related_keys = {
            'C_major': ['A_minor', 'G_major', 'F_major'],
            'G_major': ['E_minor', 'C_major', 'D_major'],
            'F_major': ['D_minor', 'C_major', 'Bb_major'],
            # 可以扩展更多调性关系
        }
        
        return key2 in related_keys.get(key1, [])


class RhythmChecker:
    """节奏检查器"""
    
    def __init__(self):
        self.rhythm_patterns = {}
        
    def load_patterns(self):
        """加载节奏模式"""
        # 常见的节奏冲突模式
        self.conflicting_patterns = [
            # 强拍冲突：两个音轨都在同一强拍上有重音
            'strong_beat_collision',
            # 节奏密度冲突：节奏过于密集
            'rhythm_density_conflict',
            # 不规则节奏冲突
            'irregular_rhythm_conflict'
        ]
        
        logger.info("节奏模式加载完成")
    
    def check_rhythm_conflict(self, track1: TrackData, track2: TrackData) -> List[ConflictReport]:
        """检查节奏冲突"""
        conflicts = []
        
        if not track1.rhythm_pattern or not track2.rhythm_pattern:
            return conflicts
        
        # 检查节奏密度冲突
        density_conflict = self._check_rhythm_density_conflict(track1, track2)
        if density_conflict:
            conflicts.append(density_conflict)
        
        # 检查节拍重叠冲突
        overlap_conflict = self._check_rhythm_overlap_conflict(track1, track2)
        if overlap_conflict:
            conflicts.append(overlap_conflict)
        
        return conflicts
    
    def _check_rhythm_density_conflict(self, track1: TrackData, track2: TrackData) -> Optional[ConflictReport]:
        """检查节奏密度冲突"""
        # 计算节奏密度（每秒音符数）
        density1 = len(track1.rhythm_pattern) / track1.duration if track1.duration > 0 else 0
        density2 = len(track2.rhythm_pattern) / track2.duration if track2.duration > 0 else 0
        
        total_density = density1 + density2
        
        # 如果总密度过高
        if total_density > 10:  # 阈值：每秒10个音符
            severity = min((total_density - 10) / 10, 1.0)
            return ConflictReport(
                conflict_type=ConflictType.RHYTHMIC,
                severity=severity,
                description=f"节奏密度过高: {total_density:.1f} 音符/秒",
                location=(0.0, min(track1.duration, track2.duration)),
                affected_tracks=[track1.track_id, track2.track_id],
                suggested_fixes=["减少音符密度", "错开节奏", "简化节奏模式"],
                auto_fixable=True
            )
        
        return None
    
    def _check_rhythm_overlap_conflict(self, track1: TrackData, track2: TrackData) -> Optional[ConflictReport]:
        """检查节奏重叠冲突"""
        # 简化的重叠检查：计算同时发声的比例
        min_len = min(len(track1.rhythm_pattern), len(track2.rhythm_pattern))
        
        if min_len == 0:
            return None
        
        # 模拟计算重叠比例（简化实现）
        overlap_count = 0
        for i in range(min_len):
            # 假设如果两个节奏值都较短，则可能重叠
            if track1.rhythm_pattern[i] < 0.5 and track2.rhythm_pattern[i] < 0.5:
                overlap_count += 1
        
        overlap_ratio = overlap_count / min_len
        
        if overlap_ratio > 0.7:  # 70%以上重叠
            return ConflictReport(
                conflict_type=ConflictType.RHYTHMIC,
                severity=overlap_ratio,
                description=f"节奏重叠过多: {overlap_ratio:.1%}",
                location=(0.0, min(track1.duration, track2.duration)),
                affected_tracks=[track1.track_id, track2.track_id],
                suggested_fixes=["错开节奏时间", "改变节奏模式", "调整音轨角色"],
                auto_fixable=True
            )
        
        return None


class StyleChecker:
    """风格检查器"""
    
    def __init__(self):
        self.style_models = {}
        
    def load_models(self):
        """加载风格检查模型"""
        # 风格特征定义
        self.style_features = {
            'classical': {
                'tempo_range': (60, 120),
                'preferred_instruments': ['violin', 'piano', 'cello', 'flute'],
                'harmony_complexity': 'high',
                'rhythm_regularity': 'high'
            },
            'jazz': {
                'tempo_range': (100, 180),
                'preferred_instruments': ['saxophone', 'piano', 'bass', 'drums'],
                'harmony_complexity': 'very_high',
                'rhythm_regularity': 'medium'
            },
            'pop': {
                'tempo_range': (80, 140),
                'preferred_instruments': ['guitar', 'piano', 'bass', 'drums'],
                'harmony_complexity': 'low',
                'rhythm_regularity': 'high'
            }
        }
        
        logger.info("风格检查模型加载完成")
    
    def check_style_consistency(self, existing_tracks: List[TrackData],
                               new_track: TrackData) -> List[ConflictReport]:
        """检查风格一致性"""
        conflicts = []
        
        if not existing_tracks:
            return conflicts
        
        # 分析现有音轨的风格
        existing_style = self._analyze_style(existing_tracks)
        
        # 分析新音轨的风格
        new_style = self._analyze_style([new_track])
        
        # 检查风格一致性
        style_conflict = self._check_style_conflict(existing_style, new_style, new_track)
        if style_conflict:
            conflicts.append(style_conflict)
        
        return conflicts
    
    def _analyze_style(self, tracks: List[TrackData]) -> Dict:
        """分析音轨风格"""
        if not tracks:
            return {}
        
        # 计算平均tempo
        tempos = [track.tempo for track in tracks if track.tempo]
        avg_tempo = np.mean(tempos) if tempos else 120
        
        # 收集乐器
        instruments = [track.instrument for track in tracks]
        
        # 简单的风格匹配
        style_scores = {}
        for style, features in self.style_features.items():
            score = 0
            
            # Tempo匹配度
            tempo_min, tempo_max = features['tempo_range']
            if tempo_min <= avg_tempo <= tempo_max:
                score += 0.4
            
            # 乐器匹配度
            instrument_matches = sum(1 for inst in instruments 
                                   if inst in features['preferred_instruments'])
            instrument_score = instrument_matches / len(instruments) if instruments else 0
            score += 0.6 * instrument_score
            
            style_scores[style] = score
        
        # 返回最匹配的风格
        best_style = max(style_scores, key=style_scores.get) if style_scores else 'unknown'
        
        return {
            'style': best_style,
            'confidence': style_scores.get(best_style, 0),
            'avg_tempo': avg_tempo,
            'instruments': instruments
        }
    
    def _check_style_conflict(self, existing_style: Dict, new_style: Dict,
                             new_track: TrackData) -> Optional[ConflictReport]:
        """检查风格冲突"""
        if not existing_style or not new_style:
            return None
        
        existing_style_name = existing_style.get('style', 'unknown')
        new_style_name = new_style.get('style', 'unknown')
        
        # 如果风格不同且置信度都较高
        existing_confidence = existing_style.get('confidence', 0)
        new_confidence = new_style.get('confidence', 0)
        
        if (existing_style_name != new_style_name and 
            existing_confidence > 0.6 and new_confidence > 0.6):
            
            severity = (existing_confidence + new_confidence) / 2
            
            return ConflictReport(
                conflict_type=ConflictType.STYLISTIC,
                severity=severity,
                description=f"风格冲突: 现有风格({existing_style_name}) vs 新音轨风格({new_style_name})",
                location=(0.0, new_track.duration),
                affected_tracks=[new_track.track_id],
                suggested_fixes=["调整乐器选择", "修改节奏模式", "改变和声进行"],
                auto_fixable=False
            )
        
        return None