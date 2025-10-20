"""
音乐结构分析器 - Week 5核心模块
Music Structure Analyzer - Core module for Week 5

实现多层音乐理解：音轨角色识别、结构解析、和声分析
"""

import numpy as np
import librosa
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from enum import Enum
import scipy.signal as signal

class TrackRole(Enum):
    """音轨角色类型"""
    MELODY = "melody"           # 主旋律
    HARMONY = "harmony"         # 和声
    BASS = "bass"              # 低音
    ACCOMPANIMENT = "accompaniment"  # 伴奏
    DECORATION = "decoration"   # 装饰音
    RHYTHM = "rhythm"          # 节奏

class MusicSection(Enum):
    """音乐段落类型"""
    INTRO = "intro"            # 前奏
    VERSE = "verse"            # 主歌
    CHORUS = "chorus"          # 副歌
    BRIDGE = "bridge"          # 桥段
    OUTRO = "outro"            # 尾声
    INSTRUMENTAL = "instrumental"  # 间奏

@dataclass
class StructureSegment:
    """结构段落"""
    section_type: MusicSection
    start_time: float
    end_time: float
    confidence: float
    characteristics: Dict[str, float]

@dataclass
class HarmonyAnalysis:
    """和声分析结果"""
    key_signature: str
    chord_progression: List[str]
    modulation_points: List[float]
    harmonic_rhythm: float
    consonance_score: float

@dataclass
class MusicStructureResult:
    """音乐结构分析结果"""
    track_roles: Dict[int, TrackRole]
    structure_segments: List[StructureSegment]
    harmony_analysis: HarmonyAnalysis
    rhythm_pattern: Dict[str, float]
    overall_form: str

class MusicStructureAnalyzer:
    """
    音乐结构分析器
    
    核心功能：
    1. 音轨角色识别
    2. 音乐结构解析 (前奏、主歌、副歌等)
    3. 和声关系分析
    4. 节奏模式识别
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.structure_model = self._load_structure_model(model_path)
        self.harmony_analyzer = HarmonyAnalyzer()
        self.rhythm_analyzer = RhythmAnalyzer()
        
    def _load_structure_model(self, model_path: Optional[str]) -> nn.Module:
        """加载结构分析模型"""
        if model_path:
            model = torch.load(model_path, map_location=self.device)
        else:
            model = StructureClassifier()
        
        model.to(self.device)
        model.eval()
        return model
    
    def analyze(self, audio_data: np.ndarray, sr: int = 22050) -> MusicStructureResult:
        """
        分析音乐结构
        
        Args:
            audio_data: 音频数据
            sr: 采样率
            
        Returns:
            MusicStructureResult: 音乐结构分析结果
        """
        # 1. 音轨角色识别
        track_roles = self._analyze_track_roles(audio_data, sr)
        
        # 2. 结构段落分析
        structure_segments = self._analyze_structure_segments(audio_data, sr)
        
        # 3. 和声分析
        harmony_analysis = self.harmony_analyzer.analyze(audio_data, sr)
        
        # 4. 节奏模式识别
        rhythm_pattern = self.rhythm_analyzer.analyze(audio_data, sr)
        
        # 5. 整体曲式识别
        overall_form = self._identify_overall_form(structure_segments)
        
        return MusicStructureResult(
            track_roles=track_roles,
            structure_segments=structure_segments,
            harmony_analysis=harmony_analysis,
            rhythm_pattern=rhythm_pattern,
            overall_form=overall_form
        )
    
    def _analyze_track_roles(self, audio_data: np.ndarray, sr: int) -> Dict[int, TrackRole]:
        """分析音轨角色"""
        # 使用音源分离技术识别不同音轨
        roles = {}
        
        # 1. 频率分析确定低音轨
        bass_energy = self._calculate_bass_energy(audio_data, sr)
        if bass_energy > 0.3:
            roles[0] = TrackRole.BASS
        
        # 2. 旋律线检测
        melody_confidence = self._detect_melody_line(audio_data, sr)
        if melody_confidence > 0.5:
            roles[1] = TrackRole.MELODY
        
        # 3. 和声检测
        harmony_presence = self._detect_harmony(audio_data, sr)
        if harmony_presence > 0.4:
            roles[2] = TrackRole.HARMONY
        
        # 4. 节奏检测
        rhythm_strength = self._detect_rhythm_elements(audio_data, sr)
        if rhythm_strength > 0.6:
            roles[3] = TrackRole.RHYTHM
        
        return roles
    
    def _analyze_structure_segments(self, audio_data: np.ndarray, sr: int) -> List[StructureSegment]:
        """分析结构段落"""
        segments = []
        
        # 1. 使用自相似矩阵检测重复结构
        chroma = librosa.feature.chroma(y=audio_data, sr=sr)
        recurrence_matrix = librosa.segment.recurrence_matrix(chroma)
        
        # 2. 检测段落边界
        boundaries = self._detect_section_boundaries(audio_data, sr)
        
        # 3. 为每个段落分类
        for i, boundary in enumerate(boundaries[:-1]):
            start_time = boundary
            end_time = boundaries[i + 1]
            
            # 提取段落特征
            segment_audio = audio_data[int(start_time * sr):int(end_time * sr)]
            section_type, confidence = self._classify_section(segment_audio, sr)
            
            # 分析段落特征
            characteristics = self._analyze_section_characteristics(segment_audio, sr)
            
            segment = StructureSegment(
                section_type=section_type,
                start_time=start_time,
                end_time=end_time,
                confidence=confidence,
                characteristics=characteristics
            )
            segments.append(segment)
        
        return segments
    
    def _detect_section_boundaries(self, audio_data: np.ndarray, sr: int) -> List[float]:
        """检测段落边界"""
        # 使用多种特征检测边界
        hop_length = 512
        
        # 1. 色度特征变化
        chroma = librosa.feature.chroma(y=audio_data, sr=sr, hop_length=hop_length)
        chroma_novelty = librosa.onset.onset_strength(S=chroma, sr=sr, hop_length=hop_length)
        
        # 2. MFCC特征变化
        mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, hop_length=hop_length)
        mfcc_novelty = librosa.onset.onset_strength(S=mfcc, sr=sr, hop_length=hop_length)
        
        # 3. 综合新颖性曲线
        combined_novelty = chroma_novelty + mfcc_novelty
        
        # 4. 峰值检测
        peaks = librosa.onset.onset_detect(
            onset_envelope=combined_novelty,
            sr=sr,
            hop_length=hop_length,
            units='time'
        )
        
        # 添加开始和结束时间
        boundaries = [0.0] + list(peaks) + [len(audio_data) / sr]
        return sorted(set(boundaries))
    
    def _classify_section(self, segment_audio: np.ndarray, sr: int) -> Tuple[MusicSection, float]:
        """分类音乐段落"""
        # 提取段落特征
        features = self._extract_section_features(segment_audio, sr)
        
        # 使用模型预测
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            predictions = self.structure_model(features_tensor)
            probabilities = torch.softmax(predictions, dim=-1)
        
        # 获取最可能的段落类型
        section_idx = torch.argmax(probabilities, dim=-1).item()
        confidence = float(torch.max(probabilities))
        
        sections = list(MusicSection)
        return sections[section_idx], confidence
    
    def _extract_section_features(self, audio_data: np.ndarray, sr: int) -> np.ndarray:
        """提取段落特征"""
        features = []
        
        # 1. 能量特征
        rms = librosa.feature.rms(y=audio_data)
        features.extend([np.mean(rms), np.std(rms)])
        
        # 2. 频谱特征
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sr)
        features.extend([np.mean(spectral_centroids), np.std(spectral_centroids)])
        
        # 3. 色度特征
        chroma = librosa.feature.chroma(y=audio_data, sr=sr)
        features.extend(np.mean(chroma, axis=1))
        
        # 4. 节奏特征
        tempo, beats = librosa.beat.beat_track(y=audio_data, sr=sr)
        features.append(tempo)
        
        return np.array(features)
    
    def _analyze_section_characteristics(self, audio_data: np.ndarray, sr: int) -> Dict[str, float]:
        """分析段落特征"""
        characteristics = {}
        
        # 动态范围
        rms = librosa.feature.rms(y=audio_data)
        characteristics['dynamic_range'] = float(np.max(rms) - np.min(rms))
        
        # 音色亮度
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr)
        characteristics['brightness'] = float(np.mean(spectral_centroid) / sr * 2)
        
        # 节奏强度
        onset_strength = librosa.onset.onset_strength(y=audio_data, sr=sr)
        characteristics['rhythmic_intensity'] = float(np.mean(onset_strength))
        
        # 和声复杂度
        chroma = librosa.feature.chroma(y=audio_data, sr=sr)
        chroma_std = np.std(chroma, axis=1)
        characteristics['harmonic_complexity'] = float(np.mean(chroma_std))
        
        return characteristics
    
    def _calculate_bass_energy(self, audio_data: np.ndarray, sr: int) -> float:
        """计算低音能量占比"""
        # 低通滤波获取低频成分
        nyquist = sr // 2
        low_cutoff = 200  # 200Hz以下为低音
        b, a = signal.butter(4, low_cutoff / nyquist, btype='low')
        bass_signal = signal.filtfilt(b, a, audio_data)
        
        # 计算能量占比
        total_energy = np.sum(audio_data ** 2)
        bass_energy = np.sum(bass_signal ** 2)
        
        return bass_energy / (total_energy + 1e-8)
    
    def _detect_melody_line(self, audio_data: np.ndarray, sr: int) -> float:
        """检测旋律线存在置信度"""
        # 使用基频跟踪检测旋律
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio_data, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr
        )
        
        # 计算旋律线的连续性和稳定性
        melody_confidence = np.mean(voiced_probs[~np.isnan(voiced_probs)])
        return float(melody_confidence)
    
    def _detect_harmony(self, audio_data: np.ndarray, sr: int) -> float:
        """检测和声存在度"""
        # 使用色度特征检测和声
        chroma = librosa.feature.chroma(y=audio_data, sr=sr)
        
        # 计算同时发声的音高数量
        active_pitches = np.sum(chroma > 0.1, axis=0)
        harmony_presence = np.mean(active_pitches > 2)  # 超过2个音高同时发声
        
        return float(harmony_presence)
    
    def _detect_rhythm_elements(self, audio_data: np.ndarray, sr: int) -> float:
        """检测节奏元素强度"""
        # 计算起始检测强度
        onset_strength = librosa.onset.onset_strength(y=audio_data, sr=sr)
        rhythm_strength = np.mean(onset_strength)
        
        return float(rhythm_strength)
    
    def _identify_overall_form(self, segments: List[StructureSegment]) -> str:
        """识别整体曲式"""
        section_sequence = [seg.section_type.value for seg in segments]
        
        # 常见曲式模式匹配
        if self._matches_pattern(section_sequence, ['intro', 'verse', 'chorus', 'verse', 'chorus']):
            return "Verse-Chorus Form"
        elif self._matches_pattern(section_sequence, ['intro', 'verse', 'bridge', 'verse']):
            return "AABA Form"
        elif len([s for s in section_sequence if s == 'verse']) >= 3:
            return "Verse Form"
        else:
            return "Free Form"
    
    def _matches_pattern(self, sequence: List[str], pattern: List[str]) -> bool:
        """检查序列是否匹配模式"""
        if len(sequence) < len(pattern):
            return False
        
        for i in range(len(sequence) - len(pattern) + 1):
            if sequence[i:i+len(pattern)] == pattern:
                return True
        return False


class HarmonyAnalyzer:
    """和声分析器"""
    
    def analyze(self, audio_data: np.ndarray, sr: int) -> HarmonyAnalysis:
        """分析和声结构"""
        # 1. 调性识别
        key_signature = self._detect_key(audio_data, sr)
        
        # 2. 和弦进行识别
        chord_progression = self._detect_chord_progression(audio_data, sr)
        
        # 3. 转调点检测
        modulation_points = self._detect_modulations(audio_data, sr)
        
        # 4. 和声节奏分析
        harmonic_rhythm = self._analyze_harmonic_rhythm(audio_data, sr)
        
        # 5. 协和度评分
        consonance_score = self._calculate_consonance_score(audio_data, sr)
        
        return HarmonyAnalysis(
            key_signature=key_signature,
            chord_progression=chord_progression,
            modulation_points=modulation_points,
            harmonic_rhythm=harmonic_rhythm,
            consonance_score=consonance_score
        )
    
    def _detect_key(self, audio_data: np.ndarray, sr: int) -> str:
        """检测调性"""
        chroma = librosa.feature.chroma(y=audio_data, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        
        # 使用Krumhansl-Schmuckler算法
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        best_correlation = -1
        best_key = 'C major'
        
        for i in range(12):
            # 大调
            major_corr = np.corrcoef(chroma_mean, np.roll(major_profile, i))[0, 1]
            if major_corr > best_correlation:
                best_correlation = major_corr
                best_key = f"{keys[i]} major"
            
            # 小调
            minor_corr = np.corrcoef(chroma_mean, np.roll(minor_profile, i))[0, 1]
            if minor_corr > best_correlation:
                best_correlation = minor_corr
                best_key = f"{keys[i]} minor"
        
        return best_key
    
    def _detect_chord_progression(self, audio_data: np.ndarray, sr: int) -> List[str]:
        """检测和弦进行"""
        # 简化的和弦识别
        hop_length = 2048
        chroma = librosa.feature.chroma(y=audio_data, sr=sr, hop_length=hop_length)
        
        # 基本三和弦模板
        chord_templates = {
            'C': [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
            'Dm': [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
            'Em': [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
            'F': [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
            'G': [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            'Am': [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
        }
        
        progression = []
        for i in range(chroma.shape[1]):
            frame_chroma = chroma[:, i]
            best_chord = 'C'
            best_score = -1
            
            for chord, template in chord_templates.items():
                score = np.dot(frame_chroma, template)
                if score > best_score:
                    best_score = score
                    best_chord = chord
            
            progression.append(best_chord)
        
        # 简化连续相同和弦
        simplified_progression = []
        for chord in progression:
            if not simplified_progression or chord != simplified_progression[-1]:
                simplified_progression.append(chord)
        
        return simplified_progression
    
    def _detect_modulations(self, audio_data: np.ndarray, sr: int) -> List[float]:
        """检测转调点"""
        # 使用滑动窗口检测调性变化
        window_size = sr * 4  # 4秒窗口
        hop_size = sr * 1     # 1秒步长
        
        modulation_points = []
        prev_key = None
        
        for start in range(0, len(audio_data) - window_size, hop_size):
            window_audio = audio_data[start:start + window_size]
            current_key = self._detect_key(window_audio, sr)
            
            if prev_key and current_key != prev_key:
                modulation_points.append(start / sr)
            
            prev_key = current_key
        
        return modulation_points
    
    def _analyze_harmonic_rhythm(self, audio_data: np.ndarray, sr: int) -> float:
        """分析和声节奏"""
        chord_progression = self._detect_chord_progression(audio_data, sr)
        
        # 计算和弦变化频率
        total_time = len(audio_data) / sr
        chord_changes = len(chord_progression)
        
        harmonic_rhythm = chord_changes / total_time  # 每秒和弦变化次数
        return float(harmonic_rhythm)
    
    def _calculate_consonance_score(self, audio_data: np.ndarray, sr: int) -> float:
        """计算协和度评分"""
        chroma = librosa.feature.chroma(y=audio_data, sr=sr)
        
        # 简化的协和度计算：基于同时发声音高的协和关系
        consonance_scores = []
        
        for i in range(chroma.shape[1]):
            frame_chroma = chroma[:, i]
            active_pitches = np.where(frame_chroma > 0.1)[0]
            
            if len(active_pitches) >= 2:
                # 计算音程关系的协和度
                intervals = []
                for j in range(len(active_pitches)):
                    for k in range(j + 1, len(active_pitches)):
                        interval = (active_pitches[k] - active_pitches[j]) % 12
                        intervals.append(interval)
                
                # 协和音程权重 (简化版本)
                consonance_weights = {0: 1.0, 7: 0.9, 5: 0.8, 4: 0.7, 3: 0.6, 8: 0.6, 9: 0.5}
                frame_consonance = np.mean([consonance_weights.get(interval, 0.3) for interval in intervals])
                consonance_scores.append(frame_consonance)
        
        return float(np.mean(consonance_scores)) if consonance_scores else 0.5


class RhythmAnalyzer:
    """节奏分析器"""
    
    def analyze(self, audio_data: np.ndarray, sr: int) -> Dict[str, float]:
        """分析节奏模式"""
        # 1. 节拍检测
        tempo, beats = librosa.beat.beat_track(y=audio_data, sr=sr)
        
        # 2. 拍子检测
        time_signature = self._detect_time_signature(audio_data, sr, beats)
        
        # 3. 节奏复杂度
        rhythmic_complexity = self._calculate_rhythmic_complexity(audio_data, sr)
        
        # 4. 同步性分析
        beat_consistency = self._analyze_beat_consistency(beats)
        
        return {
            'tempo': float(tempo),
            'time_signature': time_signature,
            'rhythmic_complexity': rhythmic_complexity,
            'beat_consistency': beat_consistency
        }
    
    def _detect_time_signature(self, audio_data: np.ndarray, sr: int, beats: np.ndarray) -> float:
        """检测拍子"""
        if len(beats) < 8:
            return 4.0  # 默认4/4拍
        
        # 分析重音模式
        beat_intervals = np.diff(beats)
        median_interval = np.median(beat_intervals)
        
        # 检测强拍模式
        onset_strength = librosa.onset.onset_strength(y=audio_data, sr=sr)
        onset_beats = librosa.frames_to_time(
            np.arange(len(onset_strength)), sr=sr, hop_length=512
        )
        
        # 简化的拍子识别
        if median_interval > 0.8:  # 慢速，可能是2/4或3/4
            return 3.0
        else:  # 快速，可能是4/4
            return 4.0
    
    def _calculate_rhythmic_complexity(self, audio_data: np.ndarray, sr: int) -> float:
        """计算节奏复杂度"""
        onset_strength = librosa.onset.onset_strength(y=audio_data, sr=sr)
        
        # 使用起始强度的变异系数作为复杂度指标
        if np.mean(onset_strength) == 0:
            return 0.0
        
        complexity = np.std(onset_strength) / np.mean(onset_strength)
        return float(min(complexity, 1.0))  # 限制在[0,1]范围
    
    def _analyze_beat_consistency(self, beats: np.ndarray) -> float:
        """分析节拍一致性"""
        if len(beats) < 3:
            return 0.0
        
        beat_intervals = np.diff(beats)
        consistency = 1.0 - (np.std(beat_intervals) / (np.mean(beat_intervals) + 1e-8))
        return float(max(0.0, consistency))


class StructureClassifier(nn.Module):
    """结构分类模型"""
    
    def __init__(self, input_dim: int = 16, num_sections: int = 6):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(32, 16),
            nn.ReLU(),
            
            nn.Linear(16, num_sections)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


# 使用示例
if __name__ == "__main__":
    analyzer = MusicStructureAnalyzer()
    
    # 测试
    try:
        audio_data, sr = librosa.load("test_audio.wav", sr=22050)
        result = analyzer.analyze(audio_data, sr)
        
        print("音乐结构分析结果:")
        print(f"音轨角色: {result.track_roles}")
        print(f"整体曲式: {result.overall_form}")
        print(f"调性: {result.harmony_analysis.key_signature}")
        print(f"节奏: {result.rhythm_pattern}")
        
    except Exception as e:
        print(f"需要音频文件进行测试: {e}")
        
        # 生成测试数据
        test_audio = np.random.randn(22050 * 10)  # 10秒测试音频
        result = analyzer.analyze(test_audio, 22050)
        print(f"随机音频测试完成 - 曲式: {result.overall_form}")