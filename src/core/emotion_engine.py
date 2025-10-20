"""
情感分析引擎 - Week 5核心模块
Emotion Analysis Engine - Core module for Week 5

实现6类情感识别与强度量化，为音乐编辑提供情感约束
"""

import numpy as np
import torch
import torch.nn as nn
import librosa
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class EmotionType(Enum):
    """情感类型枚举"""
    HAPPY = "happy"
    SAD = "sad"
    CALM = "calm"
    EXCITED = "excited"
    MELANCHOLY = "melancholy"
    ANGRY = "angry"

@dataclass
class EmotionResult:
    """情感分析结果"""
    primary_emotion: EmotionType
    emotion_scores: Dict[EmotionType, float]
    intensity: float
    confidence: float
    temporal_trajectory: Optional[np.ndarray] = None

class EmotionAnalysisEngine:
    """
    情感分析引擎
    
    核心功能：
    1. 6类情感识别 (快乐、悲伤、平静、激动、忧郁、愤怒)
    2. 情感强度量化 (0-1连续值)
    3. 时序情感轨迹追踪
    4. 情感约束生成
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        self.feature_extractor = AudioFeatureExtractor()
        
    def _load_model(self, model_path: Optional[str]) -> nn.Module:
        """加载情感分析模型"""
        if model_path:
            # 加载预训练模型
            model = torch.load(model_path, map_location=self.device)
        else:
            # 创建默认模型架构
            model = EmotionClassifier()
        
        model.to(self.device)
        model.eval()
        return model
    
    def analyze(self, audio_data: np.ndarray, sr: int = 22050) -> EmotionResult:
        """
        分析音频的情感特征
        
        Args:
            audio_data: 音频数据
            sr: 采样率
            
        Returns:
            EmotionResult: 情感分析结果
        """
        # 1. 音频特征提取
        features = self.feature_extractor.extract(audio_data, sr)
        
        # 2. 情感识别
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            emotion_logits = self.model(features_tensor)
            emotion_probs = torch.softmax(emotion_logits, dim=-1)
        
        # 3. 结果解析
        emotion_scores = self._parse_emotion_scores(emotion_probs)
        primary_emotion = max(emotion_scores, key=emotion_scores.get)
        intensity = self._calculate_intensity(emotion_probs)
        confidence = float(torch.max(emotion_probs))
        
        # 4. 时序轨迹分析
        temporal_trajectory = self._analyze_temporal_trajectory(audio_data, sr)
        
        return EmotionResult(
            primary_emotion=primary_emotion,
            emotion_scores=emotion_scores,
            intensity=intensity,
            confidence=confidence,
            temporal_trajectory=temporal_trajectory
        )
    
    def _parse_emotion_scores(self, emotion_probs: torch.Tensor) -> Dict[EmotionType, float]:
        """解析情感得分"""
        emotion_list = list(EmotionType)
        scores = {}
        probs = emotion_probs.cpu().numpy().flatten()
        
        for i, emotion in enumerate(emotion_list):
            scores[emotion] = float(probs[i])
            
        return scores
    
    def _calculate_intensity(self, emotion_probs: torch.Tensor) -> float:
        """计算情感强度"""
        # 使用熵来计算情感强度
        probs = emotion_probs.cpu().numpy().flatten()
        entropy = -np.sum(probs * np.log(probs + 1e-8))
        max_entropy = np.log(len(probs))
        
        # 强度 = 1 - 标准化熵 (越集中强度越高)
        intensity = 1.0 - (entropy / max_entropy)
        return float(intensity)
    
    def _analyze_temporal_trajectory(self, audio_data: np.ndarray, sr: int) -> np.ndarray:
        """分析时序情感轨迹"""
        # 将音频分段分析
        segment_length = sr * 2  # 2秒一段
        hop_length = sr // 2     # 0.5秒步长
        
        trajectory = []
        for start in range(0, len(audio_data) - segment_length, hop_length):
            segment = audio_data[start:start + segment_length]
            
            # 简化的情感分析 (快速版本)
            features = self.feature_extractor.extract_basic(segment, sr)
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
                emotion_logits = self.model(features_tensor)
                emotion_probs = torch.softmax(emotion_logits, dim=-1)
                intensity = self._calculate_intensity(emotion_probs)
                trajectory.append(intensity)
        
        return np.array(trajectory)

    def generate_emotion_constraint(self, emotion_result: EmotionResult) -> Dict:
        """
        生成情感约束条件
        用于指导后续的音乐编辑操作
        """
        constraint = {
            "target_emotion": emotion_result.primary_emotion.value,
            "intensity_range": [
                max(0.0, emotion_result.intensity - 0.1),
                min(1.0, emotion_result.intensity + 0.1)
            ],
            "preserve_emotions": [
                emotion.value for emotion, score in emotion_result.emotion_scores.items()
                if score > 0.2  # 保持强度超过0.2的情感
            ],
            "avoid_emotions": [
                emotion.value for emotion, score in emotion_result.emotion_scores.items()
                if score < 0.05  # 避免强度低于0.05的情感
            ],
            "temporal_stability": np.std(emotion_result.temporal_trajectory) < 0.3
        }
        
        return constraint


class AudioFeatureExtractor:
    """音频特征提取器"""
    
    def extract(self, audio_data: np.ndarray, sr: int) -> np.ndarray:
        """提取完整的音频特征"""
        features = []
        
        # 1. MFCC特征
        mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
        features.append(np.mean(mfcc, axis=1))
        features.append(np.std(mfcc, axis=1))
        
        # 2. 色度特征
        chroma = librosa.feature.chroma(y=audio_data, sr=sr)
        features.append(np.mean(chroma, axis=1))
        
        # 3. 频谱对比度
        contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sr)
        features.append(np.mean(contrast, axis=1))
        
        # 4. 零交叉率
        zcr = librosa.feature.zero_crossing_rate(audio_data)
        features.append([np.mean(zcr)])
        
        # 5. 频谱质心
        centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr)
        features.append([np.mean(centroid)])
        
        # 6. 频谱滚降
        rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)
        features.append([np.mean(rolloff)])
        
        return np.concatenate(features)
    
    def extract_basic(self, audio_data: np.ndarray, sr: int) -> np.ndarray:
        """提取基础特征 (快速版本)"""
        # 仅提取关键特征用于实时分析
        mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=8)
        chroma = librosa.feature.chroma(y=audio_data, sr=sr)
        
        features = np.concatenate([
            np.mean(mfcc, axis=1),
            np.mean(chroma, axis=1)
        ])
        
        return features


class EmotionClassifier(nn.Module):
    """情感分类模型"""
    
    def __init__(self, input_dim: int = 48, num_emotions: int = 6):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            
            nn.Linear(32, num_emotions)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


# 使用示例和测试
if __name__ == "__main__":
    # 创建情感分析引擎
    emotion_engine = EmotionAnalysisEngine()
    
    # 加载测试音频
    audio_file = "test_audio.wav"  # 替换为实际音频文件
    try:
        audio_data, sr = librosa.load(audio_file, sr=22050)
        
        # 分析情感
        result = emotion_engine.analyze(audio_data, sr)
        
        print(f"主要情感: {result.primary_emotion.value}")
        print(f"情感强度: {result.intensity:.3f}")
        print(f"置信度: {result.confidence:.3f}")
        print("\n各情感得分:")
        for emotion, score in result.emotion_scores.items():
            print(f"  {emotion.value}: {score:.3f}")
        
        # 生成情感约束
        constraint = emotion_engine.generate_emotion_constraint(result)
        print(f"\n情感约束: {constraint}")
        
    except Exception as e:
        print(f"测试需要音频文件: {e}")
        
        # 生成随机测试数据
        test_audio = np.random.randn(22050 * 3)  # 3秒随机音频
        result = emotion_engine.analyze(test_audio, 22050)
        print(f"随机音频测试 - 主要情感: {result.primary_emotion.value}")