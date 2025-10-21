"""
核心模块包
Core Modules Package

System-6 核心模块：
- 情感分析引擎 (emotion_engine.py)
- 音乐结构分析器 (music_analyzer.py)  
- 智能音轨操作器 (track_operator.py)
- 主控制器 (controller.py)
"""

from .emotion_engine import (
    EmotionAnalysisEngine, 
    EmotionResult, 
    EmotionType
)

from .music_analyzer import (
    MusicStructureAnalyzer,
    MusicStructureResult,
    TrackRole,
    MusicSection
)

from .track_operator import (
    IntelligentTrackOperator,
    TrackOperation,
    OperationType,
    OperationResult
)

__all__ = [
    'EmotionAnalysisEngine',
    'EmotionResult', 
    'EmotionType',
    'MusicStructureAnalyzer',
    'MusicStructureResult',
    'TrackRole',
    'MusicSection',
    'IntelligentTrackOperator',
    'TrackOperation',
    'OperationType', 
    'OperationResult'
]