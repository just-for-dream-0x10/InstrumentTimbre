"""

Core Modules Package

System-6 ：
-  (emotion_engine.py)
-  (music_analyzer.py)  
-  (track_operator.py)
-  (controller.py)
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