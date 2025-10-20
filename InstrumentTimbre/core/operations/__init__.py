"""
智能音轨操作引擎模块

该模块包含Week 7开发的核心音轨操作功能：
- 音轨生成模块
- 音轨替换模块  
- 音轨修复模块
- 情感驱动配器
- 实时冲突检测
"""

from .track_generation_engine import TrackGenerationEngine
from .track_replacement_engine import TrackReplacementEngine
from .track_repair_engine import TrackRepairEngine
from .emotion_driven_orchestrator import EmotionDrivenOrchestrator
from .conflict_detector import RealTimeConflictDetector
from .operation_dispatcher import OperationDispatcher

__all__ = [
    'TrackGenerationEngine',
    'TrackReplacementEngine', 
    'TrackRepairEngine',
    'EmotionDrivenOrchestrator',
    'RealTimeConflictDetector',
    'OperationDispatcher'
]