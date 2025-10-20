"""
音轨操作引擎的基础抽象类
定义所有引擎的通用接口和基础功能
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import logging
from .data_structures import (
    TrackData, OperationResult, EmotionConstraints, 
    MusicConstraints, ConflictReport
)

logger = logging.getLogger(__name__)


class BaseOperationEngine(ABC):
    """音轨操作引擎基类"""
    
    def __init__(self, engine_name: str):
        self.engine_name = engine_name
        self.is_initialized = False
        self._cache = {}  # 缓存机制
        
    @abstractmethod
    def initialize(self) -> bool:
        """
        初始化引擎
        
        Returns:
            是否初始化成功
        """
        pass
    
    @abstractmethod
    def process(self, *args, **kwargs) -> OperationResult:
        """
        处理操作的核心方法，由子类实现
        
        Returns:
            操作结果
        """
        pass
    
    def validate_inputs(self, **kwargs) -> List[str]:
        """
        验证输入参数
        
        Returns:
            错误列表，空列表表示无错误
        """
        errors = []
        
        # 检查必需的约束
        if 'emotion_constraints' in kwargs:
            emotion_constraints = kwargs['emotion_constraints']
            if emotion_constraints and not isinstance(emotion_constraints, EmotionConstraints):
                errors.append("emotion_constraints类型错误")
        
        if 'music_constraints' in kwargs:
            music_constraints = kwargs['music_constraints']
            if music_constraints and not isinstance(music_constraints, MusicConstraints):
                errors.append("music_constraints类型错误")
        
        return errors
    
    def get_cache_key(self, **kwargs) -> str:
        """生成缓存键"""
        # 简单的缓存键生成策略
        key_parts = []
        for k, v in sorted(kwargs.items()):
            if isinstance(v, (str, int, float, bool)):
                key_parts.append(f"{k}:{v}")
        return f"{self.engine_name}:" + "|".join(key_parts)
    
    def get_from_cache(self, cache_key: str) -> Optional[Any]:
        """从缓存获取结果"""
        return self._cache.get(cache_key)
    
    def save_to_cache(self, cache_key: str, result: Any):
        """保存结果到缓存"""
        # 限制缓存大小
        if len(self._cache) > 100:
            # 删除最旧的一半
            keys_to_delete = list(self._cache.keys())[:50]
            for key in keys_to_delete:
                del self._cache[key]
        
        self._cache[cache_key] = result
    
    def create_success_result(self, generated_track: TrackData, 
                            quality_metrics: Dict[str, float] = None,
                            conflicts: List[ConflictReport] = None,
                            metadata: Dict = None) -> OperationResult:
        """创建成功的操作结果"""
        if quality_metrics is None:
            quality_metrics = {}
        if conflicts is None:
            conflicts = []
        if metadata is None:
            metadata = {}
            
        return OperationResult(
            success=True,
            generated_track=generated_track,
            quality_score=quality_metrics.get('quality_score', 0.8),
            emotion_consistency=quality_metrics.get('emotion_consistency', 0.8),
            harmonic_correctness=quality_metrics.get('harmonic_correctness', 0.8),
            conflicts=conflicts,
            metadata={
                'engine': self.engine_name,
                **metadata
            }
        )
    
    def create_failure_result(self, error_message: str, 
                            warnings: List[str] = None) -> OperationResult:
        """创建失败的操作结果"""
        if warnings is None:
            warnings = []
            
        return OperationResult(
            success=False,
            warnings=[error_message] + warnings,
            metadata={'engine': self.engine_name}
        )
    
    def log_operation(self, operation_name: str, **kwargs):
        """记录操作日志"""
        logger.info(f"[{self.engine_name}] {operation_name}: {kwargs}")


class BaseGenerationEngine(BaseOperationEngine):
    """音轨生成引擎基类"""
    
    def __init__(self, engine_name: str):
        super().__init__(engine_name)
        
    @abstractmethod
    def generate_track(self, instrument: str, role: str, 
                      constraints: Dict, **kwargs) -> OperationResult:
        """
        生成音轨的抽象方法
        
        Args:
            instrument: 目标乐器
            role: 音轨角色
            constraints: 约束条件
            
        Returns:
            操作结果
        """
        pass
    
    def validate_generation_inputs(self, instrument: str, role: str,
                                 **kwargs) -> List[str]:
        """验证生成相关的输入"""
        errors = self.validate_inputs(**kwargs)
        
        if not instrument:
            errors.append("乐器名称不能为空")
        if not role:
            errors.append("音轨角色不能为空")
            
        return errors


class BaseReplacementEngine(BaseOperationEngine):
    """音轨替换引擎基类"""
    
    def __init__(self, engine_name: str):
        super().__init__(engine_name)
        
    @abstractmethod
    def replace_track(self, original_track: TrackData, 
                     target_instrument: str, **kwargs) -> OperationResult:
        """
        替换音轨的抽象方法
        
        Args:
            original_track: 原始音轨
            target_instrument: 目标乐器
            
        Returns:
            操作结果
        """
        pass
    
    def validate_replacement_inputs(self, original_track: TrackData,
                                  target_instrument: str, **kwargs) -> List[str]:
        """验证替换相关的输入"""
        errors = self.validate_inputs(**kwargs)
        
        if not original_track or not original_track.is_valid():
            errors.append("原始音轨无效")
        if not target_instrument:
            errors.append("目标乐器不能为空")
            
        return errors


class BaseRepairEngine(BaseOperationEngine):
    """音轨修复引擎基类"""
    
    def __init__(self, engine_name: str):
        super().__init__(engine_name)
        
    @abstractmethod
    def repair_track(self, problematic_track: TrackData, 
                    **kwargs) -> OperationResult:
        """
        修复音轨的抽象方法
        
        Args:
            problematic_track: 有问题的音轨
            
        Returns:
            操作结果
        """
        pass
    
    def validate_repair_inputs(self, problematic_track: TrackData,
                             **kwargs) -> List[str]:
        """验证修复相关的输入"""
        errors = self.validate_inputs(**kwargs)
        
        if not problematic_track or not problematic_track.is_valid():
            errors.append("要修复的音轨无效")
            
        return errors


class BaseConflictDetector(BaseOperationEngine):
    """冲突检测器基类"""
    
    def __init__(self, engine_name: str):
        super().__init__(engine_name)
        
    @abstractmethod
    def detect_conflicts(self, existing_tracks: List[TrackData],
                        new_track: TrackData) -> List[ConflictReport]:
        """
        检测冲突的抽象方法
        
        Args:
            existing_tracks: 现有音轨列表
            new_track: 新音轨
            
        Returns:
            冲突报告列表
        """
        pass
    
    def validate_conflict_detection_inputs(self, existing_tracks: List[TrackData],
                                         new_track: TrackData) -> List[str]:
        """验证冲突检测相关的输入"""
        errors = []
        
        if not existing_tracks:
            errors.append("现有音轨列表不能为空")
        
        if not new_track or not new_track.is_valid():
            errors.append("新音轨无效")
            
        for i, track in enumerate(existing_tracks):
            if not track.is_valid():
                errors.append(f"现有音轨{i}无效")
                
        return errors


# 工具函数
def create_default_constraints() -> Dict:
    """创建默认约束"""
    from .data_structures import EmotionType, TrackRole
    
    emotion_constraints = EmotionConstraints(
        primary_emotion=EmotionType.CALM,
        intensity=0.6
    )
    
    music_constraints = MusicConstraints(
        key="C_major",
        time_signature="4/4", 
        tempo=120
    )
    
    return {
        'emotion': emotion_constraints,
        'music': music_constraints
    }