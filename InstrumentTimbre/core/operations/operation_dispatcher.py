"""
操作分发器 - 统一的音轨操作入口
负责解析用户请求，分发到对应的处理引擎
"""

import time
import logging
from typing import Dict, List, Optional, Union
from .data_structures import (
    OperationType, OperationRequest, OperationResult, 
    EmotionConstraints, MusicConstraints, TrackData
)

logger = logging.getLogger(__name__)


class OperationDispatcher:
    """操作分发器 - 音轨操作的统一入口"""
    
    def __init__(self):
        self.track_generator = None  # 延迟初始化
        self.track_replacer = None
        self.track_repairer = None
        self.orchestrator = None
        self.conflict_detector = None
        
        self._initialized = False
        
    def initialize_engines(self):
        """初始化所有处理引擎"""
        if self._initialized:
            return
            
        try:
            from .track_generation_engine import TrackGenerationEngine
            from .track_replacement_engine import TrackReplacementEngine
            from .track_repair_engine import TrackRepairEngine
            from .emotion_driven_orchestrator import EmotionDrivenOrchestrator
            from .conflict_detector import RealTimeConflictDetector
            
            self.track_generator = TrackGenerationEngine()
            self.track_replacer = TrackReplacementEngine()
            self.track_repairer = TrackRepairEngine()
            self.orchestrator = EmotionDrivenOrchestrator()
            self.conflict_detector = RealTimeConflictDetector()
            
            self._initialized = True
            logger.info("所有音轨操作引擎初始化完成")
            
        except ImportError as e:
            logger.error(f"引擎初始化失败: {e}")
            raise
    
    def process_request(self, request: OperationRequest, 
                       current_tracks: List[TrackData]) -> OperationResult:
        """
        处理音轨操作请求
        
        Args:
            request: 操作请求
            current_tracks: 当前音轨列表
            
        Returns:
            操作结果
        """
        if not self._initialized:
            self.initialize_engines()
        
        # 验证请求
        if not request.validate():
            return OperationResult(
                success=False,
                warnings=["请求参数无效"]
            )
        
        start_time = time.time()
        
        try:
            # 分发到对应的处理引擎
            if request.operation_type == OperationType.GENERATE:
                result = self._handle_generation(request, current_tracks)
            elif request.operation_type == OperationType.REPLACE:
                result = self._handle_replacement(request, current_tracks)
            elif request.operation_type == OperationType.REPAIR:
                result = self._handle_repair(request, current_tracks)
            else:
                return OperationResult(
                    success=False,
                    warnings=[f"不支持的操作类型: {request.operation_type}"]
                )
            
            # 记录处理时间
            result.processing_time = time.time() - start_time
            
            # 执行实时冲突检测
            if result.success and result.generated_track:
                conflicts = self.conflict_detector.detect_conflicts(
                    current_tracks, result.generated_track
                )
                result.conflicts.extend(conflicts)
            
            logger.info(f"操作完成: {request.operation_type.value}, "
                       f"用时: {result.processing_time:.2f}秒, "
                       f"成功: {result.success}")
            
            return result
            
        except Exception as e:
            logger.error(f"操作处理失败: {e}")
            return OperationResult(
                success=False,
                processing_time=time.time() - start_time,
                warnings=[f"处理错误: {str(e)}"]
            )
    
    def _handle_generation(self, request: OperationRequest, 
                          current_tracks: List[TrackData]) -> OperationResult:
        """处理音轨生成请求"""
        logger.info(f"开始生成音轨: {request.target_instrument} ({request.target_role.value})")
        
        # 使用情感驱动配器获取配器建议
        if request.emotion_constraints:
            orchestration = self.orchestrator.get_orchestration_suggestion(
                request.emotion_constraints,
                request.target_instrument,
                current_tracks
            )
            logger.info(f"获得配器建议: {orchestration}")
        
        # 调用音轨生成引擎
        return self.track_generator.generate_track(
            instrument=request.target_instrument,
            role=request.target_role,
            emotion_constraints=request.emotion_constraints,
            music_constraints=request.music_constraints,
            current_tracks=current_tracks,
            intensity=request.intensity
        )
    
    def _handle_replacement(self, request: OperationRequest,
                           current_tracks: List[TrackData]) -> OperationResult:
        """处理音轨替换请求"""
        logger.info(f"开始替换音轨: 目标乐器 {request.target_instrument}")
        
        # 找到要替换的音轨
        target_track = None
        if request.reference_track:
            target_track = next(
                (track for track in current_tracks 
                 if track.track_id == request.reference_track),
                None
            )
        
        if not target_track:
            return OperationResult(
                success=False,
                warnings=["未找到要替换的音轨"]
            )
        
        # 调用音轨替换引擎
        return self.track_replacer.replace_track(
            original_track=target_track,
            target_instrument=request.target_instrument,
            emotion_constraints=request.emotion_constraints,
            music_constraints=request.music_constraints,
            preserve_function=request.preserve_original
        )
    
    def _handle_repair(self, request: OperationRequest,
                      current_tracks: List[TrackData]) -> OperationResult:
        """处理音轨修复请求"""
        logger.info("开始音轨修复")
        
        # 找到要修复的音轨
        target_track = None
        if request.reference_track:
            target_track = next(
                (track for track in current_tracks 
                 if track.track_id == request.reference_track),
                None
            )
        
        if not target_track:
            return OperationResult(
                success=False,
                warnings=["未找到要修复的音轨"]
            )
        
        # 调用音轨修复引擎
        return self.track_repairer.repair_track(
            problematic_track=target_track,
            other_tracks=current_tracks,
            constraints={
                'emotion': request.emotion_constraints,
                'music': request.music_constraints
            }
        )
    
    def parse_natural_language_request(self, text: str, 
                                     current_tracks: List[TrackData]) -> Optional[OperationRequest]:
        """
        解析自然语言请求
        
        Args:
            text: 用户的自然语言描述
            current_tracks: 当前音轨
            
        Returns:
            解析后的操作请求，如果无法解析则返回None
        """
        text = text.lower().strip()
        
        # 操作类型判断
        if any(keyword in text for keyword in ["添加", "加入", "增加", "生成"]):
            operation_type = OperationType.GENERATE
        elif any(keyword in text for keyword in ["替换", "换成", "改成"]):
            operation_type = OperationType.REPLACE
        elif any(keyword in text for keyword in ["修复", "修正", "纠正", "优化"]):
            operation_type = OperationType.REPAIR
        else:
            logger.warning(f"无法识别操作类型: {text}")
            return None
        
        # 乐器识别
        instrument_map = {
            "小提琴": "violin",
            "大提琴": "cello", 
            "钢琴": "piano",
            "吉他": "guitar",
            "长笛": "flute",
            "萨克斯": "saxophone",
            "鼓": "drums",
            "贝斯": "bass"
        }
        
        target_instrument = "violin"  # 默认
        for cn_name, en_name in instrument_map.items():
            if cn_name in text or en_name in text:
                target_instrument = en_name
                break
        
        # 角色识别
        role_map = {
            "主旋律": "melody",
            "和声": "harmony", 
            "低音": "bass",
            "节奏": "rhythm",
            "伴奏": "accompaniment"
        }
        
        target_role = "harmony"  # 默认
        for cn_role, en_role in role_map.items():
            if cn_role in text or en_role in text:
                target_role = en_role
                break
        
        # 创建请求对象
        request = OperationRequest(
            operation_type=operation_type,
            target_instrument=target_instrument,
            target_role=TrackRole(target_role)
        )
        
        logger.info(f"解析请求: {text} -> {operation_type.value} {target_instrument} {target_role}")
        
        return request
    
    def get_status(self) -> Dict:
        """获取分发器状态"""
        return {
            "initialized": self._initialized,
            "engines": {
                "track_generator": self.track_generator is not None,
                "track_replacer": self.track_replacer is not None,
                "track_repairer": self.track_repairer is not None,
                "orchestrator": self.orchestrator is not None,
                "conflict_detector": self.conflict_detector is not None
            }
        }


# 全局实例 - 单例模式
_dispatcher_instance = None

def get_operation_dispatcher() -> OperationDispatcher:
    """获取操作分发器实例（单例）"""
    global _dispatcher_instance
    if _dispatcher_instance is None:
        _dispatcher_instance = OperationDispatcher()
        _dispatcher_instance.initialize_engines()
    return _dispatcher_instance


# 便捷函数
def intelligent_track_operation(user_request: Union[str, OperationRequest], 
                               current_tracks: List[TrackData],
                               emotion_constraints: Optional[EmotionConstraints] = None,
                               music_constraints: Optional[MusicConstraints] = None) -> OperationResult:
    """
    智能音轨操作的便捷入口函数
    
    Args:
        user_request: 用户请求（自然语言字符串或结构化请求对象）
        current_tracks: 当前音轨列表
        emotion_constraints: 情感约束（可选）
        music_constraints: 音乐约束（可选）
        
    Returns:
        操作结果
    """
    dispatcher = get_operation_dispatcher()
    
    # 处理不同类型的请求
    if isinstance(user_request, str):
        # 自然语言请求
        request = dispatcher.parse_natural_language_request(user_request, current_tracks)
        if not request:
            return OperationResult(
                success=False,
                warnings=["无法解析用户请求"]
            )
    else:
        # 结构化请求
        request = user_request
    
    # 添加约束（如果提供）
    if emotion_constraints:
        request.emotion_constraints = emotion_constraints
    if music_constraints:
        request.music_constraints = music_constraints
    
    # 执行操作
    return dispatcher.process_request(request, current_tracks)