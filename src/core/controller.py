"""
智能音乐编辑主控制器 - System-6核心模块
Main Controller for Intelligent Music Editing System

整合所有核心功能，提供统一的接口
"""

import numpy as np
import librosa
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

from .emotion_engine import EmotionAnalysisEngine, EmotionResult
from .music_analyzer import MusicStructureAnalyzer, MusicStructureResult
from .track_operator import IntelligentTrackOperator, TrackOperation, OperationType, TrackRole

@dataclass
class MusicEditRequest:
    """音乐编辑请求"""
    audio_data: np.ndarray
    sr: int
    operation_type: str  # "add", "replace", "modify", "delete", "enhance"
    target_role: str     # "bass", "melody", "harmony", "rhythm"
    parameters: Dict[str, Any]
    preserve_emotion: bool = True
    quality_threshold: float = 0.7

@dataclass
class MusicEditResponse:
    """音乐编辑响应"""
    success: bool
    result_audio: Optional[np.ndarray]
    original_analysis: Dict[str, Any]
    final_analysis: Dict[str, Any]
    operation_log: List[str]
    quality_metrics: Dict[str, float]
    recommendations: List[str]

class MusicEditingController:
    """
    智能音乐编辑主控制器
    
    核心功能：
    1. 统一的音乐编辑接口
    2. 情感一致性保证
    3. 质量控制和验证
    4. 操作历史管理
    5. 智能建议生成
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        
        # 初始化核心组件
        self.emotion_engine = EmotionAnalysisEngine()
        self.structure_analyzer = MusicStructureAnalyzer()
        self.track_operator = IntelligentTrackOperator()
        
        # 操作历史
        self.operation_history: List[Dict] = []
        
        # 日志设置
        self.logger = self._setup_logger()
        
        self.logger.info("智能音乐编辑控制器初始化完成")
    
    def edit_music(self, request: MusicEditRequest) -> MusicEditResponse:
        """
        执行音乐编辑操作
        
        Args:
            request: 音乐编辑请求
            
        Returns:
            MusicEditResponse: 编辑结果
        """
        self.logger.info(f"开始音乐编辑操作: {request.operation_type} - {request.target_role}")
        
        operation_log = []
        
        try:
            # 1. 分析原始音频
            operation_log.append("正在分析原始音频...")
            original_analysis = self._analyze_audio(request.audio_data, request.sr)
            operation_log.append(f"原始分析完成 - 情感: {original_analysis['emotion']['primary']}")
            
            # 2. 验证请求参数
            validation_result = self._validate_request(request, original_analysis)
            if not validation_result['valid']:
                return MusicEditResponse(
                    success=False,
                    result_audio=None,
                    original_analysis=original_analysis,
                    final_analysis={},
                    operation_log=operation_log + [f"参数验证失败: {validation_result['reason']}"],
                    quality_metrics={},
                    recommendations=validation_result.get('recommendations', [])
                )
            
            # 3. 创建操作对象
            operation = self._create_operation(request, original_analysis)
            operation_log.append(f"创建操作: {operation.operation_type.value}")
            
            # 4. 执行操作
            operation_log.append("执行音轨操作...")
            operation_result = self.track_operator.operate(
                request.audio_data, operation, request.sr
            )
            
            if not operation_result.success:
                return MusicEditResponse(
                    success=False,
                    result_audio=None,
                    original_analysis=original_analysis,
                    final_analysis={},
                    operation_log=operation_log + [operation_result.operation_log],
                    quality_metrics={},
                    recommendations=self._generate_failure_recommendations(operation_result)
                )
            
            # 5. 分析结果音频
            operation_log.append("分析结果音频...")
            final_analysis = self._analyze_audio(operation_result.new_audio, request.sr)
            
            # 6. 质量验证
            quality_check = self._verify_quality(
                original_analysis, final_analysis, operation_result, request
            )
            operation_log.append(f"质量验证: {quality_check['status']}")
            
            # 7. 记录操作历史
            self._record_operation(request, operation_result, original_analysis, final_analysis)
            
            # 8. 生成建议
            recommendations = self._generate_recommendations(
                original_analysis, final_analysis, operation_result
            )
            
            self.logger.info("音乐编辑操作完成")
            
            return MusicEditResponse(
                success=True,
                result_audio=operation_result.new_audio,
                original_analysis=original_analysis,
                final_analysis=final_analysis,
                operation_log=operation_log + [operation_result.operation_log],
                quality_metrics=operation_result.quality_metrics,
                recommendations=recommendations
            )
            
        except Exception as e:
            error_msg = f"操作执行失败: {str(e)}"
            self.logger.error(error_msg)
            
            return MusicEditResponse(
                success=False,
                result_audio=None,
                original_analysis={},
                final_analysis={},
                operation_log=operation_log + [error_msg],
                quality_metrics={},
                recommendations=["请检查输入参数", "建议降低操作复杂度"]
            )
    
    def _analyze_audio(self, audio_data: np.ndarray, sr: int) -> Dict[str, Any]:
        """分析音频的完整特征"""
        # 1. 情感分析
        emotion_result = self.emotion_engine.analyze(audio_data, sr)
        
        # 2. 结构分析
        structure_result = self.structure_analyzer.analyze(audio_data, sr)
        
        # 3. 基础特征分析
        basic_features = self._extract_basic_features(audio_data, sr)
        
        return {
            'emotion': {
                'primary': emotion_result.primary_emotion.value,
                'scores': {e.value: score for e, score in emotion_result.emotion_scores.items()},
                'intensity': emotion_result.intensity,
                'confidence': emotion_result.confidence
            },
            'structure': {
                'track_roles': {k: v.value for k, v in structure_result.track_roles.items()},
                'overall_form': structure_result.overall_form,
                'key_signature': structure_result.harmony_analysis.key_signature,
                'tempo': structure_result.rhythm_pattern.get('tempo', 120)
            },
            'features': basic_features
        }
    
    def _extract_basic_features(self, audio_data: np.ndarray, sr: int) -> Dict[str, float]:
        """提取基础音频特征"""
        features = {}
        
        # 时长
        features['duration'] = len(audio_data) / sr
        
        # 响度
        rms = librosa.feature.rms(y=audio_data)
        features['loudness'] = float(np.mean(rms))
        
        # 频谱质心
        centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr)
        features['brightness'] = float(np.mean(centroid))
        
        # 零交叉率
        zcr = librosa.feature.zero_crossing_rate(audio_data)
        features['zcr'] = float(np.mean(zcr))
        
        # 谐波能量比
        harmonic, percussive = librosa.effects.hpss(audio_data)
        harmonic_energy = np.sum(harmonic ** 2)
        total_energy = np.sum(audio_data ** 2)
        features['harmonic_ratio'] = float(harmonic_energy / (total_energy + 1e-8))
        
        return features
    
    def _validate_request(self, request: MusicEditRequest, analysis: Dict) -> Dict[str, Any]:
        """验证编辑请求"""
        validation_result = {'valid': True, 'reason': '', 'recommendations': []}
        
        # 1. 检查音频长度
        if request.audio_data.shape[0] < request.sr * 2:  # 少于2秒
            validation_result.update({
                'valid': False,
                'reason': '音频长度过短（少于2秒）',
                'recommendations': ['请提供至少2秒的音频']
            })
            return validation_result
        
        # 2. 检查操作类型
        valid_operations = ['add', 'replace', 'modify', 'delete', 'enhance']
        if request.operation_type not in valid_operations:
            validation_result.update({
                'valid': False,
                'reason': f'不支持的操作类型: {request.operation_type}',
                'recommendations': [f'支持的操作: {", ".join(valid_operations)}']
            })
            return validation_result
        
        # 3. 检查目标角色
        valid_roles = ['bass', 'melody', 'harmony', 'rhythm', 'accompaniment', 'decoration']
        if request.target_role not in valid_roles:
            validation_result.update({
                'valid': False,
                'reason': f'不支持的音轨角色: {request.target_role}',
                'recommendations': [f'支持的角色: {", ".join(valid_roles)}']
            })
            return validation_result
        
        # 4. 检查音频质量
        if analysis['features']['loudness'] < 0.01:  # 音频太小声
            validation_result['recommendations'].append('建议增加音频音量')
        
        # 5. 检查情感一致性需求
        if request.preserve_emotion and analysis['emotion']['confidence'] < 0.5:
            validation_result['recommendations'].append('原始音频情感不够明确，可能影响情感保持效果')
        
        return validation_result
    
    def _create_operation(self, request: MusicEditRequest, analysis: Dict) -> TrackOperation:
        """创建操作对象"""
        # 操作类型映射
        operation_type_map = {
            'add': OperationType.ADD,
            'replace': OperationType.REPLACE,
            'modify': OperationType.MODIFY,
            'delete': OperationType.DELETE,
            'enhance': OperationType.ENHANCE
        }
        
        # 角色映射
        role_map = {
            'bass': TrackRole.BASS,
            'melody': TrackRole.MELODY,
            'harmony': TrackRole.HARMONY,
            'rhythm': TrackRole.RHYTHM,
            'accompaniment': TrackRole.ACCOMPANIMENT,
            'decoration': TrackRole.DECORATION
        }
        
        # 生成情感约束
        emotion_constraint = {}
        if request.preserve_emotion:
            emotion_constraint = {
                'target_emotion': analysis['emotion']['primary'],
                'intensity_range': [
                    max(0.0, analysis['emotion']['intensity'] - 0.1),
                    min(1.0, analysis['emotion']['intensity'] + 0.1)
                ]
            }
        
        return TrackOperation(
            operation_type=operation_type_map[request.operation_type],
            target_role=role_map[request.target_role],
            parameters=request.parameters,
            emotion_constraint=emotion_constraint,
            confidence=0.8
        )
    
    def _verify_quality(self, 
                       original: Dict, 
                       final: Dict, 
                       operation_result, 
                       request: MusicEditRequest) -> Dict[str, Any]:
        """验证编辑质量"""
        quality_check = {'status': 'passed', 'issues': [], 'scores': {}}
        
        # 1. 情感保持度检查
        if request.preserve_emotion:
            emotion_preservation = operation_result.emotion_preservation
            quality_check['scores']['emotion_preservation'] = emotion_preservation
            
            if emotion_preservation < 0.7:
                quality_check['issues'].append('情感保持度较低')
                if emotion_preservation < 0.5:
                    quality_check['status'] = 'warning'
        
        # 2. 音频质量检查
        quality_metrics = operation_result.quality_metrics
        for metric, score in quality_metrics.items():
            quality_check['scores'][metric] = score
            if score < request.quality_threshold:
                quality_check['issues'].append(f'{metric}质量低于阈值')
                quality_check['status'] = 'warning'
        
        # 3. 整体质量评分
        overall_score = np.mean(list(quality_check['scores'].values()))
        quality_check['scores']['overall'] = overall_score
        
        if overall_score < 0.5:
            quality_check['status'] = 'failed'
        
        return quality_check
    
    def _record_operation(self, request, result, original_analysis, final_analysis):
        """记录操作历史"""
        record = {
            'timestamp': np.datetime64('now'),
            'operation_type': request.operation_type,
            'target_role': request.target_role,
            'parameters': request.parameters,
            'original_emotion': original_analysis['emotion']['primary'],
            'final_emotion': final_analysis['emotion']['primary'],
            'emotion_preservation': result.emotion_preservation,
            'quality_metrics': result.quality_metrics,
            'success': result.success
        }
        
        self.operation_history.append(record)
        
        # 保持历史记录在合理范围内
        if len(self.operation_history) > 100:
            self.operation_history = self.operation_history[-50:]
    
    def _generate_recommendations(self, original, final, operation_result) -> List[str]:
        """生成智能建议"""
        recommendations = []
        
        # 基于情感保持度的建议
        emotion_preservation = operation_result.emotion_preservation
        if emotion_preservation < 0.8:
            recommendations.append("建议调整操作参数以更好地保持原始情感")
        
        # 基于质量指标的建议
        quality_metrics = operation_result.quality_metrics
        for metric, score in quality_metrics.items():
            if score < 0.7:
                recommendations.append(f"建议优化{metric}相关参数")
        
        # 基于历史操作的建议
        if len(self.operation_history) > 5:
            recent_operations = self.operation_history[-5:]
            success_rate = sum(1 for op in recent_operations if op['success']) / len(recent_operations)
            
            if success_rate < 0.8:
                recommendations.append("建议降低操作复杂度或调整参数")
        
        # 基于音乐特征的建议
        if final['emotion']['confidence'] < 0.6:
            recommendations.append("结果音频情感特征不够明确，建议进一步调整")
        
        return recommendations if recommendations else ["操作效果良好，无需调整"]
    
    def _generate_failure_recommendations(self, operation_result) -> List[str]:
        """生成失败时的建议"""
        return [
            "检查输入音频的质量和格式",
            "尝试简化操作参数",
            "确保目标角色与音频内容匹配",
            "考虑分步骤进行复杂操作"
        ]
    
    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            'emotion_preservation_threshold': 0.7,
            'quality_threshold': 0.6,
            'max_operation_complexity': 0.8,
            'log_level': 'INFO'
        }
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger('MusicEditingController')
        logger.setLevel(getattr(logging, self.config.get('log_level', 'INFO')))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def get_operation_history(self) -> List[Dict]:
        """获取操作历史"""
        return self.operation_history.copy()
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        if self.operation_history:
            recent_success_rate = np.mean([
                op['success'] for op in self.operation_history[-10:]
            ])
            avg_emotion_preservation = np.mean([
                op['emotion_preservation'] for op in self.operation_history[-10:]
                if op['success']
            ])
        else:
            recent_success_rate = 1.0
            avg_emotion_preservation = 1.0
        
        return {
            'total_operations': len(self.operation_history),
            'recent_success_rate': recent_success_rate,
            'avg_emotion_preservation': avg_emotion_preservation,
            'system_health': 'healthy' if recent_success_rate > 0.8 else 'warning'
        }


# 使用示例
if __name__ == "__main__":
    # 创建控制器
    controller = MusicEditingController()
    
    # 生成测试音频
    sr = 22050
    duration = 5
    test_audio = np.random.randn(sr * duration) * 0.1
    
    # 创建编辑请求
    request = MusicEditRequest(
        audio_data=test_audio,
        sr=sr,
        operation_type="add",
        target_role="bass",
        parameters={'instrument': 'bass_guitar', 'volume': 0.6},
        preserve_emotion=True,
        quality_threshold=0.7
    )
    
    # 执行编辑
    response = controller.edit_music(request)
    
    print(f"编辑成功: {response.success}")
    print(f"原始情感: {response.original_analysis.get('emotion', {}).get('primary', 'N/A')}")
    print(f"操作日志: {response.operation_log}")
    print(f"建议: {response.recommendations}")
    
    # 获取系统状态
    status = controller.get_system_status()
    print(f"系统状态: {status}")