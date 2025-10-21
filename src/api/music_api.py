"""
音乐AI API接口 - System-6
Music AI API Interface

提供RESTful API接口，便于集成和使用
"""

from flask import Flask, request, jsonify, send_file
import numpy as np
import librosa
import io
import soundfile as sf
import base64
from typing import Dict, Any

from ..core.controller import MusicEditingController, MusicEditRequest

app = Flask(__name__)
controller = MusicEditingController()

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查"""
    status = controller.get_system_status()
    return jsonify({
        'status': 'healthy',
        'system': status,
        'message': '智能音乐编辑系统运行正常'
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_audio():
    """分析音频情感和结构"""
    try:
        # 获取音频数据
        audio_data, sr = _get_audio_from_request(request)
        
        # 分析音频
        analysis = controller._analyze_audio(audio_data, sr)
        
        return jsonify({
            'success': True,
            'analysis': analysis,
            'message': '音频分析完成'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': '音频分析失败'
        }), 400

@app.route('/api/edit', methods=['POST'])
def edit_music():
    """编辑音乐"""
    try:
        # 获取请求数据
        data = request.get_json()
        audio_data, sr = _get_audio_from_request(request)
        
        # 创建编辑请求
        edit_request = MusicEditRequest(
            audio_data=audio_data,
            sr=sr,
            operation_type=data.get('operation_type', 'add'),
            target_role=data.get('target_role', 'bass'),
            parameters=data.get('parameters', {}),
            preserve_emotion=data.get('preserve_emotion', True),
            quality_threshold=data.get('quality_threshold', 0.7)
        )
        
        # 执行编辑
        response = controller.edit_music(edit_request)
        
        # 准备响应
        result = {
            'success': response.success,
            'original_analysis': response.original_analysis,
            'final_analysis': response.final_analysis,
            'operation_log': response.operation_log,
            'quality_metrics': response.quality_metrics,
            'recommendations': response.recommendations
        }
        
        # 如果成功，添加音频数据
        if response.success and response.result_audio is not None:
            # 将音频编码为base64
            audio_buffer = io.BytesIO()
            sf.write(audio_buffer, response.result_audio, sr, format='WAV')
            audio_base64 = base64.b64encode(audio_buffer.getvalue()).decode()
            result['result_audio'] = audio_base64
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': '音乐编辑失败'
        }), 400

@app.route('/api/operations', methods=['GET'])
def get_supported_operations():
    """获取支持的操作类型"""
    return jsonify({
        'operations': {
            'add': '添加音轨',
            'replace': '替换音轨', 
            'modify': '修改音轨',
            'delete': '删除音轨',
            'enhance': '增强音轨'
        },
        'roles': {
            'bass': '低音',
            'melody': '主旋律',
            'harmony': '和声',
            'rhythm': '节奏',
            'accompaniment': '伴奏',
            'decoration': '装饰音'
        },
        'emotions': {
            'happy': '快乐',
            'sad': '悲伤',
            'calm': '平静',
            'excited': '激动',
            'melancholy': '忧郁',
            'angry': '愤怒'
        }
    })

@app.route('/api/history', methods=['GET'])
def get_operation_history():
    """获取操作历史"""
    history = controller.get_operation_history()
    
    # 转换numpy类型为Python原生类型
    serializable_history = []
    for record in history:
        serializable_record = {}
        for key, value in record.items():
            if isinstance(value, np.ndarray):
                serializable_record[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                serializable_record[key] = value.item()
            else:
                serializable_record[key] = value
        serializable_history.append(serializable_record)
    
    return jsonify({
        'history': serializable_history,
        'total_count': len(history)
    })

def _get_audio_from_request(request) -> tuple:
    """从请求中获取音频数据"""
    if 'audio' not in request.files:
        # 尝试从JSON中获取base64编码的音频
        data = request.get_json()
        if data and 'audio_base64' in data:
            audio_bytes = base64.b64decode(data['audio_base64'])
            audio_buffer = io.BytesIO(audio_bytes)
            audio_data, sr = librosa.load(audio_buffer, sr=22050)
            return audio_data, sr
        else:
            raise ValueError("未找到音频数据")
    
    # 从文件获取音频
    audio_file = request.files['audio']
    audio_data, sr = librosa.load(audio_file, sr=22050)
    return audio_data, sr

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)