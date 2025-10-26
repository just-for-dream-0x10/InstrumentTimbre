"""
AI API - System-6
Music AI API Interface

RESTful API，
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
    """Health check"""
    status = controller.get_system_status()
    return jsonify({
        'status': 'healthy',
        'system': status,
        'message': ''
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_audio():
    """"""
    try:
        # 
        audio_data, sr = _get_audio_from_request(request)
        
        # 
        analysis = controller._analyze_audio(audio_data, sr)
        
        return jsonify({
            'success': True,
            'analysis': analysis,
            'message': ''
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': ''
        }), 400

@app.route('/api/edit', methods=['POST'])
def edit_music():
    """"""
    try:
        # 
        data = request.get_json()
        audio_data, sr = _get_audio_from_request(request)
        
        # create
        edit_request = MusicEditRequest(
            audio_data=audio_data,
            sr=sr,
            operation_type=data.get('operation_type', 'add'),
            target_role=data.get('target_role', 'bass'),
            parameters=data.get('parameters', {}),
            preserve_emotion=data.get('preserve_emotion', True),
            quality_threshold=data.get('quality_threshold', 0.7)
        )
        
        # 
        response = controller.edit_music(edit_request)
        
        # 
        result = {
            'success': response.success,
            'original_analysis': response.original_analysis,
            'final_analysis': response.final_analysis,
            'operation_log': response.operation_log,
            'quality_metrics': response.quality_metrics,
            'recommendations': response.recommendations
        }
        
        # success，
        if response.success and response.result_audio is not None:
            # base64
            audio_buffer = io.BytesIO()
            sf.write(audio_buffer, response.result_audio, sr, format='WAV')
            audio_base64 = base64.b64encode(audio_buffer.getvalue()).decode()
            result['result_audio'] = audio_base64
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': ''
        }), 400

@app.route('/api/operations', methods=['GET'])
def get_supported_operations():
    """Operation"""
    return jsonify({
        'operations': {
            'add': '',
            'replace': '', 
            'modify': '',
            'delete': '',
            'enhance': ''
        },
        'roles': {
            'bass': 'bass',
            'melody': '',
            'harmony': 'harmony',
            'rhythm': 'rhythm',
            'accompaniment': '',
            'decoration': ''
        },
        'emotions': {
            'happy': '',
            'sad': '',
            'calm': '',
            'excited': '',
            'melancholy': '',
            'angry': ''
        }
    })

@app.route('/api/history', methods=['GET'])
def get_operation_history():
    """Operation"""
    history = controller.get_operation_history()
    
    # numpyPython
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
    """"""
    if 'audio' not in request.files:
        # JSONbase64
        data = request.get_json()
        if data and 'audio_base64' in data:
            audio_bytes = base64.b64decode(data['audio_base64'])
            audio_buffer = io.BytesIO(audio_bytes)
            audio_data, sr = librosa.load(audio_buffer, sr=22050)
            return audio_data, sr
        else:
            raise ValueError("")
    
    # 
    audio_file = request.files['audio']
    audio_data, sr = librosa.load(audio_file, sr=22050)
    return audio_data, sr

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)