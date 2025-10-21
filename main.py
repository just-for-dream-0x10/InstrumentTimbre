"""
智能音乐编辑与优化AI系统 - 主入口
Intelligent Music Editing and Optimization AI System - Main Entry

System-6 基础架构实现
基于情感驱动的音乐编辑和优化AI工具
"""

import numpy as np
import librosa
import argparse
import os
import sys
from pathlib import Path

# 添加src到路径
sys.path.append(str(Path(__file__).parent / "src"))

from src.core.controller import MusicEditingController, MusicEditRequest
from src.core.emotion_engine import EmotionType
from src.core.track_operator import OperationType, TrackRole

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='智能音乐编辑与优化AI系统')
    parser.add_argument('--mode', choices=['demo', 'api', 'analyze', 'edit'], 
                       default='demo', help='运行模式')
    parser.add_argument('--input', type=str, help='输入音频文件路径')
    parser.add_argument('--output', type=str, help='输出音频文件路径')
    parser.add_argument('--operation', choices=['add', 'replace', 'modify', 'delete', 'enhance'],
                       default='add', help='操作类型')
    parser.add_argument('--role', choices=['bass', 'melody', 'harmony', 'rhythm'],
                       default='bass', help='目标音轨角色')
    parser.add_argument('--instrument', type=str, default='bass_guitar', help='乐器类型')
    parser.add_argument('--volume', type=float, default=0.6, help='音量大小')
    
    args = parser.parse_args()
    
    if args.mode == 'demo':
        run_demo()
    elif args.mode == 'api':
        run_api_server()
    elif args.mode == 'analyze':
        if not args.input:
            print("分析模式需要指定输入文件 --input")
            return
        analyze_audio_file(args.input)
    elif args.mode == 'edit':
        if not args.input:
            print("编辑模式需要指定输入文件 --input")
            return
        edit_audio_file(args.input, args.output, args.operation, args.role, 
                       args.instrument, args.volume)

def run_demo():
    """运行演示"""
    print("🎵 智能音乐编辑与优化AI系统 - 演示模式")
    print("=" * 50)
    
    # 创建控制器
    controller = MusicEditingController()
    
    # 生成测试音频（简单的和弦进行）
    print("1. 生成测试音频...")
    sr = 22050
    duration = 8
    
    # 创建简单的C大调和弦进行
    t = np.linspace(0, duration, sr * duration)
    
    # C - Am - F - G 和弦进行
    audio = np.zeros_like(t)
    chord_duration = duration / 4
    
    # C大调和弦 (C-E-G)
    c_chord = (np.sin(2 * np.pi * 261.63 * t) + 
               np.sin(2 * np.pi * 329.63 * t) + 
               np.sin(2 * np.pi * 392.00 * t)) / 3
    
    # 添加到前2秒
    mask1 = (t >= 0) & (t < chord_duration)
    audio[mask1] = c_chord[mask1] * 0.3
    
    # Am和弦 (A-C-E)
    am_chord = (np.sin(2 * np.pi * 220.00 * t) + 
                np.sin(2 * np.pi * 261.63 * t) + 
                np.sin(2 * np.pi * 329.63 * t)) / 3
    
    mask2 = (t >= chord_duration) & (t < 2 * chord_duration)
    audio[mask2] = am_chord[mask2] * 0.3
    
    # F大调和弦 (F-A-C)
    f_chord = (np.sin(2 * np.pi * 174.61 * t) + 
               np.sin(2 * np.pi * 220.00 * t) + 
               np.sin(2 * np.pi * 261.63 * t)) / 3
    
    mask3 = (t >= 2 * chord_duration) & (t < 3 * chord_duration)
    audio[mask3] = f_chord[mask3] * 0.3
    
    # G大调和弦 (G-B-D)
    g_chord = (np.sin(2 * np.pi * 196.00 * t) + 
               np.sin(2 * np.pi * 246.94 * t) + 
               np.sin(2 * np.pi * 293.66 * t)) / 3
    
    mask4 = (t >= 3 * chord_duration) & (t <= duration)
    audio[mask4] = g_chord[mask4] * 0.3
    
    print("✅ 测试音频生成完成 (C-Am-F-G和弦进行)")
    
    # 2. 分析原始音频
    print("\n2. 分析原始音频...")
    original_analysis = controller._analyze_audio(audio, sr)
    
    print(f"   情感: {original_analysis['emotion']['primary']}")
    print(f"   强度: {original_analysis['emotion']['intensity']:.3f}")
    print(f"   置信度: {original_analysis['emotion']['confidence']:.3f}")
    print(f"   调性: {original_analysis['structure']['key_signature']}")
    print(f"   曲式: {original_analysis['structure']['overall_form']}")
    
    # 3. 执行音轨添加操作
    print("\n3. 执行音轨添加操作 (添加低音线)...")
    
    request = MusicEditRequest(
        audio_data=audio,
        sr=sr,
        operation_type="add",
        target_role="bass",
        parameters={
            'instrument': 'bass_guitar',
            'volume': 0.5,
            'style': 'walking_bass'
        },
        preserve_emotion=True,
        quality_threshold=0.7
    )
    
    response = controller.edit_music(request)
    
    print(f"   操作成功: {response.success}")
    if response.success:
        print(f"   质量评分: {response.quality_metrics}")
        print(f"   情感保持度: {response.final_analysis.get('emotion', {}).get('confidence', 0):.3f}")
        print(f"   建议: {response.recommendations}")
        
        # 保存结果音频
        if response.result_audio is not None:
            output_file = "demo_output.wav"
            import soundfile as sf
            sf.write(output_file, response.result_audio, sr)
            print(f"   结果音频已保存到: {output_file}")
    else:
        print(f"   操作失败: {response.operation_log}")
    
    # 4. 系统状态
    print("\n4. 系统状态:")
    status = controller.get_system_status()
    for key, value in status.items():
        print(f"   {key}: {value}")
    
    print("\n🎉 演示完成!")

def analyze_audio_file(input_file: str):
    """分析音频文件"""
    print(f"🔍 分析音频文件: {input_file}")
    
    if not os.path.exists(input_file):
        print(f"❌ 文件不存在: {input_file}")
        return
    
    try:
        # 加载音频
        audio_data, sr = librosa.load(input_file, sr=22050)
        print(f"✅ 音频加载成功 - 时长: {len(audio_data)/sr:.2f}秒")
        
        # 创建控制器并分析
        controller = MusicEditingController()
        analysis = controller._analyze_audio(audio_data, sr)
        
        # 显示分析结果
        print("\n📊 分析结果:")
        print(f"情感分析:")
        print(f"  主要情感: {analysis['emotion']['primary']}")
        print(f"  情感强度: {analysis['emotion']['intensity']:.3f}")
        print(f"  置信度: {analysis['emotion']['confidence']:.3f}")
        
        print(f"\n音乐结构:")
        print(f"  调性: {analysis['structure']['key_signature']}")
        print(f"  节拍: {analysis['structure']['tempo']:.1f} BPM")
        print(f"  曲式: {analysis['structure']['overall_form']}")
        
        print(f"\n音频特征:")
        for feature, value in analysis['features'].items():
            print(f"  {feature}: {value:.4f}")
        
    except Exception as e:
        print(f"❌ 分析失败: {str(e)}")

def edit_audio_file(input_file: str, output_file: str, operation: str, 
                   role: str, instrument: str, volume: float):
    """编辑音频文件"""
    print(f"🎛️ 编辑音频文件: {input_file}")
    print(f"操作: {operation}, 角色: {role}, 乐器: {instrument}")
    
    if not os.path.exists(input_file):
        print(f"❌ 文件不存在: {input_file}")
        return
    
    try:
        # 加载音频
        audio_data, sr = librosa.load(input_file, sr=22050)
        print(f"✅ 音频加载成功")
        
        # 创建编辑请求
        request = MusicEditRequest(
            audio_data=audio_data,
            sr=sr,
            operation_type=operation,
            target_role=role,
            parameters={
                'instrument': instrument,
                'volume': volume
            },
            preserve_emotion=True,
            quality_threshold=0.7
        )
        
        # 执行编辑
        controller = MusicEditingController()
        response = controller.edit_music(request)
        
        if response.success:
            print("✅ 编辑成功!")
            
            # 保存结果
            if output_file and response.result_audio is not None:
                import soundfile as sf
                sf.write(output_file, response.result_audio, sr)
                print(f"📁 结果已保存到: {output_file}")
            
            # 显示质量指标
            print(f"📊 质量指标: {response.quality_metrics}")
            print(f"💡 建议: {response.recommendations}")
            
        else:
            print(f"❌ 编辑失败: {response.operation_log}")
            
    except Exception as e:
        print(f"❌ 编辑失败: {str(e)}")

def run_api_server():
    """运行API服务器"""
    print("🚀 启动API服务器...")
    print("API地址: http://localhost:5000")
    print("健康检查: http://localhost:5000/api/health")
    
    try:
        from src.api.music_api import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except ImportError:
        print("❌ 无法启动API服务器，请确保安装了Flask")
    except Exception as e:
        print(f"❌ API服务器启动失败: {str(e)}")

if __name__ == "__main__":
    main()