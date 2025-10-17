#!/usr/bin/env python3
"""
将长音频文件切割成短片段用于训练
"""

import os
import sys
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path

def split_audio_files(input_dir, output_dir, clip_duration=3.0, hop_duration=1.5):
    """
    将长音频文件切割成短片段
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录  
        clip_duration: 片段长度（秒）
        hop_duration: 跳跃长度（秒）
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    audio_files = list(input_dir.glob('*.wav'))
    print(f"找到 {len(audio_files)} 个音频文件")
    
    total_clips = 0
    
    for audio_file in audio_files:
        print(f"\n处理: {audio_file.name}")
        
        # 加载音频
        try:
            audio_data, sr = librosa.load(str(audio_file), sr=22050)
            duration = len(audio_data) / sr
            print(f"  原始长度: {duration:.1f}秒")
            
            # 推断类别
            filename = audio_file.stem.lower()
            if 'erhu' in filename:
                class_name = 'erhu'
            elif 'pipa' in filename:
                class_name = 'pipa'
            elif 'piano' in filename:
                class_name = 'piano'
            elif 'bass' in filename:
                class_name = 'bass'
            elif 'drum' in filename:
                class_name = 'drums'
            elif 'vocal' in filename:
                class_name = 'vocals'
            else:
                class_name = 'mixed'
            
            # 创建类别目录
            class_dir = output_dir / class_name
            class_dir.mkdir(exist_ok=True)
            
            # 切割音频
            clip_samples = int(clip_duration * sr)
            hop_samples = int(hop_duration * sr)
            
            clip_count = 0
            for start_sample in range(0, len(audio_data) - clip_samples, hop_samples):
                end_sample = start_sample + clip_samples
                clip_data = audio_data[start_sample:end_sample]
                
                # 跳过太安静的片段
                if np.max(np.abs(clip_data)) < 0.01:
                    continue
                
                # 保存片段
                clip_filename = f"{audio_file.stem}_clip_{clip_count:03d}.wav"
                clip_path = class_dir / clip_filename
                
                sf.write(str(clip_path), clip_data, sr)
                clip_count += 1
                total_clips += 1
            
            print(f"  生成 {clip_count} 个片段 -> {class_name}/")
            
        except Exception as e:
            print(f"  ❌ 处理失败: {e}")
    
    print(f"\n🎉 完成！总共生成 {total_clips} 个训练片段")
    
    # 统计各类别数量
    print("\n📊 各类别片段数量:")
    for class_dir in output_dir.iterdir():
        if class_dir.is_dir():
            count = len(list(class_dir.glob('*.wav')))
            print(f"  {class_dir.name}: {count} 个片段")

if __name__ == '__main__':
    input_dir = sys.argv[1] if len(sys.argv) > 1 else '../data/samples'
    output_dir = sys.argv[2] if len(sys.argv) > 2 else '../data/clips'
    
    print("🎵 音频片段生成工具")
    print(f"输入: {input_dir}")
    print(f"输出: {output_dir}")
    print(f"片段长度: 3秒")
    
    split_audio_files(input_dir, output_dir)