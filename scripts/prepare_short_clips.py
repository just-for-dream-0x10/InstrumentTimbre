#!/usr/bin/env python3
"""

"""

import os
import sys
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path

def split_audio_files(input_dir, output_dir, clip_duration=3.0, hop_duration=1.5):
    """
    
    
    Args:
        input_dir: 
        output_dir:   
        clip_duration: （）
        hop_duration: （）
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    audio_files = list(input_dir.glob('*.wav'))
    print(f" {len(audio_files)} ")
    
    total_clips = 0
    
    for audio_file in audio_files:
        print(f"\n: {audio_file.name}")
        
        # load
        try:
            audio_data, sr = librosa.load(str(audio_file), sr=22050)
            duration = len(audio_data) / sr
            print(f"  : {duration:.1f}")
            
            # 
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
            
            # create
            class_dir = output_dir / class_name
            class_dir.mkdir(exist_ok=True)
            
            # 
            clip_samples = int(clip_duration * sr)
            hop_samples = int(hop_duration * sr)
            
            clip_count = 0
            for start_sample in range(0, len(audio_data) - clip_samples, hop_samples):
                end_sample = start_sample + clip_samples
                clip_data = audio_data[start_sample:end_sample]
                
                # 
                if np.max(np.abs(clip_data)) < 0.01:
                    continue
                
                # save
                clip_filename = f"{audio_file.stem}_clip_{clip_count:03d}.wav"
                clip_path = class_dir / clip_filename
                
                sf.write(str(clip_path), clip_data, sr)
                clip_count += 1
                total_clips += 1
            
            print(f"   {clip_count}  -> {class_name}/")
            
        except Exception as e:
            print(f"  ❌ : {e}")
    
    print(f"\n🎉 ！ {total_clips} ")
    
    # 
    print("\n📊 :")
    for class_dir in output_dir.iterdir():
        if class_dir.is_dir():
            count = len(list(class_dir.glob('*.wav')))
            print(f"  {class_dir.name}: {count} ")

if __name__ == '__main__':
    input_dir = sys.argv[1] if len(sys.argv) > 1 else '../data/samples'
    output_dir = sys.argv[2] if len(sys.argv) > 2 else '../data/clips'
    
    print("🎵 ")
    print(f": {input_dir}")
    print(f": {output_dir}")
    print(f": 3")
    
    split_audio_files(input_dir, output_dir)