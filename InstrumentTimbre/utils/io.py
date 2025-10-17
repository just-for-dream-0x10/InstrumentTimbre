"""
Input/Output utilities for audio files
"""

import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import List, Tuple, Union, Optional

def load_audio(file_path: Union[str, Path], 
               sample_rate: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """Load audio file using librosa"""
    try:
        audio_data, sr = librosa.load(str(file_path), sr=sample_rate)
        return audio_data, sr
    except Exception as e:
        raise RuntimeError(f"Failed to load audio file {file_path}: {e}")

def save_audio(audio_data: np.ndarray, 
               file_path: Union[str, Path], 
               sample_rate: int) -> None:
    """Save audio data to file"""
    try:
        sf.write(str(file_path), audio_data, sample_rate)
    except Exception as e:
        raise RuntimeError(f"Failed to save audio file {file_path}: {e}")

def find_audio_files(directory: Union[str, Path], 
                    extensions: Optional[List[str]] = None,
                    recursive: bool = True) -> List[Path]:
    """Find all audio files in directory"""
    if extensions is None:
        extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
    
    directory = Path(directory)
    audio_files = []
    
    if recursive:
        pattern = '**/*'
    else:
        pattern = '*'
    
    for file_path in directory.glob(pattern):
        if file_path.is_file() and file_path.suffix.lower() in extensions:
            audio_files.append(file_path)
    
    return sorted(audio_files)

def get_audio_info(file_path: Union[str, Path]) -> dict:
    """Get basic audio file information"""
    try:
        info = sf.info(str(file_path))
        return {
            'duration': info.duration,
            'sample_rate': info.samplerate,
            'channels': info.channels,
            'frames': info.frames,
            'format': info.format,
            'subtype': info.subtype
        }
    except Exception as e:
        raise RuntimeError(f"Failed to get audio info for {file_path}: {e}")