"""
Batch predictor for efficient processing of multiple audio files
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from .predictor import InstrumentPredictor

class BatchPredictor:
    """
    Batch predictor for processing multiple audio files efficiently
    """
    
    def __init__(self, model_path: str, config: Optional[Dict[str, Any]] = None,
                 device: str = 'auto', max_workers: int = 4):
        """
        Initialize batch predictor
        
        Args:
            model_path: Path to trained model
            config: Configuration dictionary
            device: Device for inference
            max_workers: Maximum number of worker threads
        """
        self.predictor = InstrumentPredictor(model_path, config, device)
        self.max_workers = max_workers
        self.logger = logging.getLogger(__name__)
    
    def predict_batch(self, audio_files: List[str], 
                     batch_size: int = 8,
                     top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Predict for a batch of audio files
        
        Args:
            audio_files: List of audio file paths
            batch_size: Batch size for processing
            top_k: Number of top predictions
            
        Returns:
            List of prediction results
        """
        results = []
        
        # Process in batches
        for i in range(0, len(audio_files), batch_size):
            batch_files = audio_files[i:i + batch_size]
            batch_results = []
            
            # Use thread pool for I/O operations
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(batch_files))) as executor:
                future_to_file = {
                    executor.submit(self._predict_single_safe, file_path, top_k): file_path
                    for file_path in batch_files
                }
                
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        result = future.result()
                        result['file'] = file_path
                        batch_results.append(result)
                    except Exception as e:
                        self.logger.warning(f"Failed to process {file_path}: {e}")
                        batch_results.append({
                            'file': file_path,
                            'error': str(e),
                            'predictions': []
                        })
            
            results.extend(batch_results)
        
        return results
    
    def _predict_single_safe(self, file_path: str, top_k: int) -> Dict[str, Any]:
        """Safely predict for a single file"""
        try:
            return self.predictor.predict_file(file_path, top_k)
        except Exception as e:
            return {
                'error': str(e),
                'predictions': []
            }
    
    def predict_directory(self, directory: str, 
                         audio_extensions: List[str] = None,
                         recursive: bool = True,
                         **kwargs) -> List[Dict[str, Any]]:
        """
        Predict for all audio files in a directory
        
        Args:
            directory: Directory path
            audio_extensions: Audio file extensions to process
            recursive: Whether to search recursively
            **kwargs: Additional arguments for predict_batch
            
        Returns:
            List of prediction results
        """
        if audio_extensions is None:
            audio_extensions = ['.wav', '.mp3', '.flac', '.m4a']
        
        directory = Path(directory)
        audio_files = []
        
        for ext in audio_extensions:
            if recursive:
                audio_files.extend(directory.glob(f'**/*{ext}'))
            else:
                audio_files.extend(directory.glob(f'*{ext}'))
        
        audio_files = [str(f) for f in audio_files]
        self.logger.info(f"Found {len(audio_files)} audio files in {directory}")
        
        return self.predict_batch(audio_files, **kwargs)