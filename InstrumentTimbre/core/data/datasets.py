"""
Dataset classes for InstrumentTimbre
"""

import os
import torch
import numpy as np
import librosa
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
import logging

from ..features.chinese import ChineseInstrumentAnalyzer

class AudioDataset(Dataset):
    """
    Base audio dataset class
    """
    
    def __init__(self, data_dir: str, feature_extractor=None, 
                 sample_rate: int = 22050, max_files: Optional[int] = None):
        """
        Initialize audio dataset
        
        Args:
            data_dir: Directory containing audio files
            feature_extractor: Feature extractor to use
            sample_rate: Target sample rate
            max_files: Maximum number of files to load (for debugging)
        """
        self.data_dir = Path(data_dir)
        self.feature_extractor = feature_extractor
        self.sample_rate = sample_rate
        self.max_files = max_files
        self.logger = logging.getLogger(__name__)
        
        # Supported audio formats
        self.audio_extensions = {'.wav', '.mp3', '.flac', '.m4a'}
        
        # Load data
        self.data_files, self.labels = self._load_data()
        self.class_names = sorted(list(set(self.labels)))
        self.label_to_idx = {label: idx for idx, label in enumerate(self.class_names)}
        
        self.logger.info(f"Loaded {len(self.data_files)} files from {len(self.class_names)} classes")
    
    def _load_data(self) -> Tuple[List[Path], List[str]]:
        """Load audio file paths and labels"""
        files = []
        labels = []
        
        # Check if data is organized in subdirectories (class folders)
        subdirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        
        if subdirs:
            # Class-based organization: data/erhu/*.wav, data/pipa/*.wav
            self.logger.info("Found class-based directory structure")
            for class_dir in subdirs:
                class_name = class_dir.name
                class_files = []
                
                for file_path in class_dir.iterdir():
                    if file_path.suffix.lower() in self.audio_extensions:
                        class_files.append(file_path)
                
                # Apply max_files limit per class if specified
                if self.max_files:
                    class_files = class_files[:self.max_files // len(subdirs)]
                
                files.extend(class_files)
                labels.extend([class_name] * len(class_files))
        else:
            # Flat organization: all files in one directory, infer class from filename
            self.logger.info("Found flat directory structure, inferring classes from filenames")
            all_files = [f for f in self.data_dir.iterdir() 
                        if f.is_file() and f.suffix.lower() in self.audio_extensions]
            
            for file_path in all_files:
                # Infer class from filename
                filename = file_path.stem.lower()
                
                # Try to detect instrument from filename
                if 'erhu' in filename or 'description' in filename:
                    class_name = 'erhu'
                elif 'pipa' in filename or 'description' in filename:
                    class_name = 'pipa'
                elif 'guzheng' in filename or 'description' in filename:
                    class_name = 'guzheng'
                elif 'dizi' in filename or 'description' in filename:
                    class_name = 'dizi'
                elif 'guqin' in filename or 'description' in filename:
                    class_name = 'guqin'
                elif 'piano' in filename:
                    class_name = 'piano'
                elif 'bass' in filename:
                    class_name = 'bass'
                elif 'drum' in filename:
                    class_name = 'drums'
                elif 'vocal' in filename:
                    class_name = 'vocals'
                else:
                    # Default classification for songs
                    class_name = 'mixed'
                
                files.append(file_path)
                labels.append(class_name)
            
            # Apply max_files limit if specified
            if self.max_files and len(files) > self.max_files:
                files = files[:self.max_files]
                labels = labels[:self.max_files]
        
        return files, labels
    
    def __len__(self) -> int:
        return len(self.data_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get item by index"""
        file_path = self.data_files[idx]
        label = self.labels[idx]
        label_idx = self.label_to_idx[label]
        
        # Load audio
        try:
            audio_data, sr = librosa.load(str(file_path), sr=self.sample_rate)
            
            # Extract features if feature extractor is provided
            if self.feature_extractor:
                features = self.feature_extractor.extract_features(audio_data, sr)
                # Convert feature dict to tensor
                feature_vector = self._features_to_tensor(features)
            else:
                # Return raw audio
                feature_vector = torch.FloatTensor(audio_data)
                
        except Exception as e:
            self.logger.warning(f"Failed to load {file_path}: {e}")
            # Return zero tensor on failure
            if self.feature_extractor:
                feature_vector = torch.zeros(self._get_feature_dim())
            else:
                feature_vector = torch.zeros(self.sample_rate)  # 1 second of zeros
        
        return feature_vector, label_idx
    
    def _features_to_tensor(self, features: Dict[str, np.ndarray]) -> torch.Tensor:
        """Convert feature dictionary to tensor"""
        feature_list = []
        
        for key in sorted(features.keys()):
            feature = features[key]
            if isinstance(feature, np.ndarray):
                if feature.ndim == 0:  # Scalar
                    feature_list.append(feature.item())
                else:  # Array
                    feature_list.extend(feature.flatten())
            else:  # Scalar
                feature_list.append(float(feature))
        
        return torch.FloatTensor(feature_list)
    
    def _get_feature_dim(self) -> int:
        """Get feature dimension"""
        if hasattr(self.feature_extractor, 'get_feature_names'):
            return len(self.feature_extractor.get_feature_names())
        return 50  # Default
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced datasets"""
        class_counts = torch.zeros(len(self.class_names))
        
        for label in self.labels:
            class_counts[self.label_to_idx[label]] += 1
        
        total_samples = len(self.labels)
        weights = total_samples / (len(self.class_names) * class_counts)
        
        return weights

class ChineseInstrumentDataset(AudioDataset):
    """
    Dataset specifically for Chinese instruments with enhanced features
    """
    
    def __init__(self, data_dir: str, config: Optional[Dict[str, Any]] = None,
                 sample_rate: int = 22050, max_files: Optional[int] = None,
                 enhanced_features: bool = True):
        """
        Initialize Chinese instrument dataset
        
        Args:
            data_dir: Directory containing instrument audio files
            config: Configuration for feature extraction
            sample_rate: Target sample rate
            max_files: Maximum files to load
            enhanced_features: Whether to use enhanced Chinese features
        """
        # Initialize Chinese feature extractor
        feature_extractor = None
        if enhanced_features:
            feature_extractor = ChineseInstrumentAnalyzer(config)
        
        super().__init__(data_dir, feature_extractor, sample_rate, max_files)
        
        # Map directory names to instrument types
        self.instrument_mapping = {
            'erhu': 'erhu',
            'pipa': 'pipa', 
            'guzheng': 'guzheng',
            'dizi': 'dizi',
            'guqin': 'guqin'
        }
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """Get item with instrument type information"""
        file_path = self.data_files[idx]
        label = self.labels[idx]
        label_idx = self.label_to_idx[label]
        
        # Determine instrument type
        instrument_type = self.instrument_mapping.get(label.lower(), None)
        
        try:
            audio_data, sr = librosa.load(str(file_path), sr=self.sample_rate)
            
            if self.feature_extractor:
                # Pass instrument type for optimized feature extraction
                features = self.feature_extractor.extract_features(
                    audio_data, sr, instrument_type=instrument_type
                )
                feature_vector = self._features_to_tensor(features)
            else:
                feature_vector = torch.FloatTensor(audio_data)
                
        except Exception as e:
            self.logger.warning(f"Failed to load {file_path}: {e}")
            if self.feature_extractor:
                feature_vector = torch.zeros(self._get_feature_dim())
            else:
                feature_vector = torch.zeros(self.sample_rate)
        
        return feature_vector, label_idx, instrument_type or 'unknown'
    
    def save_features_cache(self, cache_path: str):
        """Save extracted features to cache for faster loading"""
        cache_data = {
            'features': [],
            'labels': [],
            'instrument_types': [],
            'class_names': self.class_names,
            'label_to_idx': self.label_to_idx
        }
        
        self.logger.info("Caching features...")
        for i in range(len(self)):
            features, label_idx, instrument_type = self[i]
            cache_data['features'].append(features.numpy())
            cache_data['labels'].append(label_idx)
            cache_data['instrument_types'].append(instrument_type)
        
        np.savez_compressed(cache_path, **cache_data)
        self.logger.info(f"Features cached to {cache_path}")
    
    def load_features_cache(self, cache_path: str) -> bool:
        """Load features from cache"""
        try:
            cache_data = np.load(cache_path, allow_pickle=True)
            self.cached_features = cache_data['features']
            self.cached_labels = cache_data['labels']
            self.cached_instrument_types = cache_data['instrument_types']
            self.class_names = cache_data['class_names'].tolist()
            self.label_to_idx = cache_data['label_to_idx'].item()
            self.use_cache = True
            self.logger.info(f"Loaded features from cache: {cache_path}")
            return True
        except Exception as e:
            self.logger.warning(f"Failed to load cache {cache_path}: {e}")
            return False