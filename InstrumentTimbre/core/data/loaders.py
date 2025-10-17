"""
Data loaders for InstrumentTimbre training and evaluation
"""

import torch
from torch.utils.data import DataLoader, random_split
from typing import Tuple, Dict, Any, Optional
import logging

from .datasets import AudioDataset, ChineseInstrumentDataset

def get_dataloader(dataset: torch.utils.data.Dataset, 
                   batch_size: int = 32,
                   shuffle: bool = True,
                   num_workers: int = 0,  # 0 for faster loading with cached features
                   pin_memory: bool = False) -> DataLoader:  # Disable for MPS compatibility
    """
    Create a DataLoader from a dataset
    
    Args:
        dataset: PyTorch dataset
        batch_size: Batch size for training
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for GPU
        
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True if shuffle else False
    )

def create_train_val_loaders(data_dir: str,
                           train_split: float = 0.8,
                           batch_size: int = 32,
                           config: Optional[Dict[str, Any]] = None,
                           use_chinese_features: bool = True,
                           **kwargs) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders
    
    Args:
        data_dir: Directory containing training data
        train_split: Fraction of data to use for training
        batch_size: Batch size
        config: Configuration for feature extraction
        use_chinese_features: Whether to use Chinese instrument features
        **kwargs: Additional arguments for DataLoader
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    logger = logging.getLogger(__name__)
    
    # Create fast dataset with pre-computed features
    from .fast_dataset import FastAudioDataset
    from ..features.fast import FastFeatureExtractor
    
    if use_chinese_features:
        fast_extractor = FastFeatureExtractor(config)
        dataset = FastAudioDataset(
            data_dir=data_dir,
            feature_extractor=fast_extractor
        )
    else:
        dataset = FastAudioDataset(data_dir=data_dir)
    
    # Split dataset
    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    logger.info(f"Created train/val split: {train_size}/{val_size} samples")
    
    # Create loaders
    train_loader = get_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        **kwargs
    )
    
    val_loader = get_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        **kwargs
    )
    
    return train_loader, val_loader