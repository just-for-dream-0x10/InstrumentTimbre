"""
Base model class for InstrumentTimbre
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import logging

class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all InstrumentTimbre models
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize base model
        
        Args:
            config: Model configuration dictionary
        """
        super().__init__()
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Common parameters
        self.num_classes = self.config.get('num_classes', 5)
        self.dropout_rate = self.config.get('dropout_rate', 0.1)
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        pass
    
    @abstractmethod
    def get_feature_dim(self) -> int:
        """
        Get expected input feature dimension
        
        Returns:
            Feature dimension
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information
        
        Returns:
            Dictionary with model metadata
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_class': self.__class__.__name__,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'num_classes': self.num_classes,
            'config': self.config
        }
    
    def save_checkpoint(self, filepath: str, epoch: int, 
                       optimizer_state: Optional[Dict] = None,
                       metrics: Optional[Dict] = None):
        """
        Save model checkpoint
        
        Args:
            filepath: Path to save checkpoint
            epoch: Current epoch
            optimizer_state: Optimizer state dict
            metrics: Training metrics
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'model_config': self.config,
            'model_class': self.__class__.__name__,
            'metrics': metrics or {}
        }
        
        if optimizer_state:
            checkpoint['optimizer_state_dict'] = optimizer_state
            
        torch.save(checkpoint, filepath)
        self.logger.info(f"Checkpoint saved to {filepath}")
    
    @classmethod
    def load_checkpoint(cls, filepath: str, map_location: str = 'cpu'):
        """
        Load model from checkpoint
        
        Args:
            filepath: Path to checkpoint file
            map_location: Device to load to
            
        Returns:
            Tuple of (model, checkpoint_data)
        """
        checkpoint = torch.load(filepath, map_location=map_location)
        
        # Create model instance
        config = checkpoint.get('model_config', {})
        model = cls(config)
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model, checkpoint
    
    def freeze_layers(self, layer_names: list):
        """
        Freeze specified layers
        
        Args:
            layer_names: List of layer names to freeze
        """
        for name, param in self.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = False
                self.logger.info(f"Frozen layer: {name}")
    
    def unfreeze_layers(self, layer_names: list):
        """
        Unfreeze specified layers
        
        Args:
            layer_names: List of layer names to unfreeze
        """
        for name, param in self.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = True
                self.logger.info(f"Unfrozen layer: {name}")
    
    def initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)