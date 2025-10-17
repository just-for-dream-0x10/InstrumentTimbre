"""
Unified Audio Emotion Analysis + Music Generation Model

This module implements a unified Transformer-based model that can handle both
analysis tasks (emotion, instrument, style recognition) and generation tasks
(music creation, style transfer, orchestration) using a shared backbone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Union
import logging

from .base import BaseModel
from .transformer import PositionalEncoding


class UnifiedMusicModel(BaseModel):
    """
    Unified model for music analysis and generation tasks
    
    Architecture:
    - Shared audio encoder (compatible with existing feature extractors)
    - Unified Transformer backbone (12B parameters)
    - Multiple lightweight task-specific heads
    - Backward compatible with existing InstrumentTimbre API
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize unified model
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        
        # Model dimensions
        self.input_dim = self.config.get('input_dim', 128)  # Compatible with existing
        self.d_model = self.config.get('d_model', 1024)     # Large unified representation
        self.nhead = self.config.get('nhead', 16)
        self.num_layers = self.config.get('num_layers', 24)  # Deep transformer
        self.dim_feedforward = self.config.get('dim_feedforward', 4096)
        
        # Task configuration
        self.enable_analysis = self.config.get('enable_analysis', True)
        self.enable_generation = self.config.get('enable_generation', False)
        self.enable_control = self.config.get('enable_control', False)
        
        # Build model components
        self._build_shared_encoder()
        self._build_unified_transformer()
        self._build_task_heads()
        
        # Initialize weights
        self.initialize_weights()
        
        self.logger.info(f"UnifiedMusicModel initialized with {self.get_model_info()['total_parameters']} parameters")
        self.logger.info(f"Tasks enabled - Analysis: {self.enable_analysis}, Generation: {self.enable_generation}, Control: {self.enable_control}")
    
    def _build_shared_encoder(self):
        """Build shared audio encoder - compatible with existing features"""
        
        # Input projection to unified representation
        self.input_projection = nn.Sequential(
            nn.Linear(self.input_dim, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.d_model // 2, self.d_model),
            nn.LayerNorm(self.d_model)
        )
        
        # Positional encoding for sequence processing
        self.pos_encoding = PositionalEncoding(self.d_model, max_len=1000)
        
    def _build_unified_transformer(self):
        """Build unified Transformer backbone"""
        
        # Enhanced transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout_rate,
            activation='gelu',  # Better for large models
            batch_first=True,
            norm_first=True     # Pre-norm for stability
        )
        
        self.unified_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers,
            norm=nn.LayerNorm(self.d_model)
        )
        
    def _build_task_heads(self):
        """Build task-specific heads"""
        
        self.task_heads = nn.ModuleDict()
        
        # Analysis heads (compatible with existing system)
        if self.enable_analysis:
            self.task_heads['emotion_classifier'] = self._build_classification_head(
                'emotion', self.config.get('num_emotion_classes', 6)
            )
            self.task_heads['instrument_classifier'] = self._build_classification_head(
                'instrument', self.num_classes  # Use existing num_classes (5 instruments)
            )
            self.task_heads['style_classifier'] = self._build_classification_head(
                'style', self.config.get('num_style_classes', 10)
            )
            self.task_heads['intensity_regressor'] = self._build_regression_head(
                'intensity', 6  # 6-dimensional emotion intensity
            )
        
        # Generation heads (new functionality)
        if self.enable_generation:
            self.task_heads['melody_generator'] = self._build_generation_head('melody')
            self.task_heads['harmony_generator'] = self._build_generation_head('harmony')
            self.task_heads['rhythm_generator'] = self._build_generation_head('rhythm')
            self.task_heads['orchestration_generator'] = self._build_generation_head('orchestration')
        
        # Control heads (new functionality)
        if self.enable_control:
            self.task_heads['style_transfer'] = self._build_control_head('style_transfer')
            self.task_heads['tempo_control'] = self._build_control_head('tempo')
            self.task_heads['key_control'] = self._build_control_head('key')
            self.task_heads['melody_preservation'] = self._build_control_head('melody_preservation')
    
    def _build_classification_head(self, task_name: str, num_classes: int) -> nn.Module:
        """Build classification head for specific task"""
        return nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.d_model // 2, num_classes)
        )
    
    def _build_regression_head(self, task_name: str, output_dim: int) -> nn.Module:
        """Build regression head for continuous outputs"""
        return nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.d_model, self.d_model // 4),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.d_model // 4, output_dim),
            nn.Sigmoid()  # Output range [0, 1]
        )
    
    def _build_generation_head(self, task_name: str) -> nn.Module:
        """Build generation head for music creation tasks"""
        return nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.d_model, self.d_model // 2)
        )
    
    def _build_control_head(self, task_name: str) -> nn.Module:
        """Build control head for music transformation tasks"""
        return nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.d_model // 2, self.d_model // 4)
        )
    
    def forward(self, x: torch.Tensor, task_type: str = 'analysis', 
                task_names: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass with task-specific outputs
        
        Args:
            x: Input tensor [batch_size, features] or [batch_size, seq_len, features]
            task_type: Type of task ('analysis', 'generation', 'control', 'unified')
            task_names: Specific task names to execute
            
        Returns:
            Dictionary of task outputs
        """
        # Handle input dimensions
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
        
        # Shared encoding
        x = self.input_projection(x)  # [batch_size, seq_len, d_model]
        x = self.pos_encoding(x)
        
        # Unified transformer processing
        unified_repr = self.unified_transformer(x)  # [batch_size, seq_len, d_model]
        
        # Global pooling for classification tasks
        pooled_repr = torch.mean(unified_repr, dim=1)  # [batch_size, d_model]
        
        # Execute specific tasks
        outputs = {}
        
        if task_type == 'analysis' or task_type == 'unified':
            outputs.update(self._forward_analysis(pooled_repr, task_names))
        
        if task_type == 'generation' or task_type == 'unified':
            outputs.update(self._forward_generation(unified_repr, task_names))
        
        if task_type == 'control' or task_type == 'unified':
            outputs.update(self._forward_control(pooled_repr, task_names))
        
        return outputs
    
    def _forward_analysis(self, x: torch.Tensor, task_names: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """Forward pass for analysis tasks"""
        outputs = {}
        
        analysis_tasks = ['emotion_classifier', 'instrument_classifier', 'style_classifier', 'intensity_regressor']
        if task_names:
            analysis_tasks = [t for t in analysis_tasks if t in task_names]
        
        for task_name in analysis_tasks:
            if task_name in self.task_heads:
                outputs[task_name] = self.task_heads[task_name](x)
        
        return outputs
    
    def _forward_generation(self, x: torch.Tensor, task_names: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """Forward pass for generation tasks"""
        outputs = {}
        
        generation_tasks = ['melody_generator', 'harmony_generator', 'rhythm_generator', 'orchestration_generator']
        if task_names:
            generation_tasks = [t for t in generation_tasks if t in task_names]
        
        # Use sequence representation for generation
        pooled_x = torch.mean(x, dim=1)  # [batch_size, d_model]
        
        for task_name in generation_tasks:
            if task_name in self.task_heads:
                outputs[task_name] = self.task_heads[task_name](pooled_x)
        
        return outputs
    
    def _forward_control(self, x: torch.Tensor, task_names: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """Forward pass for control tasks"""
        outputs = {}
        
        control_tasks = ['style_transfer', 'tempo_control', 'key_control', 'melody_preservation']
        if task_names:
            control_tasks = [t for t in control_tasks if t in task_names]
        
        for task_name in control_tasks:
            if task_name in self.task_heads:
                outputs[task_name] = self.task_heads[task_name](x)
        
        return outputs
    
    def get_feature_dim(self) -> int:
        """Get expected input feature dimension"""
        return self.input_dim
    
    def freeze_backbone(self):
        """Freeze the unified transformer backbone"""
        for param in self.unified_transformer.parameters():
            param.requires_grad = False
        for param in self.input_projection.parameters():
            param.requires_grad = False
        self.logger.info("Unified transformer backbone frozen")
    
    def unfreeze_backbone(self):
        """Unfreeze the unified transformer backbone"""
        for param in self.unified_transformer.parameters():
            param.requires_grad = True
        for param in self.input_projection.parameters():
            param.requires_grad = True
        self.logger.info("Unified transformer backbone unfrozen")
    
    def get_task_heads(self) -> List[str]:
        """Get list of available task heads"""
        return list(self.task_heads.keys())