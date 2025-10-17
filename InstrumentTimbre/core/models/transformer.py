"""
Transformer-based classifier for Chinese instrument timbre analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Any, Optional

from .base import BaseModel

class TransformerClassifier(BaseModel):
    """
    Transformer-based classifier for Chinese instrument timbre analysis
    
    This model uses self-attention mechanisms to capture long-range dependencies
    in audio features, particularly useful for analyzing the complex timbral
    relationships in Chinese traditional instruments.
    
    Features:
    - Multi-head self-attention for feature interaction modeling
    - Positional encoding for sequence-aware processing
    - Layer normalization and residual connections
    - Configurable depth and attention heads
    - Support for both single-frame and sequence inputs
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Transformer classifier
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        
        # Model parameters
        self.input_dim = self.config.get('input_dim', 128)
        self.d_model = self.config.get('d_model', 256)
        self.nhead = self.config.get('nhead', 8)
        self.num_layers = self.config.get('num_layers', 6)
        self.dim_feedforward = self.config.get('dim_feedforward', 512)
        
        # Input projection
        self.input_projection = nn.Linear(self.input_dim, self.d_model)
        
        # Positional encoding (placeholder for sequence data)
        self.pos_encoding = PositionalEncoding(self.d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout_rate,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.d_model // 2, self.num_classes)
        )
        
        # Initialize weights
        self.initialize_weights()
        
        self.logger.info(f"Transformer Classifier initialized with {self.get_model_info()['total_parameters']} parameters")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor [batch_size, features] or [batch_size, seq_len, features]
            
        Returns:
            Output logits [batch_size, num_classes]
        """
        # Handle different input dimensions
        if x.dim() == 2:
            # [batch_size, features] -> [batch_size, 1, features]
            x = x.unsqueeze(1)
        
        # Project to model dimension
        x = self.input_projection(x)  # [batch_size, seq_len, d_model]
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply transformer
        x = self.transformer(x)  # [batch_size, seq_len, d_model]
        
        # Global average pooling over sequence dimension
        x = torch.mean(x, dim=1)  # [batch_size, d_model]
        
        # Classification
        output = self.classifier(x)  # [batch_size, num_classes]
        
        return output
    
    def get_feature_dim(self) -> int:
        """Get expected input feature dimension"""
        return self.input_dim

class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Initialize positional encoding
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
        """
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Tensor with positional encoding added
        """
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :].transpose(0, 1)
        return x