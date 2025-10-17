"""
CNN-based classifier for Chinese instrument timbre analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional

from .base import BaseModel

class CNNClassifier(BaseModel):
    """
    CNN-based classifier optimized for Chinese instrument features
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize CNN classifier
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        
        # Model parameters
        self.input_dim = self.config.get('input_dim', 128)
        self.hidden_dims = self.config.get('hidden_dims', [256, 512, 256, 128])
        self.kernel_sizes = self.config.get('kernel_sizes', [3, 3, 3, 3])
        self.pool_sizes = self.config.get('pool_sizes', [2, 2, 2, 2])
        self.use_batch_norm = self.config.get('use_batch_norm', True)
        self.use_residual = self.config.get('use_residual', False)
        
        # Build model layers
        self._build_feature_extractor()
        self._build_classifier()
        
        # Initialize weights
        self.initialize_weights()
        
        self.logger.info(f"CNN Classifier initialized with {self.get_model_info()['total_parameters']} parameters")
    
    def _build_feature_extractor(self):
        """Build CNN feature extraction layers"""
        layers = []
        
        # Reshape input to add channel dimension if needed
        # Assuming input is [batch_size, features] -> [batch_size, 1, features]
        
        in_channels = 1
        current_size = self.input_dim
        
        for i, (hidden_dim, kernel_size, pool_size) in enumerate(
            zip(self.hidden_dims, self.kernel_sizes, self.pool_sizes)
        ):
            # Convolutional layer
            conv_layer = nn.Conv1d(
                in_channels=in_channels,
                out_channels=hidden_dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2
            )
            layers.append(conv_layer)
            
            # Batch normalization
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation
            layers.append(nn.ReLU(inplace=True))
            
            # Pooling
            if pool_size > 1:
                layers.append(nn.MaxPool1d(kernel_size=pool_size))
                current_size = current_size // pool_size
            
            # Dropout
            layers.append(nn.Dropout(self.dropout_rate))
            
            in_channels = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        self.feature_size = current_size * self.hidden_dims[-1]
        
        # Global average pooling option
        if self.config.get('use_global_pooling', False):
            self.global_pool = nn.AdaptiveAvgPool1d(1)
            self.feature_size = self.hidden_dims[-1]
        else:
            self.global_pool = None
    
    def _build_classifier(self):
        """Build classification head"""
        classifier_layers = []
        
        # Flatten layer is handled in forward pass
        
        # Fully connected layers
        fc_dims = self.config.get('fc_dims', [512, 256])
        
        in_features = self.feature_size
        for fc_dim in fc_dims:
            classifier_layers.extend([
                nn.Linear(in_features, fc_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout_rate)
            ])
            in_features = fc_dim
        
        # Final classification layer
        classifier_layers.append(nn.Linear(in_features, self.num_classes))
        
        self.classifier = nn.Sequential(*classifier_layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor [batch_size, features]
            
        Returns:
            Output logits [batch_size, num_classes]
        """
        # Reshape for 1D convolution: [batch_size, features] -> [batch_size, 1, features]
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Extract features using CNN
        features = self.feature_extractor(x)
        
        # Apply global pooling if configured
        if self.global_pool is not None:
            features = self.global_pool(features)
            features = features.squeeze(-1)  # Remove the last dimension
        else:
            # Flatten features
            features = features.view(features.size(0), -1)
        
        # Classification
        output = self.classifier(features)
        
        return output
    
    def get_feature_dim(self) -> int:
        """Get expected input feature dimension"""
        return self.input_dim
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract feature representations (before classification)
        
        Args:
            x: Input tensor
            
        Returns:
            Feature representations
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        features = self.feature_extractor(x)
        
        if self.global_pool is not None:
            features = self.global_pool(features)
            features = features.squeeze(-1)
        else:
            features = features.view(features.size(0), -1)
        
        return features

class ResidualBlock(nn.Module):
    """
    Residual block for CNN
    """
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, stride: int = 1):
        """
        Initialize residual block
        
        Args:
            in_channels: Input channels
            out_channels: Output channels
            kernel_size: Convolution kernel size
            stride: Convolution stride
        """
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              stride=stride, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 
                              padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Skip connection
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.skip = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        identity = self.skip(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += identity
        out = F.relu(out)
        
        return out

class EnhancedCNNClassifier(CNNClassifier):
    """
    Enhanced CNN with residual connections and attention
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize enhanced CNN"""
        # Set default configuration for enhanced model
        default_config = {
            'use_residual': True,
            'use_attention': True,
            'use_batch_norm': True,
            'use_global_pooling': True
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(default_config)
    
    def _build_feature_extractor(self):
        """Build enhanced feature extractor with residual blocks"""
        if not self.use_residual:
            return super()._build_feature_extractor()
        
        layers = []
        in_channels = 1
        current_size = self.input_dim
        
        for i, (hidden_dim, kernel_size, pool_size) in enumerate(
            zip(self.hidden_dims, self.kernel_sizes, self.pool_sizes)
        ):
            # Residual block
            layers.append(ResidualBlock(in_channels, hidden_dim, kernel_size))
            
            # Pooling
            if pool_size > 1:
                layers.append(nn.MaxPool1d(kernel_size=pool_size))
                current_size = current_size // pool_size
            
            # Dropout
            layers.append(nn.Dropout(self.dropout_rate))
            
            in_channels = hidden_dim
        
        # Attention mechanism
        if self.config.get('use_attention', False):
            layers.append(ChannelAttention(self.hidden_dims[-1]))
        
        self.feature_extractor = nn.Sequential(*layers)
        
        if self.config.get('use_global_pooling', True):
            self.global_pool = nn.AdaptiveAvgPool1d(1)
            self.feature_size = self.hidden_dims[-1]
        else:
            self.feature_size = current_size * self.hidden_dims[-1]
            self.global_pool = None

class ChannelAttention(nn.Module):
    """
    Channel attention mechanism
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        """
        Initialize channel attention
        
        Args:
            channels: Number of input channels
            reduction: Reduction ratio for attention
        """
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        b, c, _ = x.size()
        
        # Average and max pooling
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        
        # Attention weights
        attention = self.sigmoid(avg_out + max_out).view(b, c, 1)
        
        return x * attention