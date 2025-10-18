"""
Music Generation Head - Week 5 Development Task

This module implements the music generation head for the unified model,
providing direct audio generation capabilities integrated with the
existing emotion analysis and instrument recognition systems.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum

# Configure logging for the module
logger = logging.getLogger(__name__)


class GenerationMode(Enum):
    """Types of music generation modes"""
    COMPLETE_GENERATION = "complete_generation"
    STYLE_TRANSFER = "style_transfer"
    MELODY_CONTINUATION = "melody_continuation"
    RHYTHM_VARIATION = "rhythm_variation"
    HARMONIZATION = "harmonization"


@dataclass
class GenerationConfig:
    """Configuration for music generation"""
    max_length: int = 1024  # Maximum generation length in frames
    temperature: float = 0.8  # Sampling temperature
    top_k: int = 50  # Top-k sampling
    top_p: float = 0.9  # Top-p (nucleus) sampling
    repetition_penalty: float = 1.1
    guidance_scale: float = 7.5  # Classifier-free guidance
    num_samples: int = 1  # Number of samples to generate
    seed: Optional[int] = None


@dataclass
class GenerationResult:
    """Result of music generation process"""
    generated_features: torch.Tensor  # Generated audio features
    generation_metadata: Dict[str, Any]  # Metadata about generation process
    quality_scores: Dict[str, float]  # Quality assessment scores
    intermediate_states: List[torch.Tensor]  # Intermediate generation states
    attention_weights: Optional[torch.Tensor] = None  # Attention visualization


class MusicGenerationHead(nn.Module):
    """
    Music generation head for unified model architecture
    
    This module provides:
    - Direct audio feature generation
    - Style-conditioned generation
    - Emotion-guided generation
    - Integration with preservation engines
    """
    
    def __init__(self, 
                 input_dim: int = 768,
                 hidden_dim: int = 512,
                 output_dim: int = 128,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 max_sequence_length: int = 1024):
        """
        Initialize music generation head
        
        Args:
            input_dim: Input feature dimension from shared backbone
            hidden_dim: Hidden dimension for generation layers
            output_dim: Output audio feature dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout probability
            max_sequence_length: Maximum sequence length
        """
        super(MusicGenerationHead, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.max_sequence_length = max_sequence_length
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.positional_encoding = self._create_positional_encoding(
            max_sequence_length, hidden_dim
        )
        
        # Generation transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, output_dim)
        
        # Condition embedding layers
        self.emotion_embedding = nn.Embedding(10, hidden_dim)  # 6 emotions + padding
        self.style_embedding = nn.Embedding(8, hidden_dim)     # Style types
        self.instrument_embedding = nn.Embedding(20, hidden_dim) # Instrument types
        
        # Generation control
        self.temperature_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Quality prediction head
        self.quality_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 4),  # melody, rhythm, harmony, overall
            nn.Sigmoid()
        )
        
        logger.info("MusicGenerationHead initialized: %d->%d->%d, %d layers",
                   input_dim, hidden_dim, output_dim, num_layers)
    
    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional encoding"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)  # Add batch dimension
    
    def forward(self, 
                shared_features: torch.Tensor,
                generation_config: Optional[GenerationConfig] = None,
                conditioning: Optional[Dict[str, torch.Tensor]] = None,
                generation_mode: GenerationMode = GenerationMode.COMPLETE_GENERATION) -> GenerationResult:
        """
        Forward pass for music generation
        
        Args:
            shared_features: Features from shared backbone
            generation_config: Configuration for generation
            conditioning: Conditioning information (emotion, style, etc.)
            generation_mode: Type of generation to perform
            
        Returns:
            GenerationResult with generated audio features
        """
        config = generation_config or GenerationConfig()
        batch_size = shared_features.size(0)
        device = shared_features.device
        
        # Project input features
        hidden_states = self.input_projection(shared_features)
        
        # Add conditioning if provided
        if conditioning:
            hidden_states = self._apply_conditioning(hidden_states, conditioning)
        
        # Generate based on mode
        if generation_mode == GenerationMode.COMPLETE_GENERATION:
            result = self._complete_generation(hidden_states, config)
        elif generation_mode == GenerationMode.STYLE_TRANSFER:
            result = self._style_transfer_generation(hidden_states, config, conditioning)
        elif generation_mode == GenerationMode.MELODY_CONTINUATION:
            result = self._melody_continuation(hidden_states, config)
        else:
            # Default to complete generation
            result = self._complete_generation(hidden_states, config)
        
        return result
    
    def _apply_conditioning(self, 
                          hidden_states: torch.Tensor,
                          conditioning: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Apply conditioning signals to hidden states"""
        
        conditioned_states = hidden_states.clone()
        
        # Apply emotion conditioning
        if 'emotion' in conditioning:
            emotion_ids = conditioning['emotion']
            emotion_embeds = self.emotion_embedding(emotion_ids)
            conditioned_states = conditioned_states + emotion_embeds.unsqueeze(1)
        
        # Apply style conditioning
        if 'style' in conditioning:
            style_ids = conditioning['style']
            style_embeds = self.style_embedding(style_ids)
            conditioned_states = conditioned_states + style_embeds.unsqueeze(1)
        
        # Apply instrument conditioning
        if 'instrument' in conditioning:
            instrument_ids = conditioning['instrument']
            instrument_embeds = self.instrument_embedding(instrument_ids)
            conditioned_states = conditioned_states + instrument_embeds.unsqueeze(1)
        
        return conditioned_states
    
    def _complete_generation(self, 
                           hidden_states: torch.Tensor,
                           config: GenerationConfig) -> GenerationResult:
        """Perform complete music generation from scratch"""
        
        batch_size, seq_len, hidden_dim = hidden_states.shape
        device = hidden_states.device
        
        # Initialize generation
        generated_sequence = torch.zeros(
            batch_size, config.max_length, self.output_dim, device=device
        )
        intermediate_states = []
        quality_scores = {'melody': 0.0, 'rhythm': 0.0, 'harmony': 0.0, 'overall': 0.0}
        
        # Auto-regressive generation
        current_length = 0
        context = hidden_states
        
        for step in range(config.max_length):
            # Add positional encoding
            pos_encoding = self.positional_encoding[:, :context.size(1), :].to(device)
            context_with_pos = context + pos_encoding
            
            # Apply transformer layers
            for layer in self.transformer_layers:
                # Create causal mask
                tgt_mask = self._generate_square_subsequent_mask(context.size(1)).to(device)
                context_with_pos = layer(context_with_pos, context_with_pos, tgt_mask=tgt_mask)
            
            # Generate next token
            next_token_logits = self.output_projection(context_with_pos[:, -1:, :])
            
            # Apply temperature and sampling
            next_token = self._sample_next_token(next_token_logits, config)
            
            # Add to generated sequence
            generated_sequence[:, step, :] = next_token.squeeze(1)
            
            # Update context for next iteration
            next_token_hidden = self.input_projection(next_token)
            context = torch.cat([context, next_token_hidden], dim=1)
            
            # Keep context length manageable
            if context.size(1) > self.max_sequence_length:
                context = context[:, -self.max_sequence_length:, :]
            
            # Store intermediate state
            if step % 50 == 0:  # Store every 50 steps
                intermediate_states.append(generated_sequence[:, :step+1, :].clone())
            
            current_length = step + 1
        
        # Predict quality scores
        final_hidden = context_with_pos.mean(dim=1)  # Global average pooling
        quality_logits = self.quality_predictor(final_hidden)
        quality_scores = {
            'melody': quality_logits[0, 0].item(),
            'rhythm': quality_logits[0, 1].item(),
            'harmony': quality_logits[0, 2].item(),
            'overall': quality_logits[0, 3].item()
        }
        
        # Create generation metadata
        metadata = {
            'generation_mode': GenerationMode.COMPLETE_GENERATION.value,
            'actual_length': current_length,
            'config': config,
            'conditioning_applied': False
        }
        
        return GenerationResult(
            generated_features=generated_sequence[:, :current_length, :],
            generation_metadata=metadata,
            quality_scores=quality_scores,
            intermediate_states=intermediate_states
        )
    
    def _style_transfer_generation(self,
                                 hidden_states: torch.Tensor,
                                 config: GenerationConfig,
                                 conditioning: Optional[Dict[str, torch.Tensor]]) -> GenerationResult:
        """Perform style transfer generation"""
        
        # For style transfer, we modify the hidden states based on target style
        # and then generate with style-specific modifications
        
        batch_size, seq_len, hidden_dim = hidden_states.shape
        device = hidden_states.device
        
        # Apply style-specific transformations
        if conditioning and 'target_style' in conditioning:
            target_style = conditioning['target_style']
            style_embeds = self.style_embedding(target_style)
            
            # Style-guided modification of hidden states
            style_gate = torch.sigmoid(self.temperature_mlp(style_embeds))
            modified_hidden = hidden_states * (1 - style_gate.unsqueeze(1)) + \
                            style_embeds.unsqueeze(1) * style_gate.unsqueeze(1)
        else:
            modified_hidden = hidden_states
        
        # Generate with modified hidden states
        modified_config = GenerationConfig(
            max_length=min(config.max_length, seq_len * 2),  # Limit length for style transfer
            temperature=config.temperature * 0.8,  # Lower temperature for more controlled generation
            top_k=config.top_k,
            top_p=config.top_p
        )
        
        result = self._complete_generation(modified_hidden, modified_config)
        result.generation_metadata['generation_mode'] = GenerationMode.STYLE_TRANSFER.value
        result.generation_metadata['conditioning_applied'] = True
        
        return result
    
    def _melody_continuation(self,
                           hidden_states: torch.Tensor,
                           config: GenerationConfig) -> GenerationResult:
        """Continue an existing melody"""
        
        # For melody continuation, we use the input as a prompt and continue from there
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Use existing melody as prompt
        prompt_length = min(seq_len, config.max_length // 2)
        continuation_length = config.max_length - prompt_length
        
        # Generate continuation
        continuation_config = GenerationConfig(
            max_length=continuation_length,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p
        )
        
        # Use the last part of hidden states as context for continuation
        continuation_context = hidden_states[:, -prompt_length:, :]
        result = self._complete_generation(continuation_context, continuation_config)
        
        # Combine prompt with continuation
        prompt_features = self.output_projection(hidden_states[:, :prompt_length, :])
        combined_features = torch.cat([prompt_features, result.generated_features], dim=1)
        
        result.generated_features = combined_features
        result.generation_metadata['generation_mode'] = GenerationMode.MELODY_CONTINUATION.value
        result.generation_metadata['prompt_length'] = prompt_length
        
        return result
    
    def _sample_next_token(self, 
                          logits: torch.Tensor,
                          config: GenerationConfig) -> torch.Tensor:
        """Sample next token using various sampling strategies"""
        
        # Apply temperature
        logits = logits / config.temperature
        
        # Apply top-k filtering
        if config.top_k > 0:
            top_k_logits, top_k_indices = torch.topk(logits, config.top_k, dim=-1)
            logits_filtered = torch.full_like(logits, float('-inf'))
            logits_filtered.scatter_(-1, top_k_indices, top_k_logits)
            logits = logits_filtered
        
        # Apply top-p (nucleus) sampling
        if config.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > config.top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')
        
        # Sample from the filtered distribution
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1)
        next_token = next_token.view(logits.size(0), logits.size(1), -1)
        
        # Convert indices back to feature vectors
        # For now, use a simple linear mapping - this would be replaced with
        # a proper vocabulary/codebook in a full implementation
        next_token_features = next_token.float() / self.output_dim
        
        return next_token_features
    
    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate causal mask for autoregressive generation"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def get_generation_capabilities(self) -> Dict[str, Any]:
        """Get information about generation capabilities"""
        return {
            'supported_modes': [mode.value for mode in GenerationMode],
            'max_sequence_length': self.max_sequence_length,
            'output_dimension': self.output_dim,
            'conditioning_types': ['emotion', 'style', 'instrument'],
            'quality_metrics': ['melody', 'rhythm', 'harmony', 'overall']
        }
    
    def estimate_generation_time(self, config: GenerationConfig) -> float:
        """Estimate generation time in seconds"""
        # Simple estimation based on sequence length and model complexity
        base_time_per_token = 0.01  # seconds
        return config.max_length * base_time_per_token
    
    def validate_generation_config(self, config: GenerationConfig) -> Dict[str, Any]:
        """Validate generation configuration"""
        validation = {
            'valid': True,
            'warnings': [],
            'adjusted_config': config
        }
        
        # Check sequence length
        if config.max_length > self.max_sequence_length:
            validation['warnings'].append(
                f"max_length ({config.max_length}) exceeds model limit ({self.max_sequence_length})"
            )
            config.max_length = self.max_sequence_length
        
        # Check temperature range
        if config.temperature <= 0 or config.temperature > 2.0:
            validation['warnings'].append(
                f"temperature ({config.temperature}) outside recommended range (0, 2.0]"
            )
            config.temperature = np.clip(config.temperature, 0.1, 2.0)
        
        # Check sampling parameters
        if config.top_k <= 0:
            validation['warnings'].append("top_k should be positive")
            config.top_k = 50
        
        if not 0 < config.top_p <= 1.0:
            validation['warnings'].append("top_p should be in range (0, 1]")
            config.top_p = np.clip(config.top_p, 0.1, 1.0)
        
        validation['adjusted_config'] = config
        return validation