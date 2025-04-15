"""
Model definitions for InstrumentTimbre
"""

from .model import InstrumentTimbreModel
from .encoders import InstrumentTimbreEncoder, ChineseInstrumentTimbreEncoder
from .decoders import InstrumentTimbreDecoder, EnhancedTimbreDecoder
from .attention import (
    SelfAttention,
    FeatureAttention,
    FrequencyAttention,
    MultiHeadAttention,
)

# Define package exports
__all__ = [
    # Main model
    "InstrumentTimbreModel",
    # Encoders
    "InstrumentTimbreEncoder",
    "ChineseInstrumentTimbreEncoder",
    # Decoders
    "InstrumentTimbreDecoder",
    "EnhancedTimbreDecoder",
    # Attention mechanisms
    "SelfAttention",
    "FeatureAttention",
    "FrequencyAttention",
    "MultiHeadAttention",
]
