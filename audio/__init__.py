"""
Audio processing modules for InstrumentTimbre package
"""

# Import core functions from processors.py
from .processors import (
    # Audio loading and saving
    load_audio,
    save_audio,
    # Feature extraction functions
    extract_features,
    extract_chinese_instrument_features,
    # Audio processing
    apply_audio_effects,
    create_effect_chain,
)

__all__ = [
    # Audio loading and saving
    "load_audio",
    "save_audio",
    # Feature extraction
    "extract_features",
    "extract_chinese_instrument_features",
    # Audio processing
    "apply_audio_effects",
    "create_effect_chain",
]
