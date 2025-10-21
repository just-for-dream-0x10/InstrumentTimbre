"""
Professional Audio Processing Module

This module provides professional-grade audio processing capabilities including:
- Intelligent mixing with automatic level balancing
- Dynamic range optimization with musical context awareness
- Spatial positioning algorithms for optimal stereo imaging
- Intelligent EQ balancing for frequency conflict resolution
- Effects processing with style-appropriate settings
- Audio quality enhancement for broadcast-ready output

Key Components:
    - IntelligentMixingEngine: Automated mixing based on musical structure
    - DynamicRangeOptimizer: Smart compression and dynamic processing
    - SpatialPositioningAlgorithm: Automatic stereo/surround positioning
    - IntelligentEQBalancer: Frequency balancing and conflict resolution
    - EffectsProcessor: Style-specific reverb, delay, and modulation
    - AudioQualityEnhancer: Professional-grade quality improvement
    - ProfessionalAudioEngine: Main processing orchestrator

Usage:
    from InstrumentTimbre.core.professional_audio import ProfessionalAudioEngine
    
    audio_engine = ProfessionalAudioEngine()
    processed_audio = audio_engine.process_tracks(tracks, musical_analysis)
"""

# Simple configuration system  
from config import Config, get_config, set_config, Quality, ModelType
from config import fast_config, high_quality_config, production_config

# Main processing engine
from .professional_audio_engine import ProfessionalAudioEngine

# Legacy config (for backward compatibility)
from .professional_audio_engine import ProcessingConfig, AudioTrackInfo

# Core processors (advanced use only)
from .base_processor import BaseAudioProcessor
from .intelligent_mixing_engine import IntelligentMixingEngine
from .dynamic_range_optimizer import DynamicRangeOptimizer
from .spatial_positioning_algorithm import SpatialPositioningAlgorithm
from .intelligent_eq_balancer import IntelligentEQBalancer
from .effects_processor import EffectsProcessor
from .audio_quality_enhancer import AudioQualityEnhancer

__all__ = [
    "ProfessionalAudioEngine",
    "IntelligentMixingEngine", 
    "DynamicRangeOptimizer",
    "SpatialPositioningAlgorithm",
    "IntelligentEQBalancer",
    "EffectsProcessor",
    "AudioQualityEnhancer"
]

__version__ = "1.0.0"