"""
Feature extraction modules for InstrumentTimbre
"""

from .base import BaseFeatureExtractor
from .chinese import ChineseInstrumentAnalyzer
from .traditional import TraditionalAudioFeatures
from .deep import DeepLearningFeatures

__all__ = [
    "BaseFeatureExtractor",
    "ChineseInstrumentAnalyzer", 
    "TraditionalAudioFeatures",
    "DeepLearningFeatures"
]