"""
Quality Assurance System for Intelligent Music AI

This module provides comprehensive quality assurance and validation
for music generation, editing, and processing operations.

Core Components:
- Harmony validation and theory compliance checking
- Emotion consistency preservation validation  
- Music theory and structural validation
- Automatic conflict resolution and coordination
- User feedback interface for complex conflicts
- Multi-dimensional quality scoring system
"""

from .harmony_validator import HarmonyValidator
from .emotion_consistency_checker import EmotionConsistencyChecker
from .music_theory_validator import MusicTheoryValidator
from .auto_coordination_engine import AutoCoordinationEngine
from .user_feedback_interface import UserFeedbackInterface
from .quality_scoring_system import QualityScoringSystem
from .quality_assurance_engine import QualityAssuranceEngine

__all__ = [
    'HarmonyValidator',
    'EmotionConsistencyChecker',
    'MusicTheoryValidator', 
    'AutoCoordinationEngine',
    'UserFeedbackInterface',
    'QualityScoringSystem',
    'QualityAssuranceEngine'
]

__version__ = "1.0.0"