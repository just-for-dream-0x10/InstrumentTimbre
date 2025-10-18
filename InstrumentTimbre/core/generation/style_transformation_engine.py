"""
Style Transformation Engine - Week 4 Development Task

This module implements style transformation algorithms that convert melodies
between different musical styles while preserving core melodic identity.
Supports Chinese traditional, Western classical, and modern pop styles.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum

from .instrumentation_engine import StyleType, InstrumentationEngine
from .melody_preservation import MelodyPreservationEngine

# Configure logging for the module
logger = logging.getLogger(__name__)


@dataclass
class StyleTransformationConfig:
    """Configuration for style transformation parameters"""
    preserve_melody_similarity: float = 0.8
    preserve_rhythm_similarity: float = 0.7
    style_intensity: float = 0.7  # How strongly to apply style characteristics
    allow_ornamentation: bool = True
    allow_rhythm_modification: bool = True
    allow_harmonic_changes: bool = True


@dataclass
class TransformationResult:
    """Result of style transformation process"""
    transformed_melody_dna: Dict[str, Any]
    style_characteristics: Dict[str, float]
    preservation_scores: Dict[str, float]
    applied_modifications: List[str]
    confidence_score: float
    warnings: List[str]


class StyleCharacteristics:
    """Database of style-specific musical characteristics"""
    
    CHINESE_TRADITIONAL = {
        'preferred_scales': ['pentatonic_major', 'pentatonic_minor'],
        'ornament_types': ['grace_note', 'slide', 'vibrato', 'bend'],
        'rhythm_patterns': ['even_subdivision', 'triplet_feel'],
        'interval_preferences': {
            'unison': 0.1, 'minor_second': 0.05, 'major_second': 0.2,
            'minor_third': 0.15, 'major_third': 0.1, 'perfect_fourth': 0.15,
            'tritone': 0.02, 'perfect_fifth': 0.15, 'minor_sixth': 0.05,
            'major_sixth': 0.08, 'minor_seventh': 0.03, 'major_seventh': 0.02
        },
        'tempo_preferences': {'slow': 0.4, 'moderate': 0.5, 'fast': 0.1},
        'dynamic_characteristics': {'gentle_swells': 0.8, 'sudden_accents': 0.2},
        'phrase_structure': {'asymmetric': 0.7, 'symmetric': 0.3}
    }
    
    WESTERN_CLASSICAL = {
        'preferred_scales': ['major', 'minor', 'dorian', 'mixolydian'],
        'ornament_types': ['trill', 'mordent', 'appoggiatura', 'acciaccatura'],
        'rhythm_patterns': ['dotted_rhythms', 'syncopation', 'regular_meter'],
        'interval_preferences': {
            'unison': 0.08, 'minor_second': 0.08, 'major_second': 0.15,
            'minor_third': 0.12, 'major_third': 0.15, 'perfect_fourth': 0.12,
            'tritone': 0.05, 'perfect_fifth': 0.12, 'minor_sixth': 0.08,
            'major_sixth': 0.1, 'minor_seventh': 0.08, 'major_seventh': 0.07
        },
        'tempo_preferences': {'slow': 0.3, 'moderate': 0.4, 'fast': 0.3},
        'dynamic_characteristics': {'crescendo_diminuendo': 0.7, 'terraced': 0.3},
        'phrase_structure': {'symmetric': 0.8, 'asymmetric': 0.2}
    }
    
    MODERN_POP = {
        'preferred_scales': ['major', 'minor', 'blues', 'pentatonic_major'],
        'ornament_types': ['bend', 'slide', 'vibrato', 'ghost_note'],
        'rhythm_patterns': ['syncopation', 'straight_eighth', 'swing_feel'],
        'interval_preferences': {
            'unison': 0.1, 'minor_second': 0.03, 'major_second': 0.18,
            'minor_third': 0.15, 'major_third': 0.18, 'perfect_fourth': 0.1,
            'tritone': 0.08, 'perfect_fifth': 0.1, 'minor_sixth': 0.05,
            'major_sixth': 0.08, 'minor_seventh': 0.1, 'major_seventh': 0.05
        },
        'tempo_preferences': {'slow': 0.2, 'moderate': 0.5, 'fast': 0.3},
        'dynamic_characteristics': {'consistent_level': 0.6, 'build_ups': 0.4},
        'phrase_structure': {'symmetric': 0.6, 'asymmetric': 0.4}
    }


class StyleTransformationEngine:
    """
    Core engine for transforming musical styles while preserving melody
    
    Implements intelligent style conversion that:
    - Maintains melodic identity and recognizability
    - Applies style-appropriate ornamentations and modifications
    - Adjusts rhythmic patterns to match target style
    - Suggests appropriate instrumentation changes
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize style transformation engine
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.melody_engine = MelodyPreservationEngine()
        self.instrumentation_engine = InstrumentationEngine()
        
        # Transformation parameters
        self.default_config = StyleTransformationConfig()
        self.style_database = {
            StyleType.CHINESE_TRADITIONAL: StyleCharacteristics.CHINESE_TRADITIONAL,
            StyleType.WESTERN_CLASSICAL: StyleCharacteristics.WESTERN_CLASSICAL,
            StyleType.MODERN_POP: StyleCharacteristics.MODERN_POP
        }
        
        logger.info("StyleTransformationEngine initialized with %d style templates",
                   len(self.style_database))
    
    def transform_style(self, melody_dna: Dict[str, Any],
                       target_style: StyleType,
                       transformation_config: Optional[StyleTransformationConfig] = None) -> TransformationResult:
        """
        Transform melody to target style while preserving core characteristics
        
        Args:
            melody_dna: Original melody DNA from preservation engine
            target_style: Target style for transformation
            transformation_config: Optional transformation configuration
            
        Returns:
            TransformationResult with transformed melody and metadata
        """
        config = transformation_config or self.default_config
        
        # Get style characteristics
        style_chars = self.style_database.get(target_style, {})
        if not style_chars:
            raise ValueError(f"Unsupported style: {target_style}")
        
        # Create working copy of melody DNA
        transformed_dna = self._deep_copy_melody_dna(melody_dna)
        applied_modifications = []
        warnings = []
        
        # Apply style transformations
        if config.allow_ornamentation:
            self._apply_style_ornamentation(transformed_dna, style_chars, applied_modifications)
        
        if config.allow_rhythm_modification:
            self._apply_rhythmic_style(transformed_dna, style_chars, applied_modifications)
        
        if config.allow_harmonic_changes:
            self._apply_harmonic_style(transformed_dna, style_chars, applied_modifications)
        
        # Apply scale/interval adjustments
        self._apply_interval_style(transformed_dna, style_chars, applied_modifications)
        
        # Validate preservation constraints
        preservation_scores = self._validate_preservation(melody_dna, transformed_dna, config)
        
        # Check if preservation requirements are met
        if preservation_scores['melody_similarity'] < config.preserve_melody_similarity:
            warnings.append(f"Melody similarity below threshold: "
                          f"{preservation_scores['melody_similarity']:.3f} < {config.preserve_melody_similarity}")
        
        if preservation_scores['rhythm_similarity'] < config.preserve_rhythm_similarity:
            warnings.append(f"Rhythm similarity below threshold: "
                          f"{preservation_scores['rhythm_similarity']:.3f} < {config.preserve_rhythm_similarity}")
        
        # Calculate style characteristics and confidence
        style_characteristics = self._analyze_style_characteristics(transformed_dna, style_chars)
        confidence_score = self._calculate_transformation_confidence(
            preservation_scores, style_characteristics, len(warnings)
        )
        
        result = TransformationResult(
            transformed_melody_dna=transformed_dna,
            style_characteristics=style_characteristics,
            preservation_scores=preservation_scores,
            applied_modifications=applied_modifications,
            confidence_score=confidence_score,
            warnings=warnings
        )
        
        logger.info("Style transformation completed: %s -> %s, confidence=%.3f",
                   "original", target_style.value, confidence_score)
        
        return result
    
    def _deep_copy_melody_dna(self, melody_dna: Dict[str, Any]) -> Dict[str, Any]:
        """Create deep copy of melody DNA for transformation"""
        import copy
        return copy.deepcopy(melody_dna)
    
    def _apply_style_ornamentation(self, melody_dna: Dict[str, Any],
                                 style_chars: Dict[str, Any],
                                 modifications: List[str]) -> None:
        """Apply style-specific ornamentations to melody"""
        ornament_types = style_chars.get('ornament_types', [])
        char_notes = melody_dna.get('characteristic_notes', [])
        
        if not ornament_types or not char_notes:
            return
        
        # Add ornamentations to characteristic notes
        ornament_count = 0
        for note in char_notes:
            # Apply ornamentations to prominent notes
            if note.get('type') in ['peak', 'sustained'] and note.get('prominence', 0) > 0.5:
                # Select appropriate ornament for style
                if 'vibrato' in ornament_types and note.get('duration', 0) > 0.5:
                    note['applied_ornaments'] = note.get('applied_ornaments', []) + ['vibrato']
                    ornament_count += 1
                elif 'grace_note' in ornament_types and note.get('type') == 'peak':
                    note['applied_ornaments'] = note.get('applied_ornaments', []) + ['grace_note']
                    ornament_count += 1
                elif 'slide' in ornament_types:
                    note['applied_ornaments'] = note.get('applied_ornaments', []) + ['slide']
                    ornament_count += 1
        
        if ornament_count > 0:
            modifications.append(f"Added {ornament_count} style-appropriate ornamentations")
    
    def _apply_rhythmic_style(self, melody_dna: Dict[str, Any],
                            style_chars: Dict[str, Any],
                            modifications: List[str]) -> None:
        """Apply style-specific rhythmic characteristics"""
        rhythm_patterns = style_chars.get('rhythm_patterns', [])
        rhythmic_skeleton = melody_dna.get('rhythmic_skeleton', {})
        
        if not rhythm_patterns or not rhythmic_skeleton:
            return
        
        # Adjust tempo based on style preferences
        tempo_prefs = style_chars.get('tempo_preferences', {})
        current_tempo = rhythmic_skeleton.get('tempo', 120)
        
        if 'slow' in tempo_prefs and tempo_prefs['slow'] > 0.5 and current_tempo > 100:
            # Slow down for styles that prefer slower tempos
            new_tempo = max(60, current_tempo * 0.8)
            rhythmic_skeleton['tempo'] = new_tempo
            modifications.append(f"Adjusted tempo from {current_tempo} to {new_tempo} BPM for style")
        
        elif 'fast' in tempo_prefs and tempo_prefs['fast'] > 0.5 and current_tempo < 120:
            # Speed up for styles that prefer faster tempos
            new_tempo = min(180, current_tempo * 1.2)
            rhythmic_skeleton['tempo'] = new_tempo
            modifications.append(f"Adjusted tempo from {current_tempo} to {new_tempo} BPM for style")
        
        # Apply rhythmic pattern characteristics
        if 'syncopation' in rhythm_patterns:
            rhythmic_skeleton['style_features'] = rhythmic_skeleton.get('style_features', []) + ['syncopation']
            modifications.append("Added syncopated rhythm patterns")
        
        if 'triplet_feel' in rhythm_patterns:
            rhythmic_skeleton['style_features'] = rhythmic_skeleton.get('style_features', []) + ['triplet_feel']
            modifications.append("Applied triplet feel to rhythm")
    
    def _apply_harmonic_style(self, melody_dna: Dict[str, Any],
                            style_chars: Dict[str, Any],
                            modifications: List[str]) -> None:
        """Apply style-specific harmonic characteristics"""
        # For now, add harmonic context information
        # Full harmonic transformation would require additional development
        harmonic_context = {
            'style_harmony': style_chars.get('preferred_scales', ['major'])[0],
            'harmonic_rhythm': 'moderate',
            'chord_complexity': 'simple' if 'pentatonic' in str(style_chars) else 'moderate'
        }
        
        melody_dna['harmonic_context'] = harmonic_context
        modifications.append(f"Added {harmonic_context['style_harmony']} harmonic context")
    
    def _apply_interval_style(self, melody_dna: Dict[str, Any],
                            style_chars: Dict[str, Any],
                            modifications: List[str]) -> None:
        """Apply style-specific interval preferences"""
        interval_prefs = style_chars.get('interval_preferences', {})
        intervals = melody_dna.get('interval_sequence', np.array([]))
        
        if len(intervals) == 0 or not interval_prefs:
            return
        
        # Analyze current interval distribution
        interval_names = ['unison', 'minor_second', 'major_second', 'minor_third', 
                         'major_third', 'perfect_fourth', 'tritone', 'perfect_fifth',
                         'minor_sixth', 'major_sixth', 'minor_seventh', 'major_seventh']
        
        # Convert semitone intervals to named intervals
        interval_counts = {}
        for interval in intervals:
            abs_interval = abs(interval) % 12
            if abs_interval < len(interval_names):
                interval_name = interval_names[int(abs_interval)]
                interval_counts[interval_name] = interval_counts.get(interval_name, 0) + 1
        
        # Check for style consistency
        total_intervals = len(intervals)
        style_violations = 0
        
        for interval_name, count in interval_counts.items():
            current_ratio = count / total_intervals
            preferred_ratio = interval_prefs.get(interval_name, 0.1)
            
            # Flag intervals that are overused relative to style preference
            if current_ratio > preferred_ratio * 2:
                style_violations += 1
        
        if style_violations > 0:
            melody_dna['style_analysis'] = {
                'interval_distribution': interval_counts,
                'style_consistency': 1.0 - (style_violations / len(interval_counts))
            }
            modifications.append(f"Analyzed interval distribution for {style_violations} style inconsistencies")
    
    def _validate_preservation(self, original_dna: Dict[str, Any],
                             transformed_dna: Dict[str, Any],
                             config: StyleTransformationConfig) -> Dict[str, float]:
        """Validate that transformation preserves essential melody characteristics"""
        
        # Use melody preservation engine to compute similarities
        melody_similarity = self.melody_engine.compute_melody_similarity(original_dna, transformed_dna)
        
        # Compute rhythm similarity
        orig_rhythm = original_dna.get('rhythmic_skeleton', {})
        trans_rhythm = transformed_dna.get('rhythmic_skeleton', {})
        rhythm_similarity = self.melody_engine._compute_rhythm_similarity(orig_rhythm, trans_rhythm)
        
        # Convert numpy arrays to floats if needed
        if hasattr(rhythm_similarity, 'item'):
            rhythm_similarity = rhythm_similarity.item()
        elif isinstance(rhythm_similarity, np.ndarray):
            rhythm_similarity = float(rhythm_similarity)
        
        # Compute structural preservation
        orig_phrases = original_dna.get('phrase_boundaries', [])
        trans_phrases = transformed_dna.get('phrase_boundaries', [])
        phrase_similarity = self.melody_engine._compute_phrase_similarity(orig_phrases, trans_phrases)
        
        return {
            'melody_similarity': float(melody_similarity),
            'rhythm_similarity': float(rhythm_similarity),
            'phrase_similarity': float(phrase_similarity),
            'overall_preservation': float(
                (melody_similarity + rhythm_similarity + phrase_similarity) / 3
            )
        }
    
    def _analyze_style_characteristics(self, melody_dna: Dict[str, Any],
                                     target_style_chars: Dict[str, Any]) -> Dict[str, float]:
        """Analyze how well the melody matches target style characteristics"""
        characteristics = {}
        
        # Analyze rhythm characteristics
        rhythmic_skeleton = melody_dna.get('rhythmic_skeleton', {})
        tempo = rhythmic_skeleton.get('tempo', 120)
        
        # Tempo style matching
        tempo_prefs = target_style_chars.get('tempo_preferences', {})
        if tempo < 80:
            characteristics['tempo_match'] = tempo_prefs.get('slow', 0.3)
        elif tempo < 120:
            characteristics['tempo_match'] = tempo_prefs.get('moderate', 0.5)
        else:
            characteristics['tempo_match'] = tempo_prefs.get('fast', 0.3)
        
        # Ornament style matching
        char_notes = melody_dna.get('characteristic_notes', [])
        ornament_count = sum(1 for note in char_notes if 'applied_ornaments' in note)
        ornament_types = target_style_chars.get('ornament_types', [])
        
        if ornament_types:
            characteristics['ornamentation_match'] = min(1.0, ornament_count / max(1, len(char_notes) * 0.3))
        else:
            characteristics['ornamentation_match'] = 1.0 if ornament_count == 0 else 0.5
        
        # Harmonic style matching
        harmonic_context = melody_dna.get('harmonic_context', {})
        if harmonic_context:
            preferred_scales = target_style_chars.get('preferred_scales', [])
            style_harmony = harmonic_context.get('style_harmony', '')
            characteristics['harmonic_match'] = 1.0 if style_harmony in preferred_scales else 0.5
        else:
            characteristics['harmonic_match'] = 0.7  # Neutral if no harmonic context
        
        # Overall style consistency
        characteristics['overall_style_match'] = np.mean(list(characteristics.values()))
        
        return characteristics
    
    def _calculate_transformation_confidence(self, preservation_scores: Dict[str, float],
                                           style_characteristics: Dict[str, float],
                                           warning_count: int) -> float:
        """Calculate overall confidence in transformation quality"""
        
        # Weight preservation scores heavily
        preservation_weight = 0.7
        style_weight = 0.3
        
        preservation_avg = preservation_scores.get('overall_preservation', 0.0)
        style_avg = style_characteristics.get('overall_style_match', 0.0)
        
        base_confidence = (preservation_weight * preservation_avg + 
                          style_weight * style_avg)
        
        # Penalize for warnings
        warning_penalty = warning_count * 0.1
        
        confidence = max(0.0, min(1.0, base_confidence - warning_penalty))
        
        return confidence
    
    def suggest_style_instrumentation(self, transformed_melody_dna: Dict[str, Any],
                                    target_style: StyleType) -> Dict[str, Any]:
        """Suggest appropriate instrumentation for the transformed style"""
        
        # Use instrumentation engine to get style-appropriate suggestions
        instrumentation_result = self.instrumentation_engine.suggest_instrumentation(
            transformed_melody_dna, target_style
        )
        
        return {
            'suggested_instruments': {
                'primary': instrumentation_result.primary_instrument,
                'secondary': instrumentation_result.secondary_instruments
            },
            'arrangement_guidelines': instrumentation_result.arrangement_map,
            'style_consistency': instrumentation_result.style_consistency,
            'reasoning': instrumentation_result.reasoning
        }
    
    def get_available_styles(self) -> List[StyleType]:
        """Get list of available transformation styles"""
        return list(self.style_database.keys())
    
    def get_style_description(self, style: StyleType) -> Dict[str, Any]:
        """Get detailed description of a style's characteristics"""
        if style not in self.style_database:
            return {}
        
        style_data = self.style_database[style]
        
        return {
            'name': style.value,
            'characteristics': {
                'scales': style_data.get('preferred_scales', []),
                'ornaments': style_data.get('ornament_types', []),
                'rhythms': style_data.get('rhythm_patterns', []),
                'tempo_preferences': style_data.get('tempo_preferences', {}),
                'phrase_structure': style_data.get('phrase_structure', {})
            },
            'cultural_context': self._get_cultural_context(style),
            'typical_instruments': self._get_typical_instruments(style)
        }
    
    def _get_cultural_context(self, style: StyleType) -> str:
        """Get cultural context description for a style"""
        context_map = {
            StyleType.CHINESE_TRADITIONAL: "Traditional Chinese music emphasizes expression, nature imagery, and pentatonic scales",
            StyleType.WESTERN_CLASSICAL: "Western classical tradition with emphasis on formal structure and harmonic development", 
            StyleType.MODERN_POP: "Contemporary popular music with accessibility and commercial appeal"
        }
        
        return context_map.get(style, "No cultural context available")
    
    def _get_typical_instruments(self, style: StyleType) -> List[str]:
        """Get typical instruments for a style"""
        instrument_map = {
            StyleType.CHINESE_TRADITIONAL: ['erhu', 'pipa', 'guzheng', 'dizi', 'guqin'],
            StyleType.WESTERN_CLASSICAL: ['violin', 'piano', 'cello', 'flute', 'clarinet'],
            StyleType.MODERN_POP: ['electric_guitar', 'piano', 'synthesizer', 'bass_guitar', 'drums']
        }
        
        return instrument_map.get(style, [])