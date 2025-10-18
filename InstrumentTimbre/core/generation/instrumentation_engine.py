"""
Instrumentation Engine - Week 4 Development Task

This module implements automatic instrumentation algorithms that intelligently
assign instruments to melodic lines while preserving musical coherence.
Supports multiple musical styles and cultural traditions.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum

# Configure logging for the module
logger = logging.getLogger(__name__)


class InstrumentFamily(Enum):
    """Enumeration of instrument families for instrumentation decisions"""
    STRINGS = "strings"
    WINDS = "winds" 
    BRASS = "brass"
    PERCUSSION = "percussion"
    KEYBOARD = "keyboard"
    TRADITIONAL_CHINESE = "traditional_chinese"


class StyleType(Enum):
    """Enumeration of supported musical styles"""
    CHINESE_TRADITIONAL = "chinese_traditional"
    WESTERN_CLASSICAL = "western_classical"
    MODERN_POP = "modern_pop"
    NEUTRAL = "neutral"


@dataclass
class InstrumentConfig:
    """Configuration data structure for instrument properties"""
    name: str
    family: InstrumentFamily
    pitch_range: Tuple[float, float]  # Hz range
    dynamic_range: Tuple[float, float]  # dB range
    timbral_characteristics: Dict[str, float]
    cultural_affinity: List[StyleType]
    expression_capabilities: Dict[str, bool]


@dataclass
class InstrumentationResult:
    """Data structure containing instrumentation results"""
    primary_instrument: str
    secondary_instruments: List[str]
    arrangement_map: Dict[str, Dict[str, Any]]
    confidence_score: float
    style_consistency: float
    reasoning: List[str]


class InstrumentationEngine:
    """
    Core engine for automatic music instrumentation
    
    Provides intelligent instrument assignment based on:
    - Melodic characteristics analysis
    - Style-specific preferences
    - Cultural musical traditions
    - Harmonic and rhythmic context
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize instrumentation engine with configuration
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self._initialize_instrument_database()
        self._setup_style_preferences()
        
        # Engine parameters
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        self.max_instruments = self.config.get('max_instruments', 4)
        self.style_weight = self.config.get('style_weight', 0.6)
        self.melody_weight = self.config.get('melody_weight', 0.4)
        
        logger.info("InstrumentationEngine initialized with %d instruments", 
                   len(self.instrument_database))
    
    def _initialize_instrument_database(self) -> None:
        """Initialize comprehensive instrument database with characteristics"""
        self.instrument_database = {
            # Traditional Chinese instruments
            'erhu': InstrumentConfig(
                name='erhu',
                family=InstrumentFamily.TRADITIONAL_CHINESE,
                pitch_range=(196.0, 1568.0),  # G3 to G6
                dynamic_range=(40.0, 85.0),
                timbral_characteristics={
                    'warmth': 0.8, 'brightness': 0.6, 'expressiveness': 0.9,
                    'vibrato_capability': 0.9, 'portamento': 0.8
                },
                cultural_affinity=[StyleType.CHINESE_TRADITIONAL],
                expression_capabilities={
                    'vibrato': True, 'portamento': True, 'dynamics': True,
                    'articulation': True, 'ornaments': True
                }
            ),
            
            'pipa': InstrumentConfig(
                name='pipa',
                family=InstrumentFamily.TRADITIONAL_CHINESE,
                pitch_range=(220.0, 2093.0),  # A3 to C7
                dynamic_range=(35.0, 90.0),
                timbral_characteristics={
                    'warmth': 0.6, 'brightness': 0.8, 'expressiveness': 0.8,
                    'attack_clarity': 0.9, 'rhythmic_precision': 0.9
                },
                cultural_affinity=[StyleType.CHINESE_TRADITIONAL],
                expression_capabilities={
                    'tremolo': True, 'bend': True, 'harmonics': True,
                    'rapid_arpeggios': True, 'percussive': True
                }
            ),
            
            'guzheng': InstrumentConfig(
                name='guzheng',
                family=InstrumentFamily.TRADITIONAL_CHINESE,
                pitch_range=(196.0, 1760.0),  # G3 to A6
                dynamic_range=(30.0, 85.0),
                timbral_characteristics={
                    'warmth': 0.7, 'brightness': 0.7, 'expressiveness': 0.8,
                    'resonance': 0.9, 'harmonic_richness': 0.8
                },
                cultural_affinity=[StyleType.CHINESE_TRADITIONAL],
                expression_capabilities={
                    'glissando': True, 'harmonics': True, 'tremolo': True,
                    'chord_techniques': True, 'pitch_bend': True
                }
            ),
            
            # Western classical instruments
            'violin': InstrumentConfig(
                name='violin',
                family=InstrumentFamily.STRINGS,
                pitch_range=(196.0, 3136.0),  # G3 to G7
                dynamic_range=(30.0, 100.0),
                timbral_characteristics={
                    'warmth': 0.7, 'brightness': 0.8, 'expressiveness': 0.9,
                    'agility': 0.9, 'dynamic_range': 0.9
                },
                cultural_affinity=[StyleType.WESTERN_CLASSICAL, StyleType.MODERN_POP],
                expression_capabilities={
                    'vibrato': True, 'portamento': True, 'sul_ponticello': True,
                    'harmonics': True, 'double_stops': True
                }
            ),
            
            'piano': InstrumentConfig(
                name='piano',
                family=InstrumentFamily.KEYBOARD,
                pitch_range=(27.5, 4186.0),  # A0 to C8
                dynamic_range=(20.0, 110.0),
                timbral_characteristics={
                    'warmth': 0.6, 'brightness': 0.7, 'expressiveness': 0.8,
                    'harmonic_capability': 1.0, 'percussive': 0.7
                },
                cultural_affinity=[StyleType.WESTERN_CLASSICAL, StyleType.MODERN_POP],
                expression_capabilities={
                    'polyphony': True, 'pedaling': True, 'dynamic_control': True,
                    'articulation': True, 'chord_voicing': True
                }
            ),
            
            # Modern instruments
            'electric_guitar': InstrumentConfig(
                name='electric_guitar',
                family=InstrumentFamily.STRINGS,
                pitch_range=(82.4, 1319.0),  # E2 to E6
                dynamic_range=(25.0, 120.0),
                timbral_characteristics={
                    'warmth': 0.5, 'brightness': 0.8, 'expressiveness': 0.8,
                    'distortion': 0.7, 'sustain': 0.8
                },
                cultural_affinity=[StyleType.MODERN_POP],
                expression_capabilities={
                    'bending': True, 'vibrato': True, 'effects': True,
                    'harmonics': True, 'palm_muting': True
                }
            ),
            
            'synthesizer': InstrumentConfig(
                name='synthesizer',
                family=InstrumentFamily.KEYBOARD,
                pitch_range=(20.0, 20000.0),  # Full audio spectrum
                dynamic_range=(0.0, 120.0),
                timbral_characteristics={
                    'warmth': 0.5, 'brightness': 0.7, 'expressiveness': 0.7,
                    'versatility': 1.0, 'effects': 1.0
                },
                cultural_affinity=[StyleType.MODERN_POP],
                expression_capabilities={
                    'modulation': True, 'filtering': True, 'effects': True,
                    'polyphony': True, 'envelope_shaping': True
                }
            )
        }
    
    def _setup_style_preferences(self) -> None:
        """Configure style-specific instrumentation preferences"""
        self.style_preferences = {
            StyleType.CHINESE_TRADITIONAL: {
                'preferred_families': [InstrumentFamily.TRADITIONAL_CHINESE],
                'avoid_families': [InstrumentFamily.BRASS],
                'ensemble_sizes': (1, 3),
                'characteristics_weights': {
                    'expressiveness': 0.4, 'cultural_authenticity': 0.3,
                    'timbral_blend': 0.2, 'technical_suitability': 0.1
                }
            },
            
            StyleType.WESTERN_CLASSICAL: {
                'preferred_families': [InstrumentFamily.STRINGS, InstrumentFamily.WINDS],
                'avoid_families': [],
                'ensemble_sizes': (2, 4),
                'characteristics_weights': {
                    'expressiveness': 0.3, 'harmonic_capability': 0.3,
                    'dynamic_range': 0.2, 'timbral_blend': 0.2
                }
            },
            
            StyleType.MODERN_POP: {
                'preferred_families': [InstrumentFamily.KEYBOARD, InstrumentFamily.STRINGS],
                'avoid_families': [InstrumentFamily.TRADITIONAL_CHINESE],
                'ensemble_sizes': (2, 4),
                'characteristics_weights': {
                    'versatility': 0.3, 'modern_appeal': 0.3,
                    'production_value': 0.2, 'accessibility': 0.2
                }
            }
        }
    
    def analyze_melody_for_instrumentation(self, melody_dna: Dict[str, Any]) -> Dict[str, float]:
        """
        Analyze melody characteristics to determine instrumentation requirements
        
        Args:
            melody_dna: Melody DNA from preservation engine
            
        Returns:
            Dictionary of instrumentation requirements and weights
        """
        requirements = {}
        
        # Extract melodic statistics
        stats = melody_dna.get('melodic_stats', {})
        intervals = melody_dna.get('interval_sequence', np.array([]))
        rhythm = melody_dna.get('rhythmic_skeleton', {})
        
        # Analyze pitch range requirements
        pitch_range = stats.get('pitch_range', 0)
        if pitch_range > 24:  # More than 2 octaves
            requirements['wide_range_capability'] = 0.8
        elif pitch_range > 12:  # More than 1 octave
            requirements['moderate_range_capability'] = 0.6
        else:
            requirements['narrow_range_capability'] = 0.4
        
        # Analyze melodic complexity
        if len(intervals) > 0:
            large_intervals = np.sum(np.abs(intervals) > 7)  # Perfect fifth or larger
            complexity_ratio = large_intervals / len(intervals)
            
            if complexity_ratio > 0.3:
                requirements['high_agility'] = 0.8
            elif complexity_ratio > 0.1:
                requirements['moderate_agility'] = 0.6
            else:
                requirements['simple_technique'] = 0.5
        
        # Analyze rhythmic requirements
        tempo = rhythm.get('tempo', 120)
        note_density = rhythm.get('note_density', 1.0)
        
        if tempo > 140 and note_density > 3.0:
            requirements['fast_articulation'] = 0.8
        elif tempo > 100 or note_density > 2.0:
            requirements['moderate_speed'] = 0.6
        else:
            requirements['slow_expressive'] = 0.7
        
        # Analyze expressiveness requirements
        char_notes = melody_dna.get('characteristic_notes', [])
        expressive_features = sum(1 for note in char_notes 
                                if note.get('type') in ['peak', 'sustained'])
        
        if expressive_features > len(char_notes) * 0.3:
            requirements['high_expressiveness'] = 0.8
        else:
            requirements['moderate_expressiveness'] = 0.5
        
        logger.debug("Melody analysis requirements: %s", requirements)
        return requirements
    
    def suggest_instrumentation(self, melody_dna: Dict[str, Any], 
                              target_style: StyleType = StyleType.NEUTRAL,
                              max_instruments: Optional[int] = None) -> InstrumentationResult:
        """
        Generate instrumentation suggestions based on melody and style
        
        Args:
            melody_dna: Melody DNA from preservation engine
            target_style: Target musical style for instrumentation
            max_instruments: Maximum number of instruments to suggest
            
        Returns:
            InstrumentationResult with suggestions and reasoning
        """
        max_instruments = max_instruments or self.max_instruments
        
        # Analyze melody requirements
        melody_requirements = self.analyze_melody_for_instrumentation(melody_dna)
        
        # Get style preferences
        style_prefs = self.style_preferences.get(target_style, {})
        
        # Score all instruments
        instrument_scores = self._score_instruments(melody_requirements, style_prefs)
        
        # Select best instruments
        selected_instruments = self._select_instruments(
            instrument_scores, max_instruments, style_prefs
        )
        
        # Generate arrangement mapping
        arrangement = self._create_arrangement_map(selected_instruments, melody_dna)
        
        # Calculate confidence and consistency scores
        confidence = self._calculate_confidence(instrument_scores, selected_instruments)
        style_consistency = self._calculate_style_consistency(selected_instruments, target_style)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(selected_instruments, melody_requirements, target_style)
        
        result = InstrumentationResult(
            primary_instrument=selected_instruments[0] if selected_instruments else 'piano',
            secondary_instruments=selected_instruments[1:] if len(selected_instruments) > 1 else [],
            arrangement_map=arrangement,
            confidence_score=confidence,
            style_consistency=style_consistency,
            reasoning=reasoning
        )
        
        logger.info("Generated instrumentation: primary=%s, secondary=%s, confidence=%.3f",
                   result.primary_instrument, result.secondary_instruments, confidence)
        
        return result
    
    def _score_instruments(self, melody_requirements: Dict[str, float],
                          style_preferences: Dict[str, Any]) -> Dict[str, float]:
        """Score each instrument based on melody and style requirements"""
        scores = {}
        
        for instrument_name, instrument_config in self.instrument_database.items():
            score = 0.0
            
            # Style compatibility scoring
            if style_preferences:
                preferred_families = style_preferences.get('preferred_families', [])
                avoid_families = style_preferences.get('avoid_families', [])
                
                if instrument_config.family in preferred_families:
                    score += 0.3
                elif instrument_config.family in avoid_families:
                    score -= 0.2
            
            # Melody requirements scoring
            for requirement, weight in melody_requirements.items():
                compatibility = self._check_requirement_compatibility(
                    instrument_config, requirement
                )
                score += compatibility * weight * 0.7
            
            # Normalize score
            scores[instrument_name] = max(0.0, min(1.0, score))
        
        return scores
    
    def _check_requirement_compatibility(self, instrument: InstrumentConfig, 
                                       requirement: str) -> float:
        """Check how well an instrument meets a specific requirement"""
        compatibility_map = {
            'wide_range_capability': self._check_pitch_range(instrument, 24),
            'moderate_range_capability': self._check_pitch_range(instrument, 12),
            'narrow_range_capability': 0.8,  # Most instruments handle narrow ranges
            'high_agility': instrument.timbral_characteristics.get('agility', 0.5),
            'moderate_agility': min(1.0, instrument.timbral_characteristics.get('agility', 0.5) + 0.3),
            'simple_technique': 0.9,  # Most instruments can play simple parts
            'fast_articulation': instrument.timbral_characteristics.get('attack_clarity', 0.5),
            'moderate_speed': 0.8,
            'slow_expressive': instrument.timbral_characteristics.get('expressiveness', 0.5),
            'high_expressiveness': instrument.timbral_characteristics.get('expressiveness', 0.5),
            'moderate_expressiveness': min(1.0, instrument.timbral_characteristics.get('expressiveness', 0.5) + 0.2)
        }
        
        return compatibility_map.get(requirement, 0.5)
    
    def _check_pitch_range(self, instrument: InstrumentConfig, required_semitones: float) -> float:
        """Check if instrument can handle required pitch range"""
        instrument_range_semitones = 12 * np.log2(instrument.pitch_range[1] / instrument.pitch_range[0])
        
        if instrument_range_semitones >= required_semitones:
            return 1.0
        else:
            return instrument_range_semitones / required_semitones
    
    def _select_instruments(self, scores: Dict[str, float], max_count: int,
                          style_preferences: Dict[str, Any]) -> List[str]:
        """Select best instruments based on scores and constraints"""
        # Sort instruments by score
        sorted_instruments = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        selected = []
        used_families = set()
        
        # Apply ensemble size constraints
        min_size, max_size = style_preferences.get('ensemble_sizes', (1, max_count))
        target_size = min(max_count, max_size)
        
        for instrument_name, score in sorted_instruments:
            if len(selected) >= target_size:
                break
                
            if score < self.confidence_threshold:
                continue
            
            instrument_config = self.instrument_database[instrument_name]
            
            # Avoid too many instruments from same family
            if instrument_config.family in used_families and len(selected) >= 2:
                continue
            
            selected.append(instrument_name)
            used_families.add(instrument_config.family)
        
        # Ensure minimum ensemble size
        if len(selected) < min_size:
            remaining = [name for name, _ in sorted_instruments if name not in selected]
            selected.extend(remaining[:min_size - len(selected)])
        
        return selected
    
    def _create_arrangement_map(self, instruments: List[str], 
                               melody_dna: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Create detailed arrangement mapping for each instrument"""
        arrangement = {}
        
        phrases = melody_dna.get('phrase_boundaries', [])
        char_notes = melody_dna.get('characteristic_notes', [])
        
        for i, instrument_name in enumerate(instruments):
            instrument_config = self.instrument_database[instrument_name]
            
            # Primary instrument gets full melody
            if i == 0:
                role = 'melody'
                phrase_assignment = list(range(len(phrases) - 1)) if len(phrases) > 1 else [0]
            # Secondary instruments get supporting roles
            else:
                role = 'harmony' if i == 1 else 'accompaniment'
                # Assign to subset of phrases
                phrase_count = max(1, (len(phrases) - 1) // 2)
                phrase_assignment = list(range(0, phrase_count))
            
            arrangement[instrument_name] = {
                'role': role,
                'assigned_phrases': phrase_assignment,
                'expression_techniques': list(instrument_config.expression_capabilities.keys()),
                'dynamic_range': instrument_config.dynamic_range,
                'timbral_focus': self._determine_timbral_focus(instrument_config, role),
                'priority': i + 1
            }
        
        return arrangement
    
    def _determine_timbral_focus(self, instrument: InstrumentConfig, role: str) -> Dict[str, float]:
        """Determine timbral emphasis based on instrument and role"""
        base_characteristics = instrument.timbral_characteristics.copy()
        
        # Adjust characteristics based on role
        if role == 'melody':
            base_characteristics['expressiveness'] *= 1.2
            base_characteristics['brightness'] *= 1.1
        elif role == 'harmony':
            base_characteristics['warmth'] *= 1.2
            base_characteristics.setdefault('blend', 0.8)
        else:  # accompaniment
            base_characteristics['warmth'] *= 1.1
            base_characteristics.setdefault('stability', 0.9)
        
        # Normalize values to [0, 1]
        for key, value in base_characteristics.items():
            base_characteristics[key] = min(1.0, max(0.0, value))
        
        return base_characteristics
    
    def _calculate_confidence(self, scores: Dict[str, float], 
                            selected: List[str]) -> float:
        """Calculate overall confidence in instrumentation choice"""
        if not selected:
            return 0.0
        
        selected_scores = [scores[name] for name in selected if name in scores]
        
        if not selected_scores:
            return 0.0
        
        # Weight primary instrument more heavily
        weights = [0.5] + [0.5 / (len(selected_scores) - 1)] * (len(selected_scores) - 1)
        
        confidence = sum(score * weight for score, weight in zip(selected_scores, weights))
        return min(1.0, confidence)
    
    def _calculate_style_consistency(self, instruments: List[str], 
                                   style: StyleType) -> float:
        """Calculate how well instruments match target style"""
        if not instruments:
            return 0.0
        
        consistencies = []
        
        for instrument_name in instruments:
            instrument_config = self.instrument_database[instrument_name]
            
            if style in instrument_config.cultural_affinity:
                consistencies.append(1.0)
            elif style == StyleType.NEUTRAL:
                consistencies.append(0.8)
            else:
                # Check family compatibility
                style_prefs = self.style_preferences.get(style, {})
                preferred_families = style_prefs.get('preferred_families', [])
                avoid_families = style_prefs.get('avoid_families', [])
                
                if instrument_config.family in preferred_families:
                    consistencies.append(0.7)
                elif instrument_config.family in avoid_families:
                    consistencies.append(0.2)
                else:
                    consistencies.append(0.5)
        
        return np.mean(consistencies)
    
    def _generate_reasoning(self, instruments: List[str], 
                          requirements: Dict[str, float],
                          style: StyleType) -> List[str]:
        """Generate human-readable reasoning for instrumentation choices"""
        reasoning = []
        
        if not instruments:
            reasoning.append("No suitable instruments found for given requirements")
            return reasoning
        
        primary = instruments[0]
        primary_config = self.instrument_database[primary]
        
        # Primary instrument reasoning
        reasoning.append(f"Selected {primary} as primary instrument due to "
                        f"high {self._get_dominant_characteristic(primary_config)} "
                        f"and {style.value} style compatibility")
        
        # Secondary instruments reasoning
        if len(instruments) > 1:
            reasoning.append(f"Added {len(instruments) - 1} supporting instruments "
                           f"for ensemble richness and harmonic support")
        
        # Requirements-based reasoning
        top_requirements = sorted(requirements.items(), key=lambda x: x[1], reverse=True)[:2]
        if top_requirements:
            req_text = ", ".join([req.replace('_', ' ') for req, _ in top_requirements])
            reasoning.append(f"Instrumentation optimized for {req_text}")
        
        return reasoning
    
    def _get_dominant_characteristic(self, instrument: InstrumentConfig) -> str:
        """Get the most prominent characteristic of an instrument"""
        characteristics = instrument.timbral_characteristics
        if not characteristics:
            return "versatility"
        
        dominant = max(characteristics.items(), key=lambda x: x[1])
        return dominant[0].replace('_', ' ')
    
    def validate_instrumentation(self, result: InstrumentationResult,
                               melody_dna: Dict[str, Any]) -> Dict[str, Any]:
        """Validate instrumentation result against melody requirements"""
        validation = {
            'overall_valid': True,
            'warnings': [],
            'suggestions': [],
            'compatibility_score': 0.0
        }
        
        # Check pitch range compatibility
        melody_stats = melody_dna.get('melodic_stats', {})
        melody_range = melody_stats.get('pitch_range', 0)
        
        primary_instrument = self.instrument_database.get(result.primary_instrument)
        if primary_instrument:
            instrument_range = 12 * np.log2(
                primary_instrument.pitch_range[1] / primary_instrument.pitch_range[0]
            )
            
            if melody_range > instrument_range:
                validation['warnings'].append(
                    f"Melody range ({melody_range:.1f} semitones) exceeds "
                    f"{result.primary_instrument} range ({instrument_range:.1f} semitones)"
                )
                validation['overall_valid'] = False
        
        # Check confidence threshold
        if result.confidence_score < self.confidence_threshold:
            validation['warnings'].append(
                f"Low confidence score ({result.confidence_score:.3f}), "
                f"consider alternative instrumentation"
            )
        
        # Check style consistency
        if result.style_consistency < 0.6:
            validation['suggestions'].append(
                "Consider instruments more aligned with target style"
            )
        
        # Calculate overall compatibility
        validation['compatibility_score'] = (
            result.confidence_score * 0.6 + result.style_consistency * 0.4
        )
        
        logger.info("Instrumentation validation: valid=%s, compatibility=%.3f",
                   validation['overall_valid'], validation['compatibility_score'])
        
        return validation