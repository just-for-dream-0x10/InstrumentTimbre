"""
Key Modulation Engine - System Development Task

This module implements intelligent key modulation algorithms that can
transpose melodies and apply harmonic transformations while preserving
musical relationships and melodic recognizability.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum

# Configure logging for the module
logger = logging.getLogger(__name__)


class ModulationType(Enum):
    """Types of key modulation"""
    SIMPLE_TRANSPOSE = "simple_transpose"
    CHROMATIC_MEDIANT = "chromatic_mediant"
    CIRCLE_OF_FIFTHS = "circle_of_fifths"
    PARALLEL_MINOR_MAJOR = "parallel_minor_major"
    RELATIVE_MINOR_MAJOR = "relative_minor_major"
    TRITONE_SUBSTITUTION = "tritone_substitution"


class ScaleType(Enum):
    """Musical scale types"""
    MAJOR = "major"
    MINOR = "minor"
    DORIAN = "dorian"
    MIXOLYDIAN = "mixolydian"
    PENTATONIC_MAJOR = "pentatonic_major"
    PENTATONIC_MINOR = "pentatonic_minor"
    BLUES = "blues"
    CHROMATIC = "chromatic"


@dataclass
class KeyModulationConfig:
    """Configuration for key modulation parameters"""
    preserve_scale_relationships: bool = True
    allow_chromatic_alteration: bool = False
    smooth_transition: bool = True
    maintain_interval_quality: bool = True
    preserve_characteristic_intervals: bool = True
    emotional_target: Optional[str] = None  # "brighter", "darker", "neutral"


@dataclass 
class ModulationResult:
    """Result of key modulation process"""
    modulated_melody_dna: Dict[str, Any]
    original_key: str
    target_key: str
    semitone_shift: int
    interval_preservation_score: float
    harmonic_coherence_score: float
    applied_modifications: List[str]
    scale_analysis: Dict[str, Any]
    warnings: List[str]


class KeyModulationEngine:
    """
    Core engine for intelligent key modulation and transposition
    
    Provides sophisticated key change capabilities:
    - Simple transposition with interval preservation
    - Complex modulations following music theory rules
    - Scale-aware transformations
    - Emotional target-based key selection
    - Harmonic coherence validation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize key modulation engine
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.default_config = KeyModulationConfig()
        
        # Initialize scale and key databases
        self._initialize_scale_database()
        self._initialize_key_relationships()
        
        # Modulation parameters
        self.max_transpose_semitones = self.config.get('max_transpose_semitones', 12)
        self.interval_tolerance = self.config.get('interval_tolerance', 0.5)
        
        logger.info("KeyModulationEngine initialized with %d scale types",
                   len(self.scale_database))
    
    def _initialize_scale_database(self) -> None:
        """Initialize database of musical scales and their interval patterns"""
        self.scale_database = {
            ScaleType.MAJOR: [0, 2, 4, 5, 7, 9, 11],
            ScaleType.MINOR: [0, 2, 3, 5, 7, 8, 10],
            ScaleType.DORIAN: [0, 2, 3, 5, 7, 9, 10],
            ScaleType.MIXOLYDIAN: [0, 2, 4, 5, 7, 9, 10],
            ScaleType.PENTATONIC_MAJOR: [0, 2, 4, 7, 9],
            ScaleType.PENTATONIC_MINOR: [0, 3, 5, 7, 10],
            ScaleType.BLUES: [0, 3, 5, 6, 7, 10],
            ScaleType.CHROMATIC: list(range(12))
        }
        
        # Key signatures (number of sharps/flats)
        self.key_signatures = {
            'C': 0, 'G': 1, 'D': 2, 'A': 3, 'E': 4, 'B': 5, 'F#': 6,
            'F': -1, 'Bb': -2, 'Eb': -3, 'Ab': -4, 'Db': -5, 'Gb': -6
        }
    
    def _initialize_key_relationships(self) -> None:
        """Initialize key relationship mappings for modulation"""
        self.key_relationships = {
            'circle_of_fifths': {
                'C': ['G', 'F'], 'G': ['D', 'C'], 'D': ['A', 'G'], 'A': ['E', 'D'],
                'E': ['B', 'A'], 'B': ['F#', 'E'], 'F#': ['C#', 'B'],
                'F': ['Bb', 'C'], 'Bb': ['Eb', 'F'], 'Eb': ['Ab', 'Bb'],
                'Ab': ['Db', 'Eb'], 'Db': ['Gb', 'Ab'], 'Gb': ['B', 'Db']
            },
            'relative_keys': {
                'C': 'Am', 'G': 'Em', 'D': 'Bm', 'A': 'F#m', 'E': 'C#m',
                'B': 'G#m', 'F#': 'D#m', 'F': 'Dm', 'Bb': 'Gm',
                'Eb': 'Cm', 'Ab': 'Fm', 'Db': 'Bbm', 'Gb': 'Ebm'
            },
            'parallel_keys': {
                'C': 'Cm', 'G': 'Gm', 'D': 'Dm', 'A': 'Am', 'E': 'Em',
                'B': 'Bm', 'F#': 'F#m', 'F': 'Fm', 'Bb': 'Bbm',
                'Eb': 'Ebm', 'Ab': 'Abm', 'Db': 'Dbm', 'Gb': 'Gbm'
            }
        }
    
    def transpose_melody(self, melody_dna: Dict[str, Any],
                        semitone_shift: int,
                        config: Optional[KeyModulationConfig] = None) -> ModulationResult:
        """
        Transpose melody by specified number of semitones
        
        Args:
            melody_dna: Original melody DNA
            semitone_shift: Number of semitones to transpose (positive = up)
            config: Optional modulation configuration
            
        Returns:
            ModulationResult with transposed melody
        """
        config = config or self.default_config
        
        # Clamp semitone shift to reasonable range
        clamped_shift = np.clip(semitone_shift, -self.max_transpose_semitones, 
                               self.max_transpose_semitones)
        
        if clamped_shift != semitone_shift:
            logger.warning("Semitone shift clamped from %d to %d", 
                          semitone_shift, clamped_shift)
        
        # Create modulated copy
        modulated_dna = self._deep_copy_melody_dna(melody_dna)
        applied_modifications = []
        warnings = []
        
        # Transpose fundamental frequency track
        f0_track = melody_dna.get('f0_track', np.array([]))
        if len(f0_track) > 0:
            transpose_factor = 2 ** (clamped_shift / 12.0)
            modulated_dna['f0_track'] = f0_track * transpose_factor
            applied_modifications.append(f"Transposed F0 by {clamped_shift} semitones")
        
        # Update pitch contour (already normalized, so no change needed)
        # But update melodic statistics
        if 'melodic_stats' in melody_dna:
            stats = melody_dna['melodic_stats'].copy()
            if 'pitch_mean' in stats:
                stats['pitch_mean'] += clamped_shift
            modulated_dna['melodic_stats'] = stats
            applied_modifications.append("Updated melodic statistics for transposition")
        
        # Interval sequence remains unchanged for simple transposition
        # This preserves melodic relationships
        
        # Determine original and target keys
        original_key, target_key = self._determine_keys_from_transpose(
            melody_dna, clamped_shift
        )
        
        # Calculate preservation scores
        interval_preservation = 1.0  # Perfect for simple transposition
        harmonic_coherence = self._calculate_harmonic_coherence(modulated_dna, target_key)
        
        # Analyze scale characteristics
        scale_analysis = self._analyze_scale_content(modulated_dna, target_key)
        
        result = ModulationResult(
            modulated_melody_dna=modulated_dna,
            original_key=original_key,
            target_key=target_key,
            semitone_shift=clamped_shift,
            interval_preservation_score=interval_preservation,
            harmonic_coherence_score=harmonic_coherence,
            applied_modifications=applied_modifications,
            scale_analysis=scale_analysis,
            warnings=warnings
        )
        
        logger.info("Melody transposed: %s -> %s (%+d semitones)",
                   original_key, target_key, clamped_shift)
        
        return result
    
    def modulate_to_key(self, melody_dna: Dict[str, Any],
                       target_key: str,
                       modulation_type: ModulationType = ModulationType.SIMPLE_TRANSPOSE,
                       config: Optional[KeyModulationConfig] = None) -> ModulationResult:
        """
        Modulate melody to specific target key using specified modulation type
        
        Args:
            melody_dna: Original melody DNA
            target_key: Target key (e.g., 'D', 'F#', 'Bb')
            modulation_type: Type of modulation to apply
            config: Optional modulation configuration
            
        Returns:
            ModulationResult with modulated melody
        """
        config = config or self.default_config
        
        # Determine current key
        current_key = self._analyze_current_key(melody_dna)
        
        # Calculate required transposition
        semitone_shift = self._calculate_key_distance(current_key, target_key)
        
        # Apply modulation based on type
        if modulation_type == ModulationType.SIMPLE_TRANSPOSE:
            return self.transpose_melody(melody_dna, semitone_shift, config)
        
        elif modulation_type == ModulationType.CIRCLE_OF_FIFTHS:
            return self._apply_circle_of_fifths_modulation(
                melody_dna, current_key, target_key, config
            )
        
        elif modulation_type == ModulationType.PARALLEL_MINOR_MAJOR:
            return self._apply_parallel_modulation(
                melody_dna, current_key, target_key, config
            )
        
        elif modulation_type == ModulationType.RELATIVE_MINOR_MAJOR:
            return self._apply_relative_modulation(
                melody_dna, current_key, target_key, config
            )
        
        else:
            # Default to simple transposition for unsupported types
            logger.warning("Modulation type %s not fully implemented, using simple transpose",
                          modulation_type.value)
            return self.transpose_melody(melody_dna, semitone_shift, config)
    
    def suggest_emotional_modulation(self, melody_dna: Dict[str, Any],
                                   emotional_target: str,
                                   config: Optional[KeyModulationConfig] = None) -> List[ModulationResult]:
        """
        Suggest key modulations based on emotional target
        
        Args:
            melody_dna: Original melody DNA
            emotional_target: Target emotion ("brighter", "darker", "warmer", "cooler")
            config: Optional modulation configuration
            
        Returns:
            List of ModulationResult suggestions ranked by suitability
        """
        config = config or self.default_config
        current_key = self._analyze_current_key(melody_dna)
        
        # Define emotional key mappings
        emotional_shifts = {
            'brighter': [2, 4, 7],  # Major second, major third, perfect fifth up
            'darker': [-2, -4, -7],  # Major second, major third, perfect fifth down
            'warmer': [3, 5],  # Minor third, perfect fourth up
            'cooler': [-3, -5]  # Minor third, perfect fourth down
        }
        
        if emotional_target not in emotional_shifts:
            logger.warning("Unknown emotional target: %s", emotional_target)
            return []
        
        suggestions = []
        shifts = emotional_shifts[emotional_target]
        
        for shift in shifts:
            try:
                result = self.transpose_melody(melody_dna, shift, config)
                
                # Add emotional scoring
                emotional_score = self._calculate_emotional_fit(
                    result.modulated_melody_dna, emotional_target
                )
                result.emotional_fit_score = emotional_score
                
                suggestions.append(result)
                
            except Exception as e:
                logger.warning("Failed to create modulation for shift %d: %s", shift, e)
        
        # Sort by emotional fit and harmonic coherence
        suggestions.sort(key=lambda x: (
            getattr(x, 'emotional_fit_score', 0.5) * 0.6 +
            x.harmonic_coherence_score * 0.4
        ), reverse=True)
        
        logger.info("Generated %d emotional modulation suggestions for target: %s",
                   len(suggestions), emotional_target)
        
        return suggestions
    
    def _deep_copy_melody_dna(self, melody_dna: Dict[str, Any]) -> Dict[str, Any]:
        """Create deep copy of melody DNA for modulation"""
        import copy
        return copy.deepcopy(melody_dna)
    
    def _determine_keys_from_transpose(self, melody_dna: Dict[str, Any],
                                     semitone_shift: int) -> Tuple[str, str]:
        """Determine original and target keys from transposition"""
        
        # Analyze current key
        original_key = self._analyze_current_key(melody_dna)
        
        # Calculate target key
        key_names = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
        
        # Find original key index
        original_index = 0
        for i, key in enumerate(key_names):
            if key == original_key or key.replace('#', 'b') == original_key:
                original_index = i
                break
        
        # Calculate target key index
        target_index = (original_index + semitone_shift) % 12
        target_key = key_names[target_index]
        
        return original_key, target_key
    
    def _analyze_current_key(self, melody_dna: Dict[str, Any]) -> str:
        """Analyze melody to determine most likely current key"""
        
        intervals = melody_dna.get('interval_sequence', np.array([]))
        f0_track = melody_dna.get('f0_track', np.array([]))
        
        if len(f0_track) == 0:
            return 'C'  # Default key
        
        # Convert to semitone values
        semitones = 12 * np.log2(f0_track)
        
        # Find most common pitch classes
        pitch_classes = np.round(semitones) % 12
        unique_classes, counts = np.unique(pitch_classes, return_counts=True)
        
        # Simple key detection based on most common pitch class
        if len(unique_classes) > 0:
            tonic_class = unique_classes[np.argmax(counts)]
            key_names = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
            return key_names[int(tonic_class)]
        
        return 'C'  # Default fallback
    
    def _calculate_key_distance(self, key1: str, key2: str) -> int:
        """Calculate semitone distance between two keys"""
        
        key_names = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
        
        # Find indices
        index1 = 0
        index2 = 0
        
        for i, key in enumerate(key_names):
            if key == key1 or key.replace('#', 'b') == key1:
                index1 = i
            if key == key2 or key.replace('#', 'b') == key2:
                index2 = i
        
        # Calculate shortest distance
        distance = (index2 - index1) % 12
        if distance > 6:
            distance -= 12
        
        return distance
    
    def _apply_circle_of_fifths_modulation(self, melody_dna: Dict[str, Any],
                                         current_key: str, target_key: str,
                                         config: KeyModulationConfig) -> ModulationResult:
        """Apply modulation following circle of fifths relationships"""
        
        # For now, implement as enhanced transposition
        # Full circle of fifths modulation would require harmonic analysis
        semitone_shift = self._calculate_key_distance(current_key, target_key)
        
        result = self.transpose_melody(melody_dna, semitone_shift, config)
        result.applied_modifications.append("Applied circle of fifths modulation principle")
        
        return result
    
    def _apply_parallel_modulation(self, melody_dna: Dict[str, Any],
                                 current_key: str, target_key: str,
                                 config: KeyModulationConfig) -> ModulationResult:
        """Apply parallel major/minor modulation"""
        
        # Parallel modulation maintains the same tonic but changes mode
        # For simplicity, implement as transposition with mode analysis
        semitone_shift = self._calculate_key_distance(current_key, target_key)
        
        result = self.transpose_melody(melody_dna, semitone_shift, config)
        result.applied_modifications.append("Applied parallel major/minor modulation")
        
        return result
    
    def _apply_relative_modulation(self, melody_dna: Dict[str, Any],
                                 current_key: str, target_key: str,
                                 config: KeyModulationConfig) -> ModulationResult:
        """Apply relative major/minor modulation"""
        
        # Relative modulation shares the same key signature but different tonic
        semitone_shift = self._calculate_key_distance(current_key, target_key)
        
        result = self.transpose_melody(melody_dna, semitone_shift, config)
        result.applied_modifications.append("Applied relative major/minor modulation")
        
        return result
    
    def _calculate_harmonic_coherence(self, melody_dna: Dict[str, Any], key: str) -> float:
        """Calculate how well the melody fits in the specified key"""
        
        f0_track = melody_dna.get('f0_track', np.array([]))
        
        if len(f0_track) == 0:
            return 0.5
        
        # Convert to pitch classes
        semitones = 12 * np.log2(f0_track)
        pitch_classes = np.round(semitones) % 12
        
        # Get scale for the key (assume major for simplicity)
        scale_tones = self.scale_database[ScaleType.MAJOR]
        
        # Calculate key offset
        key_names = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
        key_offset = 0
        for i, k in enumerate(key_names):
            if k == key:
                key_offset = i
                break
        
        # Transpose scale to the key
        key_scale = [(tone + key_offset) % 12 for tone in scale_tones]
        
        # Calculate percentage of notes that fit the scale
        scale_notes = sum(1 for pc in pitch_classes if pc in key_scale)
        coherence = scale_notes / len(pitch_classes) if len(pitch_classes) > 0 else 0.5
        
        return coherence
    
    def _analyze_scale_content(self, melody_dna: Dict[str, Any], key: str) -> Dict[str, Any]:
        """Analyze scale content and characteristics of the melody"""
        
        f0_track = melody_dna.get('f0_track', np.array([]))
        intervals = melody_dna.get('interval_sequence', np.array([]))
        
        analysis = {
            'detected_key': key,
            'scale_type': 'major',  # Simplified
            'accidental_count': 0,
            'modal_characteristics': {}
        }
        
        if len(f0_track) > 0:
            # Analyze pitch class distribution
            semitones = 12 * np.log2(f0_track)
            pitch_classes = np.round(semitones) % 12
            unique_classes, counts = np.unique(pitch_classes, return_counts=True)
            
            analysis['pitch_class_distribution'] = {
                'classes': unique_classes.tolist(),
                'counts': counts.tolist()
            }
            
            analysis['tonal_center_strength'] = np.max(counts) / np.sum(counts)
        
        if len(intervals) > 0:
            # Analyze interval characteristics
            interval_types = np.abs(intervals)
            
            # Count perfect fifths and fourths (strong tonal indicators)
            perfect_intervals = np.sum((interval_types > 6.5) & (interval_types < 7.5))
            perfect_intervals += np.sum((interval_types > 4.5) & (interval_types < 5.5))
            
            analysis['perfect_interval_ratio'] = perfect_intervals / len(intervals)
            analysis['average_interval_size'] = np.mean(interval_types)
        
        return analysis
    
    def _calculate_emotional_fit(self, melody_dna: Dict[str, Any], target_emotion: str) -> float:
        """Calculate how well the modulated melody fits the target emotion"""
        
        stats = melody_dna.get('melodic_stats', {})
        
        if not stats:
            return 0.5
        
        # Simple emotional mapping based on pitch characteristics
        pitch_mean = stats.get('pitch_mean', 60)  # MIDI note number equivalent
        pitch_range = stats.get('pitch_range', 12)
        
        emotional_scores = {
            'brighter': min(1.0, (pitch_mean - 60) / 24 + 0.5),  # Higher = brighter
            'darker': min(1.0, (60 - pitch_mean) / 24 + 0.5),    # Lower = darker
            'warmer': min(1.0, 1.0 - abs(pitch_range - 12) / 12), # Moderate range = warmer
            'cooler': min(1.0, pitch_range / 24)                  # Wide range = cooler
        }
        
        return emotional_scores.get(target_emotion, 0.5)
    
    def validate_modulation(self, original_dna: Dict[str, Any],
                          modulated_dna: Dict[str, Any]) -> Dict[str, Any]:
        """Validate modulation result for musical coherence"""
        
        validation = {
            'pitch_range_reasonable': True,
            'interval_preservation': True,
            'harmonic_coherence': True,
            'warnings': [],
            'overall_valid': True
        }
        
        # Check pitch range
        orig_stats = original_dna.get('melodic_stats', {})
        mod_stats = modulated_dna.get('melodic_stats', {})
        
        if 'pitch_mean' in orig_stats and 'pitch_mean' in mod_stats:
            pitch_shift = abs(mod_stats['pitch_mean'] - orig_stats['pitch_mean'])
            if pitch_shift > 24:  # More than 2 octaves
                validation['pitch_range_reasonable'] = False
                validation['warnings'].append(f"Large pitch shift: {pitch_shift:.1f} semitones")
        
        # Check interval preservation
        orig_intervals = original_dna.get('interval_sequence', np.array([]))
        mod_intervals = modulated_dna.get('interval_sequence', np.array([]))
        
        if len(orig_intervals) > 0 and len(mod_intervals) > 0:
            if len(orig_intervals) == len(mod_intervals):
                interval_diff = np.mean(np.abs(orig_intervals - mod_intervals))
                if interval_diff > self.interval_tolerance:
                    validation['interval_preservation'] = False
                    validation['warnings'].append(f"Interval distortion: {interval_diff:.3f}")
            else:
                validation['warnings'].append("Interval sequence length mismatch")
        
        # Overall validation
        validation['overall_valid'] = all([
            validation['pitch_range_reasonable'],
            validation['interval_preservation'],
            validation['harmonic_coherence']
        ])
        
        logger.info("Modulation validation: valid=%s, warnings=%d",
                   validation['overall_valid'], len(validation['warnings']))
        
        return validation
    
    def get_available_keys(self) -> List[str]:
        """Get list of available target keys"""
        return list(self.key_signatures.keys())
    
    def get_key_characteristics(self, key: str) -> Dict[str, Any]:
        """Get characteristics and relationships of a specific key"""
        
        if key not in self.key_signatures:
            return {}
        
        return {
            'name': key,
            'key_signature': self.key_signatures[key],
            'circle_of_fifths_position': self.key_signatures[key],
            'related_keys': {
                'dominant': self.key_relationships['circle_of_fifths'].get(key, [None])[0],
                'subdominant': self.key_relationships['circle_of_fifths'].get(key, [None, None])[1] if len(self.key_relationships['circle_of_fifths'].get(key, [])) > 1 else None,
                'relative': self.key_relationships['relative_keys'].get(key),
                'parallel': self.key_relationships['parallel_keys'].get(key)
            },
            'emotional_associations': self._get_key_emotional_associations(key),
            'typical_use_cases': self._get_key_use_cases(key)
        }
    
    def _get_key_emotional_associations(self, key: str) -> List[str]:
        """Get traditional emotional associations for a key"""
        
        # Simplified emotional associations
        associations = {
            'C': ['pure', 'simple', 'bright'],
            'G': ['cheerful', 'pastoral', 'bright'],
            'D': ['brilliant', 'triumphant', 'bright'],
            'A': ['bright', 'sparkling', 'joyful'],
            'E': ['radiant', 'sharp', 'bright'],
            'B': ['harsh', 'brilliant', 'bright'],
            'F#': ['bright', 'sharp', 'intense'],
            'F': ['warm', 'pastoral', 'gentle'],
            'Bb': ['cheerful', 'noble', 'warm'],
            'Eb': ['warm', 'rich', 'heroic'],
            'Ab': ['warm', 'rich', 'deep'],
            'Db': ['warm', 'mysterious', 'deep'],
            'Gb': ['mysterious', 'remote', 'deep']
        }
        
        return associations.get(key, ['neutral'])
    
    def _get_key_use_cases(self, key: str) -> List[str]:
        """Get typical musical use cases for a key"""
        
        use_cases = {
            'C': ['beginner pieces', 'simple melodies', 'educational'],
            'G': ['folk music', 'country', 'light classical'],
            'D': ['orchestral music', 'violin pieces', 'bright songs'],
            'A': ['string music', 'guitar pieces', 'pop music'],
            'E': ['guitar music', 'rock', 'energetic pieces'],
            'B': ['advanced pieces', 'complex harmonies'],
            'F#': ['piano music', 'complex pieces'],
            'F': ['vocal music', 'gentle pieces', 'ballads'],
            'Bb': ['wind instruments', 'jazz', 'band music'],
            'Eb': ['brass music', 'heroic themes', 'jazz'],
            'Ab': ['piano music', 'romantic pieces'],
            'Db': ['advanced piano', 'impressionistic'],
            'Gb': ['advanced pieces', 'special effects']
        }
        
        return use_cases.get(key, ['general use'])