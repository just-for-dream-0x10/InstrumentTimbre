"""
Rhythm Adjustment Engine - System Development Task

This module implements intelligent rhythm adjustment algorithms that can
modify tempo, timing, and rhythmic patterns while preserving melodic
recognizability and musical coherence.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum

# Configure logging for the module
logger = logging.getLogger(__name__)


class RhythmicStyle(Enum):
    """Enumeration of rhythmic style types"""
    STRAIGHT = "straight"
    SWING = "swing"
    SHUFFLE = "shuffle"
    LATIN = "latin"
    TRIPLET = "triplet"
    SYNCOPATED = "syncopated"


class TempoAdjustmentType(Enum):
    """Types of tempo adjustments"""
    PROPORTIONAL = "proportional"  # Scale by factor
    ABSOLUTE = "absolute"  # Set to specific BPM
    ADAPTIVE = "adaptive"  # Adjust based on content


@dataclass
class RhythmAdjustmentConfig:
    """Configuration for rhythm adjustment parameters"""
    preserve_beat_structure: bool = True
    allow_micro_timing: bool = True
    swing_ratio: float = 0.6  # For swing feel (0.5 = straight, 0.67 = full swing)
    syncopation_intensity: float = 0.3
    max_tempo_change: float = 0.5  # Maximum proportional change
    quantization_strength: float = 0.8
    groove_preservation: float = 0.7


@dataclass
class RhythmAdjustmentResult:
    """Result of rhythm adjustment process"""
    adjusted_rhythmic_skeleton: Dict[str, Any]
    tempo_changes: List[Dict[str, Any]]
    applied_adjustments: List[str]
    timing_modifications: Dict[str, List[float]]
    preservation_score: float
    groove_characteristics: Dict[str, float]
    warnings: List[str]


class RhythmAdjustmentEngine:
    """
    Core engine for intelligent rhythm adjustment
    
    Provides sophisticated rhythm modification capabilities:
    - Tempo scaling with beat preservation
    - Rhythmic feel transformation (straight to swing, etc.)
    - Micro-timing adjustments for groove
    - Syncopation and accent pattern modification
    - Beat quantization and humanization
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize rhythm adjustment engine
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.default_config = RhythmAdjustmentConfig()
        
        # Engine parameters
        self.sample_rate = self.config.get('sample_rate', 22050)
        self.hop_length = self.config.get('hop_length', 512)
        self.beat_tracking_window = self.config.get('beat_tracking_window', 2.0)
        
        # Rhythmic pattern templates
        self._initialize_rhythm_templates()
        
        logger.info("RhythmAdjustmentEngine initialized")
    
    def _initialize_rhythm_templates(self) -> None:
        """Initialize templates for different rhythmic styles"""
        self.rhythm_templates = {
            RhythmicStyle.STRAIGHT: {
                'eighth_note_timing': [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
                'accent_pattern': [1.0, 0.6, 0.8, 0.6, 1.0, 0.6, 0.8, 0.6],
                'micro_timing_variance': 0.02
            },
            RhythmicStyle.SWING: {
                'eighth_note_timing': [0.0, 0.67, 1.0, 1.67, 2.0, 2.67, 3.0, 3.67],
                'accent_pattern': [1.0, 0.5, 0.8, 0.5, 1.0, 0.5, 0.8, 0.5],
                'micro_timing_variance': 0.05
            },
            RhythmicStyle.SHUFFLE: {
                'eighth_note_timing': [0.0, 0.75, 1.0, 1.75, 2.0, 2.75, 3.0, 3.75],
                'accent_pattern': [1.0, 0.4, 0.8, 0.4, 1.0, 0.4, 0.8, 0.4],
                'micro_timing_variance': 0.08
            },
            RhythmicStyle.SYNCOPATED: {
                'eighth_note_timing': [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
                'accent_pattern': [0.8, 1.0, 0.6, 1.0, 0.8, 1.0, 0.6, 1.0],
                'micro_timing_variance': 0.03
            }
        }
    
    def adjust_tempo(self, rhythmic_skeleton: Dict[str, Any],
                    tempo_adjustment: Union[float, int],
                    adjustment_type: TempoAdjustmentType = TempoAdjustmentType.PROPORTIONAL,
                    config: Optional[RhythmAdjustmentConfig] = None) -> RhythmAdjustmentResult:
        """
        Adjust tempo while preserving rhythmic relationships
        
        Args:
            rhythmic_skeleton: Original rhythmic structure
            tempo_adjustment: New tempo or scaling factor
            adjustment_type: Type of tempo adjustment to apply
            config: Optional adjustment configuration
            
        Returns:
            RhythmAdjustmentResult with adjusted rhythm
        """
        config = config or self.default_config
        
        # Extract current tempo and timing information
        current_tempo = rhythmic_skeleton.get('tempo', 120)
        onset_times = rhythmic_skeleton.get('onset_times', [])
        beat_times = rhythmic_skeleton.get('beat_times', [])
        
        # Calculate new tempo
        if adjustment_type == TempoAdjustmentType.PROPORTIONAL:
            # Clamp tempo change to maximum allowed
            clamped_factor = np.clip(tempo_adjustment, 
                                   1.0 - config.max_tempo_change,
                                   1.0 + config.max_tempo_change)
            new_tempo = current_tempo * clamped_factor
        elif adjustment_type == TempoAdjustmentType.ABSOLUTE:
            new_tempo = float(tempo_adjustment)
        else:  # ADAPTIVE
            new_tempo = self._calculate_adaptive_tempo(rhythmic_skeleton, tempo_adjustment)
        
        # Create adjusted rhythmic skeleton
        adjusted_skeleton = rhythmic_skeleton.copy()
        tempo_ratio = new_tempo / current_tempo
        
        # Adjust timing arrays
        if len(onset_times) > 0:
            adjusted_skeleton['onset_times'] = [t / tempo_ratio for t in onset_times]
        
        if len(beat_times) > 0:
            adjusted_skeleton['beat_times'] = [t / tempo_ratio for t in beat_times]
        
        # Update tempo
        adjusted_skeleton['tempo'] = new_tempo
        
        # Recalculate derived timing measures
        self._recalculate_timing_measures(adjusted_skeleton)
        
        # Track applied changes
        tempo_changes = [{
            'original_tempo': current_tempo,
            'new_tempo': new_tempo,
            'ratio': tempo_ratio,
            'adjustment_type': adjustment_type.value
        }]
        
        applied_adjustments = [f"Tempo adjusted from {float(current_tempo):.1f} to {float(new_tempo):.1f} BPM"]
        
        # Calculate preservation score
        preservation_score = self._calculate_tempo_preservation_score(
            current_tempo, new_tempo, config
        )
        
        # Analyze groove characteristics
        groove_characteristics = self._analyze_groove_characteristics(adjusted_skeleton)
        
        # Generate warnings if needed
        warnings = []
        if abs(tempo_ratio - 1.0) > config.max_tempo_change:
            warnings.append(f"Tempo change exceeds recommended maximum: {tempo_ratio:.2f}")
        
        result = RhythmAdjustmentResult(
            adjusted_rhythmic_skeleton=adjusted_skeleton,
            tempo_changes=tempo_changes,
            applied_adjustments=applied_adjustments,
            timing_modifications={'tempo_scaling': [tempo_ratio]},
            preservation_score=preservation_score,
            groove_characteristics=groove_characteristics,
            warnings=warnings
        )
        
        logger.info("Tempo adjustment completed: %.1f -> %.1f BPM (ratio: %.3f)",
                   float(current_tempo), float(new_tempo), float(tempo_ratio))
        
        return result
    
    def apply_rhythmic_style(self, rhythmic_skeleton: Dict[str, Any],
                           target_style: RhythmicStyle,
                           config: Optional[RhythmAdjustmentConfig] = None) -> RhythmAdjustmentResult:
        """
        Apply specific rhythmic feel or style to the rhythm
        
        Args:
            rhythmic_skeleton: Original rhythmic structure
            target_style: Target rhythmic style to apply
            config: Optional adjustment configuration
            
        Returns:
            RhythmAdjustmentResult with style adjustments
        """
        config = config or self.default_config
        
        if target_style not in self.rhythm_templates:
            raise ValueError(f"Unsupported rhythmic style: {target_style}")
        
        style_template = self.rhythm_templates[target_style]
        adjusted_skeleton = rhythmic_skeleton.copy()
        applied_adjustments = []
        timing_modifications = {}
        
        # Apply style-specific timing adjustments
        onset_times = rhythmic_skeleton.get('onset_times', [])
        if len(onset_times) > 0 and config.allow_micro_timing:
            # Adjust micro-timing based on style
            adjusted_onsets = self._apply_style_timing(onset_times, style_template, config)
            adjusted_skeleton['onset_times'] = adjusted_onsets
            timing_modifications['onset_adjustments'] = adjusted_onsets
            applied_adjustments.append(f"Applied {target_style.value} timing feel")
        
        # Apply accent patterns
        if target_style == RhythmicStyle.SWING:
            adjusted_skeleton['swing_ratio'] = config.swing_ratio
            applied_adjustments.append(f"Applied swing feel with ratio {config.swing_ratio:.2f}")
        
        elif target_style == RhythmicStyle.SYNCOPATED:
            adjusted_skeleton['syncopation_level'] = config.syncopation_intensity
            applied_adjustments.append(f"Applied syncopation intensity {config.syncopation_intensity:.2f}")
        
        # Add style-specific characteristics
        adjusted_skeleton['rhythmic_style'] = target_style.value
        adjusted_skeleton['style_template'] = style_template
        
        # Calculate preservation score
        preservation_score = self._calculate_style_preservation_score(
            rhythmic_skeleton, adjusted_skeleton, config
        )
        
        # Analyze groove characteristics
        groove_characteristics = self._analyze_groove_characteristics(adjusted_skeleton)
        
        result = RhythmAdjustmentResult(
            adjusted_rhythmic_skeleton=adjusted_skeleton,
            tempo_changes=[],
            applied_adjustments=applied_adjustments,
            timing_modifications=timing_modifications,
            preservation_score=preservation_score,
            groove_characteristics=groove_characteristics,
            warnings=[]
        )
        
        logger.info("Rhythmic style applied: %s, preservation: %.3f",
                   target_style.value, preservation_score)
        
        return result
    
    def quantize_rhythm(self, rhythmic_skeleton: Dict[str, Any],
                       grid_resolution: float = 0.25,
                       config: Optional[RhythmAdjustmentConfig] = None) -> RhythmAdjustmentResult:
        """
        Quantize rhythm to a specific grid while preserving musical feel
        
        Args:
            rhythmic_skeleton: Original rhythmic structure
            grid_resolution: Quantization grid (0.25 = 16th notes, 0.5 = 8th notes)
            config: Optional adjustment configuration
            
        Returns:
            RhythmAdjustmentResult with quantized rhythm
        """
        config = config or self.default_config
        
        adjusted_skeleton = rhythmic_skeleton.copy()
        onset_times = rhythmic_skeleton.get('onset_times', [])
        beat_times = rhythmic_skeleton.get('beat_times', [])
        
        if len(onset_times) == 0 or len(beat_times) == 0:
            logger.warning("Insufficient timing data for quantization")
            return self._create_empty_result(adjusted_skeleton)
        
        # Calculate beat interval for grid
        if len(beat_times) > 1:
            beat_interval = np.mean(np.diff(beat_times))
        else:
            tempo = rhythmic_skeleton.get('tempo', 120)
            beat_interval = 60.0 / tempo
        
        grid_interval = beat_interval * grid_resolution
        
        # Quantize onset times
        quantized_onsets = []
        timing_deviations = []
        
        for onset in onset_times:
            # Find nearest grid point
            grid_position = round(onset / grid_interval) * grid_interval
            
            # Apply quantization strength
            quantized_onset = (onset * (1.0 - config.quantization_strength) +
                             grid_position * config.quantization_strength)
            
            quantized_onsets.append(quantized_onset)
            timing_deviations.append(abs(onset - quantized_onset))
        
        adjusted_skeleton['onset_times'] = quantized_onsets
        adjusted_skeleton['quantization_grid'] = grid_resolution
        
        # Calculate quantization statistics
        mean_deviation = np.mean(timing_deviations)
        max_deviation = np.max(timing_deviations) if timing_deviations else 0.0
        
        applied_adjustments = [
            f"Quantized to {grid_resolution} note grid",
            f"Mean timing deviation: {mean_deviation:.3f}s"
        ]
        
        timing_modifications = {
            'quantized_onsets': quantized_onsets,
            'timing_deviations': timing_deviations
        }
        
        # Calculate preservation score
        preservation_score = 1.0 - (mean_deviation / beat_interval)
        preservation_score = max(0.0, preservation_score)
        
        # Analyze groove characteristics
        groove_characteristics = self._analyze_groove_characteristics(adjusted_skeleton)
        
        # Generate warnings
        warnings = []
        if max_deviation > beat_interval * 0.25:
            warnings.append(f"Large timing deviation detected: {max_deviation:.3f}s")
        
        result = RhythmAdjustmentResult(
            adjusted_rhythmic_skeleton=adjusted_skeleton,
            tempo_changes=[],
            applied_adjustments=applied_adjustments,
            timing_modifications=timing_modifications,
            preservation_score=preservation_score,
            groove_characteristics=groove_characteristics,
            warnings=warnings
        )
        
        logger.info("Rhythm quantized: grid=%.2f, deviation=%.3f, preservation=%.3f",
                   grid_resolution, mean_deviation, preservation_score)
        
        return result
    
    def _apply_style_timing(self, onset_times: List[float],
                          style_template: Dict[str, Any],
                          config: RhythmAdjustmentConfig) -> List[float]:
        """Apply style-specific micro-timing adjustments to onset times"""
        
        if len(onset_times) == 0:
            return onset_times
        
        timing_pattern = style_template.get('eighth_note_timing', [])
        micro_variance = style_template.get('micro_timing_variance', 0.02)
        
        # Calculate beat period from onset times
        if len(onset_times) > 1:
            estimated_beat_period = np.median(np.diff(onset_times))
        else:
            estimated_beat_period = 0.5  # Default to moderate tempo
        
        adjusted_onsets = []
        
        for i, onset in enumerate(onset_times):
            # Map to pattern position
            pattern_pos = i % len(timing_pattern)
            pattern_timing = timing_pattern[pattern_pos]
            
            # Calculate beat-relative position
            beat_position = (onset % estimated_beat_period) / estimated_beat_period
            
            # Apply style timing adjustment
            style_adjustment = (pattern_timing - beat_position) * estimated_beat_period * 0.1
            
            # Add micro-timing variance
            micro_adjustment = np.random.normal(0, micro_variance * estimated_beat_period)
            
            adjusted_onset = onset + style_adjustment + micro_adjustment
            adjusted_onsets.append(adjusted_onset)
        
        return adjusted_onsets
    
    def _calculate_adaptive_tempo(self, rhythmic_skeleton: Dict[str, Any],
                                target_energy: float) -> float:
        """Calculate adaptive tempo based on musical content and target energy"""
        
        current_tempo = rhythmic_skeleton.get('tempo', 120)
        note_density = rhythmic_skeleton.get('note_density', 1.0)
        
        # Base tempo adjustment on note density and target energy
        if target_energy > 0.8:  # High energy
            tempo_factor = 1.2 + (note_density - 1.0) * 0.1
        elif target_energy > 0.5:  # Medium energy
            tempo_factor = 1.0 + (note_density - 1.0) * 0.05
        else:  # Low energy
            tempo_factor = 0.8 + (note_density - 1.0) * 0.05
        
        adaptive_tempo = current_tempo * tempo_factor
        
        # Clamp to reasonable range
        return np.clip(adaptive_tempo, 60, 200)
    
    def _recalculate_timing_measures(self, rhythmic_skeleton: Dict[str, Any]) -> None:
        """Recalculate derived timing measures after tempo adjustment"""
        
        onset_times = rhythmic_skeleton.get('onset_times', [])
        
        if len(onset_times) > 1:
            # Recalculate inter-onset intervals
            ioi = np.diff(onset_times)
            rhythmic_skeleton['inter_onset_intervals'] = ioi.tolist()
            
            # Recalculate note density
            total_duration = onset_times[-1] - onset_times[0]
            note_density = len(onset_times) / max(total_duration, 1.0)
            rhythmic_skeleton['note_density'] = note_density
    
    def _calculate_tempo_preservation_score(self, original_tempo: float,
                                          new_tempo: float,
                                          config: RhythmAdjustmentConfig) -> float:
        """Calculate how well tempo change preserves musical character"""
        
        ratio = new_tempo / original_tempo
        
        # Score based on how extreme the change is
        if 0.8 <= ratio <= 1.25:  # Moderate change
            base_score = 0.9
        elif 0.5 <= ratio <= 2.0:  # Large but acceptable change
            base_score = 0.7
        else:  # Extreme change
            base_score = 0.4
        
        # Adjust for maximum allowed change
        if abs(ratio - 1.0) > config.max_tempo_change:
            base_score *= 0.5
        
        return base_score
    
    def _calculate_style_preservation_score(self, original_skeleton: Dict[str, Any],
                                          adjusted_skeleton: Dict[str, Any],
                                          config: RhythmAdjustmentConfig) -> float:
        """Calculate preservation score for style adjustments"""
        
        # Compare timing deviations
        orig_onsets = original_skeleton.get('onset_times', [])
        adj_onsets = adjusted_skeleton.get('onset_times', [])
        
        if not orig_onsets or not adj_onsets or len(orig_onsets) != len(adj_onsets):
            return 0.5  # Neutral score if cannot compare
        
        # Calculate timing differences
        timing_diffs = [abs(o - a) for o, a in zip(orig_onsets, adj_onsets)]
        mean_diff = np.mean(timing_diffs)
        
        # Estimate beat interval
        if len(orig_onsets) > 1:
            beat_interval = np.median(np.diff(orig_onsets))
        else:
            beat_interval = 0.5
        
        # Score based on relative timing change
        relative_change = mean_diff / beat_interval
        preservation_score = max(0.0, 1.0 - relative_change * 2)
        
        return preservation_score
    
    def _analyze_groove_characteristics(self, rhythmic_skeleton: Dict[str, Any]) -> Dict[str, float]:
        """Analyze groove characteristics of the rhythm"""
        
        characteristics = {}
        onset_times = rhythmic_skeleton.get('onset_times', [])
        
        if len(onset_times) > 2:
            # Calculate timing regularity
            ioi = np.diff(onset_times)
            timing_regularity = 1.0 - (np.std(ioi) / max(np.mean(ioi), 0.01))
            characteristics['timing_regularity'] = max(0.0, min(1.0, timing_regularity))
            
            # Calculate rhythmic complexity
            unique_intervals = len(set(np.round(ioi, 2).tolist()))
            complexity = min(1.0, unique_intervals / len(ioi))
            characteristics['rhythmic_complexity'] = complexity
            
        else:
            characteristics['timing_regularity'] = 0.5
            characteristics['rhythmic_complexity'] = 0.5
        
        # Extract style-specific characteristics
        tempo = rhythmic_skeleton.get('tempo', 120)
        characteristics['tempo_energy'] = min(1.0, tempo / 140.0)
        
        # Check for swing characteristics
        swing_ratio = rhythmic_skeleton.get('swing_ratio', 0.5)
        characteristics['swing_feel'] = abs(swing_ratio - 0.5) * 2
        
        # Check for syncopation
        syncopation_level = rhythmic_skeleton.get('syncopation_level', 0.0)
        characteristics['syncopation'] = syncopation_level
        
        return characteristics
    
    def _create_empty_result(self, skeleton: Dict[str, Any]) -> RhythmAdjustmentResult:
        """Create empty result for error cases"""
        return RhythmAdjustmentResult(
            adjusted_rhythmic_skeleton=skeleton,
            tempo_changes=[],
            applied_adjustments=[],
            timing_modifications={},
            preservation_score=1.0,
            groove_characteristics={},
            warnings=["Insufficient data for rhythm adjustment"]
        )
    
    def validate_rhythm_adjustment(self, original_skeleton: Dict[str, Any],
                                 adjusted_skeleton: Dict[str, Any]) -> Dict[str, Any]:
        """Validate rhythm adjustment results"""
        
        validation = {
            'timing_coherence': True,
            'tempo_reasonable': True,
            'preservation_adequate': True,
            'warnings': [],
            'overall_valid': True
        }
        
        # Check tempo reasonableness
        adjusted_tempo = adjusted_skeleton.get('tempo', 120)
        if adjusted_tempo < 40 or adjusted_tempo > 250:
            validation['tempo_reasonable'] = False
            validation['warnings'].append(f"Unreasonable tempo: {adjusted_tempo} BPM")
        
        # Check timing coherence
        onset_times = adjusted_skeleton.get('onset_times', [])
        if len(onset_times) > 1:
            # Convert to numpy array for safe comparison
            onset_array = np.array(onset_times)
            is_sorted = np.all(onset_array[:-1] <= onset_array[1:])
            if not is_sorted:
                validation['timing_coherence'] = False
                validation['warnings'].append("Onset times not in chronological order")
        
        # Check overall validity
        validation['overall_valid'] = all([
            validation['timing_coherence'],
            validation['tempo_reasonable'],
            validation['preservation_adequate']
        ])
        
        logger.info("Rhythm adjustment validation: valid=%s, warnings=%d",
                   validation['overall_valid'], len(validation['warnings']))
        
        return validation