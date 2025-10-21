"""
Generation Quality Evaluator - System Development Task

This module implements comprehensive quality assessment for generated music,
evaluating preservation, musical coherence, style consistency, and overall quality.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import scipy.stats as stats

# Configure logging for the module
logger = logging.getLogger(__name__)


class QualityMetric(Enum):
    """Types of quality metrics"""
    MELODY_PRESERVATION = "melody_preservation"
    RHYTHM_PRESERVATION = "rhythm_preservation"
    STYLE_CONSISTENCY = "style_consistency"
    MUSICAL_COHERENCE = "musical_coherence"
    INSTRUMENTATION_SUITABILITY = "instrumentation_suitability"
    OVERALL_QUALITY = "overall_quality"


@dataclass
class QualityThresholds:
    """Quality thresholds for different criteria"""
    excellent: float = 0.9
    good: float = 0.75
    acceptable: float = 0.6
    poor: float = 0.4


@dataclass
class QualityReport:
    """Comprehensive quality assessment report"""
    overall_score: float
    metric_scores: Dict[str, float]
    quality_grades: Dict[str, str]
    detailed_analysis: Dict[str, Dict[str, Any]]
    recommendations: List[str]
    warnings: List[str]
    passed_thresholds: Dict[str, bool]


class GenerationQualityEvaluator:
    """
    Comprehensive quality evaluator for generated music
    
    Provides multi-dimensional quality assessment:
    - Preservation metrics (melody, rhythm, structure)
    - Musical coherence and listenability
    - Style consistency and authenticity
    - Technical quality indicators
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize quality evaluator
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.thresholds = QualityThresholds()
        
        # Metric weights for overall score calculation
        self.metric_weights = self.config.get('metric_weights', {
            QualityMetric.MELODY_PRESERVATION.value: 0.3,
            QualityMetric.RHYTHM_PRESERVATION.value: 0.2,
            QualityMetric.STYLE_CONSISTENCY.value: 0.2,
            QualityMetric.MUSICAL_COHERENCE.value: 0.2,
            QualityMetric.INSTRUMENTATION_SUITABILITY.value: 0.1
        })
        
        logger.info("GenerationQualityEvaluator initialized")
    
    def evaluate_generation(self, 
                          original_dna: Dict[str, Any],
                          generated_dna: Dict[str, Any],
                          generation_metadata: Dict[str, Any]) -> QualityReport:
        """
        Perform comprehensive quality evaluation
        
        Args:
            original_dna: Original melody DNA
            generated_dna: Generated/transformed melody DNA
            generation_metadata: Metadata from generation process
            
        Returns:
            QualityReport with detailed assessment
        """
        logger.info("Starting comprehensive quality evaluation")
        
        metric_scores = {}
        detailed_analysis = {}
        warnings = []
        
        # Evaluate each quality metric
        try:
            # 1. Melody preservation
            melody_score, melody_details = self._evaluate_melody_preservation(
                original_dna, generated_dna
            )
            metric_scores[QualityMetric.MELODY_PRESERVATION.value] = melody_score
            detailed_analysis[QualityMetric.MELODY_PRESERVATION.value] = melody_details
            
            # 2. Rhythm preservation
            rhythm_score, rhythm_details = self._evaluate_rhythm_preservation(
                original_dna, generated_dna
            )
            metric_scores[QualityMetric.RHYTHM_PRESERVATION.value] = rhythm_score
            detailed_analysis[QualityMetric.RHYTHM_PRESERVATION.value] = rhythm_details
            
            # 3. Style consistency
            style_score, style_details = self._evaluate_style_consistency(
                generated_dna, generation_metadata
            )
            metric_scores[QualityMetric.STYLE_CONSISTENCY.value] = style_score
            detailed_analysis[QualityMetric.STYLE_CONSISTENCY.value] = style_details
            
            # 4. Musical coherence
            coherence_score, coherence_details = self._evaluate_musical_coherence(
                generated_dna
            )
            metric_scores[QualityMetric.MUSICAL_COHERENCE.value] = coherence_score
            detailed_analysis[QualityMetric.MUSICAL_COHERENCE.value] = coherence_details
            
            # 5. Instrumentation suitability
            inst_score, inst_details = self._evaluate_instrumentation_suitability(
                generated_dna, generation_metadata
            )
            metric_scores[QualityMetric.INSTRUMENTATION_SUITABILITY.value] = inst_score
            detailed_analysis[QualityMetric.INSTRUMENTATION_SUITABILITY.value] = inst_details
            
        except Exception as e:
            logger.error("Quality evaluation failed: %s", e)
            warnings.append(f"Evaluation error: {str(e)}")
            # Provide default scores in case of failure
            for metric in QualityMetric:
                if metric.value not in metric_scores:
                    metric_scores[metric.value] = 0.5
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(metric_scores)
        
        # Generate quality grades
        quality_grades = self._assign_quality_grades(metric_scores)
        
        # Check threshold compliance
        passed_thresholds = self._check_thresholds(metric_scores)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            metric_scores, detailed_analysis, generation_metadata
        )
        
        # Create quality report
        report = QualityReport(
            overall_score=overall_score,
            metric_scores=metric_scores,
            quality_grades=quality_grades,
            detailed_analysis=detailed_analysis,
            recommendations=recommendations,
            warnings=warnings,
            passed_thresholds=passed_thresholds
        )
        
        logger.info("Quality evaluation completed: overall=%.3f", overall_score)
        return report
    
    def _evaluate_melody_preservation(self, original_dna: Dict[str, Any], 
                                    generated_dna: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Evaluate how well the melody is preserved"""
        
        details = {}
        
        # Extract key melodic features
        orig_intervals = original_dna.get('interval_sequence', np.array([]))
        gen_intervals = generated_dna.get('interval_sequence', np.array([]))
        
        orig_contour = original_dna.get('pitch_contour', np.array([]))
        gen_contour = generated_dna.get('pitch_contour', np.array([]))
        
        # 1. Interval sequence similarity
        if len(orig_intervals) > 0 and len(gen_intervals) > 0:
            min_len = min(len(orig_intervals), len(gen_intervals))
            interval_correlation = np.corrcoef(
                orig_intervals[:min_len], gen_intervals[:min_len]
            )[0, 1] if min_len > 1 else 0.0
            
            # Handle NaN correlation
            if np.isnan(interval_correlation):
                interval_correlation = 0.0
            
            details['interval_correlation'] = interval_correlation
        else:
            interval_correlation = 0.0
            details['interval_correlation'] = 0.0
        
        # 2. Contour shape similarity
        if len(orig_contour) > 0 and len(gen_contour) > 0:
            # Resample to same length for comparison
            target_len = min(len(orig_contour), len(gen_contour), 100)
            orig_resampled = np.interp(
                np.linspace(0, 1, target_len),
                np.linspace(0, 1, len(orig_contour)),
                orig_contour
            )
            gen_resampled = np.interp(
                np.linspace(0, 1, target_len),
                np.linspace(0, 1, len(gen_contour)),
                gen_contour
            )
            
            contour_correlation = np.corrcoef(orig_resampled, gen_resampled)[0, 1]
            if np.isnan(contour_correlation):
                contour_correlation = 0.0
            
            details['contour_correlation'] = contour_correlation
        else:
            contour_correlation = 0.0
            details['contour_correlation'] = 0.0
        
        # 3. Characteristic notes preservation
        orig_char = original_dna.get('characteristic_notes', [])
        gen_char = generated_dna.get('characteristic_notes', [])
        
        if len(orig_char) > 0:
            # Count preserved characteristic note types
            orig_types = set(note.get('type', 'unknown') for note in orig_char)
            gen_types = set(note.get('type', 'unknown') for note in gen_char)
            
            preserved_types = len(orig_types.intersection(gen_types))
            char_preservation = preserved_types / len(orig_types)
        else:
            char_preservation = 1.0  # No characteristic notes to preserve
        
        details['characteristic_notes_preservation'] = char_preservation
        
        # 4. Overall melody preservation score
        preservation_components = [
            interval_correlation * 0.4,
            contour_correlation * 0.4,
            char_preservation * 0.2
        ]
        
        melody_score = sum(max(0, comp) for comp in preservation_components)
        melody_score = max(0.0, min(1.0, melody_score))
        
        details['preservation_components'] = preservation_components
        details['melody_preservation_score'] = melody_score
        
        return melody_score, details
    
    def _evaluate_rhythm_preservation(self, original_dna: Dict[str, Any],
                                    generated_dna: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Evaluate rhythm preservation quality"""
        
        details = {}
        
        orig_rhythm = original_dna.get('rhythmic_skeleton', {})
        gen_rhythm = generated_dna.get('rhythmic_skeleton', {})
        
        # 1. Tempo similarity
        orig_tempo = orig_rhythm.get('tempo', 120)
        gen_tempo = gen_rhythm.get('tempo', 120)
        
        tempo_ratio = min(orig_tempo, gen_tempo) / max(orig_tempo, gen_tempo)
        details['tempo_similarity'] = tempo_ratio
        
        # 2. Note density similarity
        orig_density = orig_rhythm.get('note_density', 1.0)
        gen_density = gen_rhythm.get('note_density', 1.0)
        
        density_ratio = min(orig_density, gen_density) / max(orig_density, gen_density)
        details['density_similarity'] = density_ratio
        
        # 3. Onset timing patterns
        orig_onsets = orig_rhythm.get('onset_times', [])
        gen_onsets = gen_rhythm.get('onset_times', [])
        
        if len(orig_onsets) > 1 and len(gen_onsets) > 1:
            # Compare inter-onset intervals
            orig_ioi = np.diff(orig_onsets)
            gen_ioi = np.diff(gen_onsets)
            
            min_len = min(len(orig_ioi), len(gen_ioi))
            if min_len > 0:
                timing_correlation = np.corrcoef(orig_ioi[:min_len], gen_ioi[:min_len])[0, 1]
                if np.isnan(timing_correlation):
                    timing_correlation = 0.0
            else:
                timing_correlation = 0.0
        else:
            timing_correlation = 0.0
        
        details['timing_correlation'] = timing_correlation
        
        # 4. Overall rhythm preservation
        rhythm_components = [
            tempo_ratio * 0.3,
            density_ratio * 0.3,
            timing_correlation * 0.4
        ]
        
        rhythm_score = sum(max(0, comp) for comp in rhythm_components)
        rhythm_score = max(0.0, min(1.0, rhythm_score))
        
        details['rhythm_components'] = rhythm_components
        details['rhythm_preservation_score'] = rhythm_score
        
        return rhythm_score, details
    
    def _evaluate_style_consistency(self, generated_dna: Dict[str, Any],
                                  generation_metadata: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Evaluate style consistency of generated music"""
        
        details = {}
        
        # Get target style information
        target_style = generation_metadata.get('target_style', 'neutral')
        style_transformation = generation_metadata.get('style_transformation', {})
        
        # 1. Style transformation confidence
        style_confidence = style_transformation.get('confidence', 0.5)
        details['style_transformation_confidence'] = style_confidence
        
        # 2. Instrumentation style consistency
        instrumentation = generation_metadata.get('instrumentation_result', {})
        inst_confidence = instrumentation.get('confidence_score', 0.5)
        details['instrumentation_confidence'] = inst_confidence
        
        # 3. Musical characteristics matching target style
        stats = generated_dna.get('melodic_stats', {})
        
        # Style-specific characteristic checks
        style_match_score = self._assess_style_characteristics(stats, target_style)
        details['style_characteristics_match'] = style_match_score
        
        # 4. Overall style consistency
        style_components = [
            style_confidence * 0.4,
            inst_confidence * 0.3,
            style_match_score * 0.3
        ]
        
        style_score = sum(max(0, comp) for comp in style_components)
        style_score = max(0.0, min(1.0, style_score))
        
        details['style_components'] = style_components
        details['style_consistency_score'] = style_score
        
        return style_score, details
    
    def _assess_style_characteristics(self, stats: Dict[str, Any], target_style: str) -> float:
        """Assess how well musical characteristics match target style"""
        
        pitch_range = stats.get('pitch_range', 12)
        pitch_mean = stats.get('pitch_mean', 60)
        
        # Style-specific scoring
        if target_style == 'chinese_traditional':
            # Prefer moderate range, expressive characteristics
            range_score = 1.0 - abs(pitch_range - 18) / 18  # Optimal around 18 semitones
            range_score = max(0, range_score)
            return range_score
            
        elif target_style == 'western_classical':
            # Prefer wider range, structured characteristics
            range_score = min(1.0, pitch_range / 24)  # Prefer wide range
            return range_score
            
        elif target_style == 'modern_pop':
            # Prefer moderate range, accessible characteristics
            range_score = 1.0 - abs(pitch_range - 12) / 12  # Optimal around 12 semitones
            range_score = max(0, range_score)
            return range_score
        
        else:
            # Neutral - any reasonable range is acceptable
            return 0.7
    
    def _evaluate_musical_coherence(self, generated_dna: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Evaluate overall musical coherence and listenability"""
        
        details = {}
        
        # 1. Pitch coherence
        stats = generated_dna.get('melodic_stats', {})
        f0_track = generated_dna.get('f0_track', np.array([]))
        
        # Check for reasonable pitch range
        pitch_range = stats.get('pitch_range', 0)
        if 6 <= pitch_range <= 36:  # 0.5 to 3 octaves
            pitch_range_score = 1.0
        elif pitch_range > 0:
            pitch_range_score = 0.7
        else:
            pitch_range_score = 0.3
        
        details['pitch_range_score'] = pitch_range_score
        
        # 2. Melodic flow (smoothness of intervals)
        intervals = generated_dna.get('interval_sequence', np.array([]))
        if len(intervals) > 0:
            large_jumps = np.sum(np.abs(intervals) > 12) / len(intervals)  # Octave+ jumps
            smoothness_score = 1.0 - large_jumps
        else:
            smoothness_score = 0.5
        
        details['melodic_smoothness'] = smoothness_score
        
        # 3. Rhythmic regularity
        rhythm = generated_dna.get('rhythmic_skeleton', {})
        onset_times = rhythm.get('onset_times', [])
        
        if len(onset_times) > 2:
            ioi = np.diff(onset_times)
            rhythm_regularity = 1.0 - (np.std(ioi) / max(np.mean(ioi), 0.01))
            rhythm_regularity = max(0, min(1, rhythm_regularity))
        else:
            rhythm_regularity = 0.5
        
        details['rhythm_regularity'] = rhythm_regularity
        
        # 4. Phrase structure coherence
        phrases = generated_dna.get('phrase_boundaries', [])
        if len(phrases) > 2:
            phrase_lengths = np.diff(phrases)
            phrase_balance = 1.0 - (np.std(phrase_lengths) / max(np.mean(phrase_lengths), 0.01))
            phrase_balance = max(0, min(1, phrase_balance))
        else:
            phrase_balance = 0.7  # Neutral if insufficient phrase data
        
        details['phrase_balance'] = phrase_balance
        
        # 5. Overall coherence score
        coherence_components = [
            pitch_range_score * 0.3,
            smoothness_score * 0.3,
            rhythm_regularity * 0.2,
            phrase_balance * 0.2
        ]
        
        coherence_score = sum(max(0, comp) for comp in coherence_components)
        coherence_score = max(0.0, min(1.0, coherence_score))
        
        details['coherence_components'] = coherence_components
        details['musical_coherence_score'] = coherence_score
        
        return coherence_score, details
    
    def _evaluate_instrumentation_suitability(self, generated_dna: Dict[str, Any],
                                             generation_metadata: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Evaluate instrumentation suitability"""
        
        details = {}
        
        instrumentation = generation_metadata.get('instrumentation_result', {})
        
        if not instrumentation:
            return 0.5, {'no_instrumentation_data': True}
        
        # 1. Instrumentation confidence
        inst_confidence = instrumentation.get('confidence_score', 0.5)
        details['instrumentation_confidence'] = inst_confidence
        
        # 2. Range compatibility
        primary_instrument = instrumentation.get('primary_instrument', '')
        stats = generated_dna.get('melodic_stats', {})
        pitch_range = stats.get('pitch_range', 12)
        
        # Simple range compatibility check
        if primary_instrument in ['erhu', 'violin']:
            # String instruments handle wide ranges well
            range_compatibility = min(1.0, pitch_range / 24)
        elif primary_instrument in ['piano', 'synthesizer']:
            # Keyboards handle any range
            range_compatibility = 1.0
        else:
            # Default compatibility
            range_compatibility = 0.8
        
        details['range_compatibility'] = range_compatibility
        
        # 3. Style appropriateness
        target_style = generation_metadata.get('target_style', 'neutral')
        style_appropriateness = self._assess_instrument_style_match(primary_instrument, target_style)
        details['style_appropriateness'] = style_appropriateness
        
        # 4. Overall instrumentation score
        inst_components = [
            inst_confidence * 0.4,
            range_compatibility * 0.3,
            style_appropriateness * 0.3
        ]
        
        inst_score = sum(max(0, comp) for comp in inst_components)
        inst_score = max(0.0, min(1.0, inst_score))
        
        details['instrumentation_components'] = inst_components
        details['instrumentation_score'] = inst_score
        
        return inst_score, details
    
    def _assess_instrument_style_match(self, instrument: str, style: str) -> float:
        """Assess how well instrument matches style"""
        
        style_instrument_map = {
            'chinese_traditional': ['erhu', 'pipa', 'guzheng'],
            'western_classical': ['violin', 'piano', 'cello'],
            'modern_pop': ['electric_guitar', 'piano', 'synthesizer']
        }
        
        appropriate_instruments = style_instrument_map.get(style, [])
        
        if instrument in appropriate_instruments:
            return 1.0
        elif style == 'neutral' or not appropriate_instruments:
            return 0.8
        else:
            return 0.6
    
    def _calculate_overall_score(self, metric_scores: Dict[str, float]) -> float:
        """Calculate weighted overall quality score"""
        
        total_score = 0.0
        total_weight = 0.0
        
        for metric, score in metric_scores.items():
            weight = self.metric_weights.get(metric, 0.1)
            total_score += score * weight
            total_weight += weight
        
        if total_weight > 0:
            return total_score / total_weight
        else:
            return 0.5
    
    def _assign_quality_grades(self, metric_scores: Dict[str, float]) -> Dict[str, str]:
        """Assign letter grades to quality metrics"""
        
        grades = {}
        
        for metric, score in metric_scores.items():
            if score >= self.thresholds.excellent:
                grades[metric] = 'A'
            elif score >= self.thresholds.good:
                grades[metric] = 'B'
            elif score >= self.thresholds.acceptable:
                grades[metric] = 'C'
            elif score >= self.thresholds.poor:
                grades[metric] = 'D'
            else:
                grades[metric] = 'F'
        
        return grades
    
    def _check_thresholds(self, metric_scores: Dict[str, float]) -> Dict[str, bool]:
        """Check which metrics pass quality thresholds"""
        
        passed = {}
        
        for metric, score in metric_scores.items():
            passed[metric] = score >= self.thresholds.acceptable
        
        return passed
    
    def _generate_recommendations(self, metric_scores: Dict[str, float],
                                detailed_analysis: Dict[str, Dict[str, Any]],
                                generation_metadata: Dict[str, Any]) -> List[str]:
        """Generate improvement recommendations based on quality scores"""
        
        recommendations = []
        
        # Check each metric for specific recommendations
        melody_score = metric_scores.get(QualityMetric.MELODY_PRESERVATION.value, 0.5)
        if melody_score < self.thresholds.acceptable:
            recommendations.append("Consider using higher melody preservation settings")
            
            melody_details = detailed_analysis.get(QualityMetric.MELODY_PRESERVATION.value, {})
            if melody_details.get('interval_correlation', 0) < 0.5:
                recommendations.append("Interval patterns significantly changed - review style transformation intensity")
        
        rhythm_score = metric_scores.get(QualityMetric.RHYTHM_PRESERVATION.value, 0.5)
        if rhythm_score < self.thresholds.acceptable:
            recommendations.append("Consider gentler rhythm adjustments")
        
        style_score = metric_scores.get(QualityMetric.STYLE_CONSISTENCY.value, 0.5)
        if style_score < self.thresholds.acceptable:
            recommendations.append("Style transformation may need refinement")
        
        coherence_score = metric_scores.get(QualityMetric.MUSICAL_COHERENCE.value, 0.5)
        if coherence_score < self.thresholds.acceptable:
            recommendations.append("Generated music may lack musical coherence")
            
            coherence_details = detailed_analysis.get(QualityMetric.MUSICAL_COHERENCE.value, {})
            if coherence_details.get('melodic_smoothness', 0) < 0.6:
                recommendations.append("Consider smoother melodic transitions")
        
        # General recommendations
        overall_score = sum(metric_scores.values()) / len(metric_scores)
        if overall_score >= self.thresholds.excellent:
            recommendations.append("Excellent generation quality achieved!")
        elif overall_score >= self.thresholds.good:
            recommendations.append("Good generation quality - minor improvements possible")
        
        return recommendations