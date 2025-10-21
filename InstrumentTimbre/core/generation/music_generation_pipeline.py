"""
Music Generation Pipeline - System Development Task

This module integrates all System and System components into a unified
music generation pipeline that can analyze input audio and generate
new music with preserved melodic identity and target style.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# Import existing components
from .melody_preservation import MelodyPreservationEngine
from .instrumentation_engine import InstrumentationEngine, StyleType
from .style_transformation_engine import StyleTransformationEngine
from .rhythm_adjustment_engine import RhythmAdjustmentEngine, RhythmicStyle
from .key_modulation_engine import KeyModulationEngine, ModulationType

# Configure logging for the module
logger = logging.getLogger(__name__)


class PipelineMode(Enum):
    """Generation pipeline modes"""
    ANALYSIS_ONLY = "analysis_only"
    STYLE_TRANSFER = "style_transfer"
    COMPLETE_REMAKE = "complete_remake"
    ENHANCEMENT = "enhancement"


@dataclass
class GenerationRequest:
    """Request for music generation"""
    input_audio: np.ndarray
    target_style: StyleType = StyleType.NEUTRAL
    preserve_melody: bool = True
    preserve_rhythm: bool = True
    tempo_adjustment: Optional[float] = None  # Factor or BPM
    key_adjustment: Optional[int] = None  # Semitones
    target_instruments: Optional[List[str]] = None
    quality_threshold: float = 0.7


@dataclass
class GenerationResult:
    """Result of music generation pipeline"""
    # Input analysis
    original_melody_dna: Dict[str, Any]
    analysis_metadata: Dict[str, Any]
    
    # Generation outputs
    generated_audio: Optional[np.ndarray] = None
    transformed_melody_dna: Optional[Dict[str, Any]] = None
    
    # Applied transformations
    style_transformation: Optional[Dict[str, Any]] = None
    instrumentation_result: Optional[Dict[str, Any]] = None
    rhythm_adjustments: Optional[Dict[str, Any]] = None
    key_modulation: Optional[Dict[str, Any]] = None
    
    # Quality assessment
    preservation_scores: Dict[str, float] = None
    quality_scores: Dict[str, float] = None
    generation_metadata: Dict[str, Any] = None
    
    # Status
    success: bool = False
    warnings: List[str] = None
    errors: List[str] = None


class MusicGenerationPipeline:
    """
    Unified music generation pipeline
    
    Integrates all System and System components:
    - Melody preservation and analysis
    - Style transformation
    - Instrumentation suggestions
    - Rhythm and tempo adjustment
    - Key modulation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize music generation pipeline
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Initialize all components
        logger.info("Initializing music generation pipeline components...")
        
        self.melody_engine = MelodyPreservationEngine(self.config.get('melody', {}))
        self.instrumentation_engine = InstrumentationEngine(self.config.get('instrumentation', {}))
        self.style_engine = StyleTransformationEngine(self.config.get('style', {}))
        self.rhythm_engine = RhythmAdjustmentEngine(self.config.get('rhythm', {}))
        self.key_engine = KeyModulationEngine(self.config.get('key', {}))
        
        # Pipeline parameters
        self.min_audio_duration = self.config.get('min_audio_duration', 3.0)
        self.max_audio_duration = self.config.get('max_audio_duration', 60.0)
        self.default_sample_rate = self.config.get('sample_rate', 22050)
        
        logger.info("MusicGenerationPipeline initialized successfully")
    
    def generate_music(self, request: GenerationRequest, mode: PipelineMode = PipelineMode.STYLE_TRANSFER) -> GenerationResult:
        """
        Main entry point for music generation
        
        Args:
            request: Generation request with input audio and parameters
            mode: Pipeline mode determining processing level
            
        Returns:
            GenerationResult with all outputs and metadata
        """
        logger.info("Starting music generation pipeline: mode=%s", mode.value)
        
        # Initialize result
        result = GenerationResult(
            original_melody_dna={},
            analysis_metadata={},
            warnings=[],
            errors=[]
        )
        
        try:
            # Step 1: Validate and preprocess input
            if not self._validate_input(request, result):
                return result
            
            # Step 2: Extract melody DNA
            result.original_melody_dna = self._extract_melody_dna(request.input_audio, result)
            if not result.original_melody_dna:
                return result
            
            # Step 3: Analyze input music
            result.analysis_metadata = self._analyze_input_music(result.original_melody_dna, result)
            
            # Step 4: Generate based on mode
            if mode == PipelineMode.ANALYSIS_ONLY:
                result.success = True
                return result
            
            elif mode == PipelineMode.STYLE_TRANSFER:
                return self._perform_style_transfer(request, result)
            
            elif mode == PipelineMode.COMPLETE_REMAKE:
                return self._perform_complete_remake(request, result)
            
            elif mode == PipelineMode.ENHANCEMENT:
                return self._perform_enhancement(request, result)
            
            else:
                result.errors.append(f"Unsupported pipeline mode: {mode.value}")
                return result
                
        except Exception as e:
            logger.error("Pipeline execution failed: %s", e)
            result.errors.append(f"Pipeline execution failed: {str(e)}")
            return result
    
    def _validate_input(self, request: GenerationRequest, result: GenerationResult) -> bool:
        """Validate input audio and request parameters"""
        
        # Check audio length
        if len(request.input_audio) == 0:
            result.errors.append("Input audio is empty")
            return False
        
        duration = len(request.input_audio) / self.default_sample_rate
        
        if duration < self.min_audio_duration:
            result.warnings.append(f"Audio duration ({duration:.1f}s) below recommended minimum ({self.min_audio_duration}s)")
        
        if duration > self.max_audio_duration:
            result.warnings.append(f"Audio duration ({duration:.1f}s) above recommended maximum ({self.max_audio_duration}s)")
            # Truncate if too long
            max_samples = int(self.max_audio_duration * self.default_sample_rate)
            request.input_audio = request.input_audio[:max_samples]
        
        # Check audio quality
        audio_rms = np.sqrt(np.mean(request.input_audio ** 2))
        if audio_rms < 0.001:
            result.warnings.append("Input audio has very low volume")
        elif audio_rms > 0.9:
            result.warnings.append("Input audio may be clipped")
        
        # Validate parameters
        if request.tempo_adjustment is not None:
            if request.tempo_adjustment <= 0:
                result.errors.append("Tempo adjustment must be positive")
                return False
        
        if request.key_adjustment is not None:
            if abs(request.key_adjustment) > 12:
                result.warnings.append("Large key adjustment may affect recognition")
        
        return True
    
    def _extract_melody_dna(self, audio: np.ndarray, result: GenerationResult) -> Dict[str, Any]:
        """Extract melody DNA from input audio"""
        
        try:
            logger.info("Extracting melody DNA...")
            melody_dna = self.melody_engine.extract_melody_dna(audio)
            
            # Validate DNA quality
            if not melody_dna:
                result.errors.append("Failed to extract melody DNA")
                return {}
            
            # Check for sufficient melodic content
            f0_track = melody_dna.get('f0_track', np.array([]))
            if len(f0_track) < 10:
                result.warnings.append("Very short melody detected")
            
            # Log extraction statistics
            stats = melody_dna.get('melodic_stats', {})
            logger.info("Melody DNA extracted: %d frames, range: %.1f semitones",
                       len(f0_track), stats.get('pitch_range', 0))
            
            return melody_dna
            
        except Exception as e:
            logger.error("Melody DNA extraction failed: %s", e)
            result.errors.append(f"Melody DNA extraction failed: {str(e)}")
            return {}
    
    def _analyze_input_music(self, melody_dna: Dict[str, Any], result: GenerationResult) -> Dict[str, Any]:
        """Analyze input music characteristics"""
        
        analysis = {
            'detected_characteristics': {},
            'complexity_scores': {},
            'suitability_scores': {},
            'recommendations': []
        }
        
        try:
            # Basic characteristics
            stats = melody_dna.get('melodic_stats', {})
            rhythm = melody_dna.get('rhythmic_skeleton', {})
            
            analysis['detected_characteristics'] = {
                'pitch_range': stats.get('pitch_range', 0),
                'pitch_mean': stats.get('pitch_mean', 0),
                'tempo': rhythm.get('tempo', 120),
                'note_density': rhythm.get('note_density', 1.0),
                'num_phrases': len(melody_dna.get('phrase_boundaries', [])) - 1
            }
            
            # Complexity analysis
            complexity = self._analyze_complexity(melody_dna)
            analysis['complexity_scores'] = complexity
            
            # Suitability for different transformations
            suitability = self._analyze_transformation_suitability(melody_dna)
            analysis['suitability_scores'] = suitability
            
            # Generate recommendations
            analysis['recommendations'] = self._generate_recommendations(complexity, suitability)
            
            logger.info("Music analysis completed: complexity=%.3f, suitability scores available",
                       complexity.get('overall', 0.5))
            
        except Exception as e:
            logger.warning("Music analysis partially failed: %s", e)
            result.warnings.append(f"Analysis warning: {str(e)}")
        
        return analysis
    
    def _analyze_complexity(self, melody_dna: Dict[str, Any]) -> Dict[str, float]:
        """Analyze melodic complexity"""
        
        complexity = {}
        
        # Interval complexity
        intervals = melody_dna.get('interval_sequence', np.array([]))
        if len(intervals) > 0:
            interval_variety = len(np.unique(np.round(intervals))) / len(intervals)
            large_intervals = np.sum(np.abs(intervals) > 7) / len(intervals)
            complexity['interval_complexity'] = (interval_variety + large_intervals) / 2
        else:
            complexity['interval_complexity'] = 0.0
        
        # Rhythmic complexity
        rhythm = melody_dna.get('rhythmic_skeleton', {})
        note_density = rhythm.get('note_density', 1.0)
        complexity['rhythmic_complexity'] = min(1.0, note_density / 5.0)
        
        # Phrase structure complexity
        phrases = melody_dna.get('phrase_boundaries', [])
        if len(phrases) > 2:
            phrase_lengths = np.diff(phrases)
            phrase_variety = np.std(phrase_lengths) / np.mean(phrase_lengths)
            complexity['phrase_complexity'] = min(1.0, phrase_variety)
        else:
            complexity['phrase_complexity'] = 0.0
        
        # Overall complexity
        complexity['overall'] = np.mean(list(complexity.values()))
        
        return complexity
    
    def _analyze_transformation_suitability(self, melody_dna: Dict[str, Any]) -> Dict[str, float]:
        """Analyze suitability for different transformations"""
        
        suitability = {}
        
        stats = melody_dna.get('melodic_stats', {})
        intervals = melody_dna.get('interval_sequence', np.array([]))
        
        # Style transfer suitability
        pitch_range = stats.get('pitch_range', 0)
        if pitch_range > 0:
            # Good range for style transfer
            suitability['style_transfer'] = min(1.0, pitch_range / 24.0)
        else:
            suitability['style_transfer'] = 0.3
        
        # Rhythm adjustment suitability
        rhythm = melody_dna.get('rhythmic_skeleton', {})
        tempo = rhythm.get('tempo', 120)
        if 60 <= tempo <= 200:  # Reasonable tempo range
            suitability['rhythm_adjustment'] = 0.9
        else:
            suitability['rhythm_adjustment'] = 0.6
        
        # Key modulation suitability
        if len(intervals) > 5:  # Sufficient melodic content
            suitability['key_modulation'] = 0.9
        else:
            suitability['key_modulation'] = 0.5
        
        # Instrumentation suitability
        char_notes = melody_dna.get('characteristic_notes', [])
        if len(char_notes) > 3:  # Clear melodic features
            suitability['instrumentation'] = 0.8
        else:
            suitability['instrumentation'] = 0.6
        
        return suitability
    
    def _generate_recommendations(self, complexity: Dict[str, float], suitability: Dict[str, float]) -> List[str]:
        """Generate recommendations based on analysis"""
        
        recommendations = []
        
        overall_complexity = complexity.get('overall', 0.5)
        
        if overall_complexity < 0.3:
            recommendations.append("Simple melody detected - consider adding ornamentations")
        elif overall_complexity > 0.7:
            recommendations.append("Complex melody detected - style transfer may need careful tuning")
        
        # Transformation-specific recommendations
        for transform, score in suitability.items():
            if score < 0.5:
                recommendations.append(f"Low suitability for {transform.replace('_', ' ')} (score: {score:.2f})")
            elif score > 0.8:
                recommendations.append(f"Excellent for {transform.replace('_', ' ')} (score: {score:.2f})")
        
        return recommendations
    
    def _perform_style_transfer(self, request: GenerationRequest, result: GenerationResult) -> GenerationResult:
        """Perform style transfer generation"""
        
        logger.info("Performing style transfer to %s", request.target_style.value)
        
        try:
            # Step 1: Style transformation
            style_result = self.style_engine.transform_style(
                result.original_melody_dna, 
                request.target_style
            )
            result.style_transformation = {
                'target_style': request.target_style.value,
                'confidence': style_result.confidence_score,
                'preservation_scores': style_result.preservation_scores,
                'applied_modifications': style_result.applied_modifications,
                'warnings': style_result.warnings
            }
            result.transformed_melody_dna = style_result.transformed_melody_dna
            
            # Step 2: Instrumentation
            if request.target_instruments:
                # Use requested instruments
                instrumentation_result = {
                    'primary_instrument': request.target_instruments[0],
                    'secondary_instruments': request.target_instruments[1:],
                    'arrangement_map': {},
                    'confidence_score': 0.8,  # Assumed since user specified
                    'reasoning': ['User-specified instruments']
                }
            else:
                # Auto-suggest instruments
                inst_result = self.instrumentation_engine.suggest_instrumentation(
                    result.transformed_melody_dna, 
                    request.target_style
                )
                instrumentation_result = {
                    'primary_instrument': inst_result.primary_instrument,
                    'secondary_instruments': inst_result.secondary_instruments,
                    'arrangement_map': inst_result.arrangement_map,
                    'confidence_score': inst_result.confidence_score,
                    'reasoning': inst_result.reasoning
                }
            
            result.instrumentation_result = instrumentation_result
            
            # Step 3: Rhythm adjustment (if requested)
            if request.tempo_adjustment:
                rhythm_skeleton = result.transformed_melody_dna.get('rhythmic_skeleton', {})
                
                if request.tempo_adjustment > 10:  # Assume it's BPM
                    from .rhythm_adjustment_engine import TempoAdjustmentType
                    rhythm_result = self.rhythm_engine.adjust_tempo(
                        rhythm_skeleton, 
                        request.tempo_adjustment,
                        TempoAdjustmentType.ABSOLUTE
                    )
                else:  # Assume it's a factor
                    from .rhythm_adjustment_engine import TempoAdjustmentType
                    rhythm_result = self.rhythm_engine.adjust_tempo(
                        rhythm_skeleton,
                        request.tempo_adjustment,
                        TempoAdjustmentType.PROPORTIONAL
                    )
                
                result.rhythm_adjustments = {
                    'original_tempo': rhythm_skeleton.get('tempo', 120),
                    'new_tempo': rhythm_result.adjusted_rhythmic_skeleton.get('tempo', 120),
                    'preservation_score': rhythm_result.preservation_score,
                    'applied_adjustments': rhythm_result.applied_adjustments
                }
                
                # Update transformed DNA with new rhythm
                result.transformed_melody_dna['rhythmic_skeleton'] = rhythm_result.adjusted_rhythmic_skeleton
            
            # Step 4: Key modulation (if requested)
            if request.key_adjustment:
                key_result = self.key_engine.transpose_melody(
                    result.transformed_melody_dna,
                    request.key_adjustment
                )
                
                result.key_modulation = {
                    'original_key': key_result.original_key,
                    'target_key': key_result.target_key,
                    'semitone_shift': key_result.semitone_shift,
                    'preservation_score': key_result.interval_preservation_score,
                    'harmonic_coherence': key_result.harmonic_coherence_score
                }
                
                # Update transformed DNA with new key
                result.transformed_melody_dna = key_result.modulated_melody_dna
            
            # Step 5: Quality assessment
            result.preservation_scores = self._assess_preservation(
                result.original_melody_dna, 
                result.transformed_melody_dna,
                request
            )
            
            result.quality_scores = self._assess_generation_quality(
                result.transformed_melody_dna,
                result.style_transformation,
                result.instrumentation_result
            )
            
            # Step 6: Final validation
            if result.preservation_scores['overall'] < request.quality_threshold:
                result.warnings.append(
                    f"Generation quality ({result.preservation_scores['overall']:.3f}) "
                    f"below threshold ({request.quality_threshold})"
                )
            
            result.success = True
            logger.info("Style transfer completed successfully")
            
        except Exception as e:
            logger.error("Style transfer failed: %s", e)
            result.errors.append(f"Style transfer failed: {str(e)}")
        
        return result
    
    def _perform_complete_remake(self, request: GenerationRequest, result: GenerationResult) -> GenerationResult:
        """Perform complete remake with more dramatic changes"""
        
        logger.info("Performing complete remake")
        
        try:
            # Start with style transfer
            result = self._perform_style_transfer(request, result)
            
            if not result.success:
                return result
            
            # Add more dramatic transformations
            
            # Apply rhythmic style change
            rhythm_skeleton = result.transformed_melody_dna.get('rhythmic_skeleton', {})
            
            # Choose rhythmic style based on target style
            if request.target_style == StyleType.CHINESE_TRADITIONAL:
                rhythmic_style = RhythmicStyle.STRAIGHT
            elif request.target_style == StyleType.WESTERN_CLASSICAL:
                rhythmic_style = RhythmicStyle.STRAIGHT
            elif request.target_style == StyleType.MODERN_POP:
                rhythmic_style = RhythmicStyle.SYNCOPATED
            else:
                rhythmic_style = RhythmicStyle.STRAIGHT
            
            rhythm_style_result = self.rhythm_engine.apply_rhythmic_style(
                rhythm_skeleton, rhythmic_style
            )
            
            # Update rhythm adjustments
            if result.rhythm_adjustments is None:
                result.rhythm_adjustments = {}
            
            result.rhythm_adjustments.update({
                'rhythmic_style': rhythmic_style.value,
                'style_preservation': rhythm_style_result.preservation_score,
                'groove_characteristics': rhythm_style_result.groove_characteristics
            })
            
            result.transformed_melody_dna['rhythmic_skeleton'] = rhythm_style_result.adjusted_rhythmic_skeleton
            
            # Suggest emotional key modulation if not already specified
            if request.key_adjustment is None:
                emotional_suggestions = self.key_engine.suggest_emotional_modulation(
                    result.transformed_melody_dna, 'brighter'
                )
                
                if emotional_suggestions:
                    best_suggestion = emotional_suggestions[0]
                    result.key_modulation = {
                        'original_key': best_suggestion.original_key,
                        'target_key': best_suggestion.target_key,
                        'semitone_shift': best_suggestion.semitone_shift,
                        'emotional_target': 'brighter',
                        'emotional_fit': getattr(best_suggestion, 'emotional_fit_score', 0.5)
                    }
                    result.transformed_melody_dna = best_suggestion.modulated_melody_dna
            
            # Re-assess quality after complete remake
            result.preservation_scores = self._assess_preservation(
                result.original_melody_dna,
                result.transformed_melody_dna, 
                request
            )
            
            result.quality_scores = self._assess_generation_quality(
                result.transformed_melody_dna,
                result.style_transformation,
                result.instrumentation_result
            )
            
            result.generation_metadata = {
                'mode': 'complete_remake',
                'transformations_applied': len([x for x in [
                    result.style_transformation,
                    result.rhythm_adjustments,
                    result.key_modulation
                ] if x is not None])
            }
            
            logger.info("Complete remake completed successfully")
            
        except Exception as e:
            logger.error("Complete remake failed: %s", e)
            result.errors.append(f"Complete remake failed: {str(e)}")
        
        return result
    
    def _perform_enhancement(self, request: GenerationRequest, result: GenerationResult) -> GenerationResult:
        """Perform subtle enhancement without major style changes"""
        
        logger.info("Performing enhancement")
        
        try:
            # Enhancement focuses on quality improvement rather than style change
            result.transformed_melody_dna = result.original_melody_dna.copy()
            
            # Subtle rhythm enhancement
            rhythm_skeleton = result.original_melody_dna.get('rhythmic_skeleton', {})
            
            # Apply quantization for cleaner timing
            rhythm_result = self.rhythm_engine.quantize_rhythm(
                rhythm_skeleton, grid_resolution=0.25
            )
            
            result.rhythm_adjustments = {
                'enhancement_type': 'quantization',
                'preservation_score': rhythm_result.preservation_score,
                'applied_adjustments': rhythm_result.applied_adjustments
            }
            
            result.transformed_melody_dna['rhythmic_skeleton'] = rhythm_result.adjusted_rhythmic_skeleton
            
            # Subtle tempo adjustment if requested
            if request.tempo_adjustment and abs(request.tempo_adjustment - 1.0) < 0.2:
                from .rhythm_adjustment_engine import TempoAdjustmentType
                tempo_result = self.rhythm_engine.adjust_tempo(
                    result.transformed_melody_dna['rhythmic_skeleton'],
                    request.tempo_adjustment,
                    TempoAdjustmentType.PROPORTIONAL
                )
                result.transformed_melody_dna['rhythmic_skeleton'] = tempo_result.adjusted_rhythmic_skeleton
                result.rhythm_adjustments['tempo_adjustment'] = request.tempo_adjustment
            
            # Instrumentation suggestions (keep original style)
            original_style = StyleType.NEUTRAL  # Default for enhancement
            inst_result = self.instrumentation_engine.suggest_instrumentation(
                result.transformed_melody_dna, original_style
            )
            
            result.instrumentation_result = {
                'primary_instrument': inst_result.primary_instrument,
                'secondary_instruments': inst_result.secondary_instruments,
                'enhancement_focus': True,
                'confidence_score': inst_result.confidence_score
            }
            
            # Quality assessment
            result.preservation_scores = self._assess_preservation(
                result.original_melody_dna,
                result.transformed_melody_dna,
                request
            )
            
            # For enhancement, preservation should be very high
            result.quality_scores = {
                'preservation_quality': result.preservation_scores['overall'],
                'enhancement_benefit': 0.8,  # Assumed benefit from enhancement
                'overall_quality': (result.preservation_scores['overall'] + 0.8) / 2
            }
            
            result.generation_metadata = {
                'mode': 'enhancement',
                'enhancement_type': 'rhythm_quantization'
            }
            
            result.success = True
            logger.info("Enhancement completed successfully")
            
        except Exception as e:
            logger.error("Enhancement failed: %s", e)
            result.errors.append(f"Enhancement failed: {str(e)}")
        
        return result
    
    def _assess_preservation(self, original_dna: Dict[str, Any], 
                           transformed_dna: Dict[str, Any],
                           request: GenerationRequest) -> Dict[str, float]:
        """Assess how well the transformation preserves the original melody"""
        
        preservation = {}
        
        try:
            # Use melody preservation engine
            melody_similarity = self.melody_engine.compute_melody_similarity(
                original_dna, transformed_dna
            )
            preservation['melody_similarity'] = float(melody_similarity)
            
            # Rhythm preservation
            orig_rhythm = original_dna.get('rhythmic_skeleton', {})
            trans_rhythm = transformed_dna.get('rhythmic_skeleton', {})
            rhythm_similarity = self.melody_engine._compute_rhythm_similarity(orig_rhythm, trans_rhythm)
            preservation['rhythm_similarity'] = float(rhythm_similarity)
            
            # Phrase structure preservation
            orig_phrases = original_dna.get('phrase_boundaries', [])
            trans_phrases = transformed_dna.get('phrase_boundaries', [])
            phrase_similarity = self.melody_engine._compute_phrase_similarity(orig_phrases, trans_phrases)
            preservation['phrase_similarity'] = float(phrase_similarity)
            
            # Overall preservation score
            weights = {
                'melody_similarity': 0.5 if request.preserve_melody else 0.2,
                'rhythm_similarity': 0.3 if request.preserve_rhythm else 0.1,
                'phrase_similarity': 0.2
            }
            
            preservation['overall'] = sum(
                preservation[key] * weight for key, weight in weights.items()
            )
            
        except Exception as e:
            logger.warning("Preservation assessment failed: %s", e)
            preservation = {
                'melody_similarity': 0.5,
                'rhythm_similarity': 0.5,
                'phrase_similarity': 0.5,
                'overall': 0.5
            }
        
        return preservation
    
    def _assess_generation_quality(self, transformed_dna: Dict[str, Any],
                                 style_result: Optional[Dict[str, Any]],
                                 instrumentation_result: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """Assess overall quality of the generation"""
        
        quality = {}
        
        # Style transformation quality
        if style_result:
            quality['style_consistency'] = style_result.get('confidence', 0.5)
        else:
            quality['style_consistency'] = 1.0  # No style change requested
        
        # Instrumentation quality
        if instrumentation_result:
            quality['instrumentation_confidence'] = instrumentation_result.get('confidence_score', 0.5)
        else:
            quality['instrumentation_confidence'] = 1.0
        
        # Musical coherence (based on melody DNA)
        stats = transformed_dna.get('melodic_stats', {})
        char_notes = transformed_dna.get('characteristic_notes', [])
        
        # Simple heuristics for musical quality
        pitch_range = stats.get('pitch_range', 0)
        if 6 <= pitch_range <= 24:  # Reasonable range
            quality['pitch_coherence'] = 0.9
        elif pitch_range > 0:
            quality['pitch_coherence'] = 0.7
        else:
            quality['pitch_coherence'] = 0.3
        
        # Characteristic note quality
        if len(char_notes) >= 3:
            quality['melodic_interest'] = 0.8
        elif len(char_notes) >= 1:
            quality['melodic_interest'] = 0.6
        else:
            quality['melodic_interest'] = 0.4
        
        # Overall quality
        quality['overall'] = np.mean(list(quality.values()))
        
        return quality
    
    def get_pipeline_capabilities(self) -> Dict[str, Any]:
        """Get information about pipeline capabilities"""
        
        return {
            'supported_modes': [mode.value for mode in PipelineMode],
            'supported_styles': [style.value for style in StyleType],
            'audio_requirements': {
                'min_duration': self.min_audio_duration,
                'max_duration': self.max_audio_duration,
                'sample_rate': self.default_sample_rate
            },
            'transformation_types': [
                'style_transfer', 'instrumentation', 'rhythm_adjustment', 
                'key_modulation', 'tempo_change'
            ],
            'quality_metrics': [
                'melody_preservation', 'rhythm_preservation', 'style_consistency',
                'instrumentation_confidence', 'overall_quality'
            ]
        }
    
    def validate_request(self, request: GenerationRequest) -> Dict[str, Any]:
        """Validate generation request before processing"""
        
        validation = {
            'valid': True,
            'warnings': [],
            'errors': []
        }
        
        # Check audio
        if len(request.input_audio) == 0:
            validation['errors'].append("Input audio is empty")
            validation['valid'] = False
        
        # Check parameters
        if request.tempo_adjustment is not None:
            if request.tempo_adjustment <= 0:
                validation['errors'].append("Tempo adjustment must be positive")
                validation['valid'] = False
            elif request.tempo_adjustment < 0.5 or request.tempo_adjustment > 2.0:
                validation['warnings'].append("Large tempo adjustment may affect quality")
        
        if request.key_adjustment is not None:
            if abs(request.key_adjustment) > 12:
                validation['warnings'].append("Large key adjustment may affect recognition")
        
        if not (0.0 <= request.quality_threshold <= 1.0):
            validation['errors'].append("Quality threshold must be between 0 and 1")
            validation['valid'] = False
        
        return validation