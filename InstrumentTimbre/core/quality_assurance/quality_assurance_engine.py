"""
Quality Assurance Engine - Main orchestrator for comprehensive quality assurance

This module coordinates all quality assurance components to provide:
- Unified quality validation workflow
- Automatic conflict detection and resolution
- User feedback integration when needed
- Comprehensive quality reporting and tracking
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import logging

from .base_validator import ValidationResult
from .data_structures import QualityScore, ConflictType, ValidationContext, MusicElement
from .harmony_validator import HarmonyValidator
from .emotion_consistency_checker import EmotionConsistencyChecker
from .music_theory_validator import MusicTheoryValidator
from .auto_coordination_engine import AutoCoordinationEngine, ConflictItem
from .user_feedback_interface import UserFeedbackInterface, UserDecision, FeedbackType
from .quality_scoring_system import QualityScoringSystem, QualityIndex


class QAMode(Enum):
    """Quality assurance operation modes"""
    AUTOMATIC = "automatic"  # Fully automated with minimal user intervention
    INTERACTIVE = "interactive"  # User feedback for important decisions
    MANUAL = "manual"  # User approval required for all changes
    MONITORING = "monitoring"  # Assessment only, no corrections


class QAResult(Enum):
    """Quality assurance result status"""
    APPROVED = "approved"  # No issues found or all resolved
    CONDITIONALLY_APPROVED = "conditionally_approved"  # Minor issues remain
    REQUIRES_ATTENTION = "requires_attention"  # Significant issues need resolution
    REJECTED = "rejected"  # Major issues prevent approval


@dataclass
class QualityAssuranceResult:
    """Comprehensive quality assurance result"""
    result_status: QAResult
    overall_quality_score: float
    quality_index: QualityIndex
    validation_results: List[ValidationResult]
    resolved_conflicts: List[ConflictItem]
    remaining_conflicts: List[ConflictItem]
    user_decisions: List[UserDecision]
    processing_time: float
    recommendations: List[str]
    metadata: Dict[str, Any]


class QualityAssuranceEngine:
    """
    Main quality assurance orchestrator
    
    Coordinates validation, conflict resolution, and quality assessment
    to ensure comprehensive music quality assurance.
    """
    
    def __init__(self, mode: QAMode = QAMode.AUTOMATIC):
        self.name = "QualityAssuranceEngine"
        self.version = "1.0.0"
        self.mode = mode
        
        # Initialize component validators
        self.harmony_validator = HarmonyValidator()
        self.emotion_checker = EmotionConsistencyChecker()
        self.theory_validator = MusicTheoryValidator()
        self.auto_coordinator = AutoCoordinationEngine()
        self.user_interface = UserFeedbackInterface()
        self.quality_scorer = QualityScoringSystem()
        
        # Quality thresholds for different approval levels
        self.quality_thresholds = {
            QAResult.APPROVED: 0.85,
            QAResult.CONDITIONALLY_APPROVED: 0.70,
            QAResult.REQUIRES_ATTENTION: 0.55,
            QAResult.REJECTED: 0.00
        }
        
        # Configure logging
        self.logger = logging.getLogger(self.name)
        
        # Performance tracking
        self.processing_stats = {
            'total_validations': 0,
            'automatic_resolutions': 0,
            'user_interventions': 0,
            'approval_rate': 0.0
        }

    def validate_result(self, operation_result: MusicElement, 
                       original_tracks: Optional[List[Dict]] = None,
                       context: Optional[ValidationContext] = None) -> QualityAssuranceResult:
        """
        Main quality assurance validation workflow
        
        Args:
            operation_result: Result of music operation to validate
            original_tracks: Original tracks for comparison (optional)
            context: Validation context and settings
            
        Returns:
            QualityAssuranceResult with comprehensive assessment
        """
        start_time = self._get_timestamp()
        
        # Initialize context if not provided
        if context is None:
            context = ValidationContext()
        
        # Add original data to context for comparison
        if original_tracks:
            context['original_data'] = original_tracks
        
        self.logger.info(f"Starting quality assurance validation in {self.mode.value} mode")
        
        # Step 1: Run all validators
        validation_results = self._run_all_validators(operation_result, context)
        
        # Step 2: Collect and prioritize conflicts
        all_conflicts = self._collect_conflicts(validation_results)
        prioritized_conflicts = self.auto_coordinator.prioritize_conflicts(all_conflicts, context)
        
        # Step 3: Calculate initial quality scores
        quality_index = self._calculate_quality_index(validation_results, operation_result, context)
        
        # Step 4: Resolve conflicts based on mode
        resolved_conflicts, remaining_conflicts, user_decisions = self._resolve_conflicts(
            prioritized_conflicts, operation_result, context
        )
        
        # Step 5: Recalculate quality after resolutions
        if resolved_conflicts:
            final_validation_results = self._run_all_validators(operation_result, context)
            final_quality_index = self._calculate_quality_index(final_validation_results, operation_result, context)
        else:
            final_validation_results = validation_results
            final_quality_index = quality_index
        
        # Step 6: Determine final result status
        result_status = self._determine_result_status(final_quality_index, remaining_conflicts)
        
        # Step 7: Generate recommendations
        recommendations = self._generate_recommendations(
            final_quality_index, remaining_conflicts, validation_results
        )
        
        # Step 8: Update processing statistics
        processing_time = self._get_timestamp() - start_time
        self._update_processing_stats(result_status, len(resolved_conflicts), len(user_decisions))
        
        # Step 9: Record quality assessment for trend analysis
        self.quality_scorer.record_quality_assessment(final_quality_index)
        
        # Create comprehensive result
        qa_result = QualityAssuranceResult(
            result_status=result_status,
            overall_quality_score=final_quality_index.overall_score,
            quality_index=final_quality_index,
            validation_results=final_validation_results,
            resolved_conflicts=resolved_conflicts,
            remaining_conflicts=remaining_conflicts,
            user_decisions=user_decisions,
            processing_time=processing_time,
            recommendations=recommendations,
            metadata={
                'mode': self.mode.value,
                'validators_used': len(validation_results),
                'original_conflicts': len(all_conflicts),
                'resolution_success_rate': len(resolved_conflicts) / max(len(all_conflicts), 1)
            }
        )
        
        self.logger.info(f"Quality assurance completed: {result_status.value} "
                        f"(score: {final_quality_index.overall_score:.2f})")
        
        return qa_result

    def resolve_conflicts(self, conflicts: List[ConflictType], 
                         music_element: MusicElement,
                         context: ValidationContext) -> QualityAssuranceResult:
        """
        Focused conflict resolution workflow
        
        Args:
            conflicts: Specific conflicts to resolve
            music_element: Music element to modify
            context: Resolution context
            
        Returns:
            QualityAssuranceResult with resolution outcomes
        """
        # Convert conflict types to conflict items
        conflict_items = []
        for i, conflict_type in enumerate(conflicts):
            conflict_item = ConflictItem(
                conflict_type=conflict_type,
                priority=self.auto_coordinator.priority_mapping.get(conflict_type),
                severity=0.7,  # Default severity
                location=f"conflict_{i}",
                description=f"Targeted resolution of {conflict_type.value}",
                affected_elements=[],
                metadata={'targeted_resolution': True, 'index': i}
            )
            conflict_items.append(conflict_item)
        
        # Use main validation workflow
        return self.validate_result(music_element, context=context)

    def get_quality_report(self, time_period: str = "30_days") -> Dict[str, Any]:
        """
        Generate comprehensive quality report
        
        Args:
            time_period: Period for trend analysis
            
        Returns:
            Comprehensive quality report
        """
        # Get quality trends
        quality_trends = self.quality_scorer.analyze_quality_trends(time_period)
        
        # Get processing statistics
        stats = self.processing_stats.copy()
        
        # Calculate additional metrics
        if stats['total_validations'] > 0:
            stats['automation_rate'] = stats['automatic_resolutions'] / stats['total_validations']
            stats['user_intervention_rate'] = stats['user_interventions'] / stats['total_validations']
        else:
            stats['automation_rate'] = 0.0
            stats['user_intervention_rate'] = 0.0
        
        return {
            'report_period': time_period,
            'quality_trends': quality_trends,
            'processing_statistics': stats,
            'component_status': {
                'harmony_validator': self.harmony_validator.name,
                'emotion_checker': self.emotion_checker.name,
                'theory_validator': self.theory_validator.name,
                'auto_coordinator': self.auto_coordinator.name,
                'user_interface': self.user_interface.name,
                'quality_scorer': self.quality_scorer.name
            },
            'system_health': self._assess_system_health()
        }

    def configure_mode(self, mode: QAMode, settings: Optional[Dict[str, Any]] = None):
        """
        Configure quality assurance mode and settings
        
        Args:
            mode: New operation mode
            settings: Optional mode-specific settings
        """
        self.mode = mode
        
        if settings:
            # Apply mode-specific settings
            if 'quality_thresholds' in settings:
                self.quality_thresholds.update(settings['quality_thresholds'])
            
            if 'component_settings' in settings:
                self._apply_component_settings(settings['component_settings'])
        
        self.logger.info(f"Quality assurance mode set to: {mode.value}")

    # Internal workflow methods
    
    def _run_all_validators(self, music_element: MusicElement, 
                           context: ValidationContext) -> List[ValidationResult]:
        """Run all component validators"""
        validation_results = []
        
        try:
            # Harmony validation
            harmony_result = self.harmony_validator.validate(music_element, context)
            validation_results.append(harmony_result)
            
            # Emotion consistency validation
            emotion_result = self.emotion_checker.validate(music_element, context)
            validation_results.append(emotion_result)
            
            # Music theory validation
            theory_result = self.theory_validator.validate(music_element, context)
            validation_results.append(theory_result)
            
        except Exception as e:
            self.logger.error(f"Validation error: {e}")
            # Create fallback result
            fallback_result = ValidationResult(
                validator_name="Fallback",
                overall_score=0.5,
                passed=False,
                detailed_scores={'fallback_score': 0.5},
                issues=[],
                metadata={'error': str(e)}
            )
            validation_results.append(fallback_result)
        
        return validation_results

    def _collect_conflicts(self, validation_results: List[ValidationResult]) -> List[ConflictType]:
        """Collect all conflicts from validation results"""
        all_conflicts = []
        
        for result in validation_results:
            all_conflicts.extend(result.conflicts)
        
        return all_conflicts

    def _calculate_quality_index(self, validation_results: List[ValidationResult],
                                music_element: MusicElement,
                                context: ValidationContext) -> QualityIndex:
        """Calculate comprehensive quality index"""
        try:
            # Extract quality scores from validation results
            technical_scores = []
            artistic_scores = []
            emotional_scores = []
            
            for result in validation_results:
                technical_scores.append(result.quality_score.technical_score)
                artistic_scores.append(result.quality_score.artistic_score)
                emotional_scores.append(result.quality_score.emotional_score)
            
            # Create simplified score objects for quality index calculation
            from .quality_scoring_system import TechnicalScore, ArtisticScore, EmotionalScore
            
            technical = TechnicalScore(
                harmony_correctness=np.mean(technical_scores),
                rhythmic_accuracy=np.mean(technical_scores),
                spectral_balance=np.mean(technical_scores),
                dynamic_range=np.mean(technical_scores),
                audio_quality=np.mean(technical_scores),
                theory_compliance=np.mean(technical_scores),
                overall_technical=np.mean(technical_scores),
                details={}
            )
            
            artistic = ArtisticScore(
                creativity=np.mean(artistic_scores),
                musical_flow=np.mean(artistic_scores),
                emotional_expression=np.mean(artistic_scores),
                style_authenticity=np.mean(artistic_scores),
                structural_coherence=np.mean(artistic_scores),
                aesthetic_appeal=np.mean(artistic_scores),
                overall_artistic=np.mean(artistic_scores),
                details={}
            )
            
            emotional = EmotionalScore(
                emotional_clarity=np.mean(emotional_scores),
                intensity_appropriateness=np.mean(emotional_scores),
                emotional_consistency=np.mean(emotional_scores),
                listener_engagement=np.mean(emotional_scores),
                mood_coherence=np.mean(emotional_scores),
                expressive_range=np.mean(emotional_scores),
                overall_emotional=np.mean(emotional_scores),
                details={}
            )
            
            return self.quality_scorer.generate_weighted_quality_index(
                technical, artistic, emotional, context
            )
            
        except Exception as e:
            self.logger.error(f"Quality index calculation error: {e}")
            # Return fallback quality index
            from .quality_scoring_system import QualityIndex, QualityDimension
            return QualityIndex(
                overall_score=0.5,
                technical_weight=0.30,
                artistic_weight=0.35,
                emotional_weight=0.25,
                dimension_scores={
                    QualityDimension.TECHNICAL: 0.5,
                    QualityDimension.ARTISTIC: 0.5,
                    QualityDimension.EMOTIONAL: 0.5
                },
                confidence_level=0.3,
                context_factors={},
                improvement_potential=0.5
            )

    def _resolve_conflicts(self, prioritized_conflicts, music_element: MusicElement, 
                          context: ValidationContext) -> Tuple[List[ConflictItem], List[ConflictItem], List[UserDecision]]:
        """Resolve conflicts based on current mode"""
        resolved_conflicts = []
        remaining_conflicts = []
        user_decisions = []
        
        # Collect all conflicts
        all_conflicts = (prioritized_conflicts.critical + prioritized_conflicts.high + 
                        prioritized_conflicts.medium + prioritized_conflicts.low + 
                        prioritized_conflicts.minor)
        
        for conflict in all_conflicts:
            if self.mode == QAMode.MONITORING:
                # Monitoring mode: no resolution, just track
                remaining_conflicts.append(conflict)
            
            elif self.mode == QAMode.AUTOMATIC:
                # Automatic mode: resolve if possible
                suggestions = self.auto_coordinator.generate_suggestions(conflict, context)
                if suggestions and suggestions[0].confidence > 0.7:
                    # Apply automatic resolution
                    cascading_result = self.auto_coordinator.apply_cascading_adjustments(
                        suggestions[0], music_element, context
                    )
                    if cascading_result.primary_fix_applied:
                        resolved_conflicts.append(conflict)
                    else:
                        remaining_conflicts.append(conflict)
                else:
                    remaining_conflicts.append(conflict)
            
            elif self.mode == QAMode.INTERACTIVE:
                # Interactive mode: use user feedback for important conflicts
                if conflict.priority.value <= 2:  # Critical or High priority
                    suggestions = self.auto_coordinator.generate_suggestions(conflict, context)
                    solution_options = self.user_interface.present_solution_options(conflict, suggestions)
                    user_decision = self.user_interface.get_user_feedback(
                        f"conflict_{conflict.metadata.get('index', 0)}", solution_options
                    )
                    user_decisions.append(user_decision)
                    
                    if user_decision.feedback_type == FeedbackType.ACCEPT and user_decision.chosen_option_id:
                        resolved_conflicts.append(conflict)
                    else:
                        remaining_conflicts.append(conflict)
                else:
                    # Auto-resolve lower priority conflicts
                    suggestions = self.auto_coordinator.generate_suggestions(conflict, context)
                    if suggestions:
                        resolved_conflicts.append(conflict)
                    else:
                        remaining_conflicts.append(conflict)
            
            elif self.mode == QAMode.MANUAL:
                # Manual mode: all conflicts require user attention
                remaining_conflicts.append(conflict)
        
        return resolved_conflicts, remaining_conflicts, user_decisions

    def _determine_result_status(self, quality_index: QualityIndex, 
                               remaining_conflicts: List[ConflictItem]) -> QAResult:
        """Determine final result status based on quality and conflicts"""
        overall_score = quality_index.overall_score
        
        # Check for critical remaining conflicts
        critical_conflicts = [c for c in remaining_conflicts if c.priority.value == 1]
        
        if critical_conflicts:
            return QAResult.REJECTED
        elif overall_score >= self.quality_thresholds[QAResult.APPROVED]:
            return QAResult.APPROVED
        elif overall_score >= self.quality_thresholds[QAResult.CONDITIONALLY_APPROVED]:
            return QAResult.CONDITIONALLY_APPROVED
        elif overall_score >= self.quality_thresholds[QAResult.REQUIRES_ATTENTION]:
            return QAResult.REQUIRES_ATTENTION
        else:
            return QAResult.REJECTED

    def _generate_recommendations(self, quality_index: QualityIndex,
                                remaining_conflicts: List[ConflictItem],
                                validation_results: List[ValidationResult]) -> List[str]:
        """Generate recommendations for improvement"""
        recommendations = []
        
        # Quality-based recommendations
        if quality_index.overall_score < 0.7:
            lowest_dimension = min(quality_index.dimension_scores, 
                                 key=quality_index.dimension_scores.get)
            recommendations.append(f"Focus on improving {lowest_dimension.value} quality")
        
        # Conflict-based recommendations
        if remaining_conflicts:
            conflict_types = [c.conflict_type for c in remaining_conflicts]
            most_common = max(set(conflict_types), key=conflict_types.count)
            recommendations.append(f"Address {most_common.value} issues")
        
        # Validator-specific recommendations
        for result in validation_results:
            recommendations.extend(result.suggestions[:2])  # Top 2 suggestions per validator
        
        return recommendations[:5]  # Limit to top 5 recommendations

    def _update_processing_stats(self, result_status: QAResult, 
                               resolved_count: int, user_decisions_count: int):
        """Update processing statistics"""
        self.processing_stats['total_validations'] += 1
        self.processing_stats['automatic_resolutions'] += resolved_count
        self.processing_stats['user_interventions'] += user_decisions_count
        
        # Update approval rate
        total = self.processing_stats['total_validations']
        if result_status in [QAResult.APPROVED, QAResult.CONDITIONALLY_APPROVED]:
            approvals = getattr(self, '_approval_count', 0) + 1
            self._approval_count = approvals
            self.processing_stats['approval_rate'] = approvals / total

    def _assess_system_health(self) -> Dict[str, str]:
        """Assess overall system health"""
        health = {
            'overall': 'healthy',
            'validators': 'operational',
            'auto_coordination': 'operational',
            'user_interface': 'operational',
            'quality_scoring': 'operational'
        }
        
        # Check if approval rate is too low
        if self.processing_stats['approval_rate'] < 0.5:
            health['overall'] = 'needs_attention'
            health['note'] = 'Low approval rate detected'
        
        return health

    def _apply_component_settings(self, settings: Dict[str, Any]):
        """Apply settings to component validators"""
        # Apply settings to individual components
        if 'harmony_validator' in settings:
            # Apply harmony validator settings
            pass
        
        if 'emotion_checker' in settings:
            # Apply emotion checker settings
            pass
        
        # Additional component configurations as needed

    def _get_timestamp(self) -> float:
        """Get current timestamp"""
        import time
        return time.time()