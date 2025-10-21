"""
Quality Scoring System - Comprehensive music quality assessment

This module provides multi-dimensional quality assessment including:
- Technical, artistic, and emotional scoring metrics
- Weighted quality index with context-aware assessment
- Benchmark comparison against reference standards
- Quality trend analysis and improvement tracking
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime

from .data_structures import QualityScore, MusicElement, ValidationContext


class QualityDimension(Enum):
    """Quality assessment dimensions"""
    TECHNICAL = "technical"
    ARTISTIC = "artistic"
    EMOTIONAL = "emotional"
    USER_EXPERIENCE = "user_experience"
    CULTURAL = "cultural"


class BenchmarkType(Enum):
    """Types of quality benchmarks"""
    PROFESSIONAL_STANDARD = "professional"
    GENRE_SPECIFIC = "genre_specific"
    USER_PREFERENCE = "user_preference"
    HISTORICAL_BASELINE = "historical"
    ADAPTIVE_TARGET = "adaptive"


@dataclass
class TechnicalScore:
    """Technical quality assessment result"""
    harmony_correctness: float
    rhythmic_accuracy: float
    spectral_balance: float
    dynamic_range: float
    audio_quality: float
    theory_compliance: float
    overall_technical: float
    details: Dict[str, float]


@dataclass
class ArtisticScore:
    """Artistic quality assessment result"""
    creativity: float
    musical_flow: float
    emotional_expression: float
    style_authenticity: float
    structural_coherence: float
    aesthetic_appeal: float
    overall_artistic: float
    details: Dict[str, float]


@dataclass
class EmotionalScore:
    """Emotional quality assessment result"""
    emotional_clarity: float
    intensity_appropriateness: float
    emotional_consistency: float
    listener_engagement: float
    mood_coherence: float
    expressive_range: float
    overall_emotional: float
    details: Dict[str, float]


@dataclass
class QualityIndex:
    """Comprehensive weighted quality index"""
    overall_score: float
    technical_weight: float
    artistic_weight: float
    emotional_weight: float
    dimension_scores: Dict[QualityDimension, float]
    confidence_level: float
    context_factors: Dict[str, float]
    improvement_potential: float


@dataclass
class BenchmarkComparison:
    """Comparison against quality benchmarks"""
    benchmark_type: BenchmarkType
    reference_score: float
    current_score: float
    relative_performance: float  # -1 to 1, where 0 is at benchmark
    percentile_ranking: float
    improvement_needed: float
    strength_areas: List[str]
    weakness_areas: List[str]


@dataclass
class QualityTrend:
    """Quality trend analysis over time"""
    time_period: str
    score_history: List[Tuple[datetime, float]]
    trend_direction: str  # "improving", "declining", "stable"
    improvement_rate: float
    consistency_score: float
    milestone_achievements: List[str]
    projected_future_quality: float


class QualityScoringSystem:
    """
    Comprehensive music quality assessment system
    
    Provides multi-dimensional quality scoring with benchmarking,
    trend analysis, and context-aware assessment.
    """
    
    def __init__(self):
        self.name = "QualityScoringSystem"
        self.version = "1.0.0"
        
        # Quality dimension weights (can be adjusted based on context)
        self.default_weights = {
            QualityDimension.TECHNICAL: 0.30,
            QualityDimension.ARTISTIC: 0.35,
            QualityDimension.EMOTIONAL: 0.25,
            QualityDimension.USER_EXPERIENCE: 0.10
        }
        
        # Benchmark standards
        self.benchmarks = {
            BenchmarkType.PROFESSIONAL_STANDARD: 0.85,
            BenchmarkType.GENRE_SPECIFIC: 0.80,
            BenchmarkType.USER_PREFERENCE: 0.75,
            BenchmarkType.HISTORICAL_BASELINE: 0.70,
            BenchmarkType.ADAPTIVE_TARGET: 0.78
        }
        
        # Quality history for trend analysis
        self.quality_history = []
        
        # Context-specific adjustments
        self.context_adjustments = {
            'chinese_traditional': {
                QualityDimension.CULTURAL: 0.15,
                QualityDimension.TECHNICAL: 0.25,
                QualityDimension.ARTISTIC: 0.35,
                QualityDimension.EMOTIONAL: 0.25
            },
            'pop_music': {
                QualityDimension.USER_EXPERIENCE: 0.20,
                QualityDimension.EMOTIONAL: 0.30,
                QualityDimension.ARTISTIC: 0.30,
                QualityDimension.TECHNICAL: 0.20
            }
        }

    def calculate_technical_score(self, music_data: MusicElement, context: ValidationContext) -> TechnicalScore:
        """
        Calculate comprehensive technical quality score
        
        Args:
            music_data: Music element to assess
            context: Assessment context
            
        Returns:
            TechnicalScore with detailed technical assessment
        """
        # Analyze harmony correctness
        harmony_score = self._assess_harmony_correctness(music_data)
        
        # Analyze rhythmic accuracy
        rhythm_score = self._assess_rhythmic_accuracy(music_data)
        
        # Analyze spectral balance
        spectral_score = self._assess_spectral_balance(music_data)
        
        # Analyze dynamic range
        dynamic_score = self._assess_dynamic_range(music_data)
        
        # Analyze audio quality
        audio_score = self._assess_audio_quality(music_data)
        
        # Analyze theory compliance
        theory_score = self._assess_theory_compliance(music_data)
        
        # Calculate weighted technical score
        technical_components = {
            'harmony': harmony_score,
            'rhythm': rhythm_score,
            'spectral': spectral_score,
            'dynamics': dynamic_score,
            'audio': audio_score,
            'theory': theory_score
        }
        
        overall_technical = np.mean(list(technical_components.values()))
        
        return TechnicalScore(
            harmony_correctness=harmony_score,
            rhythmic_accuracy=rhythm_score,
            spectral_balance=spectral_score,
            dynamic_range=dynamic_score,
            audio_quality=audio_score,
            theory_compliance=theory_score,
            overall_technical=overall_technical,
            details=technical_components
        )

    def calculate_artistic_score(self, music_data: MusicElement, context: ValidationContext) -> ArtisticScore:
        """
        Calculate comprehensive artistic quality score
        
        Args:
            music_data: Music element to assess
            context: Assessment context
            
        Returns:
            ArtisticScore with detailed artistic assessment
        """
        # Assess creativity and innovation
        creativity_score = self._assess_creativity(music_data)
        
        # Assess musical flow and phrasing
        flow_score = self._assess_musical_flow(music_data)
        
        # Assess emotional expression
        expression_score = self._assess_emotional_expression(music_data)
        
        # Assess style authenticity
        style_score = self._assess_style_authenticity(music_data, context)
        
        # Assess structural coherence
        structure_score = self._assess_structural_coherence(music_data)
        
        # Assess aesthetic appeal
        aesthetic_score = self._assess_aesthetic_appeal(music_data)
        
        # Calculate weighted artistic score
        artistic_components = {
            'creativity': creativity_score,
            'flow': flow_score,
            'expression': expression_score,
            'style': style_score,
            'structure': structure_score,
            'aesthetics': aesthetic_score
        }
        
        overall_artistic = np.mean(list(artistic_components.values()))
        
        return ArtisticScore(
            creativity=creativity_score,
            musical_flow=flow_score,
            emotional_expression=expression_score,
            style_authenticity=style_score,
            structural_coherence=structure_score,
            aesthetic_appeal=aesthetic_score,
            overall_artistic=overall_artistic,
            details=artistic_components
        )

    def calculate_emotional_score(self, music_data: MusicElement, context: ValidationContext) -> EmotionalScore:
        """
        Calculate comprehensive emotional quality score
        
        Args:
            music_data: Music element to assess
            context: Assessment context
            
        Returns:
            EmotionalScore with detailed emotional assessment
        """
        # Assess emotional clarity
        clarity_score = self._assess_emotional_clarity(music_data)
        
        # Assess intensity appropriateness
        intensity_score = self._assess_intensity_appropriateness(music_data, context)
        
        # Assess emotional consistency
        consistency_score = self._assess_emotional_consistency(music_data)
        
        # Assess listener engagement potential
        engagement_score = self._assess_listener_engagement(music_data)
        
        # Assess mood coherence
        mood_score = self._assess_mood_coherence(music_data)
        
        # Assess expressive range
        range_score = self._assess_expressive_range(music_data)
        
        # Calculate weighted emotional score
        emotional_components = {
            'clarity': clarity_score,
            'intensity': intensity_score,
            'consistency': consistency_score,
            'engagement': engagement_score,
            'mood': mood_score,
            'range': range_score
        }
        
        overall_emotional = np.mean(list(emotional_components.values()))
        
        return EmotionalScore(
            emotional_clarity=clarity_score,
            intensity_appropriateness=intensity_score,
            emotional_consistency=consistency_score,
            listener_engagement=engagement_score,
            mood_coherence=mood_score,
            expressive_range=range_score,
            overall_emotional=overall_emotional,
            details=emotional_components
        )

    def generate_weighted_quality_index(self, technical: TechnicalScore, 
                                      artistic: ArtisticScore,
                                      emotional: EmotionalScore,
                                      context: ValidationContext) -> QualityIndex:
        """
        Generate context-aware weighted quality index
        
        Args:
            technical: Technical quality scores
            artistic: Artistic quality scores
            emotional: Emotional quality scores
            context: Assessment context for weighting
            
        Returns:
            QualityIndex with comprehensive assessment
        """
        # Determine appropriate weights based on context
        weights = self._determine_context_weights(context)
        
        # Calculate dimension scores
        dimension_scores = {
            QualityDimension.TECHNICAL: technical.overall_technical,
            QualityDimension.ARTISTIC: artistic.overall_artistic,
            QualityDimension.EMOTIONAL: emotional.overall_emotional
        }
        
        # Calculate user experience score (combination of factors)
        ux_score = self._calculate_user_experience_score(technical, artistic, emotional)
        dimension_scores[QualityDimension.USER_EXPERIENCE] = ux_score
        
        # Calculate weighted overall score
        overall_score = sum(
            dimension_scores[dim] * weights.get(dim, 0.25)
            for dim in dimension_scores
        )
        
        # Calculate confidence level
        confidence = self._calculate_confidence_level(technical, artistic, emotional)
        
        # Identify context factors
        context_factors = self._extract_context_factors(context)
        
        # Estimate improvement potential
        improvement_potential = self._estimate_improvement_potential(dimension_scores)
        
        return QualityIndex(
            overall_score=overall_score,
            technical_weight=weights.get(QualityDimension.TECHNICAL, 0.30),
            artistic_weight=weights.get(QualityDimension.ARTISTIC, 0.35),
            emotional_weight=weights.get(QualityDimension.EMOTIONAL, 0.25),
            dimension_scores=dimension_scores,
            confidence_level=confidence,
            context_factors=context_factors,
            improvement_potential=improvement_potential
        )

    def compare_to_benchmarks(self, quality_index: QualityIndex, 
                            context: ValidationContext) -> List[BenchmarkComparison]:
        """
        Compare quality against various benchmarks
        
        Args:
            quality_index: Current quality assessment
            context: Context for benchmark selection
            
        Returns:
            List of benchmark comparisons
        """
        comparisons = []
        current_score = quality_index.overall_score
        
        # Compare against relevant benchmarks
        relevant_benchmarks = self._select_relevant_benchmarks(context)
        
        for benchmark_type in relevant_benchmarks:
            reference_score = self.benchmarks[benchmark_type]
            
            # Calculate relative performance
            relative_performance = (current_score - reference_score) / reference_score
            
            # Calculate percentile ranking (simplified)
            percentile = self._calculate_percentile_ranking(current_score, benchmark_type)
            
            # Calculate improvement needed
            improvement_needed = max(0, reference_score - current_score)
            
            # Identify strengths and weaknesses
            strengths, weaknesses = self._analyze_strengths_weaknesses(
                quality_index.dimension_scores, benchmark_type
            )
            
            comparison = BenchmarkComparison(
                benchmark_type=benchmark_type,
                reference_score=reference_score,
                current_score=current_score,
                relative_performance=relative_performance,
                percentile_ranking=percentile,
                improvement_needed=improvement_needed,
                strength_areas=strengths,
                weakness_areas=weaknesses
            )
            
            comparisons.append(comparison)
        
        return comparisons

    def analyze_quality_trends(self, time_period: str = "30_days") -> QualityTrend:
        """
        Analyze quality trends over time
        
        Args:
            time_period: Period for trend analysis
            
        Returns:
            QualityTrend with trend analysis
        """
        # Filter history by time period
        relevant_history = self._filter_history_by_period(self.quality_history, time_period)
        
        if len(relevant_history) < 2:
            return QualityTrend(
                time_period=time_period,
                score_history=relevant_history,
                trend_direction="insufficient_data",
                improvement_rate=0.0,
                consistency_score=0.5,
                milestone_achievements=[],
                projected_future_quality=0.5
            )
        
        # Calculate trend direction
        trend_direction = self._calculate_trend_direction(relevant_history)
        
        # Calculate improvement rate
        improvement_rate = self._calculate_improvement_rate(relevant_history)
        
        # Calculate consistency score
        consistency = self._calculate_consistency_score(relevant_history)
        
        # Identify milestone achievements
        milestones = self._identify_milestones(relevant_history)
        
        # Project future quality
        projected_quality = self._project_future_quality(relevant_history)
        
        return QualityTrend(
            time_period=time_period,
            score_history=relevant_history,
            trend_direction=trend_direction,
            improvement_rate=improvement_rate,
            consistency_score=consistency,
            milestone_achievements=milestones,
            projected_future_quality=projected_quality
        )

    def record_quality_assessment(self, quality_index: QualityIndex):
        """Record quality assessment for trend analysis"""
        timestamp = datetime.now()
        self.quality_history.append((timestamp, quality_index.overall_score))
        
        # Keep only recent history (e.g., last 100 assessments)
        if len(self.quality_history) > 100:
            self.quality_history = self.quality_history[-100:]

    # Helper methods for quality assessment
    
    def _assess_harmony_correctness(self, music_data: MusicElement) -> float:
        """Assess harmonic correctness"""
        # Simplified harmony assessment
        if hasattr(music_data, 'harmony_quality'):
            return music_data.harmony_quality
        return 0.75  # Default reasonable score

    def _assess_rhythmic_accuracy(self, music_data: MusicElement) -> float:
        """Assess rhythmic accuracy"""
        if hasattr(music_data, 'rhythm_quality'):
            return music_data.rhythm_quality
        return 0.80  # Default score

    def _assess_spectral_balance(self, music_data: MusicElement) -> float:
        """Assess spectral balance"""
        return 0.75  # Simplified assessment

    def _assess_dynamic_range(self, music_data: MusicElement) -> float:
        """Assess dynamic range"""
        return 0.70  # Simplified assessment

    def _assess_audio_quality(self, music_data: MusicElement) -> float:
        """Assess audio quality"""
        return 0.85  # Simplified assessment

    def _assess_theory_compliance(self, music_data: MusicElement) -> float:
        """Assess music theory compliance"""
        if hasattr(music_data, 'theory_compliance'):
            return music_data.theory_compliance
        return 0.78  # Default score

    def _assess_creativity(self, music_data: MusicElement) -> float:
        """Assess creativity and innovation"""
        return 0.65  # Simplified assessment

    def _assess_musical_flow(self, music_data: MusicElement) -> float:
        """Assess musical flow and phrasing"""
        return 0.80  # Simplified assessment

    def _assess_emotional_expression(self, music_data: MusicElement) -> float:
        """Assess emotional expression"""
        if hasattr(music_data, 'emotion_score'):
            return music_data.emotion_score
        return 0.75  # Default score

    def _assess_style_authenticity(self, music_data: MusicElement, context: ValidationContext) -> float:
        """Assess style authenticity"""
        return 0.82  # Simplified assessment

    def _assess_structural_coherence(self, music_data: MusicElement) -> float:
        """Assess structural coherence"""
        return 0.78  # Simplified assessment

    def _assess_aesthetic_appeal(self, music_data: MusicElement) -> float:
        """Assess aesthetic appeal"""
        return 0.73  # Simplified assessment

    def _assess_emotional_clarity(self, music_data: MusicElement) -> float:
        """Assess emotional clarity"""
        return 0.80  # Simplified assessment

    def _assess_intensity_appropriateness(self, music_data: MusicElement, context: ValidationContext) -> float:
        """Assess intensity appropriateness"""
        return 0.85  # Simplified assessment

    def _assess_emotional_consistency(self, music_data: MusicElement) -> float:
        """Assess emotional consistency"""
        return 0.75  # Simplified assessment

    def _assess_listener_engagement(self, music_data: MusicElement) -> float:
        """Assess listener engagement potential"""
        return 0.70  # Simplified assessment

    def _assess_mood_coherence(self, music_data: MusicElement) -> float:
        """Assess mood coherence"""
        return 0.82  # Simplified assessment

    def _assess_expressive_range(self, music_data: MusicElement) -> float:
        """Assess expressive range"""
        return 0.68  # Simplified assessment

    def _determine_context_weights(self, context: ValidationContext) -> Dict[QualityDimension, float]:
        """Determine appropriate weights based on context"""
        style = context.get('style', 'general')
        
        if style in self.context_adjustments:
            return self.context_adjustments[style]
        else:
            return self.default_weights

    def _calculate_user_experience_score(self, technical: TechnicalScore, 
                                       artistic: ArtisticScore, 
                                       emotional: EmotionalScore) -> float:
        """Calculate user experience score"""
        # Combine factors that affect user experience
        ux_factors = [
            artistic.musical_flow,
            emotional.listener_engagement,
            technical.audio_quality,
            artistic.aesthetic_appeal
        ]
        return np.mean(ux_factors)

    def _calculate_confidence_level(self, technical: TechnicalScore, 
                                  artistic: ArtisticScore, 
                                  emotional: EmotionalScore) -> float:
        """Calculate confidence level in the assessment"""
        # Base confidence on consistency of scores
        all_scores = [
            technical.overall_technical,
            artistic.overall_artistic,
            emotional.overall_emotional
        ]
        
        # Lower variance indicates higher confidence
        variance = np.var(all_scores)
        confidence = max(0.5, 1.0 - variance)
        return min(confidence, 1.0)

    def _extract_context_factors(self, context: ValidationContext) -> Dict[str, float]:
        """Extract relevant context factors"""
        return {
            'style_specificity': context.get('style_weight', 0.5),
            'user_preference_weight': context.get('user_preference_weight', 0.5),
            'cultural_sensitivity': context.get('cultural_weight', 0.3)
        }

    def _estimate_improvement_potential(self, dimension_scores: Dict[QualityDimension, float]) -> float:
        """Estimate potential for quality improvement"""
        # Find the dimension with the lowest score as improvement opportunity
        min_score = min(dimension_scores.values())
        max_possible_improvement = 1.0 - min_score
        
        # Consider how much room for improvement exists
        avg_score = np.mean(list(dimension_scores.values()))
        improvement_potential = (1.0 - avg_score) * 0.8  # 80% of theoretical maximum
        
        return improvement_potential

    def _select_relevant_benchmarks(self, context: ValidationContext) -> List[BenchmarkType]:
        """Select relevant benchmarks based on context"""
        benchmarks = [BenchmarkType.PROFESSIONAL_STANDARD]
        
        if context.get('genre'):
            benchmarks.append(BenchmarkType.GENRE_SPECIFIC)
        
        if context.get('user_preference_available'):
            benchmarks.append(BenchmarkType.USER_PREFERENCE)
        
        benchmarks.append(BenchmarkType.ADAPTIVE_TARGET)
        
        return benchmarks

    def _calculate_percentile_ranking(self, score: float, benchmark_type: BenchmarkType) -> float:
        """Calculate percentile ranking against benchmark"""
        # Simplified percentile calculation
        reference = self.benchmarks[benchmark_type]
        
        if score >= reference:
            return 75 + (score - reference) * 100  # 75-100th percentile
        else:
            return (score / reference) * 75  # 0-75th percentile

    def _analyze_strengths_weaknesses(self, dimension_scores: Dict[QualityDimension, float], 
                                    benchmark_type: BenchmarkType) -> Tuple[List[str], List[str]]:
        """Analyze strengths and weaknesses relative to benchmark"""
        benchmark_score = self.benchmarks[benchmark_type]
        
        strengths = []
        weaknesses = []
        
        for dimension, score in dimension_scores.items():
            if score > benchmark_score + 0.05:
                strengths.append(dimension.value)
            elif score < benchmark_score - 0.05:
                weaknesses.append(dimension.value)
        
        return strengths, weaknesses

    def _filter_history_by_period(self, history: List[Tuple[datetime, float]], period: str) -> List[Tuple[datetime, float]]:
        """Filter quality history by time period"""
        # Simplified filtering - would implement actual date filtering
        return history[-10:] if history else []

    def _calculate_trend_direction(self, history: List[Tuple[datetime, float]]) -> str:
        """Calculate trend direction from history"""
        if len(history) < 2:
            return "insufficient_data"
        
        scores = [score for _, score in history]
        first_half = np.mean(scores[:len(scores)//2])
        second_half = np.mean(scores[len(scores)//2:])
        
        if second_half > first_half + 0.05:
            return "improving"
        elif second_half < first_half - 0.05:
            return "declining"
        else:
            return "stable"

    def _calculate_improvement_rate(self, history: List[Tuple[datetime, float]]) -> float:
        """Calculate rate of improvement"""
        if len(history) < 2:
            return 0.0
        
        scores = [score for _, score in history]
        return (scores[-1] - scores[0]) / len(scores)

    def _calculate_consistency_score(self, history: List[Tuple[datetime, float]]) -> float:
        """Calculate consistency score"""
        if len(history) < 2:
            return 0.5
        
        scores = [score for _, score in history]
        variance = np.var(scores)
        return max(0.0, 1.0 - variance * 2)  # Lower variance = higher consistency

    def _identify_milestones(self, history: List[Tuple[datetime, float]]) -> List[str]:
        """Identify milestone achievements"""
        milestones = []
        
        if not history:
            return milestones
        
        scores = [score for _, score in history]
        max_score = max(scores)
        
        if max_score > 0.9:
            milestones.append("Achieved excellent quality (>90%)")
        elif max_score > 0.8:
            milestones.append("Achieved good quality (>80%)")
        elif max_score > 0.7:
            milestones.append("Achieved acceptable quality (>70%)")
        
        return milestones

    def _project_future_quality(self, history: List[Tuple[datetime, float]]) -> float:
        """Project future quality based on trends"""
        if len(history) < 3:
            return history[-1][1] if history else 0.5
        
        scores = [score for _, score in history]
        
        # Simple linear projection
        recent_trend = (scores[-1] - scores[-3]) / 2
        projected = scores[-1] + recent_trend
        
        return max(0.0, min(1.0, projected))