"""
Emotion Consistency Checker - Ensure emotional expression preservation

This module provides comprehensive emotion consistency validation including:
- Emotional coherence analysis across tracks
- Intensity preservation during modifications
- Temporal emotion tracking for stability
- Style-emotion matching verification
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum

from .base_validator import BaseValidator, ValidationResult
from .data_structures import (
    QualityScore, ConflictType, ResolutionSuggestion,
    ValidationContext, MusicElement
)


class EmotionType(Enum):
    """Core emotion types for analysis"""
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    PEACEFUL = "peaceful"
    ENERGETIC = "energetic"
    MELANCHOLIC = "melancholic"
    ROMANTIC = "romantic"
    MYSTERIOUS = "mysterious"
    TRIUMPHANT = "triumphant"
    NEUTRAL = "neutral"


class IntensityLevel(Enum):
    """Emotional intensity levels"""
    VERY_LOW = 1
    LOW = 2
    MODERATE = 3
    HIGH = 4
    VERY_HIGH = 5


@dataclass
class EmotionalProfile:
    """Emotional profile of a music segment"""
    primary_emotion: EmotionType
    secondary_emotion: Optional[EmotionType]
    intensity: IntensityLevel
    confidence: float
    temporal_stability: float
    features: Dict[str, float]  # Audio/musical features that support emotion


@dataclass
class CoherenceResult:
    """Result of emotional coherence analysis"""
    coherence_score: float
    emotion_conflicts: List[Tuple[str, str]]  # (track1, track2) pairs with conflicts
    dominant_emotion: EmotionType
    emotion_distribution: Dict[EmotionType, float]
    consistency_issues: List[str]


@dataclass
class IntensityResult:
    """Result of intensity preservation analysis"""
    preservation_score: float
    intensity_drift: float
    critical_sections: List[Tuple[float, float]]  # Time ranges with intensity issues
    original_intensity_profile: List[float]
    modified_intensity_profile: List[float]
    recommendations: List[str]


@dataclass
class TemporalEmotionResult:
    """Result of temporal emotion tracking"""
    stability_score: float
    emotion_changes: List[Tuple[float, EmotionType, EmotionType]]  # (time, from_emotion, to_emotion)
    appropriate_transitions: int
    jarring_transitions: int
    emotional_arc_quality: float


@dataclass
class StyleEmotionMatchResult:
    """Result of style-emotion matching analysis"""
    match_score: float
    style_emotion_compatibility: float
    identified_mismatches: List[str]
    cultural_appropriateness: float
    genre_consistency: float


class EmotionConsistencyChecker(BaseValidator):
    """
    Comprehensive emotion consistency validation system
    
    Ensures emotional expression preservation and coherence across
    music operations including track modifications and combinations.
    """
    
    def __init__(self):
        super().__init__("EmotionConsistencyChecker")
        self.name = "EmotionConsistencyChecker"
        self.version = "1.0.0"
        
        # Emotion compatibility matrix (how well emotions work together)
        self.emotion_compatibility = {
            EmotionType.HAPPY: {
                EmotionType.ENERGETIC: 0.9,
                EmotionType.TRIUMPHANT: 0.8,
                EmotionType.ROMANTIC: 0.7,
                EmotionType.PEACEFUL: 0.6,
                EmotionType.SAD: 0.2,
                EmotionType.MELANCHOLIC: 0.1,
                EmotionType.ANGRY: 0.3
            },
            EmotionType.SAD: {
                EmotionType.MELANCHOLIC: 0.9,
                EmotionType.PEACEFUL: 0.7,
                EmotionType.ROMANTIC: 0.6,
                EmotionType.MYSTERIOUS: 0.5,
                EmotionType.HAPPY: 0.2,
                EmotionType.ENERGETIC: 0.1,
                EmotionType.TRIUMPHANT: 0.1
            },
            EmotionType.ENERGETIC: {
                EmotionType.HAPPY: 0.9,
                EmotionType.TRIUMPHANT: 0.8,
                EmotionType.ANGRY: 0.7,
                EmotionType.ROMANTIC: 0.5,
                EmotionType.SAD: 0.1,
                EmotionType.PEACEFUL: 0.3,
                EmotionType.MELANCHOLIC: 0.1
            },
            EmotionType.PEACEFUL: {
                EmotionType.ROMANTIC: 0.8,
                EmotionType.MELANCHOLIC: 0.6,
                EmotionType.MYSTERIOUS: 0.7,
                EmotionType.HAPPY: 0.6,
                EmotionType.ANGRY: 0.1,
                EmotionType.ENERGETIC: 0.3,
                EmotionType.TRIUMPHANT: 0.4
            }
        }
        
        # Style-emotion appropriateness mapping
        self.style_emotion_mapping = {
            'classical': [EmotionType.PEACEFUL, EmotionType.ROMANTIC, EmotionType.MELANCHOLIC, EmotionType.TRIUMPHANT],
            'pop': [EmotionType.HAPPY, EmotionType.ENERGETIC, EmotionType.ROMANTIC, EmotionType.SAD],
            'jazz': [EmotionType.PEACEFUL, EmotionType.ROMANTIC, EmotionType.MELANCHOLIC, EmotionType.MYSTERIOUS],
            'rock': [EmotionType.ENERGETIC, EmotionType.ANGRY, EmotionType.TRIUMPHANT, EmotionType.HAPPY],
            'folk': [EmotionType.PEACEFUL, EmotionType.MELANCHOLIC, EmotionType.ROMANTIC, EmotionType.HAPPY],
            'chinese_traditional': [EmotionType.PEACEFUL, EmotionType.MELANCHOLIC, EmotionType.MYSTERIOUS, EmotionType.ROMANTIC]
        }

    def validate(self, music_element: MusicElement, context: ValidationContext) -> ValidationResult:
        """
        Main validation entry point for emotion consistency
        
        Args:
            music_element: Music data to validate
            context: Validation context and settings
            
        Returns:
            ValidationResult with emotion consistency assessment
        """
        emotion_scores = []
        conflicts = []
        suggestions = []
        
        # Extract emotional profiles from tracks
        if hasattr(music_element, 'tracks') and music_element.tracks:
            coherence_result = self.check_emotional_coherence(music_element.tracks)
            emotion_scores.append(coherence_result.coherence_score)
            
            if coherence_result.consistency_issues:
                conflicts.extend([
                    ConflictType.EMOTION_MISMATCH for _ in coherence_result.consistency_issues
                ])
        
        # Check intensity preservation if original data available
        if hasattr(context, 'original_data') and context.original_data:
            intensity_result = self.validate_intensity_preservation(
                context.original_data, music_element
            )
            emotion_scores.append(intensity_result.preservation_score)
            
            if intensity_result.preservation_score < 0.7:
                conflicts.append(ConflictType.INTENSITY_LOSS)
                suggestions.extend(intensity_result.recommendations)
        
        # Check temporal emotion stability
        if hasattr(music_element, 'temporal_data') and music_element.temporal_data:
            temporal_result = self.track_temporal_emotions(music_element.temporal_data)
            emotion_scores.append(temporal_result.stability_score)
            
            if temporal_result.jarring_transitions > 0:
                conflicts.append(ConflictType.EMOTION_DISCONTINUITY)
        
        # Check style-emotion matching
        if hasattr(music_element, 'style') and music_element.style:
            style_result = self.verify_style_emotion_match(
                music_element.style, 
                self._extract_dominant_emotion(music_element)
            )
            emotion_scores.append(style_result.match_score)
            
            if style_result.identified_mismatches:
                conflicts.append(ConflictType.STYLE_EMOTION_MISMATCH)
        
        # Calculate overall emotion consistency score
        overall_score = np.mean(emotion_scores) if emotion_scores else 0.5
        
        # Generate quality score
        quality_score = QualityScore(
            technical_score=overall_score * 0.7,  # Emotion is more artistic than technical
            artistic_score=overall_score,
            emotional_score=overall_score,
            overall_score=overall_score,
            confidence_level=min(overall_score + 0.1, 1.0)
        )
        
        return ValidationResult(
            validator_name=self.name,
            overall_score=overall_score,
            passed=overall_score >= 0.7,
            detailed_scores={'emotion_score': overall_score},
            issues=[],
            metadata={
                'emotion_aspects_analyzed': len(emotion_scores),
                'individual_scores': emotion_scores,
                'dominant_emotion': self._extract_dominant_emotion(music_element),
                'quality_score': quality_score.__dict__,
                'conflicts': [c.value for c in conflicts],
                'suggestions': suggestions
            }
        )

    def check_emotional_coherence(self, tracks: List[Dict]) -> CoherenceResult:
        """
        Analyze emotional coherence across multiple tracks
        
        Args:
            tracks: List of track data dictionaries
            
        Returns:
            CoherenceResult with coherence analysis
        """
        if not tracks:
            return CoherenceResult(
                coherence_score=0.5,
                emotion_conflicts=[],
                dominant_emotion=EmotionType.NEUTRAL,
                emotion_distribution={},
                consistency_issues=["No tracks to analyze"]
            )
        
        # Extract emotional profiles for each track
        track_emotions = []
        for i, track in enumerate(tracks):
            emotion_profile = self._extract_emotion_profile(track)
            track_emotions.append((f"track_{i}", emotion_profile))
        
        # Analyze coherence between tracks
        emotion_conflicts = []
        compatibility_scores = []
        
        for i in range(len(track_emotions)):
            for j in range(i + 1, len(track_emotions)):
                track1_name, emotion1 = track_emotions[i]
                track2_name, emotion2 = track_emotions[j]
                
                compatibility = self._calculate_emotion_compatibility(
                    emotion1.primary_emotion, emotion2.primary_emotion
                )
                compatibility_scores.append(compatibility)
                
                if compatibility < 0.4:  # Low compatibility threshold
                    emotion_conflicts.append((track1_name, track2_name))
        
        # Calculate overall coherence score
        coherence_score = np.mean(compatibility_scores) if compatibility_scores else 0.5
        
        # Determine dominant emotion
        emotion_counts = {}
        for _, emotion_profile in track_emotions:
            emotion = emotion_profile.primary_emotion
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        dominant_emotion = max(emotion_counts, key=emotion_counts.get) if emotion_counts else EmotionType.NEUTRAL
        
        # Calculate emotion distribution
        total_tracks = len(track_emotions)
        emotion_distribution = {
            emotion: count / total_tracks 
            for emotion, count in emotion_counts.items()
        }
        
        # Identify consistency issues
        consistency_issues = []
        if len(emotion_conflicts) > 0:
            consistency_issues.append(f"Found {len(emotion_conflicts)} emotion conflicts between tracks")
        
        if coherence_score < 0.6:
            consistency_issues.append("Low overall emotional coherence across tracks")
        
        return CoherenceResult(
            coherence_score=coherence_score,
            emotion_conflicts=emotion_conflicts,
            dominant_emotion=dominant_emotion,
            emotion_distribution=emotion_distribution,
            consistency_issues=consistency_issues
        )

    def validate_intensity_preservation(self, original_data: Dict, modified_data: Dict) -> IntensityResult:
        """
        Validate preservation of emotional intensity during modifications
        
        Args:
            original_data: Original music data
            modified_data: Modified music data
            
        Returns:
            IntensityResult with preservation analysis
        """
        # Extract intensity profiles
        original_intensity = self._extract_intensity_profile(original_data)
        modified_intensity = self._extract_intensity_profile(modified_data)
        
        # Calculate preservation score
        if len(original_intensity) == 0 or len(modified_intensity) == 0:
            return IntensityResult(
                preservation_score=0.5,
                intensity_drift=0.0,
                critical_sections=[],
                original_intensity_profile=original_intensity,
                modified_intensity_profile=modified_intensity,
                recommendations=["Insufficient data for intensity analysis"]
            )
        
        # Align profiles if different lengths
        min_length = min(len(original_intensity), len(modified_intensity))
        original_aligned = original_intensity[:min_length]
        modified_aligned = modified_intensity[:min_length]
        
        # Calculate preservation metrics
        intensity_differences = np.array(modified_aligned) - np.array(original_aligned)
        mean_drift = np.mean(np.abs(intensity_differences))
        preservation_score = max(0.0, 1.0 - mean_drift / 2.0)  # Normalize to 0-1
        
        # Identify critical sections with significant intensity changes
        critical_sections = []
        for i, diff in enumerate(intensity_differences):
            if abs(diff) > 0.5:  # Significant intensity change threshold
                start_time = i * (len(original_aligned) / min_length)
                end_time = (i + 1) * (len(original_aligned) / min_length)
                critical_sections.append((start_time, end_time))
        
        # Generate recommendations
        recommendations = []
        if mean_drift > 0.3:
            recommendations.append("Consider adjusting volume/dynamics to preserve intensity")
        if len(critical_sections) > 2:
            recommendations.append("Multiple sections show intensity loss - review overall processing")
        if preservation_score < 0.6:
            recommendations.append("Emotional intensity significantly altered - consider alternative approach")
        
        return IntensityResult(
            preservation_score=preservation_score,
            intensity_drift=mean_drift,
            critical_sections=critical_sections,
            original_intensity_profile=original_intensity,
            modified_intensity_profile=modified_intensity,
            recommendations=recommendations
        )

    def track_temporal_emotions(self, temporal_data: List[Dict]) -> TemporalEmotionResult:
        """
        Track emotional changes over time for stability analysis
        
        Args:
            temporal_data: Time-series emotion data
            
        Returns:
            TemporalEmotionResult with temporal analysis
        """
        if not temporal_data:
            return TemporalEmotionResult(
                stability_score=0.5,
                emotion_changes=[],
                appropriate_transitions=0,
                jarring_transitions=0,
                emotional_arc_quality=0.5
            )
        
        # Extract emotion sequence
        emotion_sequence = []
        for segment in temporal_data:
            emotion = self._determine_segment_emotion(segment)
            time = segment.get('time', 0)
            emotion_sequence.append((time, emotion))
        
        # Analyze emotion changes
        emotion_changes = []
        appropriate_transitions = 0
        jarring_transitions = 0
        
        for i in range(len(emotion_sequence) - 1):
            current_time, current_emotion = emotion_sequence[i]
            next_time, next_emotion = emotion_sequence[i + 1]
            
            if current_emotion != next_emotion:
                emotion_changes.append((current_time, current_emotion, next_emotion))
                
                # Evaluate transition appropriateness
                compatibility = self._calculate_emotion_compatibility(current_emotion, next_emotion)
                if compatibility > 0.6:
                    appropriate_transitions += 1
                elif compatibility < 0.3:
                    jarring_transitions += 1
        
        # Calculate stability score
        total_transitions = len(emotion_changes)
        if total_transitions == 0:
            stability_score = 1.0  # No changes = perfect stability
        else:
            stability_score = appropriate_transitions / total_transitions
        
        # Evaluate emotional arc quality
        arc_quality = self._evaluate_emotional_arc(emotion_sequence)
        
        return TemporalEmotionResult(
            stability_score=stability_score,
            emotion_changes=emotion_changes,
            appropriate_transitions=appropriate_transitions,
            jarring_transitions=jarring_transitions,
            emotional_arc_quality=arc_quality
        )

    def verify_style_emotion_match(self, style: str, emotion: EmotionType) -> StyleEmotionMatchResult:
        """
        Verify compatibility between musical style and emotional expression
        
        Args:
            style: Musical style/genre
            emotion: Detected emotion
            
        Returns:
            StyleEmotionMatchResult with matching analysis
        """
        # Get appropriate emotions for the style
        appropriate_emotions = self.style_emotion_mapping.get(style.lower(), [])
        
        # Calculate match score
        if emotion in appropriate_emotions:
            match_score = 0.9
            style_emotion_compatibility = 1.0
            identified_mismatches = []
        else:
            # Find closest appropriate emotion
            closest_emotion = self._find_closest_appropriate_emotion(emotion, appropriate_emotions)
            if closest_emotion:
                compatibility = self._calculate_emotion_compatibility(emotion, closest_emotion)
                match_score = compatibility * 0.7  # Penalty for not being directly appropriate
                style_emotion_compatibility = compatibility
                identified_mismatches = [f"Emotion {emotion.value} not typical for {style} style"]
            else:
                match_score = 0.3
                style_emotion_compatibility = 0.3
                identified_mismatches = [f"Emotion {emotion.value} incompatible with {style} style"]
        
        # Cultural appropriateness (simplified)
        cultural_appropriateness = self._assess_cultural_appropriateness(style, emotion)
        
        # Genre consistency
        genre_consistency = match_score  # Simplified - could be more sophisticated
        
        return StyleEmotionMatchResult(
            match_score=match_score,
            style_emotion_compatibility=style_emotion_compatibility,
            identified_mismatches=identified_mismatches,
            cultural_appropriateness=cultural_appropriateness,
            genre_consistency=genre_consistency
        )

    # Helper methods for internal calculations
    
    def _extract_emotion_profile(self, track_data: Dict) -> EmotionalProfile:
        """Extract emotional profile from track data"""
        # Simplified emotion extraction - would use actual analysis in production
        features = track_data.get('features', {})
        
        # Determine primary emotion based on features
        energy = features.get('energy', 0.5)
        valence = features.get('valence', 0.5)
        tempo = features.get('tempo', 120)
        
        # Simple emotion mapping based on energy and valence
        if valence > 0.6 and energy > 0.6:
            primary_emotion = EmotionType.HAPPY
        elif valence > 0.6 and energy < 0.4:
            primary_emotion = EmotionType.PEACEFUL
        elif valence < 0.4 and energy > 0.6:
            primary_emotion = EmotionType.ANGRY
        elif valence < 0.4 and energy < 0.4:
            primary_emotion = EmotionType.SAD
        else:
            primary_emotion = EmotionType.NEUTRAL
        
        # Determine intensity based on energy and tempo
        intensity_score = (energy + min(tempo / 140, 1.0)) / 2
        if intensity_score > 0.8:
            intensity = IntensityLevel.VERY_HIGH
        elif intensity_score > 0.6:
            intensity = IntensityLevel.HIGH
        elif intensity_score > 0.4:
            intensity = IntensityLevel.MODERATE
        elif intensity_score > 0.2:
            intensity = IntensityLevel.LOW
        else:
            intensity = IntensityLevel.VERY_LOW
        
        return EmotionalProfile(
            primary_emotion=primary_emotion,
            secondary_emotion=None,  # Could be enhanced
            intensity=intensity,
            confidence=0.8,  # Would be calculated based on feature reliability
            temporal_stability=0.7,  # Would be calculated from temporal analysis
            features=features
        )

    def _calculate_emotion_compatibility(self, emotion1: EmotionType, emotion2: EmotionType) -> float:
        """Calculate compatibility score between two emotions"""
        if emotion1 == emotion2:
            return 1.0
        
        # Check compatibility matrix
        if emotion1 in self.emotion_compatibility:
            return self.emotion_compatibility[emotion1].get(emotion2, 0.5)
        elif emotion2 in self.emotion_compatibility:
            return self.emotion_compatibility[emotion2].get(emotion1, 0.5)
        else:
            return 0.5  # Default neutral compatibility

    def _extract_intensity_profile(self, music_data: Dict) -> List[float]:
        """Extract intensity profile over time"""
        # Simplified intensity extraction
        if 'segments' in music_data:
            intensities = []
            for segment in music_data['segments']:
                energy = segment.get('energy', 0.5)
                loudness = segment.get('loudness', -20)
                # Normalize loudness to 0-1 range (assuming -60 to 0 dB range)
                normalized_loudness = max(0, (loudness + 60) / 60)
                intensity = (energy + normalized_loudness) / 2
                intensities.append(intensity)
            return intensities
        else:
            # Fallback: create synthetic intensity profile
            return [0.5] * 10

    def _determine_segment_emotion(self, segment: Dict) -> EmotionType:
        """Determine emotion for a single segment"""
        # Simplified segment emotion determination
        features = segment.get('features', {})
        valence = features.get('valence', 0.5)
        energy = features.get('energy', 0.5)
        
        if valence > 0.6 and energy > 0.6:
            return EmotionType.HAPPY
        elif valence > 0.6 and energy < 0.4:
            return EmotionType.PEACEFUL
        elif valence < 0.4 and energy > 0.6:
            return EmotionType.ANGRY
        elif valence < 0.4 and energy < 0.4:
            return EmotionType.SAD
        else:
            return EmotionType.NEUTRAL

    def _evaluate_emotional_arc(self, emotion_sequence: List[Tuple[float, EmotionType]]) -> float:
        """Evaluate the quality of the emotional arc/journey"""
        if len(emotion_sequence) < 3:
            return 0.5
        
        # Simple arc evaluation - looks for coherent emotional journey
        emotions = [emotion for _, emotion in emotion_sequence]
        
        # Check for too many rapid changes
        changes = sum(1 for i in range(len(emotions) - 1) if emotions[i] != emotions[i + 1])
        change_ratio = changes / len(emotions)
        
        if change_ratio > 0.7:  # Too many changes
            return 0.3
        elif change_ratio < 0.1:  # Too static
            return 0.6
        else:  # Good balance
            return 0.8

    def _find_closest_appropriate_emotion(self, emotion: EmotionType, appropriate_emotions: List[EmotionType]) -> Optional[EmotionType]:
        """Find the closest appropriate emotion from a list"""
        if not appropriate_emotions:
            return None
        
        best_emotion = None
        best_compatibility = 0
        
        for appropriate_emotion in appropriate_emotions:
            compatibility = self._calculate_emotion_compatibility(emotion, appropriate_emotion)
            if compatibility > best_compatibility:
                best_compatibility = compatibility
                best_emotion = appropriate_emotion
        
        return best_emotion

    def _assess_cultural_appropriateness(self, style: str, emotion: EmotionType) -> float:
        """Assess cultural appropriateness of emotion for style"""
        # Simplified cultural assessment
        if style.lower() == 'chinese_traditional':
            # Traditional Chinese music often emphasizes subtlety and restraint
            if emotion in [EmotionType.PEACEFUL, EmotionType.MELANCHOLIC, EmotionType.MYSTERIOUS]:
                return 1.0
            elif emotion in [EmotionType.ROMANTIC, EmotionType.HAPPY]:
                return 0.8
            elif emotion in [EmotionType.ENERGETIC, EmotionType.ANGRY]:
                return 0.4
            else:
                return 0.6
        else:
            # For other styles, assume general appropriateness
            return 0.8

    def _extract_dominant_emotion(self, music_element: MusicElement) -> EmotionType:
        """Extract the dominant emotion from a music element"""
        # Simplified extraction - would use comprehensive analysis in production
        if hasattr(music_element, 'tracks') and music_element.tracks:
            emotions = []
            for track in music_element.tracks:
                emotion_profile = self._extract_emotion_profile(track)
                emotions.append(emotion_profile.primary_emotion)
            
            # Return most common emotion
            if emotions:
                emotion_counts = {}
                for emotion in emotions:
                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                return max(emotion_counts, key=emotion_counts.get)
        
        return EmotionType.NEUTRAL