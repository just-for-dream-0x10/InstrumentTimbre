"""
Harmony Validator - Multi-level music harmony checking algorithm

This module provides comprehensive harmony validation including:
- Interval analysis and dissonance detection
- Chord progression validation with music theory compliance
- Voice leading analysis for smooth transitions
- Harmonic rhythm validation for appropriate change rates
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


class IntervalType(Enum):
    """Musical interval types for analysis"""
    UNISON = 0
    MINOR_SECOND = 1
    MAJOR_SECOND = 2
    MINOR_THIRD = 3
    MAJOR_THIRD = 4
    PERFECT_FOURTH = 5
    TRITONE = 6
    PERFECT_FIFTH = 7
    MINOR_SIXTH = 8
    MAJOR_SIXTH = 9
    MINOR_SEVENTH = 10
    MAJOR_SEVENTH = 11
    OCTAVE = 12


class DissonanceLevel(Enum):
    """Dissonance levels for interval classification"""
    CONSONANT = "consonant"
    MILD_DISSONANT = "mild_dissonant"
    MODERATE_DISSONANT = "moderate_dissonant"
    STRONG_DISSONANT = "strong_dissonant"


@dataclass
class IntervalValidationResult:
    """Result of interval analysis"""
    interval_type: IntervalType
    dissonance_level: DissonanceLevel
    requires_resolution: bool
    resolution_suggestions: List[str]
    confidence_score: float


@dataclass
class ChordProgressionResult:
    """Result of chord progression validation"""
    functional_harmony_score: float
    voice_leading_score: float
    resolution_quality: float
    theory_compliance: float
    identified_issues: List[str]
    improvement_suggestions: List[str]


@dataclass
class VoiceLeadingResult:
    """Result of voice leading analysis"""
    smoothness_score: float
    parallel_motion_issues: List[Tuple[int, int]]  # (voice1, voice2) pairs
    contrary_motion_score: float
    leap_analysis: Dict[str, float]
    overall_quality: float


@dataclass
class HarmonicRhythmResult:
    """Result of harmonic rhythm validation"""
    change_frequency_score: float
    rhythmic_consistency: float
    phrasing_alignment: float
    cadential_strength: float
    overall_rhythm_quality: float


class HarmonyValidator(BaseValidator):
    """
    Comprehensive harmony validation system
    
    Provides multi-level harmony checking including interval analysis,
    chord progression validation, voice leading, and harmonic rhythm.
    """
    
    def __init__(self):
        super().__init__("HarmonyValidator")
        self.name = "HarmonyValidator"
        self.version = "1.0.0"
        
        # Dissonance classification mapping
        self.dissonance_map = {
            IntervalType.UNISON: DissonanceLevel.CONSONANT,
            IntervalType.MINOR_SECOND: DissonanceLevel.STRONG_DISSONANT,
            IntervalType.MAJOR_SECOND: DissonanceLevel.MODERATE_DISSONANT,
            IntervalType.MINOR_THIRD: DissonanceLevel.CONSONANT,
            IntervalType.MAJOR_THIRD: DissonanceLevel.CONSONANT,
            IntervalType.PERFECT_FOURTH: DissonanceLevel.MILD_DISSONANT,
            IntervalType.TRITONE: DissonanceLevel.STRONG_DISSONANT,
            IntervalType.PERFECT_FIFTH: DissonanceLevel.CONSONANT,
            IntervalType.MINOR_SIXTH: DissonanceLevel.CONSONANT,
            IntervalType.MAJOR_SIXTH: DissonanceLevel.CONSONANT,
            IntervalType.MINOR_SEVENTH: DissonanceLevel.MODERATE_DISSONANT,
            IntervalType.MAJOR_SEVENTH: DissonanceLevel.STRONG_DISSONANT,
            IntervalType.OCTAVE: DissonanceLevel.CONSONANT,
        }
        
        # Chord progression patterns (simplified functional harmony)
        self.functional_progressions = {
            'authentic_cadence': ['V', 'I'],
            'plagal_cadence': ['IV', 'I'],
            'deceptive_cadence': ['V', 'vi'],
            'circle_of_fifths': ['vi', 'ii', 'V', 'I'],
            'common_pop': ['I', 'V', 'vi', 'IV']
        }

    def validate(self, music_element: MusicElement, context: ValidationContext) -> ValidationResult:
        """
        Main validation entry point
        
        Args:
            music_element: Music data to validate
            context: Validation context and settings
            
        Returns:
            ValidationResult with harmony assessment
        """
        harmony_scores = []
        conflicts = []
        suggestions = []
        
        # Analyze different harmony aspects
        if hasattr(music_element, 'notes') and music_element.notes:
            interval_result = self.validate_intervals(music_element.notes)
            harmony_scores.append(interval_result.confidence_score)
            
        if hasattr(music_element, 'chords') and music_element.chords:
            chord_result = self.validate_chord_progression(music_element.chords)
            harmony_scores.append(chord_result.theory_compliance)
            
        if hasattr(music_element, 'voices') and music_element.voices:
            voice_result = self.analyze_voice_leading(music_element.voices)
            harmony_scores.append(voice_result.overall_quality)
            
        if hasattr(music_element, 'harmonic_rhythm') and music_element.harmonic_rhythm:
            rhythm_result = self.validate_harmonic_rhythm(music_element.harmonic_rhythm)
            harmony_scores.append(rhythm_result.overall_rhythm_quality)
        
        # Calculate overall harmony score
        overall_score = np.mean(harmony_scores) if harmony_scores else 0.5
        
        # Generate quality score
        quality_score = QualityScore(
            technical_score=overall_score,
            artistic_score=overall_score * 0.9,  # Slightly lower artistic weight
            emotional_score=overall_score * 0.8,  # Harmony affects emotion moderately
            overall_score=overall_score,
            confidence_level=min(overall_score + 0.1, 1.0)
        )
        
        return ValidationResult(
            validator_name=self.name,
            overall_score=overall_score,
            passed=overall_score >= 0.7,
            detailed_scores={'harmony_score': overall_score},
            issues=[],  # Convert conflicts to issues if needed
            metadata={
                'harmony_aspects_analyzed': len(harmony_scores),
                'individual_scores': harmony_scores,
                'quality_score': quality_score.__dict__,
                'conflicts': [c.value for c in conflicts],
                'suggestions': suggestions
            }
        )

    def validate_intervals(self, notes: List[Dict]) -> IntervalValidationResult:
        """
        Analyze intervals between notes for dissonance and resolution needs
        
        Args:
            notes: List of note dictionaries with pitch information
            
        Returns:
            IntervalValidationResult with detailed analysis
        """
        if len(notes) < 2:
            return IntervalValidationResult(
                interval_type=IntervalType.UNISON,
                dissonance_level=DissonanceLevel.CONSONANT,
                requires_resolution=False,
                resolution_suggestions=[],
                confidence_score=1.0
            )
        
        dissonance_scores = []
        resolution_suggestions = []
        
        for i in range(len(notes) - 1):
            current_note = notes[i]
            next_note = notes[i + 1]
            
            # Calculate interval (simplified - assumes MIDI note numbers)
            interval_semitones = abs(current_note.get('pitch', 60) - next_note.get('pitch', 60)) % 12
            interval_type = IntervalType(interval_semitones)
            
            # Classify dissonance
            dissonance_level = self.dissonance_map[interval_type]
            
            # Calculate dissonance score (lower is more dissonant)
            dissonance_score = self._calculate_dissonance_score(dissonance_level)
            dissonance_scores.append(dissonance_score)
            
            # Generate resolution suggestions for dissonances
            if dissonance_level in [DissonanceLevel.MODERATE_DISSONANT, DissonanceLevel.STRONG_DISSONANT]:
                suggestions = self._generate_interval_resolution_suggestions(interval_type)
                resolution_suggestions.extend(suggestions)
        
        # Determine overall interval quality
        avg_dissonance_score = np.mean(dissonance_scores)
        most_dissonant_interval = IntervalType(
            np.argmin([self._calculate_dissonance_score(self.dissonance_map[IntervalType(i % 12)]) 
                      for i in range(len(notes) - 1)])
        )
        
        requires_resolution = avg_dissonance_score < 0.6
        
        return IntervalValidationResult(
            interval_type=most_dissonant_interval,
            dissonance_level=self.dissonance_map[most_dissonant_interval],
            requires_resolution=requires_resolution,
            resolution_suggestions=resolution_suggestions,
            confidence_score=avg_dissonance_score
        )

    def validate_chord_progression(self, chords: List[str]) -> ChordProgressionResult:
        """
        Validate chord progression for music theory compliance
        
        Args:
            chords: List of chord symbols (e.g., ['C', 'Am', 'F', 'G'])
            
        Returns:
            ChordProgressionResult with detailed analysis
        """
        if len(chords) < 2:
            return ChordProgressionResult(
                functional_harmony_score=0.5,
                voice_leading_score=0.5,
                resolution_quality=0.5,
                theory_compliance=0.5,
                identified_issues=["Insufficient chord progression length"],
                improvement_suggestions=["Add more chords for meaningful progression"]
            )
        
        # Analyze functional harmony
        functional_score = self._analyze_functional_harmony(chords)
        
        # Analyze voice leading quality
        voice_leading_score = self._analyze_chord_voice_leading(chords)
        
        # Analyze resolution quality
        resolution_score = self._analyze_chord_resolutions(chords)
        
        # Calculate theory compliance
        theory_score = (functional_score + voice_leading_score + resolution_score) / 3
        
        # Identify issues
        issues = []
        suggestions = []
        
        if functional_score < 0.6:
            issues.append("Weak functional harmony progression")
            suggestions.append("Consider using stronger tonal relationships (I-V-I, ii-V-I)")
            
        if voice_leading_score < 0.6:
            issues.append("Poor voice leading between chords")
            suggestions.append("Minimize voice movement and avoid parallel fifths/octaves")
            
        if resolution_score < 0.6:
            issues.append("Weak chord resolutions")
            suggestions.append("Strengthen cadential passages and dominant resolutions")
        
        return ChordProgressionResult(
            functional_harmony_score=functional_score,
            voice_leading_score=voice_leading_score,
            resolution_quality=resolution_score,
            theory_compliance=theory_score,
            identified_issues=issues,
            improvement_suggestions=suggestions
        )

    def analyze_voice_leading(self, voices: List[List[Dict]]) -> VoiceLeadingResult:
        """
        Analyze voice leading for smoothness and proper motion
        
        Args:
            voices: List of voice parts, each containing note dictionaries
            
        Returns:
            VoiceLeadingResult with detailed analysis
        """
        if len(voices) < 2:
            return VoiceLeadingResult(
                smoothness_score=0.5,
                parallel_motion_issues=[],
                contrary_motion_score=0.5,
                leap_analysis={},
                overall_quality=0.5
            )
        
        # Analyze smoothness (minimal voice movement)
        smoothness_score = self._calculate_voice_smoothness(voices)
        
        # Detect parallel motion issues
        parallel_issues = self._detect_parallel_motion(voices)
        
        # Analyze contrary motion
        contrary_score = self._analyze_contrary_motion(voices)
        
        # Analyze leaps in individual voices
        leap_analysis = self._analyze_voice_leaps(voices)
        
        # Calculate overall quality
        overall_quality = (
            smoothness_score * 0.4 +
            (1.0 - len(parallel_issues) * 0.1) * 0.3 +  # Penalty for parallel issues
            contrary_score * 0.2 +
            np.mean(list(leap_analysis.values())) * 0.1 if leap_analysis else 0.0
        )
        
        return VoiceLeadingResult(
            smoothness_score=smoothness_score,
            parallel_motion_issues=parallel_issues,
            contrary_motion_score=contrary_score,
            leap_analysis=leap_analysis,
            overall_quality=min(max(overall_quality, 0.0), 1.0)
        )

    def validate_harmonic_rhythm(self, harmonic_changes: List[Dict]) -> HarmonicRhythmResult:
        """
        Validate harmonic rhythm for appropriate change frequency and phrasing
        
        Args:
            harmonic_changes: List of harmonic change events with timing
            
        Returns:
            HarmonicRhythmResult with detailed analysis
        """
        if not harmonic_changes:
            return HarmonicRhythmResult(
                change_frequency_score=0.5,
                rhythmic_consistency=0.5,
                phrasing_alignment=0.5,
                cadential_strength=0.5,
                overall_rhythm_quality=0.5
            )
        
        # Analyze change frequency
        frequency_score = self._analyze_harmonic_change_frequency(harmonic_changes)
        
        # Analyze rhythmic consistency
        consistency_score = self._analyze_rhythmic_consistency(harmonic_changes)
        
        # Analyze phrasing alignment
        phrasing_score = self._analyze_phrasing_alignment(harmonic_changes)
        
        # Analyze cadential strength
        cadential_score = self._analyze_cadential_strength(harmonic_changes)
        
        # Calculate overall harmonic rhythm quality
        overall_quality = np.mean([frequency_score, consistency_score, phrasing_score, cadential_score])
        
        return HarmonicRhythmResult(
            change_frequency_score=frequency_score,
            rhythmic_consistency=consistency_score,
            phrasing_alignment=phrasing_score,
            cadential_strength=cadential_score,
            overall_rhythm_quality=overall_quality
        )

    # Helper methods for internal calculations
    
    def _calculate_dissonance_score(self, dissonance_level: DissonanceLevel) -> float:
        """Convert dissonance level to numerical score"""
        scores = {
            DissonanceLevel.CONSONANT: 1.0,
            DissonanceLevel.MILD_DISSONANT: 0.8,
            DissonanceLevel.MODERATE_DISSONANT: 0.5,
            DissonanceLevel.STRONG_DISSONANT: 0.2
        }
        return scores[dissonance_level]

    def _generate_interval_resolution_suggestions(self, interval_type: IntervalType) -> List[str]:
        """Generate resolution suggestions for dissonant intervals"""
        suggestions = {
            IntervalType.MINOR_SECOND: ["Resolve down by step", "Move to unison or third"],
            IntervalType.MAJOR_SEVENTH: ["Resolve up by step to octave", "Move down to sixth"],
            IntervalType.TRITONE: ["Resolve outward to sixth", "Resolve inward to third"],
            IntervalType.PERFECT_FOURTH: ["Resolve down to third", "Move to fifth or sixth"]
        }
        return suggestions.get(interval_type, [])

    def _analyze_functional_harmony(self, chords: List[str]) -> float:
        """Analyze functional harmony strength of chord progression"""
        # Simplified analysis - check for common progressions
        progression_str = '-'.join(chords)
        
        score = 0.5  # Base score
        
        # Check for common functional progressions
        for pattern_name, pattern in self.functional_progressions.items():
            pattern_str = '-'.join(pattern)
            if pattern_str in progression_str:
                score += 0.2
        
        # Check for tonic-dominant relationships
        if any(chord in ['V', 'V7'] for chord in chords) and 'I' in chords:
            score += 0.2
            
        return min(score, 1.0)

    def _analyze_chord_voice_leading(self, chords: List[str]) -> float:
        """Analyze voice leading quality between chords"""
        # Simplified analysis based on chord relationships
        score = 0.7  # Base score for reasonable voice leading
        
        for i in range(len(chords) - 1):
            current = chords[i]
            next_chord = chords[i + 1]
            
            # Penalize distant chord relationships
            if self._chords_are_distant(current, next_chord):
                score -= 0.1
        
        return max(score, 0.0)

    def _analyze_chord_resolutions(self, chords: List[str]) -> float:
        """Analyze quality of chord resolutions"""
        score = 0.6  # Base score
        
        for i in range(len(chords) - 1):
            current = chords[i]
            next_chord = chords[i + 1]
            
            # Strong resolutions
            if (current == 'V' and next_chord == 'I') or (current == 'V7' and next_chord == 'I'):
                score += 0.2
            elif current == 'IV' and next_chord == 'I':
                score += 0.1
        
        return min(score, 1.0)

    def _chords_are_distant(self, chord1: str, chord2: str) -> bool:
        """Simple check if chords are harmonically distant"""
        # Simplified - could be enhanced with actual harmonic distance calculation
        distant_pairs = [('C', 'F#'), ('G', 'Db'), ('D', 'Ab')]
        return (chord1, chord2) in distant_pairs or (chord2, chord1) in distant_pairs

    def _calculate_voice_smoothness(self, voices: List[List[Dict]]) -> float:
        """Calculate smoothness of voice movement"""
        if not voices or len(voices[0]) < 2:
            return 0.5
        
        total_movement = 0
        movement_count = 0
        
        for voice in voices:
            for i in range(len(voice) - 1):
                current_pitch = voice[i].get('pitch', 60)
                next_pitch = voice[i + 1].get('pitch', 60)
                movement = abs(next_pitch - current_pitch)
                total_movement += movement
                movement_count += 1
        
        avg_movement = total_movement / movement_count if movement_count > 0 else 0
        
        # Score based on average movement (smaller is better)
        if avg_movement <= 2:  # Stepwise motion
            return 1.0
        elif avg_movement <= 4:  # Small leaps
            return 0.8
        elif avg_movement <= 7:  # Medium leaps
            return 0.6
        else:  # Large leaps
            return 0.3

    def _detect_parallel_motion(self, voices: List[List[Dict]]) -> List[Tuple[int, int]]:
        """Detect parallel fifths and octaves between voices"""
        parallel_issues = []
        
        if len(voices) < 2:
            return parallel_issues
        
        for i in range(len(voices)):
            for j in range(i + 1, len(voices)):
                voice1 = voices[i]
                voice2 = voices[j]
                
                min_length = min(len(voice1), len(voice2))
                
                for k in range(min_length - 1):
                    interval1 = abs(voice1[k].get('pitch', 60) - voice2[k].get('pitch', 60)) % 12
                    interval2 = abs(voice1[k + 1].get('pitch', 60) - voice2[k + 1].get('pitch', 60)) % 12
                    
                    # Check for parallel fifths (7 semitones) or octaves (0 semitones)
                    if (interval1 == interval2) and (interval1 in [0, 7]):
                        parallel_issues.append((i, j))
        
        return parallel_issues

    def _analyze_contrary_motion(self, voices: List[List[Dict]]) -> float:
        """Analyze the presence of contrary motion between voices"""
        if len(voices) < 2:
            return 0.5
        
        contrary_motion_count = 0
        total_motion_pairs = 0
        
        for i in range(len(voices)):
            for j in range(i + 1, len(voices)):
                voice1 = voices[i]
                voice2 = voices[j]
                
                min_length = min(len(voice1), len(voice2))
                
                for k in range(min_length - 1):
                    pitch1_current = voice1[k].get('pitch', 60)
                    pitch1_next = voice1[k + 1].get('pitch', 60)
                    pitch2_current = voice2[k].get('pitch', 60)
                    pitch2_next = voice2[k + 1].get('pitch', 60)
                    
                    direction1 = pitch1_next - pitch1_current
                    direction2 = pitch2_next - pitch2_current
                    
                    if direction1 * direction2 < 0:  # Opposite directions
                        contrary_motion_count += 1
                    
                    total_motion_pairs += 1
        
        return contrary_motion_count / total_motion_pairs if total_motion_pairs > 0 else 0.5

    def _analyze_voice_leaps(self, voices: List[List[Dict]]) -> Dict[str, float]:
        """Analyze leaps in individual voices"""
        leap_analysis = {}
        
        for i, voice in enumerate(voices):
            if len(voice) < 2:
                leap_analysis[f'voice_{i}'] = 0.5
                continue
            
            large_leaps = 0
            total_intervals = 0
            
            for j in range(len(voice) - 1):
                current_pitch = voice[j].get('pitch', 60)
                next_pitch = voice[j + 1].get('pitch', 60)
                interval = abs(next_pitch - current_pitch)
                
                if interval > 7:  # Leap larger than perfect fifth
                    large_leaps += 1
                
                total_intervals += 1
            
            # Score based on proportion of large leaps (fewer is better)
            leap_ratio = large_leaps / total_intervals if total_intervals > 0 else 0
            leap_analysis[f'voice_{i}'] = max(0.0, 1.0 - leap_ratio * 2)
        
        return leap_analysis

    def _analyze_harmonic_change_frequency(self, harmonic_changes: List[Dict]) -> float:
        """Analyze the frequency of harmonic changes"""
        if len(harmonic_changes) < 2:
            return 0.5
        
        # Calculate average time between changes
        time_intervals = []
        for i in range(len(harmonic_changes) - 1):
            current_time = harmonic_changes[i].get('time', 0)
            next_time = harmonic_changes[i + 1].get('time', 0)
            interval = next_time - current_time
            time_intervals.append(interval)
        
        avg_interval = np.mean(time_intervals)
        
        # Optimal harmonic rhythm is around 1-2 beats per change
        if 0.5 <= avg_interval <= 2.0:
            return 1.0
        elif 0.25 <= avg_interval <= 4.0:
            return 0.8
        else:
            return 0.5

    def _analyze_rhythmic_consistency(self, harmonic_changes: List[Dict]) -> float:
        """Analyze consistency of harmonic rhythm patterns"""
        if len(harmonic_changes) < 3:
            return 0.5
        
        # Calculate variance in timing intervals
        time_intervals = []
        for i in range(len(harmonic_changes) - 1):
            current_time = harmonic_changes[i].get('time', 0)
            next_time = harmonic_changes[i + 1].get('time', 0)
            interval = next_time - current_time
            time_intervals.append(interval)
        
        variance = np.var(time_intervals) if time_intervals else 0
        
        # Lower variance indicates more consistent rhythm
        if variance < 0.1:
            return 1.0
        elif variance < 0.5:
            return 0.8
        elif variance < 1.0:
            return 0.6
        else:
            return 0.4

    def _analyze_phrasing_alignment(self, harmonic_changes: List[Dict]) -> float:
        """Analyze alignment of harmonic changes with musical phrases"""
        # Simplified analysis - assumes changes on strong beats are better
        strong_beat_changes = 0
        
        for change in harmonic_changes:
            beat_position = change.get('beat_position', 0)
            if beat_position in [0, 2]:  # Strong beats in 4/4 time
                strong_beat_changes += 1
        
        alignment_ratio = strong_beat_changes / len(harmonic_changes) if harmonic_changes else 0
        return alignment_ratio

    def _analyze_cadential_strength(self, harmonic_changes: List[Dict]) -> float:
        """Analyze the strength of cadential passages"""
        cadential_score = 0.5  # Base score
        
        # Look for cadential patterns in the harmonic changes
        for i in range(len(harmonic_changes) - 1):
            current_chord = harmonic_changes[i].get('chord', '')
            next_chord = harmonic_changes[i + 1].get('chord', '')
            
            # Strong cadences
            if current_chord == 'V' and next_chord == 'I':
                cadential_score += 0.3
            elif current_chord == 'IV' and next_chord == 'I':
                cadential_score += 0.2
        
        return min(cadential_score, 1.0)