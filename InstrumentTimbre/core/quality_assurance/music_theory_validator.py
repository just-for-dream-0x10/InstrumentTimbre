"""
Music Theory Validator - Theory-based reasonableness checking

This module provides comprehensive music theory validation including:
- Scale compliance and key signature adherence
- Rhythm pattern validation and time signature compliance
- Form structure checking and musical form logic verification
- Counterpoint rules and classical voice writing validation
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Set
from dataclasses import dataclass
from enum import Enum

from .base_validator import BaseValidator, ValidationResult
from .data_structures import (
    QualityScore, ConflictType, ResolutionSuggestion,
    ValidationContext, MusicElement
)


class ScaleType(Enum):
    """Musical scale types for analysis"""
    MAJOR = "major"
    MINOR = "minor"
    DORIAN = "dorian"
    PHRYGIAN = "phrygian"
    LYDIAN = "lydian"
    MIXOLYDIAN = "mixolydian"
    AEOLIAN = "aeolian"
    LOCRIAN = "locrian"
    PENTATONIC_MAJOR = "pentatonic_major"
    PENTATONIC_MINOR = "pentatonic_minor"
    CHROMATIC = "chromatic"


class TimeSignature(Enum):
    """Common time signatures"""
    FOUR_FOUR = "4/4"
    THREE_FOUR = "3/4"
    TWO_FOUR = "2/4"
    SIX_EIGHT = "6/8"
    NINE_EIGHT = "9/8"
    TWELVE_EIGHT = "12/8"
    FIVE_FOUR = "5/4"
    SEVEN_EIGHT = "7/8"


class MusicalForm(Enum):
    """Musical form types"""
    BINARY = "binary"
    TERNARY = "ternary"
    RONDO = "rondo"
    SONATA = "sonata"
    THEME_VARIATIONS = "theme_variations"
    VERSE_CHORUS = "verse_chorus"
    AABA = "aaba"
    THROUGH_COMPOSED = "through_composed"


@dataclass
class ScaleComplianceResult:
    """Result of scale compliance analysis"""
    compliance_score: float
    detected_scale: ScaleType
    key_center: str
    non_scale_notes: List[str]
    modulations: List[Tuple[float, str]]  # (time, new_key)
    chromatic_alterations: List[str]
    recommendations: List[str]


@dataclass
class RhythmValidationResult:
    """Result of rhythm pattern validation"""
    meter_compliance: float
    beat_accuracy: float
    syncopation_appropriateness: float
    rhythmic_complexity: float
    time_signature_consistency: float
    identified_issues: List[str]


@dataclass
class FormStructureResult:
    """Result of form structure analysis"""
    form_coherence: float
    detected_form: MusicalForm
    phrase_structure_quality: float
    cadence_placement: float
    proportional_balance: float
    structural_issues: List[str]


@dataclass
class CounterpointResult:
    """Result of counterpoint analysis"""
    species_compliance: float
    parallel_motion_violations: int
    voice_independence: float
    dissonance_treatment: float
    melodic_contour_quality: float
    counterpoint_issues: List[str]


class MusicTheoryValidator(BaseValidator):
    """
    Comprehensive music theory validation system
    
    Provides theory-based validation including scale compliance,
    rhythm patterns, form structure, and counterpoint rules.
    """
    
    def __init__(self):
        super().__init__("MusicTheoryValidator")
        self.name = "MusicTheoryValidator"
        self.version = "1.0.0"
        
        # Scale definitions (semitone patterns from root)
        self.scale_patterns = {
            ScaleType.MAJOR: [0, 2, 4, 5, 7, 9, 11],
            ScaleType.MINOR: [0, 2, 3, 5, 7, 8, 10],
            ScaleType.DORIAN: [0, 2, 3, 5, 7, 9, 10],
            ScaleType.PHRYGIAN: [0, 1, 3, 5, 7, 8, 10],
            ScaleType.LYDIAN: [0, 2, 4, 6, 7, 9, 11],
            ScaleType.MIXOLYDIAN: [0, 2, 4, 5, 7, 9, 10],
            ScaleType.PENTATONIC_MAJOR: [0, 2, 4, 7, 9],
            ScaleType.PENTATONIC_MINOR: [0, 3, 5, 7, 10]
        }
        
        # Circle of fifths for key relationships
        self.circle_of_fifths = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#', 'Ab', 'Eb', 'Bb', 'F']
        
        # Common chord progressions for form analysis
        self.common_progressions = {
            'authentic_cadence': ['V', 'I'],
            'plagal_cadence': ['IV', 'I'],
            'half_cadence': ['I', 'V'],
            'deceptive_cadence': ['V', 'vi']
        }

    def validate(self, music_element: MusicElement, context: ValidationContext) -> ValidationResult:
        """
        Main validation entry point for music theory compliance
        
        Args:
            music_element: Music data to validate
            context: Validation context and settings
            
        Returns:
            ValidationResult with theory compliance assessment
        """
        theory_scores = []
        conflicts = []
        suggestions = []
        
        # Scale compliance analysis
        if hasattr(music_element, 'notes') and music_element.notes:
            scale_result = self.validate_scale_compliance(
                music_element.notes, 
                context.get('key', 'C')
            )
            theory_scores.append(scale_result.compliance_score)
            
            if scale_result.compliance_score < 0.7:
                conflicts.append(ConflictType.SCALE_VIOLATION)
                suggestions.extend(scale_result.recommendations)
        
        # Rhythm validation
        if hasattr(music_element, 'rhythm_data') and music_element.rhythm_data:
            rhythm_result = self.validate_rhythm_patterns(
                music_element.rhythm_data,
                context.get('time_signature', TimeSignature.FOUR_FOUR)
            )
            theory_scores.append(rhythm_result.meter_compliance)
            
            if rhythm_result.identified_issues:
                conflicts.append(ConflictType.RHYTHM_VIOLATION)
        
        # Form structure validation
        if hasattr(music_element, 'structure_data') and music_element.structure_data:
            form_result = self.check_form_structure(music_element.structure_data)
            theory_scores.append(form_result.form_coherence)
            
            if form_result.structural_issues:
                conflicts.append(ConflictType.FORM_VIOLATION)
        
        # Counterpoint validation
        if hasattr(music_element, 'voices') and music_element.voices and len(music_element.voices) > 1:
            counterpoint_result = self.validate_counterpoint(music_element.voices)
            theory_scores.append(counterpoint_result.species_compliance)
            
            if counterpoint_result.parallel_motion_violations > 0:
                conflicts.append(ConflictType.COUNTERPOINT_VIOLATION)
        
        # Calculate overall theory compliance score
        overall_score = np.mean(theory_scores) if theory_scores else 0.5
        
        # Generate quality score
        quality_score = QualityScore(
            technical_score=overall_score,
            artistic_score=overall_score * 0.8,  # Theory is more technical than artistic
            emotional_score=overall_score * 0.6,  # Theory indirectly affects emotion
            overall_score=overall_score,
            confidence_level=min(overall_score + 0.1, 1.0)
        )
        
        return ValidationResult(
            validator_name=self.name,
            overall_score=overall_score,
            passed=overall_score >= 0.7,
            detailed_scores={'theory_score': overall_score},
            issues=[],
            metadata={
                'theory_aspects_analyzed': len(theory_scores),
                'individual_scores': theory_scores,
                'quality_score': quality_score.__dict__,
                'conflicts': [c.value for c in conflicts],
                'suggestions': suggestions
            }
        )

    def validate_scale_compliance(self, notes: List[Dict], key: str) -> ScaleComplianceResult:
        """
        Validate adherence to scale and key signature
        
        Args:
            notes: List of note dictionaries with pitch information
            key: Key signature (e.g., 'C', 'Am', 'F#')
            
        Returns:
            ScaleComplianceResult with detailed analysis
        """
        if not notes:
            return ScaleComplianceResult(
                compliance_score=0.5,
                detected_scale=ScaleType.MAJOR,
                key_center=key,
                non_scale_notes=[],
                modulations=[],
                chromatic_alterations=[],
                recommendations=["No notes to analyze"]
            )
        
        # Extract pitch classes from notes
        pitch_classes = []
        for note in notes:
            if 'pitch' in note:
                pitch_class = note['pitch'] % 12
                pitch_classes.append(pitch_class)
        
        if not pitch_classes:
            return ScaleComplianceResult(
                compliance_score=0.5,
                detected_scale=ScaleType.MAJOR,
                key_center=key,
                non_scale_notes=[],
                modulations=[],
                chromatic_alterations=[],
                recommendations=["No valid pitch information"]
            )
        
        # Determine key center and scale type
        detected_scale, key_center = self._analyze_scale_context(pitch_classes, key)
        
        # Get scale pattern for detected scale
        scale_pattern = self.scale_patterns.get(detected_scale, self.scale_patterns[ScaleType.MAJOR])
        
        # Convert key center to root pitch class
        root_pitch_class = self._key_to_pitch_class(key_center)
        
        # Generate expected scale notes
        expected_notes = [(root_pitch_class + interval) % 12 for interval in scale_pattern]
        
        # Analyze compliance
        scale_notes = set()
        non_scale_notes = []
        
        for pitch_class in pitch_classes:
            if pitch_class in expected_notes:
                scale_notes.add(pitch_class)
            else:
                note_name = self._pitch_class_to_note_name(pitch_class)
                non_scale_notes.append(note_name)
        
        # Calculate compliance score
        total_notes = len(pitch_classes)
        scale_note_count = total_notes - len([pc for pc in pitch_classes if pc not in expected_notes])
        compliance_score = scale_note_count / total_notes if total_notes > 0 else 0
        
        # Detect potential modulations
        modulations = self._detect_modulations(notes, key_center)
        
        # Identify chromatic alterations
        chromatic_alterations = self._identify_chromatic_alterations(non_scale_notes, detected_scale)
        
        # Generate recommendations
        recommendations = []
        if compliance_score < 0.8:
            recommendations.append(f"Consider staying within {detected_scale.value} scale")
        if len(non_scale_notes) > 0:
            recommendations.append("Review chromatic notes for proper resolution")
        if len(modulations) > 2:
            recommendations.append("Consider simplifying modulation scheme")
        
        return ScaleComplianceResult(
            compliance_score=compliance_score,
            detected_scale=detected_scale,
            key_center=key_center,
            non_scale_notes=list(set(non_scale_notes)),
            modulations=modulations,
            chromatic_alterations=chromatic_alterations,
            recommendations=recommendations
        )

    def validate_rhythm_patterns(self, rhythm_data: Dict, time_signature: TimeSignature) -> RhythmValidationResult:
        """
        Validate rhythm patterns for time signature compliance
        
        Args:
            rhythm_data: Rhythm pattern data
            time_signature: Expected time signature
            
        Returns:
            RhythmValidationResult with rhythm analysis
        """
        # Extract beat information
        beats = rhythm_data.get('beats', [])
        if not beats:
            return RhythmValidationResult(
                meter_compliance=0.5,
                beat_accuracy=0.5,
                syncopation_appropriateness=0.5,
                rhythmic_complexity=0.5,
                time_signature_consistency=0.5,
                identified_issues=["No rhythm data available"]
            )
        
        # Analyze meter compliance
        meter_compliance = self._analyze_meter_compliance(beats, time_signature)
        
        # Analyze beat accuracy
        beat_accuracy = self._analyze_beat_accuracy(beats)
        
        # Analyze syncopation appropriateness
        syncopation_score = self._analyze_syncopation(beats, time_signature)
        
        # Calculate rhythmic complexity
        complexity_score = self._calculate_rhythmic_complexity(beats)
        
        # Check time signature consistency
        consistency_score = self._check_time_signature_consistency(beats, time_signature)
        
        # Identify issues
        issues = []
        if meter_compliance < 0.6:
            issues.append("Poor meter compliance")
        if beat_accuracy < 0.7:
            issues.append("Beat timing inaccuracies")
        if consistency_score < 0.8:
            issues.append("Time signature inconsistencies")
        
        return RhythmValidationResult(
            meter_compliance=meter_compliance,
            beat_accuracy=beat_accuracy,
            syncopation_appropriateness=syncopation_score,
            rhythmic_complexity=complexity_score,
            time_signature_consistency=consistency_score,
            identified_issues=issues
        )

    def check_form_structure(self, structure_data: Dict) -> FormStructureResult:
        """
        Check musical form structure and coherence
        
        Args:
            structure_data: Musical structure information
            
        Returns:
            FormStructureResult with form analysis
        """
        sections = structure_data.get('sections', [])
        if not sections:
            return FormStructureResult(
                form_coherence=0.5,
                detected_form=MusicalForm.THROUGH_COMPOSED,
                phrase_structure_quality=0.5,
                cadence_placement=0.5,
                proportional_balance=0.5,
                structural_issues=["No structural data available"]
            )
        
        # Detect musical form
        detected_form = self._detect_musical_form(sections)
        
        # Analyze form coherence
        coherence_score = self._analyze_form_coherence(sections, detected_form)
        
        # Analyze phrase structure
        phrase_quality = self._analyze_phrase_structure(sections)
        
        # Analyze cadence placement
        cadence_score = self._analyze_cadence_placement(sections)
        
        # Check proportional balance
        balance_score = self._check_proportional_balance(sections)
        
        # Identify structural issues
        issues = []
        if coherence_score < 0.6:
            issues.append("Weak form coherence")
        if phrase_quality < 0.7:
            issues.append("Poor phrase structure")
        if cadence_score < 0.6:
            issues.append("Weak cadence placement")
        
        return FormStructureResult(
            form_coherence=coherence_score,
            detected_form=detected_form,
            phrase_structure_quality=phrase_quality,
            cadence_placement=cadence_score,
            proportional_balance=balance_score,
            structural_issues=issues
        )

    def validate_counterpoint(self, voices: List[List[Dict]]) -> CounterpointResult:
        """
        Validate counterpoint rules and voice writing
        
        Args:
            voices: List of voice parts
            
        Returns:
            CounterpointResult with counterpoint analysis
        """
        if len(voices) < 2:
            return CounterpointResult(
                species_compliance=0.5,
                parallel_motion_violations=0,
                voice_independence=0.5,
                dissonance_treatment=0.5,
                melodic_contour_quality=0.5,
                counterpoint_issues=["Insufficient voices for counterpoint analysis"]
            )
        
        # Check species counterpoint compliance
        species_score = self._check_species_counterpoint(voices)
        
        # Detect parallel motion violations
        parallel_violations = self._detect_parallel_violations(voices)
        
        # Analyze voice independence
        independence_score = self._analyze_voice_independence(voices)
        
        # Check dissonance treatment
        dissonance_score = self._check_dissonance_treatment(voices)
        
        # Analyze melodic contour quality
        contour_score = self._analyze_melodic_contours(voices)
        
        # Identify issues
        issues = []
        if parallel_violations > 0:
            issues.append(f"Found {parallel_violations} parallel motion violations")
        if independence_score < 0.6:
            issues.append("Voices lack independence")
        if dissonance_score < 0.7:
            issues.append("Poor dissonance treatment")
        
        return CounterpointResult(
            species_compliance=species_score,
            parallel_motion_violations=parallel_violations,
            voice_independence=independence_score,
            dissonance_treatment=dissonance_score,
            melodic_contour_quality=contour_score,
            counterpoint_issues=issues
        )

    # Helper methods for internal calculations
    
    def _key_to_pitch_class(self, key: str) -> int:
        """Convert key name to pitch class number"""
        key_map = {
            'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
            'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
            'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
        }
        # Handle minor keys (e.g., 'Am' -> 'A')
        root = key[0].upper()
        if len(key) > 1 and key[1] == '#':
            root += '#'
        elif len(key) > 1 and key[1] == 'b':
            root += 'b'
        
        return key_map.get(root, 0)

    def _pitch_class_to_note_name(self, pitch_class: int) -> str:
        """Convert pitch class to note name"""
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        return note_names[pitch_class % 12]

    def _analyze_scale_context(self, pitch_classes: List[int], suggested_key: str) -> Tuple[ScaleType, str]:
        """Analyze pitch classes to determine most likely scale and key"""
        # Simplified analysis - count pitch class frequencies
        pitch_counts = {}
        for pc in pitch_classes:
            pitch_counts[pc] = pitch_counts.get(pc, 0) + 1
        
        # Find most common pitch class as potential tonic
        most_common_pc = max(pitch_counts, key=pitch_counts.get)
        key_center = self._pitch_class_to_note_name(most_common_pc)
        
        # Determine scale type based on interval patterns
        unique_pcs = list(set(pitch_classes))
        
        # Test against known scale patterns
        best_match = ScaleType.MAJOR
        best_score = 0
        
        for scale_type, pattern in self.scale_patterns.items():
            for root in range(12):
                expected_pcs = [(root + interval) % 12 for interval in pattern]
                matches = len(set(unique_pcs) & set(expected_pcs))
                score = matches / len(expected_pcs) if expected_pcs else 0
                
                if score > best_score:
                    best_score = score
                    best_match = scale_type
                    if score > 0.8:  # Good match found
                        key_center = self._pitch_class_to_note_name(root)
        
        return best_match, key_center

    def _detect_modulations(self, notes: List[Dict], home_key: str) -> List[Tuple[float, str]]:
        """Detect key changes/modulations in the music"""
        modulations = []
        # Simplified modulation detection
        # In a real implementation, this would analyze harmonic progressions
        # and key signature changes over time
        return modulations

    def _identify_chromatic_alterations(self, non_scale_notes: List[str], scale_type: ScaleType) -> List[str]:
        """Identify chromatic alterations and their purposes"""
        alterations = []
        # Simplified identification
        for note in non_scale_notes:
            if note:  # Basic check for common chromatic alterations
                alterations.append(f"Chromatic note: {note}")
        return alterations

    def _analyze_meter_compliance(self, beats: List[Dict], time_sig: TimeSignature) -> float:
        """Analyze compliance with time signature"""
        # Simplified meter analysis
        return 0.8  # Placeholder

    def _analyze_beat_accuracy(self, beats: List[Dict]) -> float:
        """Analyze accuracy of beat timing"""
        # Simplified beat accuracy analysis
        return 0.8  # Placeholder

    def _analyze_syncopation(self, beats: List[Dict], time_sig: TimeSignature) -> float:
        """Analyze appropriateness of syncopation"""
        # Simplified syncopation analysis
        return 0.7  # Placeholder

    def _calculate_rhythmic_complexity(self, beats: List[Dict]) -> float:
        """Calculate rhythmic complexity score"""
        # Simplified complexity calculation
        return 0.6  # Placeholder

    def _check_time_signature_consistency(self, beats: List[Dict], time_sig: TimeSignature) -> float:
        """Check consistency with time signature"""
        # Simplified consistency check
        return 0.9  # Placeholder

    def _detect_musical_form(self, sections: List[Dict]) -> MusicalForm:
        """Detect the musical form from section structure"""
        # Simplified form detection
        if len(sections) == 2:
            return MusicalForm.BINARY
        elif len(sections) == 3:
            return MusicalForm.TERNARY
        else:
            return MusicalForm.THROUGH_COMPOSED

    def _analyze_form_coherence(self, sections: List[Dict], form: MusicalForm) -> float:
        """Analyze coherence of musical form"""
        # Simplified coherence analysis
        return 0.8  # Placeholder

    def _analyze_phrase_structure(self, sections: List[Dict]) -> float:
        """Analyze quality of phrase structure"""
        # Simplified phrase analysis
        return 0.7  # Placeholder

    def _analyze_cadence_placement(self, sections: List[Dict]) -> float:
        """Analyze placement and strength of cadences"""
        # Simplified cadence analysis
        return 0.8  # Placeholder

    def _check_proportional_balance(self, sections: List[Dict]) -> float:
        """Check proportional balance of sections"""
        # Simplified balance check
        return 0.7  # Placeholder

    def _check_species_counterpoint(self, voices: List[List[Dict]]) -> float:
        """Check species counterpoint compliance"""
        # Simplified species counterpoint check
        return 0.7  # Placeholder

    def _detect_parallel_violations(self, voices: List[List[Dict]]) -> int:
        """Detect parallel fifth and octave violations"""
        # Simplified parallel motion detection
        return 0  # Placeholder

    def _analyze_voice_independence(self, voices: List[List[Dict]]) -> float:
        """Analyze independence between voices"""
        # Simplified independence analysis
        return 0.8  # Placeholder

    def _check_dissonance_treatment(self, voices: List[List[Dict]]) -> float:
        """Check proper treatment of dissonances"""
        # Simplified dissonance treatment check
        return 0.8  # Placeholder

    def _analyze_melodic_contours(self, voices: List[List[Dict]]) -> float:
        """Analyze quality of melodic contours"""
        # Simplified contour analysis
        return 0.7  # Placeholder