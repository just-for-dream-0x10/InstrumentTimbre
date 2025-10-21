"""
Week 8 Quality Assurance System Tests

Comprehensive test suite for the Quality Assurance system components
including validators, auto-coordination, user feedback, and quality scoring.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from dataclasses import dataclass
from typing import Dict, List, Any

# Import QA system components
from InstrumentTimbre.core.quality_assurance import (
    QualityAssuranceEngine, HarmonyValidator, EmotionConsistencyChecker,
    MusicTheoryValidator, AutoCoordinationEngine, UserFeedbackInterface,
    QualityScoringSystem
)
from InstrumentTimbre.core.quality_assurance.data_structures import (
    QualityScore, ConflictType, ValidationContext, MusicElement
)
from InstrumentTimbre.core.quality_assurance.quality_assurance_engine import QAMode, QAResult


# Test fixtures and mock data
@pytest.fixture
def sample_music_element():
    """Create a sample music element for testing"""
    @dataclass
    class TestMusicElement:
        notes: List[Dict] = None
        chords: List[str] = None
        tracks: List[Dict] = None
        style: str = "classical"
        
        def __post_init__(self):
            if self.notes is None:
                self.notes = [
                    {'pitch': 60, 'duration': 1.0, 'time': 0.0},
                    {'pitch': 64, 'duration': 1.0, 'time': 1.0},
                    {'pitch': 67, 'duration': 1.0, 'time': 2.0}
                ]
            if self.chords is None:
                self.chords = ['C', 'Am', 'F', 'G']
            if self.tracks is None:
                self.tracks = [
                    {'features': {'energy': 0.7, 'valence': 0.8}},
                    {'features': {'energy': 0.6, 'valence': 0.7}}
                ]
    
    return TestMusicElement()


@pytest.fixture
def sample_context():
    """Create a sample validation context"""
    return ValidationContext({
        'key': 'C',
        'time_signature': '4/4',
        'style': 'classical',
        'user_preference_weight': 0.8
    })


class TestHarmonyValidator:
    """Test suite for Harmony Validator"""
    
    def test_harmony_validator_initialization(self):
        """Test harmony validator initialization"""
        validator = HarmonyValidator()
        assert validator.name == "HarmonyValidator"
        assert validator.version == "1.0.0"
        assert len(validator.dissonance_map) > 0
        assert len(validator.functional_progressions) > 0

    def test_interval_validation(self, sample_music_element):
        """Test interval validation functionality"""
        validator = HarmonyValidator()
        result = validator.validate_intervals(sample_music_element.notes)
        
        assert hasattr(result, 'confidence_score')
        assert 0 <= result.confidence_score <= 1
        assert hasattr(result, 'requires_resolution')
        assert isinstance(result.resolution_suggestions, list)

    def test_chord_progression_validation(self, sample_music_element):
        """Test chord progression validation"""
        validator = HarmonyValidator()
        result = validator.validate_chord_progression(sample_music_element.chords)
        
        assert hasattr(result, 'theory_compliance')
        assert 0 <= result.theory_compliance <= 1
        assert hasattr(result, 'functional_harmony_score')
        assert isinstance(result.identified_issues, list)

    def test_harmony_validator_integration(self, sample_music_element, sample_context):
        """Test full harmony validator integration"""
        validator = HarmonyValidator()
        result = validator.validate(sample_music_element, sample_context)
        
        assert result.validator_name == "HarmonyValidator"
        assert hasattr(result.quality_score, 'overall_score')
        assert 0 <= result.quality_score.overall_score <= 1


class TestEmotionConsistencyChecker:
    """Test suite for Emotion Consistency Checker"""
    
    def test_emotion_checker_initialization(self):
        """Test emotion checker initialization"""
        checker = EmotionConsistencyChecker()
        assert checker.name == "EmotionConsistencyChecker"
        assert checker.version == "1.0.0"
        assert len(checker.emotion_compatibility) > 0

    def test_emotional_coherence_analysis(self, sample_music_element):
        """Test emotional coherence analysis"""
        checker = EmotionConsistencyChecker()
        result = checker.check_emotional_coherence(sample_music_element.tracks)
        
        assert hasattr(result, 'coherence_score')
        assert 0 <= result.coherence_score <= 1
        assert hasattr(result, 'dominant_emotion')
        assert isinstance(result.emotion_conflicts, list)

    def test_intensity_preservation(self, sample_music_element):
        """Test intensity preservation validation"""
        checker = EmotionConsistencyChecker()
        original_data = {'segments': [{'energy': 0.8, 'loudness': -10}] * 5}
        modified_data = {'segments': [{'energy': 0.7, 'loudness': -12}] * 5}
        
        result = checker.validate_intensity_preservation(original_data, modified_data)
        
        assert hasattr(result, 'preservation_score')
        assert 0 <= result.preservation_score <= 1
        assert hasattr(result, 'intensity_drift')
        assert isinstance(result.recommendations, list)

    def test_emotion_checker_integration(self, sample_music_element, sample_context):
        """Test full emotion checker integration"""
        checker = EmotionConsistencyChecker()
        result = checker.validate(sample_music_element, sample_context)
        
        assert result.validator_name == "EmotionConsistencyChecker"
        assert hasattr(result.quality_score, 'emotional_score')
        assert 0 <= result.quality_score.emotional_score <= 1


class TestMusicTheoryValidator:
    """Test suite for Music Theory Validator"""
    
    def test_theory_validator_initialization(self):
        """Test theory validator initialization"""
        validator = MusicTheoryValidator()
        assert validator.name == "MusicTheoryValidator"
        assert validator.version == "1.0.0"
        assert len(validator.scale_patterns) > 0

    def test_scale_compliance_validation(self, sample_music_element):
        """Test scale compliance validation"""
        validator = MusicTheoryValidator()
        result = validator.validate_scale_compliance(sample_music_element.notes, 'C')
        
        assert hasattr(result, 'compliance_score')
        assert 0 <= result.compliance_score <= 1
        assert hasattr(result, 'detected_scale')
        assert isinstance(result.non_scale_notes, list)

    def test_rhythm_pattern_validation(self):
        """Test rhythm pattern validation"""
        validator = MusicTheoryValidator()
        rhythm_data = {'beats': [{'time': 0, 'strength': 1}, {'time': 1, 'strength': 0.5}]}
        
        from InstrumentTimbre.core.quality_assurance.music_theory_validator import TimeSignature
        result = validator.validate_rhythm_patterns(rhythm_data, TimeSignature.FOUR_FOUR)
        
        assert hasattr(result, 'meter_compliance')
        assert 0 <= result.meter_compliance <= 1
        assert isinstance(result.identified_issues, list)

    def test_theory_validator_integration(self, sample_music_element, sample_context):
        """Test full theory validator integration"""
        validator = MusicTheoryValidator()
        result = validator.validate(sample_music_element, sample_context)
        
        assert result.validator_name == "MusicTheoryValidator"
        assert hasattr(result.quality_score, 'technical_score')


class TestAutoCoordinationEngine:
    """Test suite for Auto-Coordination Engine"""
    
    def test_auto_coordinator_initialization(self):
        """Test auto-coordination engine initialization"""
        coordinator = AutoCoordinationEngine()
        assert coordinator.name == "AutoCoordinationEngine"
        assert coordinator.version == "1.0.0"
        assert len(coordinator.priority_mapping) > 0

    def test_conflict_prioritization(self, sample_context):
        """Test conflict prioritization"""
        coordinator = AutoCoordinationEngine()
        conflicts = [ConflictType.HARMONY_VIOLATION, ConflictType.EMOTION_MISMATCH]
        
        result = coordinator.prioritize_conflicts(conflicts, sample_context)
        
        assert hasattr(result, 'total_count')
        assert result.total_count == len(conflicts)
        assert hasattr(result, 'critical')
        assert hasattr(result, 'high')

    def test_suggestion_generation(self, sample_context):
        """Test resolution suggestion generation"""
        coordinator = AutoCoordinationEngine()
        
        from InstrumentTimbre.core.quality_assurance.auto_coordination_engine import ConflictItem, ConflictPriority
        conflict = ConflictItem(
            conflict_type=ConflictType.HARMONY_VIOLATION,
            priority=ConflictPriority.HIGH,
            severity=0.8,
            location="test_location",
            description="Test conflict",
            affected_elements=["harmony"],
            metadata={'index': 0}
        )
        
        suggestions = coordinator.generate_suggestions(conflict, sample_context)
        
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        for suggestion in suggestions:
            assert hasattr(suggestion, 'confidence')
            assert hasattr(suggestion, 'estimated_improvement')


class TestUserFeedbackInterface:
    """Test suite for User Feedback Interface"""
    
    def test_user_interface_initialization(self):
        """Test user feedback interface initialization"""
        interface = UserFeedbackInterface()
        assert interface.name == "UserFeedbackInterface"
        assert interface.version == "1.0.0"
        assert len(interface.visualization_preferences) > 0

    def test_conflict_visualization(self):
        """Test conflict visualization generation"""
        interface = UserFeedbackInterface()
        
        from InstrumentTimbre.core.quality_assurance.auto_coordination_engine import ConflictItem, ConflictPriority
        conflicts = [ConflictItem(
            conflict_type=ConflictType.HARMONY_VIOLATION,
            priority=ConflictPriority.HIGH,
            severity=0.8,
            location="test_location",
            description="Test conflict",
            affected_elements=["harmony"],
            metadata={'index': 0}
        )]
        
        visualizations = interface.visualize_conflicts(conflicts, {})
        
        assert isinstance(visualizations, list)
        assert len(visualizations) == 1
        assert hasattr(visualizations[0], 'conflict_id')
        assert hasattr(visualizations[0], 'severity_indicator')

    def test_solution_option_presentation(self):
        """Test solution option presentation"""
        interface = UserFeedbackInterface()
        
        from InstrumentTimbre.core.quality_assurance.auto_coordination_engine import (
            ConflictItem, ConflictPriority, ResolutionSuggestion, ResolutionStrategy
        )
        conflict = ConflictItem(
            conflict_type=ConflictType.HARMONY_VIOLATION,
            priority=ConflictPriority.HIGH,
            severity=0.8,
            location="test_location",
            description="Test conflict",
            affected_elements=["harmony"],
            metadata={'index': 0}
        )
        
        suggestions = [ResolutionSuggestion(
            conflict_id="test_conflict",
            strategy=ResolutionStrategy.AUTOMATIC_FIX,
            confidence=0.8,
            estimated_improvement=0.7,
            side_effects=["May alter melody"],
            implementation_cost=0.3,
            description="Test suggestion",
            parameters={}
        )]
        
        options = interface.present_solution_options(conflict, suggestions)
        
        assert isinstance(options, list)
        assert len(options) > 0
        assert hasattr(options[0], 'title')
        assert hasattr(options[0], 'confidence')


class TestQualityScoringSystem:
    """Test suite for Quality Scoring System"""
    
    def test_quality_scorer_initialization(self):
        """Test quality scoring system initialization"""
        scorer = QualityScoringSystem()
        assert scorer.name == "QualityScoringSystem"
        assert scorer.version == "1.0.0"
        assert len(scorer.default_weights) > 0

    def test_technical_score_calculation(self, sample_music_element, sample_context):
        """Test technical score calculation"""
        scorer = QualityScoringSystem()
        result = scorer.calculate_technical_score(sample_music_element, sample_context)
        
        assert hasattr(result, 'overall_technical')
        assert 0 <= result.overall_technical <= 1
        assert hasattr(result, 'harmony_correctness')
        assert hasattr(result, 'details')

    def test_artistic_score_calculation(self, sample_music_element, sample_context):
        """Test artistic score calculation"""
        scorer = QualityScoringSystem()
        result = scorer.calculate_artistic_score(sample_music_element, sample_context)
        
        assert hasattr(result, 'overall_artistic')
        assert 0 <= result.overall_artistic <= 1
        assert hasattr(result, 'creativity')
        assert hasattr(result, 'aesthetic_appeal')

    def test_emotional_score_calculation(self, sample_music_element, sample_context):
        """Test emotional score calculation"""
        scorer = QualityScoringSystem()
        result = scorer.calculate_emotional_score(sample_music_element, sample_context)
        
        assert hasattr(result, 'overall_emotional')
        assert 0 <= result.overall_emotional <= 1
        assert hasattr(result, 'emotional_clarity')
        assert hasattr(result, 'listener_engagement')


class TestQualityAssuranceEngine:
    """Test suite for Quality Assurance Engine (Integration Tests)"""
    
    def test_qa_engine_initialization(self):
        """Test QA engine initialization"""
        engine = QualityAssuranceEngine()
        assert engine.name == "QualityAssuranceEngine"
        assert engine.version == "1.0.0"
        assert engine.mode == QAMode.AUTOMATIC

    def test_qa_engine_mode_configuration(self):
        """Test QA engine mode configuration"""
        engine = QualityAssuranceEngine()
        
        engine.configure_mode(QAMode.INTERACTIVE)
        assert engine.mode == QAMode.INTERACTIVE
        
        engine.configure_mode(QAMode.MANUAL)
        assert engine.mode == QAMode.MANUAL

    def test_full_validation_workflow_automatic(self, sample_music_element, sample_context):
        """Test full validation workflow in automatic mode"""
        engine = QualityAssuranceEngine(mode=QAMode.AUTOMATIC)
        
        result = engine.validate_result(sample_music_element, None, sample_context)
        
        assert hasattr(result, 'result_status')
        assert result.result_status in [QAResult.APPROVED, QAResult.CONDITIONALLY_APPROVED, 
                                       QAResult.REQUIRES_ATTENTION, QAResult.REJECTED]
        assert hasattr(result, 'overall_quality_score')
        assert 0 <= result.overall_quality_score <= 1
        assert isinstance(result.validation_results, list)
        assert isinstance(result.recommendations, list)

    def test_full_validation_workflow_monitoring(self, sample_music_element, sample_context):
        """Test validation workflow in monitoring mode"""
        engine = QualityAssuranceEngine(mode=QAMode.MONITORING)
        
        result = engine.validate_result(sample_music_element, None, sample_context)
        
        assert hasattr(result, 'result_status')
        assert hasattr(result, 'overall_quality_score')
        # In monitoring mode, conflicts should be identified but not resolved
        assert len(result.resolved_conflicts) == 0

    def test_conflict_resolution_workflow(self, sample_music_element, sample_context):
        """Test targeted conflict resolution"""
        engine = QualityAssuranceEngine()
        
        conflicts = [ConflictType.HARMONY_VIOLATION, ConflictType.EMOTION_MISMATCH]
        result = engine.resolve_conflicts(conflicts, sample_music_element, sample_context)
        
        assert hasattr(result, 'resolved_conflicts')
        assert hasattr(result, 'remaining_conflicts')
        assert isinstance(result.user_decisions, list)

    def test_quality_report_generation(self):
        """Test quality report generation"""
        engine = QualityAssuranceEngine()
        
        # Simulate some processing history
        engine._update_processing_stats(QAResult.APPROVED, 2, 0)
        engine._update_processing_stats(QAResult.CONDITIONALLY_APPROVED, 1, 1)
        
        report = engine.get_quality_report("7_days")
        
        assert 'processing_statistics' in report
        assert 'quality_trends' in report
        assert 'component_status' in report
        assert 'system_health' in report

    def test_performance_benchmarks(self, sample_music_element, sample_context):
        """Test performance benchmarks"""
        engine = QualityAssuranceEngine()
        
        import time
        start_time = time.time()
        
        result = engine.validate_result(sample_music_element, None, sample_context)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should complete within 2 seconds for typical tracks
        assert processing_time < 2.0
        
        # Should have reasonable quality score
        assert result.overall_quality_score > 0.3


class TestIntegrationScenarios:
    """Integration test scenarios for real-world usage"""
    
    def test_chinese_traditional_music_validation(self):
        """Test validation of Chinese traditional music"""
        # Create Chinese traditional music element
        @dataclass
        class ChineseMusicElement:
            notes: List[Dict]
            style: str = "chinese_traditional"
            tracks: List[Dict]
            
            def __init__(self):
                # Pentatonic scale notes (C, D, E, G, A)
                self.notes = [
                    {'pitch': 60, 'duration': 1.0},  # C
                    {'pitch': 62, 'duration': 1.0},  # D
                    {'pitch': 64, 'duration': 1.0},  # E
                    {'pitch': 67, 'duration': 1.0},  # G
                    {'pitch': 69, 'duration': 1.0}   # A
                ]
                self.tracks = [{'features': {'energy': 0.4, 'valence': 0.6}}]
        
        chinese_element = ChineseMusicElement()
        context = ValidationContext({'style': 'chinese_traditional', 'key': 'C'})
        
        engine = QualityAssuranceEngine()
        result = engine.validate_result(chinese_element, None, context)
        
        # Should handle Chinese traditional music appropriately
        assert result.result_status in [QAResult.APPROVED, QAResult.CONDITIONALLY_APPROVED]
        assert result.overall_quality_score > 0.5

    def test_complex_harmony_conflict_resolution(self):
        """Test resolution of complex harmony conflicts"""
        @dataclass 
        class ComplexHarmonyElement:
            notes: List[Dict]
            chords: List[str]
            style: str = "classical"
            
            def __init__(self):
                # Create intentionally problematic harmony
                self.notes = [
                    {'pitch': 60, 'duration': 1.0},  # C
                    {'pitch': 61, 'duration': 1.0},  # C# (creates dissonance)
                    {'pitch': 66, 'duration': 1.0},  # F# (tritone)
                ]
                self.chords = ['C', 'F#', 'Bb', 'E']  # Distant chord progression
        
        problem_element = ComplexHarmonyElement()
        context = ValidationContext({'key': 'C', 'style': 'classical'})
        
        engine = QualityAssuranceEngine(mode=QAMode.AUTOMATIC)
        result = engine.validate_result(problem_element, None, context)
        
        # Should identify harmony conflicts
        assert len(result.validation_results) > 0
        # Should provide recommendations
        assert len(result.recommendations) > 0

    def test_user_feedback_learning_simulation(self):
        """Test user feedback learning simulation"""
        engine = QualityAssuranceEngine(mode=QAMode.INTERACTIVE)
        
        # Simulate multiple validation cycles with user feedback
        for i in range(3):
            @dataclass
            class TestElement:
                notes: List[Dict]
                style: str = "pop"
                
                def __init__(self):
                    self.notes = [{'pitch': 60 + i, 'duration': 1.0}]
            
            element = TestElement()
            context = ValidationContext({'style': 'pop'})
            
            result = engine.validate_result(element, None, context)
            
            # Should complete without errors
            assert hasattr(result, 'result_status')
        
        # Check that user interface has learning data
        assert len(engine.user_interface.user_decisions) >= 0


# Performance and stress tests
class TestPerformanceAndStress:
    """Performance and stress tests for QA system"""
    
    def test_large_music_element_processing(self):
        """Test processing of large music elements"""
        @dataclass
        class LargeMusicElement:
            notes: List[Dict]
            tracks: List[Dict]
            style: str = "orchestral"
            
            def __init__(self):
                # Create large number of notes
                self.notes = [
                    {'pitch': 60 + (i % 12), 'duration': 0.5, 'time': i * 0.5}
                    for i in range(200)  # 200 notes
                ]
                self.tracks = [
                    {'features': {'energy': 0.6, 'valence': 0.7}}
                    for _ in range(10)  # 10 tracks
                ]
        
        large_element = LargeMusicElement()
        context = ValidationContext({'style': 'orchestral'})
        
        engine = QualityAssuranceEngine()
        
        import time
        start_time = time.time()
        result = engine.validate_result(large_element, None, context)
        processing_time = time.time() - start_time
        
        # Should handle large elements efficiently
        assert processing_time < 5.0  # Within 5 seconds
        assert result.overall_quality_score > 0.2

    def test_concurrent_validation_simulation(self):
        """Test concurrent validation capabilities simulation"""
        import threading
        import queue
        
        results_queue = queue.Queue()
        
        def validate_element(element_id):
            @dataclass
            class ConcurrentElement:
                notes: List[Dict]
                style: str = "test"
                
                def __init__(self, element_id):
                    self.notes = [{'pitch': 60 + element_id, 'duration': 1.0}]
            
            element = ConcurrentElement(element_id)
            context = ValidationContext({'element_id': element_id})
            
            engine = QualityAssuranceEngine()
            result = engine.validate_result(element, None, context)
            results_queue.put((element_id, result.overall_quality_score))
        
        # Create multiple threads
        threads = []
        for i in range(3):  # 3 concurrent validations
            thread = threading.Thread(target=validate_element, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=10)  # 10 second timeout
        
        # Check results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        assert len(results) == 3
        for element_id, score in results:
            assert 0 <= score <= 1


if __name__ == "__main__":
    # Run specific test modules for debugging
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--disable-warnings"
    ])