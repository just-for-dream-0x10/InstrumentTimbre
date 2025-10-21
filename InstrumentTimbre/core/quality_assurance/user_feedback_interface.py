"""
User Feedback Interface - Interactive conflict resolution

This module provides user-friendly conflict resolution including:
- Clear conflict visualization and problem presentation
- Multiple solution options with preview capabilities
- Before/after comparison system
- User preference learning for adaptive decision making
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum

from .data_structures import ConflictType, ResolutionSuggestion
from .auto_coordination_engine import ConflictItem, ResolutionStrategy


class FeedbackType(Enum):
    """Types of user feedback"""
    ACCEPT = "accept"
    REJECT = "reject"
    MODIFY = "modify"
    POSTPONE = "postpone"


class VisualizationType(Enum):
    """Types of conflict visualization"""
    AUDIO_WAVEFORM = "audio_waveform"
    SCORE_NOTATION = "score_notation"
    HARMONIC_ANALYSIS = "harmonic_analysis"
    EMOTIONAL_TIMELINE = "emotional_timeline"
    TEXT_DESCRIPTION = "text_description"


@dataclass
class ConflictVisualization:
    """Visualization data for a conflict"""
    conflict_id: str
    visualization_type: VisualizationType
    visual_data: Dict[str, Any]
    description: str
    severity_indicator: str
    affected_regions: List[Tuple[float, float]]  # Time ranges
    suggested_focus_areas: List[str]


@dataclass
class SolutionOption:
    """User-presentable solution option"""
    option_id: str
    title: str
    description: str
    confidence: float
    estimated_improvement: float
    trade_offs: List[str]
    preview_available: bool
    implementation_time: str
    reversible: bool
    metadata: Dict[str, Any]


@dataclass
class PreviewResult:
    """Preview comparison result"""
    original_data: Dict[str, Any]
    modified_data: Dict[str, Any]
    difference_highlights: List[str]
    quality_comparison: Dict[str, Tuple[float, float]]  # (before, after)
    user_metrics: Dict[str, float]
    visual_comparison: Optional[Dict[str, Any]]


@dataclass
class UserDecision:
    """User's decision on a conflict resolution"""
    conflict_id: str
    chosen_option_id: Optional[str]
    feedback_type: FeedbackType
    custom_modifications: Dict[str, Any]
    reasoning: Optional[str]
    confidence_in_decision: float
    timestamp: float


@dataclass
class PreferenceLearningResult:
    """Result of learning from user preferences"""
    learned_patterns: List[str]
    preference_weights: Dict[str, float]
    decision_confidence: float
    prediction_accuracy: float
    recommendation_adjustments: List[str]


class UserFeedbackInterface:
    """
    Interactive conflict resolution interface
    
    Provides clear conflict presentation, solution options,
    preview capabilities, and user preference learning.
    """
    
    def __init__(self):
        self.name = "UserFeedbackInterface"
        self.version = "1.0.0"
        
        # User preference history
        self.user_decisions = []
        self.learned_preferences = {}
        
        # Visualization preferences
        self.visualization_preferences = {
            ConflictType.HARMONY_VIOLATION: VisualizationType.HARMONIC_ANALYSIS,
            ConflictType.EMOTION_MISMATCH: VisualizationType.EMOTIONAL_TIMELINE,
            ConflictType.RHYTHM_VIOLATION: VisualizationType.AUDIO_WAVEFORM,
            ConflictType.SCALE_VIOLATION: VisualizationType.SCORE_NOTATION
        }
        
        # Quality metrics for user understanding
        self.user_metrics = {
            'musicality': 'How musical does it sound?',
            'emotion_clarity': 'How clear is the emotional expression?',
            'technical_quality': 'How technically correct is it?',
            'personal_preference': 'How much do you like it?',
            'style_consistency': 'How consistent is the style?'
        }

    def visualize_conflicts(self, conflicts: List[ConflictItem], context: Dict[str, Any]) -> List[ConflictVisualization]:
        """
        Create clear visualizations of conflicts for user understanding
        
        Args:
            conflicts: List of conflicts to visualize
            context: Additional context for visualization
            
        Returns:
            List of conflict visualizations
        """
        visualizations = []
        
        for conflict in conflicts:
            # Determine best visualization type for this conflict
            viz_type = self.visualization_preferences.get(
                conflict.conflict_type, 
                VisualizationType.TEXT_DESCRIPTION
            )
            
            # Generate visualization data
            visual_data = self._generate_visual_data(conflict, viz_type, context)
            
            # Create severity indicator
            severity_indicator = self._create_severity_indicator(conflict.severity)
            
            # Identify affected regions
            affected_regions = self._identify_affected_regions(conflict, context)
            
            # Generate focus areas
            focus_areas = self._generate_focus_areas(conflict)
            
            visualization = ConflictVisualization(
                conflict_id=f"conflict_{conflict.metadata.get('index', 0)}",
                visualization_type=viz_type,
                visual_data=visual_data,
                description=self._create_user_friendly_description(conflict),
                severity_indicator=severity_indicator,
                affected_regions=affected_regions,
                suggested_focus_areas=focus_areas
            )
            
            visualizations.append(visualization)
        
        return visualizations

    def present_solution_options(self, conflict: ConflictItem, 
                                suggestions: List[ResolutionSuggestion]) -> List[SolutionOption]:
        """
        Present solution options in user-friendly format
        
        Args:
            conflict: Conflict to resolve
            suggestions: Resolution suggestions from auto-coordination
            
        Returns:
            List of user-presentable solution options
        """
        solution_options = []
        
        for i, suggestion in enumerate(suggestions):
            # Create user-friendly title and description
            title = self._create_solution_title(suggestion)
            description = self._create_solution_description(suggestion, conflict)
            
            # Determine if preview is available
            preview_available = self._can_generate_preview(suggestion)
            
            # Estimate implementation time
            implementation_time = self._estimate_implementation_time(suggestion)
            
            # Check if solution is reversible
            reversible = self._is_solution_reversible(suggestion)
            
            # Extract trade-offs in user-friendly language
            trade_offs = self._extract_user_friendly_trade_offs(suggestion)
            
            option = SolutionOption(
                option_id=f"option_{i}_{suggestion.conflict_id}",
                title=title,
                description=description,
                confidence=suggestion.confidence,
                estimated_improvement=suggestion.estimated_improvement,
                trade_offs=trade_offs,
                preview_available=preview_available,
                implementation_time=implementation_time,
                reversible=reversible,
                metadata={
                    'original_suggestion': suggestion,
                    'complexity': self._assess_solution_complexity(suggestion)
                }
            )
            
            solution_options.append(option)
        
        # Sort by user preference if available
        if self.learned_preferences:
            solution_options = self._sort_by_user_preference(solution_options)
        
        return solution_options

    def generate_preview(self, original_data: Dict[str, Any], 
                        proposed_solution: SolutionOption) -> PreviewResult:
        """
        Generate before/after preview for solution comparison
        
        Args:
            original_data: Original music data
            proposed_solution: Solution to preview
            
        Returns:
            PreviewResult with comparison data
        """
        # Apply proposed solution to generate modified data
        modified_data = self._apply_solution_for_preview(original_data, proposed_solution)
        
        # Identify key differences
        differences = self._identify_key_differences(original_data, modified_data)
        
        # Calculate quality metrics comparison
        quality_comparison = self._calculate_quality_comparison(original_data, modified_data)
        
        # Generate user-friendly metrics
        user_metrics = self._calculate_user_metrics(original_data, modified_data)
        
        # Create visual comparison if supported
        visual_comparison = self._create_visual_comparison(original_data, modified_data, proposed_solution)
        
        return PreviewResult(
            original_data=original_data,
            modified_data=modified_data,
            difference_highlights=differences,
            quality_comparison=quality_comparison,
            user_metrics=user_metrics,
            visual_comparison=visual_comparison
        )

    def learn_user_preferences(self, decisions: List[UserDecision]) -> PreferenceLearningResult:
        """
        Learn from user decisions to improve future recommendations
        
        Args:
            decisions: History of user decisions
            
        Returns:
            PreferenceLearningResult with learning outcomes
        """
        # Add new decisions to history
        self.user_decisions.extend(decisions)
        
        # Analyze decision patterns
        patterns = self._analyze_decision_patterns(self.user_decisions)
        
        # Update preference weights
        new_weights = self._update_preference_weights(self.user_decisions)
        self.learned_preferences.update(new_weights)
        
        # Calculate prediction accuracy
        accuracy = self._calculate_prediction_accuracy(decisions)
        
        # Generate recommendation adjustments
        adjustments = self._generate_recommendation_adjustments(patterns, new_weights)
        
        # Estimate decision confidence
        confidence = self._estimate_decision_confidence(decisions)
        
        return PreferenceLearningResult(
            learned_patterns=patterns,
            preference_weights=new_weights,
            decision_confidence=confidence,
            prediction_accuracy=accuracy,
            recommendation_adjustments=adjustments
        )

    def get_user_feedback(self, conflict_id: str, solution_options: List[SolutionOption]) -> UserDecision:
        """
        Collect user feedback on conflict resolution (simulated for automated testing)
        
        Args:
            conflict_id: ID of the conflict being resolved
            solution_options: Available solution options
            
        Returns:
            UserDecision with user's choice
        """
        # In a real implementation, this would present UI and wait for user input
        # For now, we'll simulate intelligent user behavior based on learned preferences
        
        if solution_options:
            # Choose the option with highest confidence if no preferences learned
            if not self.learned_preferences:
                chosen_option = max(solution_options, key=lambda x: x.confidence)
            else:
                chosen_option = self._predict_user_choice(solution_options)
            
            return UserDecision(
                conflict_id=conflict_id,
                chosen_option_id=chosen_option.option_id,
                feedback_type=FeedbackType.ACCEPT,
                custom_modifications={},
                reasoning="Simulated user decision based on learned preferences",
                confidence_in_decision=0.8,
                timestamp=np.datetime64('now').astype(float)
            )
        else:
            return UserDecision(
                conflict_id=conflict_id,
                chosen_option_id=None,
                feedback_type=FeedbackType.POSTPONE,
                custom_modifications={},
                reasoning="No suitable options available",
                confidence_in_decision=0.9,
                timestamp=np.datetime64('now').astype(float)
            )

    # Helper methods for internal processing
    
    def _generate_visual_data(self, conflict: ConflictItem, viz_type: VisualizationType, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate visualization data based on type"""
        if viz_type == VisualizationType.HARMONIC_ANALYSIS:
            return {
                'chord_progression': ['C', 'Am', 'F', 'G'],
                'problematic_chords': [1, 3],
                'suggested_replacements': ['Dm', 'Em']
            }
        elif viz_type == VisualizationType.EMOTIONAL_TIMELINE:
            return {
                'timeline': [0, 1, 2, 3, 4],
                'emotions': ['happy', 'sad', 'happy', 'angry', 'peaceful'],
                'conflict_points': [1, 3]
            }
        elif viz_type == VisualizationType.AUDIO_WAVEFORM:
            return {
                'waveform_data': np.random.randn(1000).tolist(),
                'problem_regions': [(100, 200), (600, 700)]
            }
        else:
            return {'description': conflict.description}

    def _create_severity_indicator(self, severity: float) -> str:
        """Create user-friendly severity indicator"""
        if severity >= 0.8:
            return "🔴 Critical - Requires immediate attention"
        elif severity >= 0.6:
            return "🟡 Important - Should be addressed"
        elif severity >= 0.4:
            return "🟠 Moderate - Consider fixing"
        else:
            return "🟢 Minor - Optional improvement"

    def _identify_affected_regions(self, conflict: ConflictItem, context: Dict[str, Any]) -> List[Tuple[float, float]]:
        """Identify time regions affected by the conflict"""
        # Simplified region identification
        location = conflict.location or "0"
        try:
            start_time = float(location.split('_')[-1]) * 2.0  # Convert to time
            return [(start_time, start_time + 2.0)]
        except:
            return [(0.0, 2.0)]

    def _generate_focus_areas(self, conflict: ConflictItem) -> List[str]:
        """Generate focus areas for user attention"""
        focus_mapping = {
            ConflictType.HARMONY_VIOLATION: ["Chord progression", "Voice leading"],
            ConflictType.EMOTION_MISMATCH: ["Track balance", "Emotional consistency"],
            ConflictType.RHYTHM_VIOLATION: ["Beat alignment", "Time signature"],
            ConflictType.SCALE_VIOLATION: ["Melody notes", "Key signature"]
        }
        return focus_mapping.get(conflict.conflict_type, ["General music structure"])

    def _create_user_friendly_description(self, conflict: ConflictItem) -> str:
        """Create user-friendly conflict description"""
        descriptions = {
            ConflictType.HARMONY_VIOLATION: "The chord progression doesn't follow traditional harmony rules, which might sound unusual to listeners.",
            ConflictType.EMOTION_MISMATCH: "The different parts of your music express conflicting emotions, creating an inconsistent feel.",
            ConflictType.RHYTHM_VIOLATION: "The rhythm doesn't align properly with the time signature, which could confuse listeners.",
            ConflictType.SCALE_VIOLATION: "Some notes don't fit the key you're working in, which might sound out of place."
        }
        return descriptions.get(conflict.conflict_type, conflict.description)

    def _create_solution_title(self, suggestion: ResolutionSuggestion) -> str:
        """Create user-friendly solution title"""
        if suggestion.strategy == ResolutionStrategy.AUTOMATIC_FIX:
            return "🔧 Auto-fix"
        elif suggestion.strategy == ResolutionStrategy.SUGGEST_ALTERNATIVES:
            return "💡 Alternative approach"
        elif suggestion.strategy == ResolutionStrategy.USER_INTERVENTION:
            return "✋ Manual guidance needed"
        else:
            return "❓ Custom solution"

    def _create_solution_description(self, suggestion: ResolutionSuggestion, conflict: ConflictItem) -> str:
        """Create detailed solution description"""
        base_description = suggestion.description
        confidence_text = f"Confidence: {suggestion.confidence:.1%}"
        improvement_text = f"Expected improvement: {suggestion.estimated_improvement:.1%}"
        
        return f"{base_description}\n\n{confidence_text}\n{improvement_text}"

    def _can_generate_preview(self, suggestion: ResolutionSuggestion) -> bool:
        """Check if preview can be generated for this suggestion"""
        return suggestion.strategy in [ResolutionStrategy.AUTOMATIC_FIX, ResolutionStrategy.SUGGEST_ALTERNATIVES]

    def _estimate_implementation_time(self, suggestion: ResolutionSuggestion) -> str:
        """Estimate time to implement the solution"""
        cost = suggestion.implementation_cost
        if cost < 0.3:
            return "< 1 second"
        elif cost < 0.6:
            return "1-5 seconds"
        else:
            return "5-15 seconds"

    def _is_solution_reversible(self, suggestion: ResolutionSuggestion) -> bool:
        """Check if the solution can be easily reversed"""
        return suggestion.strategy != ResolutionStrategy.USER_INTERVENTION

    def _extract_user_friendly_trade_offs(self, suggestion: ResolutionSuggestion) -> List[str]:
        """Convert technical trade-offs to user-friendly language"""
        friendly_trade_offs = []
        for trade_off in suggestion.side_effects:
            if "melodic" in trade_off.lower():
                friendly_trade_offs.append("May change the melody slightly")
            elif "mood" in trade_off.lower():
                friendly_trade_offs.append("Could affect the overall mood")
            elif "character" in trade_off.lower():
                friendly_trade_offs.append("Might alter the musical character")
            else:
                friendly_trade_offs.append(trade_off)
        return friendly_trade_offs

    def _assess_solution_complexity(self, suggestion: ResolutionSuggestion) -> str:
        """Assess complexity level of the solution"""
        if suggestion.implementation_cost < 0.3:
            return "Simple"
        elif suggestion.implementation_cost < 0.6:
            return "Moderate"
        else:
            return "Complex"

    def _sort_by_user_preference(self, options: List[SolutionOption]) -> List[SolutionOption]:
        """Sort options based on learned user preferences"""
        # Simplified preference sorting
        def preference_score(option):
            base_score = option.confidence * option.estimated_improvement
            
            # Adjust based on learned preferences
            complexity = option.metadata.get('complexity', 'Moderate')
            if self.learned_preferences.get('prefers_simple', False) and complexity == 'Simple':
                base_score *= 1.2
            elif self.learned_preferences.get('prefers_complex', False) and complexity == 'Complex':
                base_score *= 1.1
            
            return base_score
        
        return sorted(options, key=preference_score, reverse=True)

    def _apply_solution_for_preview(self, original_data: Dict[str, Any], solution: SolutionOption) -> Dict[str, Any]:
        """Apply solution to generate preview data"""
        # Simplified solution application for preview
        modified_data = original_data.copy()
        
        # Simulate modifications based on solution type
        if "harmony" in solution.description.lower():
            modified_data['harmony_modified'] = True
        elif "emotion" in solution.description.lower():
            modified_data['emotion_adjusted'] = True
        
        return modified_data

    def _identify_key_differences(self, original: Dict[str, Any], modified: Dict[str, Any]) -> List[str]:
        """Identify key differences between original and modified"""
        differences = []
        
        for key in modified:
            if key not in original:
                differences.append(f"Added: {key}")
            elif original.get(key) != modified.get(key):
                differences.append(f"Modified: {key}")
        
        return differences

    def _calculate_quality_comparison(self, original: Dict[str, Any], modified: Dict[str, Any]) -> Dict[str, Tuple[float, float]]:
        """Calculate quality metrics comparison"""
        return {
            'harmony_quality': (0.6, 0.8),
            'emotion_consistency': (0.5, 0.9),
            'technical_correctness': (0.7, 0.8),
            'overall_musicality': (0.6, 0.85)
        }

    def _calculate_user_metrics(self, original: Dict[str, Any], modified: Dict[str, Any]) -> Dict[str, float]:
        """Calculate user-friendly metrics"""
        return {
            'musicality': 0.85,
            'emotion_clarity': 0.8,
            'technical_quality': 0.9,
            'personal_preference': 0.75,
            'style_consistency': 0.8
        }

    def _create_visual_comparison(self, original: Dict[str, Any], modified: Dict[str, Any], solution: SolutionOption) -> Optional[Dict[str, Any]]:
        """Create visual comparison data"""
        if solution.preview_available:
            return {
                'comparison_type': 'side_by_side',
                'highlight_changes': True,
                'difference_markers': ['harmony_change_at_0.5s', 'emotion_adjustment_at_1.2s']
            }
        return None

    def _analyze_decision_patterns(self, decisions: List[UserDecision]) -> List[str]:
        """Analyze patterns in user decisions"""
        patterns = []
        
        accept_rate = len([d for d in decisions if d.feedback_type == FeedbackType.ACCEPT]) / len(decisions) if decisions else 0
        
        if accept_rate > 0.8:
            patterns.append("User generally accepts automatic suggestions")
        elif accept_rate < 0.3:
            patterns.append("User prefers manual control over automatic fixes")
        
        return patterns

    def _update_preference_weights(self, decisions: List[UserDecision]) -> Dict[str, float]:
        """Update preference weights based on decisions"""
        weights = {}
        
        # Analyze decision confidence
        avg_confidence = np.mean([d.confidence_in_decision for d in decisions]) if decisions else 0.5
        
        if avg_confidence > 0.8:
            weights['decision_confidence'] = 'high'
        else:
            weights['decision_confidence'] = 'moderate'
        
        return weights

    def _calculate_prediction_accuracy(self, recent_decisions: List[UserDecision]) -> float:
        """Calculate how accurately we can predict user preferences"""
        # Simplified accuracy calculation
        return 0.75 if recent_decisions else 0.5

    def _generate_recommendation_adjustments(self, patterns: List[str], weights: Dict[str, float]) -> List[str]:
        """Generate adjustments to recommendation system"""
        adjustments = []
        
        for pattern in patterns:
            if "automatic" in pattern:
                adjustments.append("Increase weight for automatic solutions")
            elif "manual" in pattern:
                adjustments.append("Prioritize user intervention options")
        
        return adjustments

    def _estimate_decision_confidence(self, decisions: List[UserDecision]) -> float:
        """Estimate overall confidence in decision predictions"""
        if not decisions:
            return 0.5
        
        avg_confidence = np.mean([d.confidence_in_decision for d in decisions])
        return min(avg_confidence + 0.1, 1.0)

    def _predict_user_choice(self, options: List[SolutionOption]) -> SolutionOption:
        """Predict which option the user would choose"""
        # Simplified prediction based on learned preferences
        if self.learned_preferences.get('decision_confidence') == 'high':
            # User is confident, prefer higher confidence options
            return max(options, key=lambda x: x.confidence)
        else:
            # User is cautious, prefer reversible options with good improvement
            reversible_options = [opt for opt in options if opt.reversible]
            if reversible_options:
                return max(reversible_options, key=lambda x: x.estimated_improvement)
            else:
                return max(options, key=lambda x: x.estimated_improvement)