"""
Quality Assurance Data Structures

Defines core data types for quality assurance operations,
validation results, and conflict resolution.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union, Tuple, Any
from enum import Enum
import numpy as np


class QualityIssueType(Enum):
    """Types of quality issues that can be detected"""
    HARMONIC_CONFLICT = "harmonic_conflict"
    EMOTION_INCONSISTENCY = "emotion_inconsistency"
    THEORY_VIOLATION = "theory_violation"
    RHYTHM_CONFLICT = "rhythm_conflict"
    DYNAMIC_IMBALANCE = "dynamic_imbalance"
    STYLE_INCONSISTENCY = "style_inconsistency"
    STRUCTURAL_ISSUE = "structural_issue"


class ValidationSeverity(Enum):
    """Severity levels for validation issues"""
    CRITICAL = "critical"      # Must be fixed
    WARNING = "warning"        # Should be addressed
    SUGGESTION = "suggestion"  # Optional improvement
    INFO = "info"             # Informational only


class ResolutionStrategy(Enum):
    """Strategies for resolving quality issues"""
    AUTO_FIX = "auto_fix"              # Automatic resolution
    USER_CHOICE = "user_choice"        # Present options to user
    MANUAL_REVIEW = "manual_review"    # Requires manual intervention
    ACCEPT_AS_IS = "accept_as_is"      # Accept current state


@dataclass
class QualityIssue:
    """Represents a specific quality issue detected in music"""
    issue_type: QualityIssueType
    severity: ValidationSeverity
    description: str
    location: Tuple[float, float]  # Time range (start, end)
    affected_tracks: List[str]
    
    # Detection details
    confidence: float = 0.0  # 0-1, confidence in issue detection
    technical_details: Dict[str, Any] = field(default_factory=dict)
    
    # Resolution information
    suggested_fixes: List[str] = field(default_factory=list)
    resolution_strategy: ResolutionStrategy = ResolutionStrategy.USER_CHOICE
    auto_fixable: bool = False
    
    def __post_init__(self):
        """Validate issue data"""
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        if self.location[0] > self.location[1]:
            raise ValueError("Invalid time range")


@dataclass
class ValidationResult:
    """Result of a validation operation"""
    validator_name: str
    overall_score: float  # 0-1, higher is better
    passed: bool
    
    # Detailed scores
    detailed_scores: Dict[str, float] = field(default_factory=dict)
    
    # Issues found
    issues: List[QualityIssue] = field(default_factory=list)
    
    # Processing information
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_critical_issues(self) -> List[QualityIssue]:
        """Get only critical issues"""
        return [issue for issue in self.issues 
                if issue.severity == ValidationSeverity.CRITICAL]
    
    def get_fixable_issues(self) -> List[QualityIssue]:
        """Get automatically fixable issues"""
        return [issue for issue in self.issues if issue.auto_fixable]
    
    def __post_init__(self):
        """Validate result data"""
        if not 0 <= self.overall_score <= 1:
            raise ValueError("Overall score must be between 0 and 1")


@dataclass
class QualityAssessment:
    """Comprehensive quality assessment result"""
    overall_quality_score: float  # 0-1 composite score
    
    # Individual validation results
    harmony_result: Optional[ValidationResult] = None
    emotion_result: Optional[ValidationResult] = None
    theory_result: Optional[ValidationResult] = None
    
    # Aggregate information
    total_issues: int = 0
    critical_issues: int = 0
    auto_fixable_issues: int = 0
    
    # Quality metrics
    technical_quality: float = 0.0    # Music theory compliance
    artistic_quality: float = 0.0     # Emotional and stylistic quality
    production_quality: float = 0.0   # Audio quality and balance
    
    # Decision information
    approved: bool = False
    requires_user_input: bool = False
    
    # Processing metadata
    assessment_time: float = 0.0
    validator_versions: Dict[str, str] = field(default_factory=dict)
    
    def get_all_issues(self) -> List[QualityIssue]:
        """Get all issues from all validation results"""
        all_issues = []
        for result in [self.harmony_result, self.emotion_result, self.theory_result]:
            if result and result.issues:
                all_issues.extend(result.issues)
        return all_issues
    
    def get_quality_summary(self) -> Dict[str, Any]:
        """Get a summary of quality metrics"""
        return {
            'overall_score': self.overall_quality_score,
            'technical_score': self.technical_quality,
            'artistic_score': self.artistic_quality,
            'production_score': self.production_quality,
            'total_issues': self.total_issues,
            'critical_issues': self.critical_issues,
            'approval_status': self.approved
        }


@dataclass
class ConflictResolution:
    """Result of conflict resolution process"""
    resolution_id: str
    original_issue: QualityIssue
    resolution_strategy: ResolutionStrategy
    
    # Resolution details
    applied_fix: Optional[str] = None
    success: bool = False
    
    # Quality impact
    before_score: float = 0.0
    after_score: float = 0.0
    improvement: float = 0.0
    
    # User interaction
    user_approved: bool = False
    user_feedback: Optional[str] = None
    
    # Technical details
    processing_time: float = 0.0
    side_effects: List[str] = field(default_factory=list)
    
    def calculate_improvement(self):
        """Calculate improvement score"""
        if self.before_score > 0:
            self.improvement = (self.after_score - self.before_score) / self.before_score
        else:
            self.improvement = self.after_score


@dataclass
class UserFeedbackRequest:
    """Request for user feedback on quality issues"""
    request_id: str
    issue: QualityIssue
    
    # Available options
    resolution_options: List[Dict[str, Any]] = field(default_factory=list)
    
    # Context information
    context_description: str = ""
    preview_available: bool = False
    
    # User interaction
    timeout_seconds: int = 300  # 5 minutes default
    requires_preview: bool = False
    
    def add_resolution_option(self, name: str, description: str, 
                            auto_applicable: bool = True, **kwargs):
        """Add a resolution option"""
        option = {
            'name': name,
            'description': description,
            'auto_applicable': auto_applicable,
            **kwargs
        }
        self.resolution_options.append(option)


@dataclass
class UserFeedbackResponse:
    """User response to feedback request"""
    request_id: str
    selected_option: Optional[str] = None
    custom_preference: Optional[str] = None
    
    # User choices
    apply_to_similar: bool = False  # Apply to similar issues
    remember_preference: bool = False  # Remember for future
    
    # Additional feedback
    satisfaction_rating: Optional[int] = None  # 1-5 scale
    comments: Optional[str] = None
    
    # Response metadata
    response_time: float = 0.0
    response_timestamp: Optional[str] = None


@dataclass
class QualityPreferences:
    """User preferences for quality assessment"""
    # Quality priorities (weights sum to 1.0)
    harmony_weight: float = 0.3
    emotion_weight: float = 0.3
    theory_weight: float = 0.2
    style_weight: float = 0.2
    
    # Tolerance levels
    harmony_tolerance: float = 0.8    # Minimum harmony score
    emotion_tolerance: float = 0.7    # Minimum emotion consistency
    theory_tolerance: float = 0.8     # Minimum theory compliance
    
    # Auto-resolution settings
    auto_fix_threshold: float = 0.9   # Auto-fix if confidence > threshold
    user_confirmation_threshold: float = 0.7  # Ask user if confidence < threshold
    
    # Style preferences
    preferred_styles: List[str] = field(default_factory=list)
    forbidden_elements: List[str] = field(default_factory=list)
    
    def validate(self) -> bool:
        """Validate preferences"""
        weights = [self.harmony_weight, self.emotion_weight, 
                  self.theory_weight, self.style_weight]
        return abs(sum(weights) - 1.0) < 0.01


# Utility functions
def create_quality_issue(issue_type: QualityIssueType, severity: ValidationSeverity,
                        description: str, location: Tuple[float, float],
                        affected_tracks: List[str], **kwargs) -> QualityIssue:
    """Utility function to create quality issues"""
    return QualityIssue(
        issue_type=issue_type,
        severity=severity,
        description=description,
        location=location,
        affected_tracks=affected_tracks,
        **kwargs
    )


def merge_validation_results(results: List[ValidationResult]) -> ValidationResult:
    """Merge multiple validation results into one"""
    if not results:
        raise ValueError("Cannot merge empty results list")
    
    # Calculate overall scores
    overall_score = np.mean([result.overall_score for result in results])
    passed = all(result.passed for result in results)
    
    # Combine issues
    all_issues = []
    for result in results:
        all_issues.extend(result.issues)
    
    # Combine detailed scores
    detailed_scores = {}
    for result in results:
        for key, value in result.detailed_scores.items():
            if key in detailed_scores:
                detailed_scores[key] = (detailed_scores[key] + value) / 2
            else:
                detailed_scores[key] = value
    
    return ValidationResult(
        validator_name="merged",
        overall_score=overall_score,
        passed=passed,
        detailed_scores=detailed_scores,
        issues=all_issues,
        processing_time=sum(result.processing_time for result in results)
    )


# Core data types needed by validators
@dataclass
class QualityScore:
    """Quality score with multiple dimensions"""
    technical_score: float
    artistic_score: float
    emotional_score: float
    overall_score: float
    confidence_level: float

class ConflictType(Enum):
    """Types of conflicts detected in music"""
    HARMONY_VIOLATION = "harmony_violation"
    SCALE_VIOLATION = "scale_violation"
    RHYTHM_VIOLATION = "rhythm_violation"
    EMOTION_MISMATCH = "emotion_mismatch"
    INTENSITY_LOSS = "intensity_loss"
    STYLE_MISMATCH = "style_mismatch"
    COUNTERPOINT_VIOLATION = "counterpoint_violation"
    FORM_VIOLATION = "form_violation"
    EMOTION_DISCONTINUITY = "emotion_discontinuity"
    STYLE_EMOTION_MISMATCH = "style_emotion_mismatch"

class ValidationContext(dict):
    """Context for validation operations"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

@dataclass 
class MusicElement:
    """Generic music element for validation"""
    notes: Optional[List[Dict]] = None
    chords: Optional[List[str]] = None
    tracks: Optional[List[Dict]] = None
    style: Optional[str] = None
    rhythm_data: Optional[Dict] = None
    structure_data: Optional[Dict] = None
    voices: Optional[List[List[Dict]]] = None
    harmonic_rhythm: Optional[List[Dict]] = None
    temporal_data: Optional[List[Dict]] = None

@dataclass
class ResolutionSuggestion:
    """Suggestion for resolving a quality issue"""
    issue_id: str
    strategy: ResolutionStrategy
    description: str
    confidence: float
    estimated_improvement: float
    implementation_details: Dict[str, Any] = field(default_factory=dict)