"""
Base Validator Classes

Abstract base classes for all quality assurance validators,
providing common interface and shared functionality.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Union
import time
import logging
from .data_structures import (
    ValidationResult, QualityIssue, QualityAssessment,
    ValidationSeverity, QualityIssueType
)

logger = logging.getLogger(__name__)


class BaseValidator(ABC):
    """Abstract base class for all quality validators"""
    
    def __init__(self, validator_name: str, version: str = "1.0.0"):
        self.validator_name = validator_name
        self.version = version
        self.is_initialized = False
        self._cache = {}
        
        # Configuration
        self.config = self._get_default_config()
        
    def initialize(self) -> bool:
        """
        Initialize the validator
        
        Returns:
            True if initialization successful, False otherwise
        """
        self.is_initialized = True
        return True
    
    @abstractmethod
    def validate(self, *args, **kwargs) -> ValidationResult:
        """
        Perform validation operation
        
        Returns:
            ValidationResult with scores and detected issues
        """
        pass
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for this validator"""
        return {
            'cache_enabled': True,
            'cache_size_limit': 100,
            'log_level': 'INFO',
            'timeout_seconds': 30
        }
    
    def configure(self, config: Dict[str, Any]):
        """Update validator configuration"""
        self.config.update(config)
        logger.info(f"{self.validator_name} configuration updated")
    
    def get_cache_key(self, *args, **kwargs) -> str:
        """Generate cache key for validation input"""
        # Simple hash-based cache key
        import hashlib
        content = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def get_from_cache(self, cache_key: str) -> Optional[ValidationResult]:
        """Retrieve result from cache"""
        if not self.config.get('cache_enabled', True):
            return None
        return self._cache.get(cache_key)
    
    def save_to_cache(self, cache_key: str, result: ValidationResult):
        """Save result to cache"""
        if not self.config.get('cache_enabled', True):
            return
            
        # Limit cache size
        if len(self._cache) >= self.config.get('cache_size_limit', 100):
            # Remove oldest entry
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        
        self._cache[cache_key] = result
    
    def create_success_result(self, overall_score: float, 
                            detailed_scores: Dict[str, float] = None,
                            issues: List[QualityIssue] = None,
                            metadata: Dict[str, Any] = None) -> ValidationResult:
        """Create a successful validation result"""
        return ValidationResult(
            validator_name=self.validator_name,
            overall_score=overall_score,
            passed=overall_score >= 0.7,  # Default threshold
            detailed_scores=detailed_scores or {},
            issues=issues or [],
            metadata={
                'validator_version': self.version,
                **(metadata or {})
            }
        )
    
    def create_failure_result(self, error_message: str,
                            issues: List[QualityIssue] = None) -> ValidationResult:
        """Create a failed validation result"""
        failure_issue = QualityIssue(
            issue_type=QualityIssueType.STRUCTURAL_ISSUE,
            severity=ValidationSeverity.CRITICAL,
            description=f"Validation failed: {error_message}",
            location=(0.0, 0.0),
            affected_tracks=[],
            confidence=1.0
        )
        
        all_issues = [failure_issue]
        if issues:
            all_issues.extend(issues)
        
        return ValidationResult(
            validator_name=self.validator_name,
            overall_score=0.0,
            passed=False,
            issues=all_issues,
            metadata={
                'validator_version': self.version,
                'error': error_message
            }
        )
    
    def log_validation(self, input_description: str, result: ValidationResult):
        """Log validation operation"""
        logger.info(
            f"[{self.validator_name}] Validated {input_description}: "
            f"Score={result.overall_score:.3f}, "
            f"Issues={len(result.issues)}, "
            f"Time={result.processing_time:.3f}s"
        )


class HarmonyValidatorBase(BaseValidator):
    """Base class for harmony validation"""
    
    def __init__(self):
        super().__init__("HarmonyValidator")
        
    @abstractmethod
    def validate_intervals(self, track1_pitches: List[float], 
                          track2_pitches: List[float]) -> ValidationResult:
        """Validate intervals between two tracks"""
        pass
    
    @abstractmethod
    def validate_chord_progression(self, chords: List[str], 
                                 key: str) -> ValidationResult:
        """Validate chord progression in given key"""
        pass
    
    @abstractmethod 
    def validate_voice_leading(self, voices: List[List[float]]) -> ValidationResult:
        """Validate voice leading between multiple voices"""
        pass


class EmotionValidatorBase(BaseValidator):
    """Base class for emotion consistency validation"""
    
    def __init__(self):
        super().__init__("EmotionValidator")
        
    @abstractmethod
    def validate_emotion_consistency(self, original_emotion: Dict[str, float],
                                   modified_emotion: Dict[str, float]) -> ValidationResult:
        """Validate emotion consistency between original and modified"""
        pass
    
    @abstractmethod
    def validate_temporal_emotion(self, emotion_timeline: List[Dict]) -> ValidationResult:
        """Validate emotion changes over time"""
        pass


class TheoryValidatorBase(BaseValidator):
    """Base class for music theory validation"""
    
    def __init__(self):
        super().__init__("TheoryValidator")
        
    @abstractmethod
    def validate_scale_compliance(self, pitches: List[float], 
                                key: str) -> ValidationResult:
        """Validate that pitches comply with scale"""
        pass
    
    @abstractmethod
    def validate_rhythm_pattern(self, rhythm_pattern: List[float],
                              time_signature: str) -> ValidationResult:
        """Validate rhythm pattern against time signature"""
        pass
    
    @abstractmethod
    def validate_form_structure(self, structure_segments: List[Dict]) -> ValidationResult:
        """Validate musical form structure"""
        pass


class CompositeValidator(BaseValidator):
    """Composite validator that combines multiple validators"""
    
    def __init__(self, validators: List[BaseValidator], name: str = "CompositeValidator"):
        super().__init__(name)
        self.validators = validators
        
    def initialize(self) -> bool:
        """Initialize all sub-validators"""
        success = True
        for validator in self.validators:
            if not validator.initialize():
                logger.error(f"Failed to initialize {validator.validator_name}")
                success = False
        
        self.is_initialized = success
        return success
    
    def validate(self, *args, **kwargs) -> ValidationResult:
        """Run all validators and combine results"""
        if not self.is_initialized:
            return self.create_failure_result("Composite validator not initialized")
        
        start_time = time.time()
        results = []
        
        # Run all validators
        for validator in self.validators:
            try:
                result = validator.validate(*args, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Validator {validator.validator_name} failed: {e}")
                error_result = validator.create_failure_result(str(e))
                results.append(error_result)
        
        # Combine results
        combined_result = self._combine_results(results)
        combined_result.processing_time = time.time() - start_time
        
        return combined_result
    
    def _combine_results(self, results: List[ValidationResult]) -> ValidationResult:
        """Combine multiple validation results"""
        if not results:
            return self.create_failure_result("No validation results to combine")
        
        # Calculate weighted overall score
        total_score = sum(result.overall_score for result in results)
        overall_score = total_score / len(results)
        
        # Combine all issues
        all_issues = []
        for result in results:
            all_issues.extend(result.issues)
        
        # Check if passed (all validators must pass)
        passed = all(result.passed for result in results)
        
        # Combine detailed scores
        detailed_scores = {}
        for result in results:
            for key, value in result.detailed_scores.items():
                score_key = f"{result.validator_name}_{key}"
                detailed_scores[score_key] = value
        
        # Combine metadata
        metadata = {
            'validator_count': len(results),
            'individual_scores': [result.overall_score for result in results],
            'individual_validators': [result.validator_name for result in results]
        }
        
        return ValidationResult(
            validator_name=self.validator_name,
            overall_score=overall_score,
            passed=passed,
            detailed_scores=detailed_scores,
            issues=all_issues,
            metadata=metadata
        )


class ValidationPipeline:
    """Pipeline for orchestrating multiple validation steps"""
    
    def __init__(self, name: str = "ValidationPipeline"):
        self.name = name
        self.steps = []
        
    def add_step(self, validator: BaseValidator, 
                 step_name: str = None, 
                 required: bool = True,
                 depends_on: List[str] = None):
        """Add a validation step to the pipeline"""
        step = {
            'validator': validator,
            'name': step_name or validator.validator_name,
            'required': required,
            'depends_on': depends_on or []
        }
        self.steps.append(step)
    
    def execute(self, *args, **kwargs) -> QualityAssessment:
        """Execute the validation pipeline"""
        start_time = time.time()
        results = {}
        
        # Execute steps in order
        for step in self.steps:
            step_name = step['name']
            validator = step['validator']
            
            # Check dependencies
            if not self._check_dependencies(step, results):
                if step['required']:
                    logger.error(f"Required step {step_name} dependencies not met")
                    break
                else:
                    logger.warning(f"Skipping optional step {step_name}")
                    continue
            
            # Execute validation
            try:
                result = validator.validate(*args, **kwargs)
                results[step_name] = result
                
                # Stop on critical failures for required steps
                if step['required'] and not result.passed:
                    critical_issues = result.get_critical_issues()
                    if critical_issues:
                        logger.error(f"Critical issues in required step {step_name}")
                        break
                        
            except Exception as e:
                logger.error(f"Step {step_name} failed: {e}")
                if step['required']:
                    break
        
        # Create comprehensive assessment
        assessment = self._create_assessment(results)
        assessment.assessment_time = time.time() - start_time
        
        return assessment
    
    def _check_dependencies(self, step: Dict, results: Dict[str, ValidationResult]) -> bool:
        """Check if step dependencies are satisfied"""
        for dep_name in step.get('depends_on', []):
            if dep_name not in results:
                return False
            if not results[dep_name].passed:
                return False
        return True
    
    def _create_assessment(self, results: Dict[str, ValidationResult]) -> QualityAssessment:
        """Create quality assessment from pipeline results"""
        # Extract specific results
        harmony_result = results.get('harmony', results.get('HarmonyValidator'))
        emotion_result = results.get('emotion', results.get('EmotionValidator'))  
        theory_result = results.get('theory', results.get('TheoryValidator'))
        
        # Calculate overall score
        scores = [result.overall_score for result in results.values()]
        overall_score = sum(scores) / len(scores) if scores else 0.0
        
        # Count issues
        all_issues = []
        for result in results.values():
            all_issues.extend(result.issues)
        
        total_issues = len(all_issues)
        critical_issues = len([issue for issue in all_issues 
                             if issue.severity == ValidationSeverity.CRITICAL])
        auto_fixable_issues = len([issue for issue in all_issues if issue.auto_fixable])
        
        # Calculate component quality scores
        technical_quality = theory_result.overall_score if theory_result else 0.0
        artistic_quality = emotion_result.overall_score if emotion_result else 0.0
        production_quality = harmony_result.overall_score if harmony_result else 0.0
        
        # Determine approval
        approved = overall_score >= 0.7 and critical_issues == 0
        requires_user_input = any(
            len(result.issues) > 0 and not all(issue.auto_fixable for issue in result.issues)
            for result in results.values()
        )
        
        return QualityAssessment(
            overall_quality_score=overall_score,
            harmony_result=harmony_result,
            emotion_result=emotion_result,
            theory_result=theory_result,
            total_issues=total_issues,
            critical_issues=critical_issues,
            auto_fixable_issues=auto_fixable_issues,
            technical_quality=technical_quality,
            artistic_quality=artistic_quality,
            production_quality=production_quality,
            approved=approved,
            requires_user_input=requires_user_input,
            validator_versions={name: result.metadata.get('validator_version', 'unknown') 
                              for name, result in results.items()}
        )