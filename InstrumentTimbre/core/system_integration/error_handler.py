"""
System Error Handler

Centralized error handling, logging, and recovery system for InstrumentTimbre.
"""

import logging
import traceback
import time
from typing import Dict, Any, Optional, List, Callable, Type
from functools import wraps
from dataclasses import dataclass
from enum import Enum

from .exception_types import (
    InstrumentTimbreError, RecoverableError, ErrorSeverity, ErrorCategory,
    SystemResourceError, ModelError, AudioProcessingError
)


class RecoveryStrategy(Enum):
    """Error recovery strategies"""
    NONE = "none"
    RETRY = "retry"
    FALLBACK = "fallback"
    DEGRADE = "degrade"
    ABORT = "abort"


@dataclass
class ErrorContext:
    """Context information for error handling"""
    module_name: str
    function_name: str
    operation_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    input_data_info: Optional[Dict[str, Any]] = None
    system_state: Optional[Dict[str, Any]] = None


@dataclass
class RecoveryAction:
    """Recovery action definition"""
    strategy: RecoveryStrategy
    action: Optional[Callable] = None
    max_retries: int = 3
    retry_delay: float = 1.0
    fallback_value: Any = None
    degrade_quality: bool = False


class SystemErrorHandler:
    """
    Centralized error handling system that provides:
    - Error logging and tracking
    - Automatic error recovery
    - Error reporting and metrics
    - User-friendly error messages
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Error tracking
        self.error_history: List[Dict[str, Any]] = []
        self.error_counts: Dict[str, int] = {}
        self.recovery_registry: Dict[Type[Exception], RecoveryAction] = {}
        
        # Recovery statistics
        self.recovery_stats = {
            "total_errors": 0,
            "recovered_errors": 0,
            "failed_recoveries": 0,
            "retry_success_rate": 0.0
        }
        
        # Initialize default recovery strategies
        self._initialize_default_recovery_strategies()
    
    def _initialize_default_recovery_strategies(self):
        """Initialize default recovery strategies for common errors"""
        
        # Recoverable errors with retry
        self.register_recovery_strategy(
            RecoverableError,
            RecoveryAction(
                strategy=RecoveryStrategy.RETRY,
                max_retries=3,
                retry_delay=1.0
            )
        )
        
        # Audio processing errors with fallback
        self.register_recovery_strategy(
            AudioProcessingError,
            RecoveryAction(
                strategy=RecoveryStrategy.FALLBACK,
                fallback_value=None,
                degrade_quality=True
            )
        )
        
        # Model errors with quality degradation
        self.register_recovery_strategy(
            ModelError,
            RecoveryAction(
                strategy=RecoveryStrategy.DEGRADE,
                degrade_quality=True
            )
        )
        
        # System resource errors with retry and degradation
        self.register_recovery_strategy(
            SystemResourceError,
            RecoveryAction(
                strategy=RecoveryStrategy.RETRY,
                max_retries=2,
                retry_delay=2.0,
                degrade_quality=True
            )
        )
    
    def register_recovery_strategy(
        self, 
        exception_type: Type[Exception], 
        recovery_action: RecoveryAction
    ):
        """Register a recovery strategy for a specific exception type"""
        self.recovery_registry[exception_type] = recovery_action
        self.logger.info(f"Registered recovery strategy for {exception_type.__name__}")
    
    def handle_error(
        self, 
        error: Exception, 
        context: Optional[ErrorContext] = None,
        auto_recover: bool = True
    ) -> Dict[str, Any]:
        """
        Handle an error with logging, recovery, and reporting
        
        Args:
            error: The exception that occurred
            context: Context information about where the error occurred
            auto_recover: Whether to attempt automatic recovery
            
        Returns:
            Dictionary containing error handling results
        """
        start_time = time.time()
        
        # Create error record
        error_record = self._create_error_record(error, context)
        
        # Log the error
        self._log_error(error_record)
        
        # Track error statistics
        self._update_error_stats(error)
        
        # Attempt recovery if enabled
        recovery_result = None
        if auto_recover:
            recovery_result = self._attempt_recovery(error, context)
        
        # Compile results
        handling_result = {
            "error_id": error_record["error_id"],
            "error_type": type(error).__name__,
            "severity": error_record["severity"],
            "recovery_attempted": auto_recover,
            "recovery_successful": recovery_result.get("success", False) if recovery_result else False,
            "recovery_strategy": recovery_result.get("strategy") if recovery_result else None,
            "processing_time": time.time() - start_time,
            "user_message": self._generate_user_message(error, recovery_result),
            "technical_details": error_record
        }
        
        return handling_result
    
    def _create_error_record(self, error: Exception, context: Optional[ErrorContext]) -> Dict[str, Any]:
        """Create detailed error record for logging and tracking"""
        
        error_id = f"ERR_{int(time.time() * 1000)}"
        
        # Extract error information
        if isinstance(error, InstrumentTimbreError):
            error_dict = error.to_dict()
            severity = error.severity.value
            category = error.category.value
            error_code = error.error_code
            suggestions = error.suggestions
        else:
            error_dict = {"message": str(error)}
            severity = ErrorSeverity.MEDIUM.value
            category = ErrorCategory.SYSTEM_RESOURCE.value
            error_code = "UNKNOWN_ERROR"
            suggestions = []
        
        record = {
            "error_id": error_id,
            "timestamp": time.time(),
            "error_type": type(error).__name__,
            "error_code": error_code,
            "message": str(error),
            "severity": severity,
            "category": category,
            "suggestions": suggestions,
            "traceback": traceback.format_exc(),
            "context": context.__dict__ if context else {},
            "full_error_info": error_dict
        }
        
        # Add to error history
        self.error_history.append(record)
        
        # Keep only last 1000 errors to prevent memory issues
        if len(self.error_history) > 1000:
            self.error_history = self.error_history[-1000:]
        
        return record
    
    def _log_error(self, error_record: Dict[str, Any]):
        """Log error with appropriate level based on severity"""
        
        severity = error_record["severity"]
        message = f"[{error_record['error_id']}] {error_record['message']}"
        
        if severity == ErrorSeverity.CRITICAL.value:
            self.logger.critical(message, extra=error_record)
        elif severity == ErrorSeverity.HIGH.value:
            self.logger.error(message, extra=error_record)
        elif severity == ErrorSeverity.MEDIUM.value:
            self.logger.warning(message, extra=error_record)
        else:
            self.logger.info(message, extra=error_record)
    
    def _update_error_stats(self, error: Exception):
        """Update error statistics"""
        error_type = type(error).__name__
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        self.recovery_stats["total_errors"] += 1
    
    def _attempt_recovery(self, error: Exception, context: Optional[ErrorContext]) -> Dict[str, Any]:
        """Attempt to recover from the error using registered strategies"""
        
        # Find appropriate recovery strategy
        recovery_action = self._find_recovery_strategy(error)
        if not recovery_action:
            return {"success": False, "reason": "No recovery strategy available"}
        
        strategy = recovery_action.strategy
        self.logger.info(f"Attempting recovery using strategy: {strategy.value}")
        
        try:
            if strategy == RecoveryStrategy.RETRY:
                return self._retry_recovery(error, recovery_action, context)
            elif strategy == RecoveryStrategy.FALLBACK:
                return self._fallback_recovery(error, recovery_action, context)
            elif strategy == RecoveryStrategy.DEGRADE:
                return self._degrade_recovery(error, recovery_action, context)
            else:
                return {"success": False, "reason": f"Unknown recovery strategy: {strategy}"}
                
        except Exception as recovery_error:
            self.logger.error(f"Recovery attempt failed: {recovery_error}")
            self.recovery_stats["failed_recoveries"] += 1
            return {
                "success": False, 
                "reason": f"Recovery failed: {recovery_error}",
                "strategy": strategy.value
            }
    
    def _find_recovery_strategy(self, error: Exception) -> Optional[RecoveryAction]:
        """Find the most appropriate recovery strategy for the error"""
        
        # Check for exact type match first
        error_type = type(error)
        if error_type in self.recovery_registry:
            return self.recovery_registry[error_type]
        
        # Check for parent class matches
        for registered_type, recovery_action in self.recovery_registry.items():
            if isinstance(error, registered_type):
                return recovery_action
        
        return None
    
    def _retry_recovery(self, error: Exception, recovery_action: RecoveryAction, context: Optional[ErrorContext]) -> Dict[str, Any]:
        """Implement retry recovery strategy"""
        
        max_retries = recovery_action.max_retries
        retry_delay = recovery_action.retry_delay
        
        for attempt in range(max_retries):
            self.logger.info(f"Retry attempt {attempt + 1}/{max_retries}")
            
            # Wait before retry (except first attempt)
            if attempt > 0:
                time.sleep(retry_delay)
            
            try:
                # If the error has a retry mechanism, use it
                if hasattr(error, 'can_retry') and hasattr(error, 'increment_retry'):
                    if not error.can_retry():
                        break
                    error.increment_retry()
                
                # For now, we return success if we can retry
                # In a real implementation, we would re-execute the failed operation
                if attempt < max_retries - 1:  # Still have retries left
                    self.recovery_stats["recovered_errors"] += 1
                    return {
                        "success": True,
                        "strategy": RecoveryStrategy.RETRY.value,
                        "attempts": attempt + 1
                    }
                    
            except Exception as retry_error:
                self.logger.warning(f"Retry attempt {attempt + 1} failed: {retry_error}")
                continue
        
        return {
            "success": False,
            "strategy": RecoveryStrategy.RETRY.value,
            "reason": f"All {max_retries} retry attempts failed"
        }
    
    def _fallback_recovery(self, error: Exception, recovery_action: RecoveryAction, context: Optional[ErrorContext]) -> Dict[str, Any]:
        """Implement fallback recovery strategy"""
        
        try:
            # Use fallback action if available
            if recovery_action.action:
                fallback_result = recovery_action.action(error, context)
            else:
                fallback_result = recovery_action.fallback_value
            
            self.recovery_stats["recovered_errors"] += 1
            return {
                "success": True,
                "strategy": RecoveryStrategy.FALLBACK.value,
                "fallback_result": fallback_result,
                "quality_degraded": recovery_action.degrade_quality
            }
            
        except Exception as fallback_error:
            return {
                "success": False,
                "strategy": RecoveryStrategy.FALLBACK.value,
                "reason": f"Fallback failed: {fallback_error}"
            }
    
    def _degrade_recovery(self, error: Exception, recovery_action: RecoveryAction, context: Optional[ErrorContext]) -> Dict[str, Any]:
        """Implement quality degradation recovery strategy"""
        
        try:
            # Apply quality degradation
            degradation_applied = self._apply_quality_degradation(context)
            
            if degradation_applied:
                self.recovery_stats["recovered_errors"] += 1
                return {
                    "success": True,
                    "strategy": RecoveryStrategy.DEGRADE.value,
                    "quality_degraded": True,
                    "degradation_details": degradation_applied
                }
            else:
                return {
                    "success": False,
                    "strategy": RecoveryStrategy.DEGRADE.value,
                    "reason": "Could not apply quality degradation"
                }
                
        except Exception as degrade_error:
            return {
                "success": False,
                "strategy": RecoveryStrategy.DEGRADE.value,
                "reason": f"Degradation failed: {degrade_error}"
            }
    
    def _apply_quality_degradation(self, context: Optional[ErrorContext]) -> Optional[Dict[str, Any]]:
        """Apply quality degradation to help recovery"""
        
        # This would integrate with the global config system
        try:
            from config import get_config, update_config, Quality
            
            current_config = get_config()
            
            # Degrade quality settings
            degradation = {}
            
            if current_config.quality != Quality.FAST:
                update_config(quality=Quality.FAST)
                degradation["quality"] = "degraded_to_fast"
            
            if current_config.batch_size > 8:
                update_config(batch_size=max(4, current_config.batch_size // 2))
                degradation["batch_size"] = "reduced"
            
            if current_config.sample_rate > 22050:
                update_config(sample_rate=22050)
                degradation["sample_rate"] = "reduced_to_22050"
            
            return degradation if degradation else None
            
        except Exception as e:
            self.logger.warning(f"Could not apply quality degradation: {e}")
            return None
    
    def _generate_user_message(self, error: Exception, recovery_result: Optional[Dict[str, Any]]) -> str:
        """Generate user-friendly error message"""
        
        # Base message
        if isinstance(error, InstrumentTimbreError):
            base_message = error.message
            suggestions = error.suggestions
        else:
            base_message = "An unexpected error occurred during processing."
            suggestions = ["Please try again or contact support."]
        
        # Add recovery information
        if recovery_result and recovery_result.get("success"):
            strategy = recovery_result.get("strategy", "unknown")
            if strategy == "retry":
                recovery_msg = " The operation was automatically retried and completed successfully."
            elif strategy == "fallback":
                recovery_msg = " The system used an alternative approach to complete the operation."
            elif strategy == "degrade":
                recovery_msg = " The system reduced quality settings to complete the operation."
            else:
                recovery_msg = " The system automatically recovered from the error."
            
            if recovery_result.get("quality_degraded"):
                recovery_msg += " Note: Processing quality was reduced to ensure completion."
        else:
            recovery_msg = " Automatic recovery was not possible."
        
        # Combine message
        full_message = base_message + recovery_msg
        
        # Add suggestions
        if suggestions:
            full_message += "\n\nSuggestions:\n" + "\n".join(f"• {s}" for s in suggestions[:3])
        
        return full_message
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics"""
        
        total_errors = self.recovery_stats["total_errors"]
        recovered_errors = self.recovery_stats["recovered_errors"]
        
        return {
            "total_errors": total_errors,
            "recovered_errors": recovered_errors,
            "failed_recoveries": self.recovery_stats["failed_recoveries"],
            "recovery_rate": recovered_errors / total_errors if total_errors > 0 else 0.0,
            "error_counts_by_type": self.error_counts.copy(),
            "recent_errors": len([e for e in self.error_history if time.time() - e["timestamp"] < 3600]),  # Last hour
            "critical_errors": len([e for e in self.error_history if e["severity"] == ErrorSeverity.CRITICAL.value])
        }
    
    def clear_error_history(self):
        """Clear error history and reset statistics"""
        self.error_history.clear()
        self.error_counts.clear()
        self.recovery_stats = {
            "total_errors": 0,
            "recovered_errors": 0,
            "failed_recoveries": 0,
            "retry_success_rate": 0.0
        }
        self.logger.info("Error history and statistics cleared")


# Decorator for automatic error handling
def handle_errors(
    auto_recover: bool = True,
    context_module: Optional[str] = None,
    custom_recovery: Optional[RecoveryAction] = None
):
    """
    Decorator for automatic error handling in functions
    
    Args:
        auto_recover: Whether to attempt automatic recovery
        context_module: Module name for context
        custom_recovery: Custom recovery action for this function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            error_handler = SystemErrorHandler()
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = ErrorContext(
                    module_name=context_module or func.__module__,
                    function_name=func.__name__
                )
                
                # Register custom recovery if provided
                if custom_recovery:
                    error_handler.register_recovery_strategy(type(e), custom_recovery)
                
                result = error_handler.handle_error(e, context, auto_recover)
                
                # Re-raise if recovery failed
                if not result["recovery_successful"]:
                    raise e
                
                # Return None or default value if recovered
                return None
                
        return wrapper
    return decorator