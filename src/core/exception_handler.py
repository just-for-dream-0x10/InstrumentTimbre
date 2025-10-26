"""
Comprehensive Exception Handling System

Provides robust error handling, recovery mechanisms, and detailed logging
for all system operations following coding standards.
"""

import logging
import traceback
import time
from typing import Dict, Any, Optional, Callable, Type, List
from dataclasses import dataclass
from enum import Enum
from functools import wraps
import asyncio


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification"""
    BUSINESS_LOGIC = "business_logic"
    SYSTEM_RESOURCE = "system_resource"
    NETWORK = "network"
    DATA_VALIDATION = "data_validation"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"


@dataclass
class ErrorContext:
    """Error context information"""
    module_name: str
    function_name: str
    operation_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    input_data_summary: Optional[Dict[str, Any]] = None
    timestamp: float = 0.0


@dataclass
class ErrorRecord:
    """Complete error record for tracking and analysis"""
    error_id: str
    error_type: str
    message: str
    severity: ErrorSeverity
    category: ErrorCategory
    context: ErrorContext
    traceback_info: str
    recovery_attempted: bool = False
    recovery_successful: bool = False
    retry_count: int = 0
    timestamp: float = 0.0


class SystemException(Exception):
    """Base system exception with enhanced context"""
    
    def __init__(
        self, 
        message: str, 
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        recoverable: bool = True
    ):
        super().__init__(message)
        self.severity = severity
        self.category = category
        self.recoverable = recoverable


class BusinessLogicException(SystemException):
    """Business logic related exceptions"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message, 
            category=ErrorCategory.BUSINESS_LOGIC,
            **kwargs
        )


class ResourceException(SystemException):
    """System resource related exceptions"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message, 
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.SYSTEM_RESOURCE,
            **kwargs
        )


class ValidationException(SystemException):
    """Data validation exceptions"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message, 
            category=ErrorCategory.DATA_VALIDATION,
            **kwargs
        )


class ConfigurationException(SystemException):
    """Configuration related exceptions"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message, 
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.CONFIGURATION,
            recoverable=False,
            **kwargs
        )


class ExceptionHandler:
    """
    Comprehensive exception handling system with recovery mechanisms,
    detailed logging, and performance monitoring.
    """
    
    def __init__(self, max_error_records: int = 1000):
        """
        Initialize exception handler
        
        Args:
            max_error_records: Maximum number of error records to keep
        """
        self.logger = logging.getLogger(__name__)
        self.max_error_records = max_error_records
        
        # Error tracking
        self.error_counter = 0
        self.error_records: List[ErrorRecord] = []
        
        # Recovery strategies
        self.recovery_strategies: Dict[Type[Exception], Callable] = {
            ResourceException: self._recover_resource_error,
            BusinessLogicException: self._recover_business_error,
            ValidationException: self._recover_validation_error
        }
        
        # Statistics
        self.error_stats = {
            "total_errors": 0,
            "recovered_errors": 0,
            "critical_errors": 0,
            "errors_by_category": {},
            "errors_by_severity": {}
        }
    
    def handle_exception(
        self, 
        error: Exception, 
        context: ErrorContext,
        attempt_recovery: bool = True
    ) -> Dict[str, Any]:
        """
        Handle exception with comprehensive error processing
        
        Args:
            error: The exception to handle
            context: Error context information
            attempt_recovery: Whether to attempt error recovery
            
        Returns:
            Error handling result with recovery information
        """
        # Generate error ID
        error_id = f"err_{self.error_counter:06d}_{int(time.time())}"
        self.error_counter += 1
        
        # Classify error
        severity, category = self._classify_error(error)
        
        # Create error record
        error_record = ErrorRecord(
            error_id=error_id,
            error_type=type(error).__name__,
            message=str(error),
            severity=severity,
            category=category,
            context=context,
            traceback_info=traceback.format_exc(),
            timestamp=time.time()
        )
        
        # Log error
        self._log_error(error_record)
        
        # Attempt recovery if enabled and error is recoverable
        recovery_result = None
        if attempt_recovery and self._is_recoverable(error):
            recovery_result = self._attempt_recovery(error, error_record)
            error_record.recovery_attempted = True
            error_record.recovery_successful = recovery_result.get("success", False)
        
        # Store error record
        self._store_error_record(error_record)
        
        # Update statistics
        self._update_error_stats(error_record)
        
        return {
            "error_id": error_id,
            "error_type": error_record.error_type,
            "message": error_record.message,
            "severity": severity.value,
            "category": category.value,
            "recoverable": self._is_recoverable(error),
            "recovery_attempted": error_record.recovery_attempted,
            "recovery_successful": error_record.recovery_successful,
            "recovery_result": recovery_result,
            "user_message": self._generate_user_message(error_record),
            "timestamp": error_record.timestamp
        }
    
    def _classify_error(self, error: Exception) -> tuple[ErrorSeverity, ErrorCategory]:
        """Classify error by severity and category"""
        
        # Handle system exceptions with predefined classification
        if isinstance(error, SystemException):
            return error.severity, error.category
        
        # Classify built-in exceptions
        if isinstance(error, (MemoryError, OSError)):
            return ErrorSeverity.CRITICAL, ErrorCategory.SYSTEM_RESOURCE
        
        elif isinstance(error, (ValueError, TypeError)):
            return ErrorSeverity.MEDIUM, ErrorCategory.DATA_VALIDATION
        
        elif isinstance(error, (ConnectionError, TimeoutError)):
            return ErrorSeverity.HIGH, ErrorCategory.NETWORK
        
        elif isinstance(error, FileNotFoundError):
            return ErrorSeverity.MEDIUM, ErrorCategory.CONFIGURATION
        
        else:
            return ErrorSeverity.MEDIUM, ErrorCategory.UNKNOWN
    
    def _is_recoverable(self, error: Exception) -> bool:
        """Determine if error is recoverable"""
        
        if isinstance(error, SystemException):
            return error.recoverable
        
        # Built-in exceptions recovery rules
        non_recoverable = (MemoryError, SystemExit, KeyboardInterrupt)
        return not isinstance(error, non_recoverable)
    
    def _attempt_recovery(
        self, 
        error: Exception, 
        error_record: ErrorRecord
    ) -> Dict[str, Any]:
        """Attempt error recovery using appropriate strategy"""
        
        self.logger.info(f"Attempting recovery for error {error_record.error_id}")
        
        try:
            # Find recovery strategy
            strategy = None
            for error_type, recovery_func in self.recovery_strategies.items():
                if isinstance(error, error_type):
                    strategy = recovery_func
                    break
            
            if not strategy:
                strategy = self._default_recovery_strategy
            
            # Execute recovery strategy
            recovery_result = strategy(error, error_record)
            
            if recovery_result.get("success", False):
                self.logger.info(f"Recovery successful for error {error_record.error_id}")
            else:
                self.logger.warning(f"Recovery failed for error {error_record.error_id}")
            
            return recovery_result
            
        except Exception as recovery_error:
            self.logger.error(
                f"Recovery attempt failed for error {error_record.error_id}: {recovery_error}"
            )
            return {
                "success": False,
                "strategy": "recovery_failed",
                "error": str(recovery_error)
            }
    
    def _recover_resource_error(
        self, 
        error: Exception, 
        error_record: ErrorRecord
    ) -> Dict[str, Any]:
        """Recovery strategy for resource errors"""
        
        # Attempt to free up resources
        try:
            import gc
            gc.collect()
            
            return {
                "success": True,
                "strategy": "resource_cleanup",
                "action": "garbage_collection_performed"
            }
        except Exception:
            return {"success": False, "strategy": "resource_cleanup"}
    
    def _recover_business_error(
        self, 
        error: Exception, 
        error_record: ErrorRecord
    ) -> Dict[str, Any]:
        """Recovery strategy for business logic errors"""
        
        # Provide fallback or default behavior
        return {
            "success": True,
            "strategy": "graceful_degradation",
            "action": "fallback_to_default_behavior",
            "recommendation": "Review input parameters and try again"
        }
    
    def _recover_validation_error(
        self, 
        error: Exception, 
        error_record: ErrorRecord
    ) -> Dict[str, Any]:
        """Recovery strategy for validation errors"""
        
        return {
            "success": False,
            "strategy": "input_correction_required",
            "action": "user_input_correction_needed",
            "recommendation": "Please correct the input data and retry"
        }
    
    def _default_recovery_strategy(
        self, 
        error: Exception, 
        error_record: ErrorRecord
    ) -> Dict[str, Any]:
        """Default recovery strategy for unknown errors"""
        
        return {
            "success": False,
            "strategy": "no_recovery_available",
            "action": "manual_intervention_required"
        }
    
    def _log_error(self, error_record: ErrorRecord) -> None:
        """Log error with appropriate level and context"""
        
        log_message = (
            f"Error {error_record.error_id}: {error_record.message} "
            f"[{error_record.category.value}:{error_record.severity.value}]"
        )
        
        extra_context = {
            "error_id": error_record.error_id,
            "error_category": error_record.category.value,
            "error_severity": error_record.severity.value,
            "module": error_record.context.module_name,
            "function": error_record.context.function_name,
            "operation_id": error_record.context.operation_id
        }
        
        if error_record.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message, extra=extra_context)
        elif error_record.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message, extra=extra_context)
        elif error_record.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message, extra=extra_context)
        else:
            self.logger.info(log_message, extra=extra_context)
    
    def _store_error_record(self, error_record: ErrorRecord) -> None:
        """Store error record for analysis"""
        
        self.error_records.append(error_record)
        
        # Maintain maximum records limit
        if len(self.error_records) > self.max_error_records:
            self.error_records = self.error_records[-self.max_error_records:]
    
    def _update_error_stats(self, error_record: ErrorRecord) -> None:
        """Update error statistics"""
        
        self.error_stats["total_errors"] += 1
        
        if error_record.recovery_successful:
            self.error_stats["recovered_errors"] += 1
        
        if error_record.severity == ErrorSeverity.CRITICAL:
            self.error_stats["critical_errors"] += 1
        
        # Update category stats
        category_key = error_record.category.value
        if category_key not in self.error_stats["errors_by_category"]:
            self.error_stats["errors_by_category"][category_key] = 0
        self.error_stats["errors_by_category"][category_key] += 1
        
        # Update severity stats
        severity_key = error_record.severity.value
        if severity_key not in self.error_stats["errors_by_severity"]:
            self.error_stats["errors_by_severity"][severity_key] = 0
        self.error_stats["errors_by_severity"][severity_key] += 1
    
    def _generate_user_message(self, error_record: ErrorRecord) -> str:
        """Generate user-friendly error message"""
        
        base_message = "An error occurred while processing your request."
        
        if error_record.category == ErrorCategory.DATA_VALIDATION:
            return f"{base_message} Please check your input data and try again."
        
        elif error_record.category == ErrorCategory.SYSTEM_RESOURCE:
            return f"{base_message} The system is currently under high load. Please try again later."
        
        elif error_record.category == ErrorCategory.NETWORK:
            return f"{base_message} There was a network connectivity issue. Please check your connection."
        
        elif error_record.category == ErrorCategory.CONFIGURATION:
            return f"{base_message} There is a configuration issue. Please contact support."
        
        else:
            return f"{base_message} Please try again or contact support if the problem persists."
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics"""
        
        return {
            **self.error_stats,
            "recovery_rate": (
                self.error_stats["recovered_errors"] / self.error_stats["total_errors"]
                if self.error_stats["total_errors"] > 0 else 0
            ),
            "critical_error_rate": (
                self.error_stats["critical_errors"] / self.error_stats["total_errors"]
                if self.error_stats["total_errors"] > 0 else 0
            ),
            "recent_errors": len([
                r for r in self.error_records 
                if time.time() - r.timestamp < 3600  # Last hour
            ])
        }
    
    def get_recent_errors(self, limit: int = 10) -> List[ErrorRecord]:
        """Get recent error records"""
        
        return sorted(
            self.error_records, 
            key=lambda r: r.timestamp, 
            reverse=True
        )[:limit]


def exception_handler(
    attempt_recovery: bool = True,
    log_traceback: bool = True,
    reraise: bool = False
):
    """
    Decorator for automatic exception handling
    
    Args:
        attempt_recovery: Whether to attempt error recovery
        log_traceback: Whether to log full traceback
        reraise: Whether to reraise exception after handling
    """
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            handler = ExceptionHandler()
            
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
                    
            except Exception as e:
                context = ErrorContext(
                    module_name=func.__module__,
                    function_name=func.__name__,
                    timestamp=time.time()
                )
                
                result = handler.handle_exception(e, context, attempt_recovery)
                
                if reraise:
                    raise
                
                return {"error": True, "error_info": result}
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            handler = ExceptionHandler()
            
            try:
                return func(*args, **kwargs)
                
            except Exception as e:
                context = ErrorContext(
                    module_name=func.__module__,
                    function_name=func.__name__,
                    timestamp=time.time()
                )
                
                result = handler.handle_exception(e, context, attempt_recovery)
                
                if reraise:
                    raise
                
                return {"error": True, "error_info": result}
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator