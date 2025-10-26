"""
Custom Exception Types for InstrumentTimbre System

Defines all custom exceptions used across the system for better error handling and debugging.
"""

from typing import Dict, Any, Optional, List
from enum import Enum


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification"""
    AUDIO_PROCESSING = "audio_processing"
    MODEL_INFERENCE = "model_inference"
    DATA_VALIDATION = "data_validation"
    CONFIGURATION = "configuration"
    SYSTEM_RESOURCE = "system_resource"
    USER_INPUT = "user_input"
    NETWORK = "network"
    FILE_IO = "file_io"


# Base Exception Classes
class InstrumentTimbreError(Exception):
    """Base exception for all InstrumentTimbre errors"""
    
    def __init__(
        self, 
        message: str,
        error_code: str = "UNKNOWN",
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.SYSTEM_RESOURCE,
        context: Optional[Dict[str, Any]] = None,
        suggestions: Optional[List[str]] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.severity = severity
        self.category = category
        self.context = context or {}
        self.suggestions = suggestions or []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization"""
        return {
            "message": self.message,
            "error_code": self.error_code,
            "severity": self.severity.value,
            "category": self.category.value,
            "context": self.context,
            "suggestions": self.suggestions
        }


# Audio Processing Exceptions
class AudioProcessingError(InstrumentTimbreError):
    """Audio processing related errors"""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.AUDIO_PROCESSING)
        kwargs.setdefault('error_code', 'AUDIO_PROC_ERROR')
        super().__init__(message, **kwargs)


class InvalidAudioFormatError(AudioProcessingError):
    """Invalid audio format or corrupted audio data"""
    
    def __init__(self, format_info: str = "", **kwargs):
        message = f"Invalid audio format: {format_info}"
        kwargs.setdefault('error_code', 'INVALID_AUDIO_FORMAT')
        kwargs.setdefault('suggestions', [
            "Check audio file format (supported: WAV, MP3, FLAC)",
            "Verify audio file is not corrupted",
            "Ensure audio has valid sample rate and channels"
        ])
        super().__init__(message, **kwargs)


class AudioProcessingTimeoutError(AudioProcessingError):
    """Audio processing operation timeout"""
    
    def __init__(self, operation: str = "", timeout: float = 0, **kwargs):
        message = f"Audio processing timeout: {operation} (timeout: {timeout}s)"
        kwargs.setdefault('error_code', 'AUDIO_TIMEOUT')
        kwargs.setdefault('severity', ErrorSeverity.HIGH)
        kwargs.setdefault('suggestions', [
            "Reduce audio file size or duration",
            "Increase timeout setting",
            "Check system resources"
        ])
        super().__init__(message, **kwargs)


# Model and Inference Exceptions
class ModelError(InstrumentTimbreError):
    """Model related errors"""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.MODEL_INFERENCE)
        kwargs.setdefault('error_code', 'MODEL_ERROR')
        super().__init__(message, **kwargs)


class ModelNotFoundError(ModelError):
    """Model file not found or not accessible"""
    
    def __init__(self, model_path: str = "", **kwargs):
        message = f"Model not found: {model_path}"
        kwargs.setdefault('error_code', 'MODEL_NOT_FOUND')
        kwargs.setdefault('severity', ErrorSeverity.CRITICAL)
        kwargs.setdefault('suggestions', [
            "Check model file path",
            "Download required model files",
            "Verify model directory permissions"
        ])
        super().__init__(message, **kwargs)


class ModelLoadError(ModelError):
    """Model loading failed"""
    
    def __init__(self, model_name: str = "", reason: str = "", **kwargs):
        message = f"Failed to load model '{model_name}': {reason}"
        kwargs.setdefault('error_code', 'MODEL_LOAD_FAILED')
        kwargs.setdefault('severity', ErrorSeverity.HIGH)
        kwargs.setdefault('suggestions', [
            "Check model file integrity",
            "Verify model compatibility",
            "Check available system memory"
        ])
        super().__init__(message, **kwargs)


class InferenceError(ModelError):
    """Model inference failed"""
    
    def __init__(self, model_name: str = "", **kwargs):
        message = f"Inference failed for model: {model_name}"
        kwargs.setdefault('error_code', 'INFERENCE_FAILED')
        kwargs.setdefault('suggestions', [
            "Check input data format",
            "Verify model is properly loaded",
            "Check GPU/CPU availability"
        ])
        super().__init__(message, **kwargs)


# Configuration and Validation Exceptions
class ConfigurationError(InstrumentTimbreError):
    """Configuration related errors"""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.CONFIGURATION)
        kwargs.setdefault('error_code', 'CONFIG_ERROR')
        super().__init__(message, **kwargs)


class InvalidConfigError(ConfigurationError):
    """Invalid configuration values"""
    
    def __init__(self, config_key: str = "", value: Any = None, **kwargs):
        message = f"Invalid configuration: {config_key} = {value}"
        kwargs.setdefault('error_code', 'INVALID_CONFIG')
        kwargs.setdefault('suggestions', [
            "Check configuration value ranges",
            "Refer to configuration documentation",
            "Use validate_config() function"
        ])
        super().__init__(message, **kwargs)


class DataValidationError(InstrumentTimbreError):
    """Data validation errors"""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.DATA_VALIDATION)
        kwargs.setdefault('error_code', 'DATA_VALIDATION_ERROR')
        super().__init__(message, **kwargs)


# System Resource Exceptions
class SystemResourceError(InstrumentTimbreError):
    """System resource related errors"""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.SYSTEM_RESOURCE)
        kwargs.setdefault('error_code', 'RESOURCE_ERROR')
        super().__init__(message, **kwargs)


class InsufficientMemoryError(SystemResourceError):
    """Insufficient system memory"""
    
    def __init__(self, required_mb: int = 0, available_mb: int = 0, **kwargs):
        message = f"Insufficient memory: required {required_mb}MB, available {available_mb}MB"
        kwargs.setdefault('error_code', 'INSUFFICIENT_MEMORY')
        kwargs.setdefault('severity', ErrorSeverity.HIGH)
        kwargs.setdefault('suggestions', [
            "Close other applications to free memory",
            "Reduce batch size or audio length",
            "Use quality=FAST for lower memory usage"
        ])
        super().__init__(message, **kwargs)


class GPUNotAvailableError(SystemResourceError):
    """GPU not available when required"""
    
    def __init__(self, **kwargs):
        message = "GPU not available but required for operation"
        kwargs.setdefault('error_code', 'GPU_NOT_AVAILABLE')
        kwargs.setdefault('suggestions', [
            "Set use_gpu=False in configuration",
            "Install GPU drivers and CUDA",
            "Check GPU availability"
        ])
        super().__init__(message, **kwargs)


# File I/O Exceptions
class FileIOError(InstrumentTimbreError):
    """File I/O related errors"""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.FILE_IO)
        kwargs.setdefault('error_code', 'FILE_IO_ERROR')
        super().__init__(message, **kwargs)


class FileNotFoundError(FileIOError):
    """File not found"""
    
    def __init__(self, filepath: str = "", **kwargs):
        message = f"File not found: {filepath}"
        kwargs.setdefault('error_code', 'FILE_NOT_FOUND')
        kwargs.setdefault('suggestions', [
            "Check file path spelling",
            "Verify file exists",
            "Check file permissions"
        ])
        super().__init__(message, **kwargs)


class FileAccessError(FileIOError):
    """File access permission denied"""
    
    def __init__(self, filepath: str = "", operation: str = "", **kwargs):
        message = f"File access denied: {operation} {filepath}"
        kwargs.setdefault('error_code', 'FILE_ACCESS_DENIED')
        kwargs.setdefault('suggestions', [
            "Check file permissions",
            "Run with appropriate privileges",
            "Ensure file is not locked by another process"
        ])
        super().__init__(message, **kwargs)


# User Input Exceptions
class UserInputError(InstrumentTimbreError):
    """User input validation errors"""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.USER_INPUT)
        kwargs.setdefault('error_code', 'USER_INPUT_ERROR')
        kwargs.setdefault('severity', ErrorSeverity.LOW)
        super().__init__(message, **kwargs)


class InvalidParameterError(UserInputError):
    """Invalid parameter provided by user"""
    
    def __init__(self, param_name: str = "", param_value: Any = None, **kwargs):
        message = f"Invalid parameter: {param_name} = {param_value}"
        kwargs.setdefault('error_code', 'INVALID_PARAMETER')
        kwargs.setdefault('suggestions', [
            "Check parameter documentation",
            "Verify parameter value range",
            "Use supported parameter values"
        ])
        super().__init__(message, **kwargs)


# Network and Connectivity Exceptions
class NetworkError(InstrumentTimbreError):
    """Network related errors"""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.NETWORK)
        kwargs.setdefault('error_code', 'NETWORK_ERROR')
        super().__init__(message, **kwargs)


class ConnectionTimeoutError(NetworkError):
    """Network connection timeout"""
    
    def __init__(self, url: str = "", timeout: float = 0, **kwargs):
        message = f"Connection timeout: {url} (timeout: {timeout}s)"
        kwargs.setdefault('error_code', 'CONNECTION_TIMEOUT')
        kwargs.setdefault('suggestions', [
            "Check internet connection",
            "Increase timeout setting",
            "Try again later"
        ])
        super().__init__(message, **kwargs)


# Module Integration Exceptions
class ModuleIntegrationError(InstrumentTimbreError):
    """Module integration and coordination errors"""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.SYSTEM_RESOURCE)
        kwargs.setdefault('error_code', 'MODULE_INTEGRATION_ERROR')
        super().__init__(message, **kwargs)


class ModuleNotAvailableError(ModuleIntegrationError):
    """Required module not available"""
    
    def __init__(self, module_name: str = "", **kwargs):
        message = f"Module not available: {module_name}"
        kwargs.setdefault('error_code', 'MODULE_NOT_AVAILABLE')
        kwargs.setdefault('suggestions', [
            "Check module installation",
            "Verify module dependencies",
            "Install required packages"
        ])
        super().__init__(message, **kwargs)


class ModuleVersionError(ModuleIntegrationError):
    """Module version compatibility error"""
    
    def __init__(self, module_name: str = "", required_version: str = "", actual_version: str = "", **kwargs):
        message = f"Module version mismatch: {module_name} requires {required_version}, got {actual_version}"
        kwargs.setdefault('error_code', 'MODULE_VERSION_MISMATCH')
        kwargs.setdefault('suggestions', [
            "Update module to required version",
            "Check compatibility matrix",
            "Downgrade to compatible version"
        ])
        super().__init__(message, **kwargs)


# Recovery and Retry Exceptions
class RecoverableError(InstrumentTimbreError):
    """Errors that can potentially be recovered from"""
    
    def __init__(self, message: str, retry_count: int = 0, max_retries: int = 3, **kwargs):
        super().__init__(message, **kwargs)
        self.retry_count = retry_count
        self.max_retries = max_retries
    
    def can_retry(self) -> bool:
        """Check if error can be retried"""
        return self.retry_count < self.max_retries
    
    def increment_retry(self):
        """Increment retry count"""
        self.retry_count += 1