"""
Exception hierarchy for InstrumentTimbre framework.

Provides specialized exceptions for timbre analysis, training, and conversion
operations with detailed error context and structured error handling.
"""


class TimbreException(Exception):
    """
    Base exception for all InstrumentTimbre related errors.

    All timbre-specific exceptions inherit from this class to provide
    consistent error handling throughout the framework.
    """

    def __init__(self, message: str, details: dict = None):
        """
        Initialize timbre exception with message and context.

        Args:
            message: Human-readable error description
            details: Additional error context as key-value pairs
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}


class TimbreAnalysisError(TimbreException):
    """
    Exception raised during timbre analysis operations.

    This includes errors in:
    - Feature extraction from audio
    - Timbre classification and recognition
    - Instrument identification failures
    - Audio preprocessing issues
    """

    pass


class TimbreTrainingError(TimbreException):
    """
    Exception raised during model training operations.

    This covers:
    - Training data loading failures
    - Model convergence issues
    - Validation errors
    - Checkpoint saving/loading problems
    """

    pass


class TimbreConversionError(TimbreException):
    """
    Exception raised during timbre conversion operations.

    This includes:
    - Style transfer failures
    - Audio synthesis errors
    - Format conversion issues
    - Quality degradation problems
    """

    pass


class ModelLoadError(TimbreException):
    """
    Exception raised when timbre models fail to load.

    This covers:
    - Missing model files
    - Incompatible model versions
    - Memory allocation failures
    - Device compatibility issues
    """

    pass


class FeatureExtractionError(TimbreException):
    """
    Exception raised during audio feature extraction.

    This includes:
    - Invalid audio format
    - Corrupted audio data
    - Feature computation failures
    - Unsupported sample rates
    """

    pass


class InstrumentRecognitionError(TimbreException):
    """
    Exception raised during instrument recognition.

    This covers:
    - Unknown instrument types
    - Ambiguous classification results
    - Low confidence predictions
    - Multiple instrument detection issues
    """

    pass
