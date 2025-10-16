"""
Core components for InstrumentTimbre framework.

This module provides fundamental infrastructure including:
- Exception handling specific to timbre analysis
- Logging system integration
- Configuration management for timbre processing
- Data models for timbre analysis results
"""

from .config import (
    AnalysisConfig,
    TimbreConfig,
    TimbreConfigManager,
    TrainingConfig,
    get_config,
)
from .exceptions import (
    FeatureExtractionError,
    InstrumentRecognitionError,
    ModelLoadError,
    TimbreAnalysisError,
    TimbreConversionError,
    TimbreException,
    TimbreTrainingError,
)
from .logger import (
    get_logger,
    log_feature_extraction,
    log_instrument_detection,
    log_training_epoch,
    setup_logging,
)
from .models import (
    AnalysisResult,
    AudioFeatures,
    BatchProcessingResult,
    ConversionResult,
    InstrumentPrediction,
    InstrumentType,
    TimbreResult,
    TrainingResult,
)

__all__ = [
    # Exceptions
    "TimbreException",
    "TimbreAnalysisError",
    "TimbreTrainingError",
    "TimbreConversionError",
    "ModelLoadError",
    "FeatureExtractionError",
    "InstrumentRecognitionError",
    # Logging
    "get_logger",
    "setup_logging",
    "log_training_epoch",
    "log_feature_extraction",
    "log_instrument_detection",
    # Configuration
    "TimbreConfig",
    "TrainingConfig",
    "AnalysisConfig",
    "get_config",
    "TimbreConfigManager",
    # Models
    "TimbreResult",
    "AnalysisResult",
    "TrainingResult",
    "ConversionResult",
    "AudioFeatures",
    "InstrumentType",
    "InstrumentPrediction",
    "BatchProcessingResult",
]
