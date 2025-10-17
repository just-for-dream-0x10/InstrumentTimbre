"""
InstrumentTimbre modules package.

Modern architecture for Chinese traditional instrument timbre analysis.
Provides unified services for timbre analysis, training, and conversion.
"""

from .core import (
    AnalysisResult,
    TimbreAnalysisError,
    TimbreConfig,
    TimbreException,
    TimbreResult,
    TimbreTrainingError,
    TrainingResult,
    get_config,
    get_logger,
)
from .services import (
    TimbreAnalysisService,
    TimbreConversionService,
    TimbreTrainingService,
)

__all__ = [
    # Core components
    "TimbreException",
    "TimbreAnalysisError",
    "TimbreTrainingError",
    "get_logger",
    "get_config",
    "TimbreConfig",
    "TimbreResult",
    "AnalysisResult",
    "TrainingResult",
    # Services
    "TimbreAnalysisService",
    "TimbreTrainingService",
    "TimbreConversionService",
]
