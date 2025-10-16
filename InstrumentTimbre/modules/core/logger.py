"""
Logging system for InstrumentTimbre framework.

Provides unified logging with performance monitoring and structured
output specifically designed for timbre analysis operations.
"""

import logging

# Import base logger from MusicAITools
import sys
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "MusicAITools"))

try:
    from modules.core.logger import (
        MusicAILogger,
        log_error_with_context,
        log_performance,
    )
except ImportError:
    # Fallback if MusicAITools not available
    class MusicAILogger:
        _instance = None
        _logger = None

        def __new__(cls):
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance

        def setup_logger(
            self, name: str = "instrumenttimbre", level: int = logging.INFO
        ):
            if self._logger is not None:
                return self._logger

            self._logger = logging.getLogger(name)
            self._logger.setLevel(level)

            if not self._logger.handlers:
                formatter = logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s"
                )

                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setLevel(level)
                console_handler.setFormatter(formatter)
                self._logger.addHandler(console_handler)

            return self._logger

        def get_logger(self):
            if self._logger is None:
                return self.setup_logger()
            return self._logger

    def log_performance(
        operation: str, duration: float, details: Dict[str, Any] = None
    ):
        logger = get_logger()
        details_str = f" | {details}" if details else ""
        logger.info(
            f"PERFORMANCE: {operation} completed in {duration:.3f}s{details_str}"
        )

    def log_error_with_context(error: Exception, context: Dict[str, Any] = None):
        logger = get_logger()
        context_str = f" | Context: {context}" if context else ""
        logger.error(f"Error: {type(error).__name__}: {str(error)}{context_str}")


class TimbreLogger:
    """
    Specialized logger for timbre analysis operations.

    Extends the base logging system with timbre-specific functionality
    including instrument recognition logging and training progress tracking.
    """

    def __init__(self):
        """Initialize timbre-specific logger."""
        self.base_logger = MusicAILogger()
        self.logger = self.base_logger.setup_logger("instrumenttimbre")

    def log_instrument_detection(
        self,
        audio_file: str,
        detected_instrument: str,
        confidence: float,
        processing_time: float,
    ):
        """
        Log instrument detection results.

        Args:
            audio_file: Path to analyzed audio file
            detected_instrument: Identified instrument name
            confidence: Detection confidence score (0.0-1.0)
            processing_time: Analysis time in seconds
        """
        self.logger.info(
            f"INSTRUMENT_DETECTION: {Path(audio_file).name} -> {detected_instrument} "
            f"(confidence: {confidence:.3f}, time: {processing_time:.3f}s)"
        )

    def log_training_epoch(
        self,
        epoch: int,
        total_epochs: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        accuracy: Optional[float] = None,
    ):
        """
        Log training epoch progress.

        Args:
            epoch: Current epoch number
            total_epochs: Total number of epochs
            train_loss: Training loss value
            val_loss: Validation loss (optional)
            accuracy: Validation accuracy (optional)
        """
        progress = f"[{epoch}/{total_epochs}]"
        loss_info = f"train_loss: {train_loss:.4f}"

        if val_loss is not None:
            loss_info += f", val_loss: {val_loss:.4f}"

        if accuracy is not None:
            loss_info += f", accuracy: {accuracy:.4f}"

        self.logger.info(f"TRAINING_EPOCH: {progress} {loss_info}")

    def log_feature_extraction(
        self, audio_file: str, features_extracted: list, extraction_time: float
    ):
        """
        Log feature extraction operations.

        Args:
            audio_file: Path to processed audio file
            features_extracted: List of extracted feature types
            extraction_time: Time taken for extraction
        """
        features_str = ", ".join(features_extracted)
        self.logger.debug(
            f"FEATURE_EXTRACTION: {Path(audio_file).name} -> [{features_str}] "
            f"(time: {extraction_time:.3f}s)"
        )

    def log_model_performance(
        self, model_name: str, dataset_size: int, accuracy: float, inference_time: float
    ):
        """
        Log model performance metrics.

        Args:
            model_name: Name of the evaluated model
            dataset_size: Size of test dataset
            accuracy: Overall accuracy score
            inference_time: Average inference time per sample
        """
        self.logger.info(
            f"MODEL_PERFORMANCE: {model_name} | "
            f"dataset: {dataset_size} samples | "
            f"accuracy: {accuracy:.4f} | "
            f"avg_inference: {inference_time:.3f}s"
        )


# Global logger instance
_timbre_logger = TimbreLogger()


def get_logger() -> logging.Logger:
    """
    Get the global timbre logger instance.

    Returns:
        Configured logger for timbre operations
    """
    return _timbre_logger.logger


def setup_logging(
    level: int = logging.INFO, log_file: Optional[Path] = None
) -> logging.Logger:
    """
    Setup timbre logging configuration.

    Args:
        level: Minimum logging level
        log_file: Optional file path for log output

    Returns:
        Configured logger instance
    """
    logger = _timbre_logger.base_logger.setup_logger(
        name="instrumenttimbre", level=level
    )
    return logger


def log_instrument_detection(
    audio_file: str, detected_instrument: str, confidence: float, processing_time: float
):
    """
    Log instrument detection results.

    Args:
        audio_file: Path to analyzed audio file
        detected_instrument: Identified instrument name
        confidence: Detection confidence score
        processing_time: Analysis time in seconds
    """
    _timbre_logger.log_instrument_detection(
        audio_file, detected_instrument, confidence, processing_time
    )


def log_training_epoch(
    epoch: int,
    total_epochs: int,
    train_loss: float,
    val_loss: Optional[float] = None,
    accuracy: Optional[float] = None,
):
    """
    Log training epoch progress.

    Args:
        epoch: Current epoch number
        total_epochs: Total number of epochs
        train_loss: Training loss value
        val_loss: Validation loss (optional)
        accuracy: Validation accuracy (optional)
    """
    _timbre_logger.log_training_epoch(
        epoch, total_epochs, train_loss, val_loss, accuracy
    )


def log_feature_extraction(
    audio_file: str, features_extracted: list, extraction_time: float
):
    """
    Log feature extraction operations.

    Args:
        audio_file: Path to processed audio file
        features_extracted: List of extracted feature types
        extraction_time: Time taken for extraction
    """
    _timbre_logger.log_feature_extraction(
        audio_file, features_extracted, extraction_time
    )
