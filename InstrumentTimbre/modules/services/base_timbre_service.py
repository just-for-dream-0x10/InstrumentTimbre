"""
Base service class for InstrumentTimbre framework.

Provides common functionality for all timbre-related services
with unified error handling, performance monitoring, and logging.
"""

# Import base service from MusicAITools
import sys
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "MusicAITools"))

try:
    from modules.core.base_service import BaseService as MusicAIBaseService
except ImportError:
    # Fallback implementation if MusicAITools not available
    class MusicAIBaseService:
        def __init__(self, service_name: str):
            self.service_name = service_name


from ..core import (
    AudioFeatures,
    InstrumentType,
    TimbreException,
    TimbreResult,
    get_config,
    get_logger,
)


class BaseTimbreService(MusicAIBaseService):
    """
    Abstract base class for all timbre processing services.

    Extends MusicAITools BaseService with timbre-specific functionality
    including Chinese instrument handling and specialized monitoring.
    """

    def __init__(self, service_name: str):
        """
        Initialize base timbre service.

        Args:
            service_name: Human-readable name for the service
        """
        super().__init__(service_name)
        self.logger = get_logger()
        self.config = get_config()
        self._timbre_stats = {
            "chinese_instruments_processed": 0,
            "western_instruments_processed": 0,
            "feature_extraction_time": 0.0,
            "model_inference_time": 0.0,
        }

    def safe_process(self, input_data: Any, **kwargs) -> TimbreResult:
        """
        Safe wrapper for timbre processing with specialized error handling.

        Args:
            input_data: Input data for processing
            **kwargs: Additional processing parameters

        Returns:
            TimbreResult: Result with timbre-specific fields
        """
        start_time = time.time()
        operation_id = self._generate_operation_id()

        try:
            self.logger.info(f"Starting {self.service_name} operation {operation_id}")

            # Pre-processing validation
            self._validate_timbre_input(input_data, **kwargs)

            # Execute main processing
            result = self.process(input_data, **kwargs)

            # Post-processing validation
            self._validate_timbre_result(result)

            # Update timing and statistics
            processing_time = time.time() - start_time
            result.processing_time = processing_time

            self._update_timbre_stats(result)

            self.logger.info(
                f"{self.service_name} operation {operation_id} completed successfully "
                f"in {processing_time:.3f}s"
            )

            return result

        except TimbreException as e:
            processing_time = time.time() - start_time

            self.logger.error(
                f"{self.service_name} operation {operation_id} failed: {e}",
                extra={"operation_id": operation_id, "error_type": type(e).__name__},
            )

            return self._create_timbre_error_result(str(e), processing_time)

        except Exception as e:
            processing_time = time.time() - start_time

            self.logger.exception(
                f"Unexpected error in {self.service_name} operation {operation_id}: {e}",
                extra={"operation_id": operation_id},
            )

            return self._create_timbre_error_result(
                f"Unexpected error: {str(e)}", processing_time
            )

    @abstractmethod
    def process(self, input_data: Any, **kwargs) -> TimbreResult:
        """
        Main processing method to be implemented by subclasses.

        Args:
            input_data: Input data for processing
            **kwargs: Additional processing parameters

        Returns:
            TimbreResult: Processing result
        """
        pass

    def _validate_timbre_input(self, input_data: Any, **kwargs) -> None:
        """
        Validate input data for timbre processing.

        Args:
            input_data: Input data to validate
            **kwargs: Additional parameters to validate

        Raises:
            TimbreException: If validation fails
        """
        # Default implementation - subclasses should override
        if input_data is None:
            raise TimbreException("Input data cannot be None")

    def _validate_timbre_result(self, result: TimbreResult) -> None:
        """
        Validate timbre processing result.

        Args:
            result: Processing result to validate

        Raises:
            TimbreException: If result validation fails
        """
        if not isinstance(result, TimbreResult):
            raise TimbreException("Process method must return TimbreResult instance")

    def _validate_audio_file(self, file_path: Union[str, Path]) -> Path:
        """
        Validate audio file for timbre processing.

        Args:
            file_path: Path to audio file

        Returns:
            Path: Validated audio file path

        Raises:
            TimbreException: If file validation fails
        """
        audio_path = Path(file_path)

        if not audio_path.exists():
            raise TimbreException(f"Audio file does not exist: {audio_path}")

        if not audio_path.is_file():
            raise TimbreException(f"Path is not a regular file: {audio_path}")

        # Check file extension
        valid_extensions = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
        if audio_path.suffix.lower() not in valid_extensions:
            raise TimbreException(f"Unsupported audio format: {audio_path.suffix}")

        return audio_path

    def _create_timbre_error_result(
        self, error_message: str, processing_time: Optional[float] = None
    ) -> TimbreResult:
        """
        Create error result for timbre operations.

        Args:
            error_message: Description of the error
            processing_time: Time spent before error occurred

        Returns:
            TimbreResult: Error result object
        """
        return TimbreResult(
            success=False,
            processing_time=processing_time or 0.0,
            error_message=error_message,
        )

    def _update_timbre_stats(self, result: TimbreResult) -> None:
        """
        Update timbre-specific processing statistics.

        Args:
            result: Processing result to extract stats from
        """
        # Track Chinese vs Western instrument processing
        if hasattr(result, "predictions"):
            # Analysis result
            chinese_detected = (
                any(pred.is_chinese for pred in result.predictions)
                if result.predictions
                else False
            )

            if chinese_detected:
                self._timbre_stats["chinese_instruments_processed"] += 1
            else:
                self._timbre_stats["western_instruments_processed"] += 1

        # Track feature extraction time if available
        if hasattr(result, "features") and result.features:
            self._timbre_stats[
                "feature_extraction_time"
            ] += result.features.extraction_time

    def _generate_operation_id(self) -> str:
        """
        Generate unique operation ID for tracking.

        Returns:
            Unique operation identifier
        """
        import uuid

        return str(uuid.uuid4())[:8]

    @contextmanager
    def _performance_monitor(self, operation_name: str):
        """
        Context manager for performance monitoring.

        Args:
            operation_name: Name of operation being monitored
        """
        start_time = time.time()

        try:
            yield
        finally:
            duration = time.time() - start_time

            self.logger.debug(
                f"Performance: {operation_name} took {duration:.3f}s",
                extra={
                    "operation": operation_name,
                    "duration": duration,
                    "service": self.service_name,
                },
            )

    def get_timbre_stats(self) -> Dict[str, Any]:
        """
        Get timbre-specific processing statistics.

        Returns:
            Dictionary with timbre processing statistics
        """
        base_stats = getattr(self, "get_stats", lambda: {})()

        timbre_stats = self._timbre_stats.copy()
        timbre_stats.update(base_stats)

        # Calculate derived statistics
        total_instruments = (
            timbre_stats["chinese_instruments_processed"]
            + timbre_stats["western_instruments_processed"]
        )

        if total_instruments > 0:
            timbre_stats["chinese_instrument_percentage"] = (
                timbre_stats["chinese_instruments_processed"] / total_instruments * 100
            )
        else:
            timbre_stats["chinese_instrument_percentage"] = 0.0

        return timbre_stats

    def reset_timbre_stats(self) -> None:
        """Reset timbre-specific statistics."""
        self._timbre_stats = {
            "chinese_instruments_processed": 0,
            "western_instruments_processed": 0,
            "feature_extraction_time": 0.0,
            "model_inference_time": 0.0,
        }

        # Reset base stats if available
        if hasattr(self, "reset_stats"):
            self.reset_stats()

    def _load_model_safely(self, model_path: Union[str, Path]) -> Any:
        """
        Safely load a timbre model with error handling.

        Args:
            model_path: Path to model file

        Returns:
            Loaded model object

        Raises:
            TimbreException: If model loading fails
        """
        model_path = Path(model_path)

        if not model_path.exists():
            raise TimbreException(f"Model file does not exist: {model_path}")

        try:
            import torch

            # Load model with appropriate device handling
            device = self._get_device()
            model = torch.load(model_path, map_location=device)

            self.logger.info(f"Successfully loaded model from {model_path}")
            return model

        except Exception as e:
            raise TimbreException(f"Failed to load model from {model_path}: {e}")

    def _get_device(self) -> str:
        """
        Get appropriate device for model inference.

        Returns:
            Device string ('cuda', 'mps', or 'cpu')
        """
        device_config = self.config.system.device

        if device_config == "auto":
            try:
                import torch

                if torch.cuda.is_available():
                    return "cuda"
                elif torch.backends.mps.is_available():
                    return "mps"
                else:
                    return "cpu"
            except ImportError:
                return "cpu"
        else:
            return device_config

    def _ensure_output_directory(self, output_path: Path) -> None:
        """
        Ensure output directory exists for timbre processing.

        Args:
            output_path: Path that should have existing parent directory
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Log directory creation
        if not output_path.parent.exists():
            self.logger.debug(f"Created output directory: {output_path.parent}")

    def _get_chinese_instrument_priority(self, instrument: InstrumentType) -> float:
        """
        Get priority boost for Chinese instruments if enabled.

        Args:
            instrument: Instrument type to check

        Returns:
            Priority boost value
        """
        if (
            self.config.analysis.prioritize_chinese_instruments
            and instrument in InstrumentType.get_chinese_instruments()
        ):
            return self.config.analysis.chinese_instrument_boost
        return 0.0
