"""
Data models for InstrumentTimbre framework.

Provides structured data classes for timbre analysis results,
audio features, and processing outcomes with type safety and validation.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .exceptions import TimbreException


class InstrumentType(Enum):
    """
    Chinese traditional instrument types with family classification.

    Organized by instrument families for hierarchical recognition
    and culturally-aware timbre analysis.
    """

    # String instruments (bowed strings)
    ERHU = "erhu"  # Chinese two-stringed violin
    VIOLIN = "violin"  # Western violin
    CELLO = "cello"  # Western cello

    # Plucked instruments (traditional Chinese)
    PIPA = "pipa"  # Chinese four-stringed lute
    GUZHENG = "guzheng"  # Chinese zither
    GUQIN = "guqin"  # Chinese seven-stringed zither
    RUAN = "ruan"  # Chinese plucked string instrument
    LIUQIN = "liuqin"  # Chinese small plucked instrument

    # Wind instruments (traditional and modern)
    DIZI = "dizi"  # Chinese bamboo flute
    XIAO = "xiao"  # Chinese vertical bamboo flute
    SUONA = "suona"  # Chinese double-reed horn
    SHENG = "sheng"  # Chinese mouth organ

    # Percussion instruments (metal and membrane)
    GONG = "gong"  # Chinese bronze gong
    DRUM = "drum"  # Various drums
    BELL = "bell"  # Various bells

    # Western instruments for comparison
    PIANO = "piano"
    GUITAR = "guitar"
    FLUTE = "flute"
    SAXOPHONE = "saxophone"

    @classmethod
    def get_chinese_instruments(cls) -> List["InstrumentType"]:
        """Get list of Chinese traditional instruments."""
        chinese_instruments = [
            cls.ERHU,
            cls.PIPA,
            cls.GUZHENG,
            cls.GUQIN,
            cls.RUAN,
            cls.LIUQIN,
            cls.DIZI,
            cls.XIAO,
            cls.SUONA,
            cls.SHENG,
            cls.GONG,
            cls.DRUM,
            cls.BELL,
        ]
        return chinese_instruments

    @classmethod
    def get_instrument_family(cls, instrument: "InstrumentType") -> str:
        """
        Get instrument family for given instrument.

        Args:
            instrument: Instrument type

        Returns:
            Family name string
        """
        string_family = [cls.ERHU, cls.VIOLIN, cls.CELLO]
        plucked_family = [cls.PIPA, cls.GUZHENG, cls.GUQIN, cls.RUAN, cls.LIUQIN]
        wind_family = [
            cls.DIZI,
            cls.XIAO,
            cls.SUONA,
            cls.SHENG,
            cls.FLUTE,
            cls.SAXOPHONE,
        ]
        percussion_family = [cls.GONG, cls.DRUM, cls.BELL]

        if instrument in string_family:
            return "string"
        elif instrument in plucked_family:
            return "plucked"
        elif instrument in wind_family:
            return "wind"
        elif instrument in percussion_family:
            return "percussion"
        elif instrument in [cls.PIANO, cls.GUITAR]:
            return "keyboard_fretted"
        else:
            return "unknown"


@dataclass
class AudioFeatures:
    """
    Container for extracted audio features.

    Stores various types of audio features with metadata
    for timbre analysis and instrument recognition.
    """

    # Spectral features
    mel_spectrogram: Optional[np.ndarray] = None
    mfcc: Optional[np.ndarray] = None
    chroma: Optional[np.ndarray] = None
    spectral_centroid: Optional[np.ndarray] = None
    spectral_rolloff: Optional[np.ndarray] = None
    spectral_bandwidth: Optional[np.ndarray] = None
    zero_crossing_rate: Optional[np.ndarray] = None

    # Temporal features
    tempo: Optional[float] = None
    beat_times: Optional[np.ndarray] = None

    # Harmonic features
    harmonic_ratio: Optional[float] = None
    fundamental_frequency: Optional[np.ndarray] = None

    # Metadata
    sample_rate: int = 22050
    duration: float = 0.0
    extraction_time: float = 0.0
    feature_types: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Post-initialization validation and feature type detection."""
        self._detect_available_features()
        self._validate_features()

    def _detect_available_features(self) -> None:
        """Detect which features are available in this instance."""
        self.feature_types = []

        if self.mel_spectrogram is not None:
            self.feature_types.append("mel_spectrogram")
        if self.mfcc is not None:
            self.feature_types.append("mfcc")
        if self.chroma is not None:
            self.feature_types.append("chroma")
        if self.spectral_centroid is not None:
            self.feature_types.append("spectral_centroid")
        if self.zero_crossing_rate is not None:
            self.feature_types.append("zero_crossing_rate")

    def _validate_features(self) -> None:
        """Validate feature dimensions and consistency."""
        if self.sample_rate <= 0:
            raise TimbreException("sample_rate must be positive")

        if self.duration < 0:
            raise TimbreException("duration cannot be negative")

    def get_feature_summary(self) -> Dict[str, Any]:
        """
        Get summary information about extracted features.

        Returns:
            Dictionary with feature summary statistics
        """
        summary = {
            "available_features": self.feature_types,
            "sample_rate": self.sample_rate,
            "duration": self.duration,
            "extraction_time": self.extraction_time,
        }

        if self.mel_spectrogram is not None:
            summary["mel_spectrogram_shape"] = self.mel_spectrogram.shape

        if self.mfcc is not None:
            summary["mfcc_shape"] = self.mfcc.shape

        if self.tempo is not None:
            summary["tempo"] = self.tempo

        return summary


@dataclass
class InstrumentPrediction:
    """
    Single instrument prediction with confidence and metadata.

    Represents a predicted instrument type with associated
    confidence score and analysis details.
    """

    instrument: InstrumentType
    confidence: float
    family: str = field(init=False)
    is_chinese: bool = field(init=False)

    def __post_init__(self):
        """Post-initialization to set derived fields."""
        self.family = InstrumentType.get_instrument_family(self.instrument)
        self.is_chinese = self.instrument in InstrumentType.get_chinese_instruments()

    def __str__(self) -> str:
        """String representation of prediction."""
        return f"{self.instrument.value} ({self.confidence:.3f})"


@dataclass
class TimbreResult:
    """
    Base result class for timbre processing operations.

    Provides common fields and functionality for all
    timbre analysis, training, and conversion results.
    """

    success: bool
    processing_time: float
    timestamp: float = field(default_factory=time.time)
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata entry."""
        self.metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value with optional default."""
        return self.metadata.get(key, default)


@dataclass
class AnalysisResult(TimbreResult):
    """
    Result of timbre analysis operation.

    Contains instrument predictions, confidence scores,
    and detailed analysis information.
    """

    # Analysis results
    predictions: List[InstrumentPrediction] = field(default_factory=list)
    features: Optional[AudioFeatures] = None
    audio_file: Optional[Path] = None

    # Analysis metadata
    analysis_method: str = "transformer"
    model_version: Optional[str] = None
    segment_results: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def top_prediction(self) -> Optional[InstrumentPrediction]:
        """Get the highest confidence prediction."""
        if not self.predictions:
            return None
        return max(self.predictions, key=lambda p: p.confidence)

    @property
    def chinese_predictions(self) -> List[InstrumentPrediction]:
        """Get predictions for Chinese instruments only."""
        return [p for p in self.predictions if p.is_chinese]

    def get_predictions_by_family(self, family: str) -> List[InstrumentPrediction]:
        """
        Get predictions filtered by instrument family.

        Args:
            family: Instrument family name

        Returns:
            List of predictions from specified family
        """
        return [p for p in self.predictions if p.family == family]

    def add_prediction(self, instrument: InstrumentType, confidence: float) -> None:
        """
        Add an instrument prediction.

        Args:
            instrument: Predicted instrument type
            confidence: Prediction confidence (0.0-1.0)
        """
        prediction = InstrumentPrediction(instrument=instrument, confidence=confidence)
        self.predictions.append(prediction)

        # Keep predictions sorted by confidence
        self.predictions.sort(key=lambda p: p.confidence, reverse=True)

    def get_confidence_summary(self) -> Dict[str, float]:
        """
        Get summary of confidence scores.

        Returns:
            Dictionary with confidence statistics
        """
        if not self.predictions:
            return {}

        confidences = [p.confidence for p in self.predictions]

        return {
            "max_confidence": max(confidences),
            "min_confidence": min(confidences),
            "mean_confidence": sum(confidences) / len(confidences),
            "confidence_spread": max(confidences) - min(confidences),
        }


@dataclass
class TrainingResult(TimbreResult):
    """
    Result of model training operation.

    Contains training metrics, model information,
    and performance statistics.
    """

    # Training metrics
    final_train_loss: float = 0.0
    final_val_loss: float = 0.0
    best_val_accuracy: float = 0.0
    epochs_completed: int = 0
    early_stopped: bool = False

    # Model information
    model_path: Optional[Path] = None
    model_size_mb: float = 0.0
    total_parameters: int = 0

    # Training history
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    val_accuracies: List[float] = field(default_factory=list)

    # Dataset information
    train_samples: int = 0
    val_samples: int = 0
    instrument_distribution: Dict[str, int] = field(default_factory=dict)

    def add_epoch_result(
        self, train_loss: float, val_loss: float, val_accuracy: float
    ) -> None:
        """
        Add training epoch result.

        Args:
            train_loss: Training loss for epoch
            val_loss: Validation loss for epoch
            val_accuracy: Validation accuracy for epoch
        """
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.val_accuracies.append(val_accuracy)

        # Update best metrics
        if val_accuracy > self.best_val_accuracy:
            self.best_val_accuracy = val_accuracy

        # Update final metrics
        self.final_train_loss = train_loss
        self.final_val_loss = val_loss
        self.epochs_completed = len(self.train_losses)

    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive training summary.

        Returns:
            Dictionary with training statistics
        """
        return {
            "epochs_completed": self.epochs_completed,
            "early_stopped": self.early_stopped,
            "best_val_accuracy": self.best_val_accuracy,
            "final_train_loss": self.final_train_loss,
            "final_val_loss": self.final_val_loss,
            "total_training_time": self.processing_time,
            "model_size_mb": self.model_size_mb,
            "total_parameters": self.total_parameters,
            "train_samples": self.train_samples,
            "val_samples": self.val_samples,
            "instrument_classes": len(self.instrument_distribution),
        }


@dataclass
class ConversionResult(TimbreResult):
    """
    Result of timbre conversion operation.

    Contains converted audio information and
    conversion quality metrics.
    """

    # Conversion outputs
    converted_audio_path: Optional[Path] = None
    source_instrument: Optional[InstrumentType] = None
    target_instrument: Optional[InstrumentType] = None

    # Quality metrics
    conversion_quality_score: float = 0.0
    spectral_similarity: float = 0.0
    perceptual_quality: float = 0.0

    # Conversion parameters
    conversion_strength: float = 1.0
    preserved_characteristics: List[str] = field(default_factory=list)

    def get_conversion_summary(self) -> Dict[str, Any]:
        """
        Get conversion operation summary.

        Returns:
            Dictionary with conversion details
        """
        return {
            "source_instrument": self.source_instrument.value
            if self.source_instrument
            else None,
            "target_instrument": self.target_instrument.value
            if self.target_instrument
            else None,
            "conversion_quality_score": self.conversion_quality_score,
            "spectral_similarity": self.spectral_similarity,
            "perceptual_quality": self.perceptual_quality,
            "conversion_strength": self.conversion_strength,
            "preserved_characteristics": self.preserved_characteristics,
            "processing_time": self.processing_time,
        }


@dataclass
class BatchProcessingResult:
    """
    Result of batch processing operations.

    Aggregates multiple individual results with
    summary statistics and error tracking.
    """

    individual_results: List[TimbreResult] = field(default_factory=list)
    total_files: int = 0
    successful_files: int = 0
    failed_files: int = 0
    total_processing_time: float = 0.0
    average_processing_time: float = 0.0

    def add_result(self, result: TimbreResult) -> None:
        """
        Add individual processing result to batch.

        Args:
            result: Individual processing result
        """
        self.individual_results.append(result)
        self.total_files += 1

        if result.success:
            self.successful_files += 1
        else:
            self.failed_files += 1

        self.total_processing_time += result.processing_time
        self.average_processing_time = self.total_processing_time / self.total_files

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_files == 0:
            return 0.0
        return (self.successful_files / self.total_files) * 100.0

    def get_failed_results(self) -> List[TimbreResult]:
        """Get list of failed processing results."""
        return [r for r in self.individual_results if not r.success]

    def get_successful_results(self) -> List[TimbreResult]:
        """Get list of successful processing results."""
        return [r for r in self.individual_results if r.success]

    def get_batch_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive batch processing summary.

        Returns:
            Dictionary with batch statistics
        """
        return {
            "total_files": self.total_files,
            "successful_files": self.successful_files,
            "failed_files": self.failed_files,
            "success_rate": self.success_rate,
            "total_processing_time": self.total_processing_time,
            "average_processing_time": self.average_processing_time,
            "throughput_files_per_second": self.total_files / self.total_processing_time
            if self.total_processing_time > 0
            else 0,
        }
