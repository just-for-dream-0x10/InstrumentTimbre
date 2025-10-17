"""
Timbre analysis service for instrument recognition.

Provides comprehensive timbre analysis with Chinese instrument specialization,
feature extraction, and confidence-based instrument recognition.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import librosa
import numpy as np
import torch
import torch.nn.functional as F

from ..core import (
    AnalysisResult,
    AudioFeatures,
    FeatureExtractionError,
    InstrumentPrediction,
    InstrumentType,
    TimbreAnalysisError,
    get_logger,
)
from .base_timbre_service import BaseTimbreService


class TimbreAnalysisService(BaseTimbreService):
    """
    Professional timbre analysis service for instrument recognition.

    Specializes in Chinese traditional instruments with advanced
    feature extraction and transformer-based classification.
    """

    def __init__(self):
        """Initialize timbre analysis service."""
        super().__init__("TimbreAnalysis")
        self.model = None
        self.feature_extractor = None
        self._load_analysis_model()

    def process(
        self,
        audio_file: str,
        extract_features: bool = True,
        return_segments: bool = False,
    ) -> AnalysisResult:
        """
        Analyze audio file for instrument timbre recognition.

        Args:
            audio_file: Path to audio file for analysis
            extract_features: Whether to extract and return audio features
            return_segments: Whether to analyze segments separately

        Returns:
            AnalysisResult: Comprehensive analysis results

        Raises:
            TimbreAnalysisError: If analysis fails
        """
        # Validate input
        audio_path = self._validate_audio_file(audio_file)

        try:
            # Load and preprocess audio
            with self._performance_monitor("audio_loading"):
                audio_data, sample_rate = self._load_audio(audio_path)

            # Extract features
            features = None
            if extract_features:
                with self._performance_monitor("feature_extraction"):
                    features = self._extract_audio_features(audio_data, sample_rate)

            # Perform instrument recognition
            with self._performance_monitor("instrument_recognition"):
                predictions = self._recognize_instruments(audio_data, sample_rate)

            # Segment analysis if requested
            segment_results = []
            if return_segments:
                with self._performance_monitor("segment_analysis"):
                    segment_results = self._analyze_segments(audio_data, sample_rate)

            # Create result
            result = AnalysisResult(
                success=True,
                processing_time=0.0,  # Will be set by safe_process
                predictions=predictions,
                features=features,
                audio_file=audio_path,
                analysis_method="transformer_chinese_specialized",
                segment_results=segment_results,
            )

            # Add metadata
            result.add_metadata("sample_rate", sample_rate)
            result.add_metadata("audio_duration", len(audio_data) / sample_rate)
            result.add_metadata(
                "chinese_instruments_detected", len(result.chinese_predictions)
            )

            self.logger.info(
                f"Analysis completed for {audio_path.name}: "
                f"{len(predictions)} predictions, "
                f"top: {result.top_prediction}"
            )

            return result

        except Exception as e:
            raise TimbreAnalysisError(f"Timbre analysis failed for {audio_path}: {e}")

    def _load_analysis_model(self) -> None:
        """Load the timbre analysis model."""
        try:
            model_path = Path(self.config.system.model_dir) / "timbre_model.pt"

            if model_path.exists():
                self.model = self._load_model_safely(model_path)
                self.model.eval()
                self.logger.info("Loaded timbre analysis model")
            else:
                self.logger.warning(f"Model not found at {model_path}, using fallback")
                self.model = None

        except Exception as e:
            self.logger.error(f"Failed to load analysis model: {e}")
            self.model = None

    def _load_audio(self, audio_path: Path) -> tuple:
        """
        Load audio file with proper preprocessing.

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            # Load with target sample rate
            target_sr = self.config.analysis.sample_rate
            audio_data, sample_rate = librosa.load(
                str(audio_path), sr=target_sr, mono=True
            )

            # Ensure minimum length
            min_duration = 0.1  # 100ms minimum
            min_samples = int(min_duration * sample_rate)

            if len(audio_data) < min_samples:
                raise TimbreAnalysisError(
                    f"Audio too short: {len(audio_data)/sample_rate:.2f}s "
                    f"(minimum: {min_duration}s)"
                )

            return audio_data, sample_rate

        except Exception as e:
            raise TimbreAnalysisError(f"Failed to load audio from {audio_path}: {e}")

    def _extract_audio_features(
        self, audio_data: np.ndarray, sample_rate: int
    ) -> AudioFeatures:
        """
        Extract comprehensive audio features for timbre analysis.

        Args:
            audio_data: Audio signal data
            sample_rate: Audio sample rate

        Returns:
            AudioFeatures: Extracted feature bundle
        """
        start_time = time.time()

        try:
            features = AudioFeatures(
                sample_rate=sample_rate, duration=len(audio_data) / sample_rate
            )

            # Extract spectral features
            if "mel" in [
                ft.value for ft in self.config.analysis.feature_types
            ] or "multi" in [ft.value for ft in self.config.analysis.feature_types]:
                features.mel_spectrogram = librosa.feature.melspectrogram(
                    y=audio_data,
                    sr=sample_rate,
                    n_fft=self.config.analysis.n_fft,
                    hop_length=self.config.analysis.hop_length,
                    n_mels=self.config.analysis.n_mels,
                )

            # Extract MFCC features
            if "mfcc" in [
                ft.value for ft in self.config.analysis.feature_types
            ] or "multi" in [ft.value for ft in self.config.analysis.feature_types]:
                features.mfcc = librosa.feature.mfcc(
                    y=audio_data,
                    sr=sample_rate,
                    n_mfcc=self.config.analysis.n_mfcc,
                    n_fft=self.config.analysis.n_fft,
                    hop_length=self.config.analysis.hop_length,
                )

            # Extract chroma features
            if "chroma" in [
                ft.value for ft in self.config.analysis.feature_types
            ] or "multi" in [ft.value for ft in self.config.analysis.feature_types]:
                features.chroma = librosa.feature.chroma_stft(
                    y=audio_data,
                    sr=sample_rate,
                    n_fft=self.config.analysis.n_fft,
                    hop_length=self.config.analysis.hop_length,
                )

            # Extract spectral features
            features.spectral_centroid = librosa.feature.spectral_centroid(
                y=audio_data, sr=sample_rate
            )

            features.spectral_rolloff = librosa.feature.spectral_rolloff(
                y=audio_data, sr=sample_rate
            )

            features.zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_data)

            # Extract tempo
            try:
                tempo, _ = librosa.beat.beat_track(y=audio_data, sr=sample_rate)
                features.tempo = float(tempo)
            except Exception:
                features.tempo = None

            # Record extraction time
            features.extraction_time = time.time() - start_time

            self.logger.debug(
                f"Extracted features: {features.feature_types} "
                f"in {features.extraction_time:.3f}s"
            )

            return features

        except Exception as e:
            raise FeatureExtractionError(f"Feature extraction failed: {e}")

    def _recognize_instruments(
        self, audio_data: np.ndarray, sample_rate: int
    ) -> List[InstrumentPrediction]:
        """
        Recognize instruments from audio using trained model.

        Args:
            audio_data: Audio signal data
            sample_rate: Audio sample rate

        Returns:
            List of instrument predictions with confidence scores
        """
        try:
            if self.model is None:
                # Fallback to simple heuristic classification
                return self._fallback_instrument_recognition(audio_data, sample_rate)

            # Prepare input for model
            model_input = self._prepare_model_input(audio_data, sample_rate)

            # Run inference
            device = self._get_device()
            model_input = model_input.to(device)

            with torch.no_grad():
                outputs = self.model(model_input)
                probabilities = F.softmax(outputs, dim=-1)

            # Convert to predictions
            predictions = self._convert_model_output_to_predictions(probabilities)

            # Apply Chinese instrument priority boost
            predictions = self._apply_chinese_instrument_boost(predictions)

            # Filter by confidence threshold
            min_confidence = self.config.analysis.min_confidence_threshold
            predictions = [p for p in predictions if p.confidence >= min_confidence]

            # Limit number of predictions
            max_candidates = self.config.analysis.max_candidates
            predictions = predictions[:max_candidates]

            return predictions

        except Exception as e:
            self.logger.error(f"Instrument recognition failed: {e}")
            return self._fallback_instrument_recognition(audio_data, sample_rate)

    def _fallback_instrument_recognition(
        self, audio_data: np.ndarray, sample_rate: int
    ) -> List[InstrumentPrediction]:
        """
        Fallback instrument recognition using heuristics.

        Args:
            audio_data: Audio signal data
            sample_rate: Audio sample rate

        Returns:
            List of heuristic-based predictions
        """
        predictions = []

        # Simple heuristic based on spectral characteristics
        spectral_centroid = np.mean(
            librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)
        )

        zcr = np.mean(librosa.feature.zero_crossing_rate(audio_data))

        # Heuristic rules for Chinese instruments
        if spectral_centroid < 1000 and zcr < 0.1:
            # Low frequency, smooth -> likely string instrument
            predictions.append(InstrumentPrediction(InstrumentType.ERHU, 0.6))
            predictions.append(InstrumentPrediction(InstrumentType.GUZHENG, 0.5))
        elif spectral_centroid > 2000:
            # High frequency -> likely wind instrument
            predictions.append(InstrumentPrediction(InstrumentType.DIZI, 0.6))
            predictions.append(InstrumentPrediction(InstrumentType.SUONA, 0.5))
        else:
            # Medium frequency -> plucked instrument
            predictions.append(InstrumentPrediction(InstrumentType.PIPA, 0.6))
            predictions.append(InstrumentPrediction(InstrumentType.GUQIN, 0.5))

        self.logger.warning("Using fallback heuristic instrument recognition")
        return predictions

    def _prepare_model_input(
        self, audio_data: np.ndarray, sample_rate: int
    ) -> torch.Tensor:
        """
        Prepare audio data for model input.

        Args:
            audio_data: Audio signal data
            sample_rate: Audio sample rate

        Returns:
            Tensor ready for model inference
        """
        # Extract mel spectrogram for model input
        mel_spec = librosa.feature.melspectrogram(
            y=audio_data,
            sr=sample_rate,
            n_fft=self.config.analysis.n_fft,
            hop_length=self.config.analysis.hop_length,
            n_mels=self.config.analysis.n_mels,
        )

        # Convert to log scale
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)

        # Normalize
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)

        # Convert to tensor and add batch dimension
        tensor = torch.FloatTensor(log_mel).unsqueeze(0)

        return tensor

    def _convert_model_output_to_predictions(
        self, probabilities: torch.Tensor
    ) -> List[InstrumentPrediction]:
        """
        Convert model output probabilities to instrument predictions.

        Args:
            probabilities: Model output probabilities

        Returns:
            List of instrument predictions
        """
        predictions = []

        # Get all instrument types (this would normally come from model metadata)
        instrument_classes = list(InstrumentType)

        probs = probabilities.cpu().numpy().flatten()

        for i, prob in enumerate(probs):
            if i < len(instrument_classes):
                instrument = instrument_classes[i]
                prediction = InstrumentPrediction(
                    instrument=instrument, confidence=float(prob)
                )
                predictions.append(prediction)

        # Sort by confidence
        predictions.sort(key=lambda p: p.confidence, reverse=True)

        return predictions

    def _apply_chinese_instrument_boost(
        self, predictions: List[InstrumentPrediction]
    ) -> List[InstrumentPrediction]:
        """
        Apply confidence boost to Chinese instruments if enabled.

        Args:
            predictions: Original predictions

        Returns:
            Predictions with applied boost
        """
        if not self.config.analysis.prioritize_chinese_instruments:
            return predictions

        boost = self.config.analysis.chinese_instrument_boost

        for prediction in predictions:
            if prediction.is_chinese:
                # Apply boost but keep confidence <= 1.0
                prediction.confidence = min(1.0, prediction.confidence + boost)

        # Re-sort after boost
        predictions.sort(key=lambda p: p.confidence, reverse=True)

        return predictions

    def _analyze_segments(
        self, audio_data: np.ndarray, sample_rate: int
    ) -> List[Dict[str, Any]]:
        """
        Analyze audio in segments for temporal analysis.

        Args:
            audio_data: Audio signal data
            sample_rate: Audio sample rate

        Returns:
            List of segment analysis results
        """
        segment_length = self.config.analysis.segment_length
        overlap_ratio = self.config.analysis.overlap_ratio

        segment_samples = int(segment_length * sample_rate)
        hop_samples = int(segment_samples * (1 - overlap_ratio))

        segment_results = []

        for start_sample in range(0, len(audio_data) - segment_samples, hop_samples):
            end_sample = start_sample + segment_samples
            segment = audio_data[start_sample:end_sample]

            # Analyze segment
            segment_predictions = self._recognize_instruments(segment, sample_rate)

            segment_result = {
                "start_time": start_sample / sample_rate,
                "end_time": end_sample / sample_rate,
                "top_prediction": segment_predictions[0]
                if segment_predictions
                else None,
                "all_predictions": segment_predictions,
            }

            segment_results.append(segment_result)

        return segment_results
