"""
Timbre conversion service for style transfer.

Provides timbre conversion capabilities between different instruments
with preservation of musical characteristics and Chinese instrument focus.
"""

import time
from pathlib import Path
from typing import Optional, Union

import librosa
import numpy as np
import soundfile as sf

from ..core import ConversionResult, InstrumentType, TimbreConversionError, get_logger
from .base_timbre_service import BaseTimbreService


class TimbreConversionService(BaseTimbreService):
    """
    Professional timbre conversion service.

    Converts audio between different instrument timbres while
    preserving musical content and applying Chinese instrument characteristics.
    """

    def __init__(self):
        """Initialize timbre conversion service."""
        super().__init__("TimbreConversion")
        self.conversion_model = None
        self._load_conversion_model()

    def process(
        self,
        input_audio: str,
        target_instrument: Union[str, InstrumentType],
        output_file: Optional[str] = None,
        conversion_strength: Optional[float] = None,
    ) -> ConversionResult:
        """
        Convert audio timbre to target instrument.

        Args:
            input_audio: Path to input audio file
            target_instrument: Target instrument for conversion
            output_file: Output file path (auto-generated if None)
            conversion_strength: Conversion strength (0.0-1.0, uses config default if None)

        Returns:
            ConversionResult: Conversion results with quality metrics

        Raises:
            TimbreConversionError: If conversion fails
        """
        try:
            # Validate inputs
            audio_path = self._validate_audio_file(input_audio)
            target_inst = self._parse_target_instrument(target_instrument)
            strength = conversion_strength or self.config.conversion.conversion_strength

            # Prepare output path
            if output_file is None:
                output_file = self._generate_output_path(audio_path, target_inst)
            output_path = Path(output_file)
            self._ensure_output_directory(output_path)

            # Load and preprocess audio
            with self._performance_monitor("audio_loading"):
                audio_data, sample_rate = self._load_audio_for_conversion(audio_path)

            # Perform timbre conversion
            with self._performance_monitor("timbre_conversion"):
                converted_audio = self._convert_timbre(
                    audio_data, sample_rate, target_inst, strength
                )

            # Post-process and save
            with self._performance_monitor("audio_saving"):
                self._save_converted_audio(converted_audio, sample_rate, output_path)

            # Calculate quality metrics
            with self._performance_monitor("quality_assessment"):
                quality_metrics = self._assess_conversion_quality(
                    audio_data, converted_audio, sample_rate
                )

            # Create result
            result = ConversionResult(
                success=True,
                processing_time=0.0,  # Will be set by safe_process
                converted_audio_path=output_path,
                target_instrument=target_inst,
                conversion_quality_score=quality_metrics.get("overall_quality", 0.0),
                spectral_similarity=quality_metrics.get("spectral_similarity", 0.0),
                conversion_strength=strength,
            )

            # Add metadata
            result.add_metadata("input_file", str(audio_path))
            result.add_metadata("target_instrument", target_inst.value)
            result.add_metadata("quality_metrics", quality_metrics)

            self.logger.info(
                f"Conversion completed: {audio_path.name} -> {target_inst.value} "
                f"(quality: {result.conversion_quality_score:.3f})"
            )

            return result

        except Exception as e:
            raise TimbreConversionError(f"Timbre conversion failed: {e}")

    def _load_conversion_model(self) -> None:
        """Load the timbre conversion model."""
        try:
            model_path = Path(self.config.system.model_dir) / "conversion_model.pt"

            if model_path.exists():
                self.conversion_model = self._load_model_safely(model_path)
                self.conversion_model.eval()
                self.logger.info("Loaded timbre conversion model")
            else:
                self.logger.warning(f"Conversion model not found at {model_path}")
                self.conversion_model = None

        except Exception as e:
            self.logger.error(f"Failed to load conversion model: {e}")
            self.conversion_model = None
