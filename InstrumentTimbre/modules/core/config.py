"""
Configuration management for InstrumentTimbre framework.

Provides centralized configuration with validation for timbre analysis,
training, and conversion operations with Chinese instrument specialization.
"""

import json
import os
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from .exceptions import TimbreException


class InstrumentFamily(Enum):
    """Chinese traditional instrument families."""

    STRING = "string"  # String instruments (erhu, pipa, guzheng)
    WIND = "wind"  # Wind instruments (dizi, xiao, suona)
    PERCUSSION = "percussion"  # Percussion instruments (gong, drum, bell)
    PLUCKED = "plucked"  # Plucked instruments (pipa, guqin, ruan)


class FeatureType(Enum):
    """Audio feature extraction types."""

    MEL_SPECTROGRAM = "mel"
    MFCC = "mfcc"
    CHROMA = "chroma"
    CONSTANT_Q = "constant_q"
    SPECTRAL_CENTROID = "spectral_centroid"
    ZERO_CROSSING_RATE = "zcr"
    MULTI_FEATURE = "multi"


@dataclass
class TrainingConfig:
    """
    Configuration for timbre model training.

    Controls training hyperparameters, data augmentation,
    and optimization settings for Chinese instrument models.
    """

    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    scheduler_step_size: int = 30
    scheduler_gamma: float = 0.1
    early_stopping_patience: int = 15
    validation_split: float = 0.2

    # Data augmentation
    enable_augmentation: bool = True
    pitch_shift_range: float = 2.0  # semitones
    time_stretch_range: float = 0.2  # ratio
    noise_factor: float = 0.01

    # Chinese instrument specific
    chinese_instruments_only: bool = True
    enable_instrument_family_loss: bool = True
    family_loss_weight: float = 0.3

    # Model architecture
    model_size: str = "base"  # tiny, base, large
    use_attention: bool = True
    dropout_rate: float = 0.1

    def validate(self) -> None:
        """
        Validate training configuration parameters.

        Raises:
            TimbreException: If parameters are invalid
        """
        if not 1 <= self.epochs <= 1000:
            raise TimbreException("epochs must be between 1 and 1000")

        if not 1 <= self.batch_size <= 256:
            raise TimbreException("batch_size must be between 1 and 256")

        if not 1e-6 <= self.learning_rate <= 1.0:
            raise TimbreException("learning_rate must be between 1e-6 and 1.0")

        if not 0.0 <= self.validation_split <= 0.5:
            raise TimbreException("validation_split must be between 0.0 and 0.5")

        if self.model_size not in ["tiny", "base", "large"]:
            raise TimbreException("model_size must be 'tiny', 'base', or 'large'")


@dataclass
class AnalysisConfig:
    """
    Configuration for timbre analysis operations.

    Controls feature extraction, model inference, and
    result post-processing for instrument recognition.
    """

    # Feature extraction
    feature_types: List[FeatureType] = field(
        default_factory=lambda: [FeatureType.MULTI_FEATURE]
    )
    sample_rate: int = 22050
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 128
    n_mfcc: int = 13

    # Analysis parameters
    segment_length: float = 3.0  # seconds
    overlap_ratio: float = 0.5
    min_confidence_threshold: float = 0.7
    max_candidates: int = 3

    # Chinese instrument focus
    prioritize_chinese_instruments: bool = True
    chinese_instrument_boost: float = 0.1

    # Post-processing
    enable_smoothing: bool = True
    smoothing_window_size: int = 5
    enable_confidence_calibration: bool = True

    def validate(self) -> None:
        """
        Validate analysis configuration parameters.

        Raises:
            TimbreException: If parameters are invalid
        """
        if not 8000 <= self.sample_rate <= 96000:
            raise TimbreException("sample_rate must be between 8000 and 96000")

        if not 256 <= self.n_fft <= 8192:
            raise TimbreException("n_fft must be between 256 and 8192")

        if not 0.1 <= self.segment_length <= 30.0:
            raise TimbreException("segment_length must be between 0.1 and 30.0 seconds")

        if not 0.0 <= self.min_confidence_threshold <= 1.0:
            raise TimbreException(
                "min_confidence_threshold must be between 0.0 and 1.0"
            )


@dataclass
class ConversionConfig:
    """
    Configuration for timbre conversion operations.

    Controls style transfer parameters and audio synthesis
    settings for converting between instrument timbres.
    """

    # Conversion parameters
    conversion_strength: float = 0.8
    preserve_dynamics: bool = True
    preserve_pitch: bool = True

    # Quality settings
    output_sample_rate: int = 44100
    output_bit_depth: int = 16
    output_format: str = "wav"

    # Chinese instrument conversion
    enable_traditional_tuning: bool = True
    apply_cultural_characteristics: bool = True

    def validate(self) -> None:
        """
        Validate conversion configuration parameters.

        Raises:
            TimbreException: If parameters are invalid
        """
        if not 0.0 <= self.conversion_strength <= 1.0:
            raise TimbreException("conversion_strength must be between 0.0 and 1.0")

        if self.output_format not in ["wav", "flac", "mp3"]:
            raise TimbreException("output_format must be 'wav', 'flac', or 'mp3'")


@dataclass
class SystemConfig:
    """
    System-level configuration for the framework.

    Controls resource management, caching, and performance
    optimization settings.
    """

    # Data paths (following rules.md)
    training_data_dir: str = "../wav"  # Upper directory wav files
    output_dir: str = "output"
    cache_dir: str = "cache"
    model_dir: str = "saved_models"

    # Performance
    device: str = "auto"  # auto, cpu, cuda, mps
    enable_mixed_precision: bool = True
    max_workers: int = 4

    # Caching
    enable_feature_cache: bool = True
    cache_size_limit_mb: int = 2000

    # Logging
    log_level: str = "INFO"
    enable_performance_logging: bool = True

    def validate(self) -> None:
        """
        Validate system configuration parameters.

        Raises:
            TimbreException: If parameters are invalid
        """
        if not 1 <= self.max_workers <= 32:
            raise TimbreException("max_workers must be between 1 and 32")

        if not 100 <= self.cache_size_limit_mb <= 10000:
            raise TimbreException("cache_size_limit_mb must be between 100 and 10000")

        if self.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            raise TimbreException("log_level must be DEBUG, INFO, WARNING, or ERROR")


@dataclass
class TimbreConfig:
    """
    Master configuration for the InstrumentTimbre framework.

    Combines all subsystem configurations with validation
    and provides unified access to all settings.
    """

    training: TrainingConfig = field(default_factory=TrainingConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    conversion: ConversionConfig = field(default_factory=ConversionConfig)
    system: SystemConfig = field(default_factory=SystemConfig)

    def validate(self) -> None:
        """
        Validate all configuration sections.

        Raises:
            TimbreException: If any subsection is invalid
        """
        self.training.validate()
        self.analysis.validate()
        self.conversion.validate()
        self.system.validate()


class TimbreConfigManager:
    """
    Configuration manager for InstrumentTimbre framework.

    Handles loading, saving, and validation of configuration
    with environment variable support and file persistence.
    """

    def __init__(self, config_file: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager.

        Args:
            config_file: Path to configuration file (YAML or JSON)
        """
        self.config_file = (
            Path(config_file) if config_file else Path("timbre_config.yaml")
        )
        self._config = None

    def load_config(self) -> TimbreConfig:
        """
        Load configuration from file and environment variables.

        Returns:
            Validated TimbreConfig instance

        Raises:
            TimbreException: If configuration loading fails
        """
        if self._config is not None:
            return self._config

        # Start with defaults
        config = TimbreConfig()

        # Load from file if exists
        if self.config_file.exists():
            try:
                config = self._load_from_file(config)
            except Exception as e:
                raise TimbreException(
                    f"Failed to load config from {self.config_file}: {e}"
                )

        # Apply environment overrides
        config = self._apply_environment_overrides(config)

        # Validate
        config.validate()

        self._config = config
        return config

    def save_config(self, config: Optional[TimbreConfig] = None) -> None:
        """
        Save configuration to file.

        Args:
            config: Configuration to save (uses current if None)

        Raises:
            TimbreException: If saving fails
        """
        if config is None:
            config = self.get_config()

        try:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)

            with open(self.config_file, "w", encoding="utf-8") as f:
                yaml.dump(asdict(config), f, default_flow_style=False, sort_keys=False)

        except Exception as e:
            raise TimbreException(f"Failed to save config to {self.config_file}: {e}")

    def get_config(self) -> TimbreConfig:
        """
        Get current configuration.

        Returns:
            Current TimbreConfig instance
        """
        if self._config is None:
            return self.load_config()
        return self._config

    def _load_from_file(self, base_config: TimbreConfig) -> TimbreConfig:
        """
        Load configuration from YAML or JSON file.

        Args:
            base_config: Base configuration to update

        Returns:
            Updated configuration
        """
        with open(self.config_file, "r", encoding="utf-8") as f:
            if self.config_file.suffix.lower() == ".yaml":
                data = yaml.safe_load(f)
            else:
                data = json.load(f)

        return self._update_config_from_dict(base_config, data)

    def _update_config_from_dict(
        self, config: TimbreConfig, data: Dict[str, Any]
    ) -> TimbreConfig:
        """
        Update configuration from dictionary data.

        Args:
            config: Configuration to update
            data: Dictionary with new values

        Returns:
            Updated configuration
        """
        for section_name, section_data in data.items():
            if hasattr(config, section_name):
                section_obj = getattr(config, section_name)
                if hasattr(section_obj, "__dict__"):
                    for key, value in section_data.items():
                        if hasattr(section_obj, key):
                            setattr(section_obj, key, value)

        return config

    def _apply_environment_overrides(self, config: TimbreConfig) -> TimbreConfig:
        """
        Apply environment variable overrides.

        Environment variables follow: TIMBRE_<SECTION>_<KEY>

        Args:
            config: Configuration to update

        Returns:
            Configuration with environment overrides
        """
        env_prefix = "TIMBRE_"

        for env_key, env_value in os.environ.items():
            if not env_key.startswith(env_prefix):
                continue

            config_path = env_key[len(env_prefix) :].lower().split("_")
            if len(config_path) < 2:
                continue

            section_name = config_path[0]
            key_name = "_".join(config_path[1:])

            if hasattr(config, section_name):
                section_obj = getattr(config, section_name)
                if hasattr(section_obj, key_name):
                    current_value = getattr(section_obj, key_name)
                    converted_value = self._convert_env_value(
                        env_value, type(current_value)
                    )
                    setattr(section_obj, key_name, converted_value)

        return config

    def _convert_env_value(self, value: str, target_type: type) -> Any:
        """
        Convert environment variable to target type.

        Args:
            value: String value from environment
            target_type: Desired type

        Returns:
            Converted value
        """
        if target_type == bool:
            return value.lower() in ("true", "1", "yes", "on")
        elif target_type == int:
            return int(value)
        elif target_type == float:
            return float(value)
        elif target_type == list:
            return [item.strip() for item in value.split(",")]
        else:
            return value


# Global configuration manager instance
_global_config_manager = TimbreConfigManager()


def get_config() -> TimbreConfig:
    """
    Get the global timbre configuration instance.

    Returns:
        Current global TimbreConfig
    """
    return _global_config_manager.get_config()


def load_config_from_file(config_file: Union[str, Path]) -> TimbreConfig:
    """
    Load configuration from specified file.

    Args:
        config_file: Path to configuration file

    Returns:
        Loaded TimbreConfig
    """
    manager = TimbreConfigManager(config_file)
    return manager.load_config()


def save_config_to_file(config: TimbreConfig, config_file: Union[str, Path]) -> None:
    """
    Save configuration to specified file.

    Args:
        config: Configuration to save
        config_file: Output file path
    """
    manager = TimbreConfigManager(config_file)
    manager.save_config(config)
