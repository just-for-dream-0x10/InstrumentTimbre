"""
ULTIMATE Unified Configuration - Replaces ALL config classes in the project
Generated automatically to consolidate 14 configuration classes from 12 files.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
from enum import Enum


class Quality(Enum):
    FAST = "fast"
    NORMAL = "normal" 
    HIGH = "high"
    ULTRA = "ultra"


class ProcessingMode(Enum):
    REAL_TIME = "real_time"
    BATCH = "batch"
    STUDIO = "studio"


@dataclass
class UltimateConfig:
    """
    Single configuration class that replaces ALL 14 config classes.
    Covers: Audio Processing, Training, Generation, Professional Audio, Operations
    """
    
    # === CORE SYSTEM ===
    quality: Quality = Quality.NORMAL
    processing_mode: ProcessingMode = ProcessingMode.BATCH
    use_gpu: bool = True
    num_workers: int = 4
    debug: bool = False
    random_seed: int = 42
    
    # === PATHS ===
    data_dir: str = "data"
    model_dir: str = "models"
    output_dir: str = "output"
    cache_dir: str = "cache"
    log_dir: str = "logs"
    
    # === AUDIO BASICS ===
    sample_rate: int = 22050
    channels: int = 2
    bit_depth: int = 16
    max_duration: float = 30.0
    buffer_size: int = 4096
    
    # === TRAINING ===
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 100
    early_stopping_patience: int = 10
    weight_decay: float = 1e-4
    dropout: float = 0.1
    
    # === MODEL ARCHITECTURE ===
    hidden_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    feature_dim: int = 512
    vocab_size: int = 10000
    
    # === AUDIO FEATURES ===
    n_mels: int = 128
    n_fft: int = 2048
    hop_length: int = 512
    win_length: Optional[int] = None
    
    # === PROFESSIONAL AUDIO ===
    target_loudness_lufs: float = -16.0
    max_peak_db: float = -1.0
    reverb_amount: float = 0.3
    compression_ratio: float = 3.0
    eq_boost_amount: float = 0.4
    stereo_width: float = 1.0
    
    # === GENERATION ===
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    max_length: int = 1024
    beam_size: int = 4
    
    # === STYLE & MUSIC ===
    key_signature: str = "C_major"
    time_signature: str = "4/4"
    tempo_bpm: float = 120.0
    style_strength: float = 0.7
    
    # === EFFECTS & PROCESSING ===
    enable_reverb: bool = True
    enable_compression: bool = True
    enable_eq: bool = True
    enable_stereo_enhancement: bool = True
    enable_noise_reduction: bool = False
    
    # === OPERATION MODES ===
    auto_repair_tracks: bool = True
    conflict_resolution: str = "auto"  # auto, manual, aggressive
    generation_strategy: str = "balanced"  # conservative, balanced, creative
    
    # === ADVANCED SETTINGS ===
    multiprocessing: bool = False
    memory_limit_gb: Optional[float] = None
    checkpoint_interval: int = 1000
    log_level: str = "INFO"
    
    # === CUSTOM OVERRIDES ===
    custom_params: Dict[str, Any] = field(default_factory=dict)


# === PRESET CONFIGURATIONS ===

def minimal_config() -> UltimateConfig:
    """Minimal configuration for basic functionality"""
    return UltimateConfig(
        quality=Quality.FAST,
        batch_size=8,
        epochs=10,
        enable_reverb=False,
        enable_compression=False,
        max_duration=10.0
    )


def development_config() -> UltimateConfig:
    """Development and testing configuration"""
    return UltimateConfig(
        debug=True,
        quality=Quality.NORMAL,
        batch_size=16,
        epochs=50,
        early_stopping_patience=5
    )


def production_config() -> UltimateConfig:
    """Production deployment configuration"""
    return UltimateConfig(
        quality=Quality.HIGH,
        processing_mode=ProcessingMode.STUDIO,
        debug=False,
        use_gpu=True,
        target_loudness_lufs=-14.0,
        max_peak_db=-0.3
    )


def real_time_config() -> UltimateConfig:
    """Real-time processing configuration"""
    return UltimateConfig(
        processing_mode=ProcessingMode.REAL_TIME,
        quality=Quality.FAST,
        buffer_size=1024,
        batch_size=1,
        enable_reverb=False
    )


def studio_config() -> UltimateConfig:
    """Studio-quality processing configuration"""
    return UltimateConfig(
        quality=Quality.ULTRA,
        processing_mode=ProcessingMode.STUDIO,
        sample_rate=48000,
        bit_depth=24,
        target_loudness_lufs=-18.0,
        reverb_amount=0.4,
        compression_ratio=2.5
    )


# === GLOBAL CONFIG MANAGEMENT ===
_global_config = UltimateConfig()

def get_config() -> UltimateConfig:
    return _global_config

def set_config(config: UltimateConfig) -> None:
    global _global_config
    _global_config = config

def update_config(**kwargs) -> None:
    global _global_config
    for key, value in kwargs.items():
        if hasattr(_global_config, key):
            setattr(_global_config, key, value)

def reset_config() -> None:
    global _global_config
    _global_config = UltimateConfig()


# === VALIDATION ===
def validate_config(config: UltimateConfig) -> bool:
    """Comprehensive configuration validation"""
    try:
        assert config.batch_size > 0
        assert 0.0 < config.learning_rate < 1.0
        assert config.epochs > 0
        assert 0.0 <= config.reverb_amount <= 1.0
        assert 0.0 <= config.compression_ratio <= 10.0
        assert config.sample_rate in [22050, 44100, 48000]
        assert -30.0 <= config.target_loudness_lufs <= 0.0
        assert 0.0 <= config.temperature <= 2.0
        return True
    except AssertionError:
        return False


# === LEGACY COMPATIBILITY ===
class ConfigAdapter:
    """Adapter to convert UltimateConfig to legacy formats"""
    
    @staticmethod
    def to_any_legacy_config(config: UltimateConfig, config_type: str) -> Dict[str, Any]:
        """Convert to any legacy configuration format"""
        base = {
            "sample_rate": config.sample_rate,
            "quality": config.quality.value,
            "use_gpu": config.use_gpu,
            "debug": config.debug
        }
        
        if "training" in config_type.lower():
            base.update({
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "epochs": config.epochs
            })
        
        if "audio" in config_type.lower():
            base.update({
                "target_loudness": config.target_loudness_lufs,
                "reverb_amount": config.reverb_amount,
                "compression_ratio": config.compression_ratio
            })
        
        if "generation" in config_type.lower():
            base.update({
                "temperature": config.temperature,
                "max_length": config.max_length,
                "top_k": config.top_k
            })
        
        return base


# Auto-initialize
set_config(UltimateConfig())
