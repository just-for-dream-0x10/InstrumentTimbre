"""
Base Audio Processor Classes

Abstract base classes and common data structures for professional audio processing.
Follows the established coding standards and provides consistent interfaces.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging


class ProcessingStage(Enum):
    """Audio processing stages for pipeline management"""
    PRE_PROCESSING = "pre_processing"
    MIXING = "mixing"
    DYNAMICS = "dynamics"
    SPATIAL = "spatial"
    EQ = "eq"
    EFFECTS = "effects"
    ENHANCEMENT = "enhancement"
    POST_PROCESSING = "post_processing"


class InstrumentType(Enum):
    """Standard instrument categorization for processing"""
    DRUMS = "drums"
    BASS = "bass"
    GUITAR = "guitar"
    PIANO = "piano"
    VOCALS = "vocals"
    STRINGS = "strings"
    BRASS = "brass"
    WOODWINDS = "woodwinds"
    SYNTH = "synth"
    PERCUSSION = "percussion"
    OTHER = "other"


class MusicalRole(Enum):
    """Musical role classification for intelligent processing"""
    LEAD_MELODY = "lead_melody"
    HARMONY = "harmony"
    RHYTHM = "rhythm"
    BASS_LINE = "bass_line"
    COUNTER_MELODY = "counter_melody"
    ACCOMPANIMENT = "accompaniment"
    TEXTURE = "texture"
    EFFECTS = "effects"


@dataclass
class AudioTrack:
    """Audio track data structure with metadata"""
    audio_data: np.ndarray
    sample_rate: int
    track_name: str
    instrument_type: InstrumentType
    musical_role: MusicalRole
    original_level: float = 0.0
    pan_position: float = 0.0  # -1.0 (left) to 1.0 (right)
    is_stereo: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate audio data on initialization"""
        if not isinstance(self.audio_data, np.ndarray):
            raise TypeError("audio_data must be numpy array")
        
        if self.audio_data.dtype not in [np.float32, np.float64]:
            self.audio_data = self.audio_data.astype(np.float32)
        
        if len(self.audio_data.shape) == 1:
            self.is_stereo = False
        elif len(self.audio_data.shape) == 2 and self.audio_data.shape[1] == 2:
            self.is_stereo = True
        else:
            raise ValueError(f"Invalid audio shape: {self.audio_data.shape}")


@dataclass
class ProcessingResult:
    """Result of audio processing operation"""
    processed_audio: np.ndarray
    processing_stage: ProcessingStage
    success: bool
    processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        """Validate processing result"""
        if self.success and self.processed_audio is None:
            raise ValueError("processed_audio cannot be None when success=True")


@dataclass
class MixingParameters:
    """Parameters for mixing operations"""
    target_lufs: float = -23.0  # Broadcast standard
    peak_limit: float = -1.0    # Peak limiting threshold
    stereo_width: float = 1.0   # Stereo width (0.0-2.0)
    room_tone_level: float = -60.0  # Room tone/noise floor
    preserve_dynamics: bool = True
    musical_style: str = "balanced"
    
    def validate(self) -> None:
        """Validate mixing parameters"""
        if not -60.0 <= self.target_lufs <= 0.0:
            raise ValueError(f"target_lufs must be between -60.0 and 0.0, got {self.target_lufs}")
        
        if not -20.0 <= self.peak_limit <= 0.0:
            raise ValueError(f"peak_limit must be between -20.0 and 0.0, got {self.peak_limit}")
        
        if not 0.0 <= self.stereo_width <= 2.0:
            raise ValueError(f"stereo_width must be between 0.0 and 2.0, got {self.stereo_width}")


@dataclass
class FrequencyBand:
    """Frequency band definition for multiband processing"""
    low_freq: float
    high_freq: float
    gain: float = 0.0
    q_factor: float = 1.0
    
    def __post_init__(self) -> None:
        """Validate frequency band parameters"""
        if self.low_freq >= self.high_freq:
            raise ValueError("low_freq must be less than high_freq")
        
        if self.low_freq < 20.0 or self.high_freq > 20000.0:
            raise ValueError("Frequency range must be within 20Hz-20kHz")


@dataclass
class SpatialPosition:
    """3D spatial position for audio placement"""
    pan: float = 0.0      # -1.0 (left) to 1.0 (right)
    depth: float = 0.0    # -1.0 (far) to 1.0 (close)
    height: float = 0.0   # -1.0 (low) to 1.0 (high)
    width: float = 0.0    # stereo width adjustment
    
    def validate(self) -> None:
        """Validate spatial position parameters"""
        for attr, value in [("pan", self.pan), ("depth", self.depth), 
                           ("height", self.height), ("width", self.width)]:
            if not -1.0 <= value <= 1.0:
                raise ValueError(f"{attr} must be between -1.0 and 1.0, got {value}")


class BaseAudioProcessor(ABC):
    """
    Abstract base class for all audio processors
    
    Provides common functionality and enforces consistent interface
    across all professional audio processing components.
    """
    
    def __init__(self, processor_name: str, sample_rate: int = 22050) -> None:
        """
        Initialize base audio processor
        
        Args:
            processor_name: Unique name for this processor
            sample_rate: Audio sample rate in Hz
        """
        self.processor_name = processor_name
        self.sample_rate = sample_rate
        self.is_initialized = False
        self.processing_count = 0
        self.total_processing_time = 0.0
        
        # Setup logging
        self.logger = logging.getLogger(f"audio_processor.{processor_name}")
        
        # Configuration storage
        self.config: Dict[str, Any] = {}
    
    @abstractmethod
    def process(self, track: AudioTrack, **kwargs) -> ProcessingResult:
        """
        Process a single audio track
        
        Args:
            track: Audio track to process
            **kwargs: Additional processing parameters
            
        Returns:
            ProcessingResult with processed audio and metadata
        """
        pass
    
    def batch_process(self, tracks: List[AudioTrack], **kwargs) -> List[ProcessingResult]:
        """
        Process multiple audio tracks
        
        Args:
            tracks: List of audio tracks to process
            **kwargs: Additional processing parameters
            
        Returns:
            List of ProcessingResult objects
        """
        results = []
        
        for track in tracks:
            try:
                result = self.process(track, **kwargs)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to process track {track.track_name}: {e}")
                # Create failure result
                failure_result = ProcessingResult(
                    processed_audio=track.audio_data,  # Return original on failure
                    processing_stage=self._get_processing_stage(),
                    success=False,
                    processing_time=0.0,
                    metadata={"error": str(e)},
                    warnings=[f"Processing failed: {str(e)}"]
                )
                results.append(failure_result)
        
        return results
    
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Initialize processor with configuration
        
        Args:
            config: Optional configuration parameters
            
        Returns:
            True if initialization successful
        """
        if config:
            self.config.update(config)
        
        # Validate sample rate
        if not 8000 <= self.sample_rate <= 192000:
            self.logger.warning(f"Unusual sample rate: {self.sample_rate}Hz")
        
        self.is_initialized = True
        self.logger.info(f"Initialized {self.processor_name} processor")
        return True
    
    def validate_input(self, track: AudioTrack) -> None:
        """
        Validate input audio track
        
        Args:
            track: Audio track to validate
            
        Raises:
            ValueError: If track is invalid
        """
        if track.audio_data is None or len(track.audio_data) == 0:
            raise ValueError("Empty audio data")
        
        if track.sample_rate != self.sample_rate:
            self.logger.warning(
                f"Sample rate mismatch: expected {self.sample_rate}, "
                f"got {track.sample_rate} for track {track.track_name}"
            )
        
        # Check for NaN or infinite values
        if np.any(np.isnan(track.audio_data)) or np.any(np.isinf(track.audio_data)):
            raise ValueError("Audio data contains NaN or infinite values")
        
        # Check dynamic range
        if np.max(np.abs(track.audio_data)) < 1e-6:
            self.logger.warning(f"Very low audio level in track {track.track_name}")
    
    def get_performance_stats(self) -> Dict[str, float]:
        """
        Get processor performance statistics
        
        Returns:
            Dictionary with performance metrics
        """
        avg_processing_time = (
            self.total_processing_time / self.processing_count 
            if self.processing_count > 0 else 0.0
        )
        
        return {
            "processor_name": self.processor_name,
            "total_processed": self.processing_count,
            "total_time": self.total_processing_time,
            "average_time": avg_processing_time,
            "sample_rate": self.sample_rate
        }
    
    def reset_stats(self) -> None:
        """Reset performance statistics"""
        self.processing_count = 0
        self.total_processing_time = 0.0
    
    @abstractmethod
    def _get_processing_stage(self) -> ProcessingStage:
        """Return the processing stage this processor handles"""
        pass
    
    def _update_stats(self, processing_time: float) -> None:
        """Update processing statistics"""
        self.processing_count += 1
        self.total_processing_time += processing_time
    
    def _validate_numpy_array(self, array: np.ndarray, name: str) -> None:
        """
        Validate numpy array for audio processing
        
        Args:
            array: Array to validate
            name: Name for error messages
        """
        if not isinstance(array, np.ndarray):
            raise TypeError(f"{name} must be numpy array")
        
        if array.dtype not in [np.float32, np.float64]:
            raise TypeError(f"{name} must be float32 or float64")
        
        if len(array.shape) > 2:
            raise ValueError(f"{name} cannot have more than 2 dimensions")


class MultiTrackProcessor(BaseAudioProcessor):
    """
    Base class for processors that handle multiple tracks simultaneously
    
    Provides additional functionality for cross-track analysis and processing.
    """
    
    @abstractmethod
    def process_multitrack(self, tracks: List[AudioTrack], **kwargs) -> List[ProcessingResult]:
        """
        Process multiple tracks with cross-track analysis
        
        Args:
            tracks: List of audio tracks to process together
            **kwargs: Additional processing parameters
            
        Returns:
            List of ProcessingResult objects
        """
        pass
    
    def analyze_track_relationships(self, tracks: List[AudioTrack]) -> Dict[str, Any]:
        """
        Analyze relationships between tracks for intelligent processing
        
        Args:
            tracks: List of tracks to analyze
            
        Returns:
            Dictionary containing relationship analysis
        """
        if len(tracks) < 2:
            return {"track_count": len(tracks), "relationships": []}
        
        relationships = []
        
        # Analyze frequency overlaps
        for i, track1 in enumerate(tracks):
            for j, track2 in enumerate(tracks[i+1:], i+1):
                # Simple correlation analysis
                if track1.audio_data.shape == track2.audio_data.shape:
                    correlation = np.corrcoef(
                        track1.audio_data.flatten(), 
                        track2.audio_data.flatten()
                    )[0, 1]
                    
                    relationships.append({
                        "track1": track1.track_name,
                        "track2": track2.track_name,
                        "correlation": float(correlation),
                        "instrument_compatibility": self._assess_instrument_compatibility(
                            track1.instrument_type, track2.instrument_type
                        )
                    })
        
        return {
            "track_count": len(tracks),
            "relationships": relationships,
            "total_duration": max(len(t.audio_data) for t in tracks) / self.sample_rate
        }
    
    def _assess_instrument_compatibility(self, inst1: InstrumentType, inst2: InstrumentType) -> float:
        """
        Assess compatibility between two instrument types
        
        Args:
            inst1: First instrument type
            inst2: Second instrument type
            
        Returns:
            Compatibility score between 0.0 and 1.0
        """
        # Compatibility matrix based on frequency ranges and musical roles
        compatibility_matrix = {
            (InstrumentType.DRUMS, InstrumentType.BASS): 0.9,
            (InstrumentType.BASS, InstrumentType.PIANO): 0.8,
            (InstrumentType.VOCALS, InstrumentType.GUITAR): 0.9,
            (InstrumentType.STRINGS, InstrumentType.PIANO): 0.8,
            (InstrumentType.BRASS, InstrumentType.WOODWINDS): 0.7,
        }
        
        # Check both directions
        pair1 = (inst1, inst2)
        pair2 = (inst2, inst1)
        
        if pair1 in compatibility_matrix:
            return compatibility_matrix[pair1]
        elif pair2 in compatibility_matrix:
            return compatibility_matrix[pair2]
        else:
            # Default compatibility for unknown pairs
            return 0.6