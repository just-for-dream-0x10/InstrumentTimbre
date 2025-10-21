"""
Professional Audio Engine - Main orchestrator for professional audio processing.

This module provides the main entry point for all professional audio processing operations,
coordinating between different specialized processors to deliver broadcast-quality output.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import logging

from .base_processor import BaseAudioProcessor
from config import Config, get_config, LegacyConfigAdapter
from .intelligent_mixing_engine import IntelligentMixingEngine
from .dynamic_range_optimizer import DynamicRangeOptimizer
from .spatial_positioning_algorithm import SpatialPositioningAlgorithm
from .intelligent_eq_balancer import IntelligentEQBalancer
from .effects_processor import EffectsProcessor
from .audio_quality_enhancer import AudioQualityEnhancer


class ProcessingPriority(Enum):
    """Processing priority levels for audio operations."""
    REAL_TIME = 1
    HIGH_QUALITY = 2
    BATCH_PROCESSING = 3


@dataclass
class ProcessingConfig:
    """Configuration for professional audio processing."""
    sample_rate: int = 22050
    bit_depth: int = 24
    processing_priority: ProcessingPriority = ProcessingPriority.HIGH_QUALITY
    enable_quality_enhancement: bool = True
    enable_spatial_processing: bool = True
    enable_dynamic_optimization: bool = True
    target_lufs: float = -16.0
    max_peak_level: float = -1.0
    stereo_width: float = 1.0
    processing_quality: str = "high"


@dataclass
class AudioTrackInfo:
    """Information about an audio track for processing."""
    track_id: str
    instrument_type: str
    role: str  # 'lead', 'rhythm', 'bass', 'percussion', 'harmony'
    importance_weight: float
    original_level: float
    frequency_range: Tuple[float, float]
    dynamic_range: float
    emotional_content: Dict[str, float]


class ProfessionalAudioEngine(BaseAudioProcessor):
    """
    Main professional audio processing engine that orchestrates all audio enhancement operations.
    
    This engine coordinates between specialized processors to deliver professional-quality
    audio output suitable for broadcast and professional music production.
    """
    
    def __init__(self, config: Optional[ProcessingConfig] = None, simple_config: Optional[Config] = None):
        """
        Initialize the professional audio engine.
        
        Args:
            config: Legacy processing configuration settings (deprecated)
            simple_config: New simple configuration system (recommended)
        """
        super().__init__("professional_audio_engine", 22050)
        
        # Use simple config if provided
        if simple_config is not None:
            self.simple_config = simple_config
            self.config = self._convert_to_legacy_config(simple_config)
        else:
            self.config = config or ProcessingConfig()
            self.simple_config = None
            
        self.logger = logging.getLogger(__name__)
        
        # Initialize specialized processors
        self._initialize_processors()
        
        # Processing state
        self.is_processing = False
        self.current_session_id: Optional[str] = None
        
    def _initialize_processors(self) -> None:
        """Initialize all specialized audio processors."""
        try:
            self.mixing_engine = IntelligentMixingEngine(self.config)
            self.dynamic_optimizer = DynamicRangeOptimizer(self.config)
            self.spatial_processor = SpatialPositioningAlgorithm(self.config)
            self.eq_balancer = IntelligentEQBalancer(self.config)
            self.effects_processor = EffectsProcessor(self.config)
            self.quality_enhancer = AudioQualityEnhancer(self.config)
            
            self.logger.info("Professional audio processors initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize processors: {e}")
            raise
    
    def process_multitrack_audio(
        self,
        tracks: Dict[str, np.ndarray],
        track_info: Dict[str, AudioTrackInfo],
        musical_analysis: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process multiple audio tracks with professional enhancement.
        
        Args:
            tracks: Dictionary of track_id -> audio_data mappings
            track_info: Dictionary of track_id -> AudioTrackInfo mappings
            musical_analysis: Optional musical structure and emotion analysis
            
        Returns:
            Tuple of (processed_stereo_audio, processing_metadata)
        """
        self.logger.info(f"Starting professional audio processing for {len(tracks)} tracks")
        self.is_processing = True
        
        try:
            # Step 1: Pre-processing validation and normalization
            validated_tracks, processing_metadata = self._validate_and_prepare_tracks(
                tracks, track_info
            )
            
            # Step 2: Intelligent mixing with level balancing
            mixed_tracks, mixing_metadata = self.mixing_engine.process_tracks(
                validated_tracks, track_info, musical_analysis
            )
            
            # Step 3: Intelligent EQ balancing for frequency separation
            eq_balanced_tracks, eq_metadata = self.eq_balancer.balance_frequency_spectrum(
                mixed_tracks, track_info, musical_analysis
            )
            
            # Step 4: Spatial positioning for stereo imaging
            spatially_positioned, spatial_metadata = self.spatial_processor.position_tracks(
                eq_balanced_tracks, track_info, musical_analysis
            )
            
            # Step 5: Effects processing for musical enhancement
            effects_processed, effects_metadata = self.effects_processor.apply_musical_effects(
                spatially_positioned, track_info, musical_analysis
            )
            
            # Step 6: Dynamic range optimization
            dynamics_optimized, dynamics_metadata = self.dynamic_optimizer.optimize_dynamics(
                effects_processed, track_info, musical_analysis
            )
            
            # Step 7: Final quality enhancement
            final_audio, quality_metadata = self.quality_enhancer.enhance_audio_quality(
                dynamics_optimized, self.config
            )
            
            # Compile comprehensive metadata
            complete_metadata = self._compile_processing_metadata(
                processing_metadata, mixing_metadata, eq_metadata,
                spatial_metadata, effects_metadata, dynamics_metadata, quality_metadata
            )
            
            self.logger.info("Professional audio processing completed successfully")
            return final_audio, complete_metadata
            
        except Exception as e:
            self.logger.error(f"Audio processing failed: {e}")
            raise
        finally:
            self.is_processing = False
    
    def _validate_and_prepare_tracks(
        self,
        tracks: Dict[str, np.ndarray],
        track_info: Dict[str, AudioTrackInfo]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Validate input tracks and prepare them for processing.
        
        Args:
            tracks: Raw audio tracks
            track_info: Track information metadata
            
        Returns:
            Tuple of (validated_tracks, validation_metadata)
        """
        validated_tracks = {}
        validation_metadata = {
            "original_track_count": len(tracks),
            "validated_track_count": 0,
            "rejected_tracks": [],
            "normalization_applied": []
        }
        
        for track_id, audio_data in tracks.items():
            try:
                # Validate audio data format
                if not isinstance(audio_data, np.ndarray):
                    raise ValueError(f"Track {track_id}: Audio data must be numpy array")
                
                if audio_data.dtype != np.float32:
                    audio_data = audio_data.astype(np.float32)
                    validation_metadata["normalization_applied"].append(track_id)
                
                # Validate audio length (reject very short clips)
                min_length = self.config.sample_rate * 0.1  # 100ms minimum
                if len(audio_data) < min_length:
                    validation_metadata["rejected_tracks"].append({
                        "track_id": track_id,
                        "reason": "Audio too short"
                    })
                    continue
                
                # Normalize audio levels
                max_amplitude = np.max(np.abs(audio_data))
                if max_amplitude > 0:
                    audio_data = audio_data / max_amplitude * 0.8  # Leave headroom
                
                validated_tracks[track_id] = audio_data
                validation_metadata["validated_track_count"] += 1
                
            except Exception as e:
                validation_metadata["rejected_tracks"].append({
                    "track_id": track_id,
                    "reason": str(e)
                })
                self.logger.warning(f"Track {track_id} validation failed: {e}")
        
        return validated_tracks, validation_metadata
    
    def _compile_processing_metadata(
        self,
        *metadata_dicts: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compile metadata from all processing stages.
        
        Args:
            *metadata_dicts: Variable number of metadata dictionaries
            
        Returns:
            Compiled comprehensive metadata
        """
        compiled_metadata = {
            "processing_chain": [
                "validation", "mixing", "eq_balancing", 
                "spatial_positioning", "effects", "dynamics", "quality_enhancement"
            ],
            "config": {
                "sample_rate": self.config.sample_rate,
                "bit_depth": self.config.bit_depth,
                "processing_priority": self.config.processing_priority.name,
                "target_lufs": self.config.target_lufs
            },
            "stage_metadata": {}
        }
        
        stage_names = compiled_metadata["processing_chain"]
        for i, metadata in enumerate(metadata_dicts):
            if i < len(stage_names):
                compiled_metadata["stage_metadata"][stage_names[i]] = metadata
        
        return compiled_metadata
    
    def get_processing_status(self) -> Dict[str, Any]:
        """
        Get current processing status information.
        
        Returns:
            Dictionary containing current status information
        """
        return {
            "is_processing": self.is_processing,
            "current_session_id": self.current_session_id,
            "config": {
                "sample_rate": self.config.sample_rate,
                "processing_priority": self.config.processing_priority.name,
                "quality_enhancement_enabled": self.config.enable_quality_enhancement
            }
        }
    
    def update_configuration(self, new_config: ProcessingConfig) -> None:
        """
        Update processing configuration and reinitialize processors if needed.
        
        Args:
            new_config: New processing configuration
        """
        old_sample_rate = self.config.sample_rate
        self.config = new_config
        
        # Reinitialize processors if sample rate changed
        if new_config.sample_rate != old_sample_rate:
            self._initialize_processors()
            self.logger.info("Processors reinitialized due to sample rate change")
        
        # Update existing processors with new config
        for processor in [self.mixing_engine, self.dynamic_optimizer, 
                         self.spatial_processor, self.eq_balancer,
                         self.effects_processor, self.quality_enhancer]:
            if hasattr(processor, 'update_config'):
                processor.update_config(new_config)
    
    def process(self, track, **kwargs):
        """Process a single audio track (required by base class)."""
        # For the main engine, delegate to multitrack processing
        from .base_processor import AudioTrack, ProcessingResult, ProcessingStage
        
        if not isinstance(track, AudioTrack):
            # Convert to AudioTrack format
            track_info = kwargs.get('track_info', {})
            audio_track = AudioTrack(
                audio_data=track,
                sample_rate=self.config.sample_rate,
                track_name="single_track",
                instrument_type=track_info.get('instrument_type', 'OTHER'),
                musical_role=track_info.get('role', 'HARMONY')
            )
        else:
            audio_track = track
        
        # Process as single-track multitrack
        tracks = {"single_track": audio_track.audio_data}
        track_info_dict = {"single_track": track_info} if 'track_info' in kwargs else {}
        
        processed_audio, metadata = self.process_multitrack_audio(
            tracks, track_info_dict, kwargs.get('musical_analysis')
        )
        
        return ProcessingResult(
            processed_audio=processed_audio,
            processing_stage=ProcessingStage.POST_PROCESSING,
            success=True,
            processing_time=0.0,
            metadata=metadata
        )
    
    def _get_processing_stage(self):
        """Return the processing stage this processor handles."""
        from .base_processor import ProcessingStage
        return ProcessingStage.POST_PROCESSING
    
    def _convert_to_legacy_config(self, simple_config: Config) -> ProcessingConfig:
        """Convert simple config to legacy format"""
        return ProcessingConfig(
            sample_rate=simple_config.sample_rate,
            bit_depth=16,
            target_lufs=simple_config.target_db,
            max_peak_level=simple_config.max_peak_db,
            stereo_width=simple_config.stereo_width,
            enable_quality_enhancement=True,
            enable_spatial_processing=simple_config.enable_spatial,
            enable_dynamic_optimization=simple_config.enable_compression
        )
    
    def process_simple(
        self,
        tracks: Dict[str, np.ndarray],
        track_info: Dict[str, TrackInfo],
        config: Optional[Config] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Simple audio processing interface using unified config
        
        Args:
            tracks: Audio tracks {track_name: audio_data}
            track_info: Simple track information {track_name: TrackInfo}
            config: Optional processing config
            
        Returns:
            Processed stereo audio and metadata
        """
        # Use provided config or default
        processing_config = config or self.simple_config
        if processing_config is None:
            # Auto recommend config
            instruments = [info.instrument for info in track_info.values()]
            processing_config = recommend_config(instruments)
        
        # Validate config
        if not validate_config(processing_config):
            self.logger.warning("Config validation failed, using default")
            processing_config = Config()
        
        # Convert to legacy format for processing
        legacy_track_info = {}
        for track_id, simple_info in track_info.items():
            legacy_track_info[track_id] = AudioTrackInfo(
                track_id=track_id,
                instrument_type=simple_info.instrument,
                role=simple_info.role,
                importance_weight=simple_info.importance,
                original_level=simple_info.level,
                frequency_range=(100, 8000),
                dynamic_range=8.0,
                emotional_content={"energy": 0.5, "valence": 0.5}
            )
        
        # Generate musical analysis
        musical_analysis = {
            "tempo": 120.0,
            "style": processing_config.style.value,
            "emotional_analysis": {"energy": 0.5, "valence": 0.5}
        }
        
        # Call main processing pipeline
        return self.process_multitrack_audio(tracks, legacy_track_info, musical_analysis)