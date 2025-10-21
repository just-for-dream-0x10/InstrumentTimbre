"""
Spatial Positioning Algorithm - Intelligent stereo/surround positioning for instruments.

This module provides intelligent spatial positioning based on musical structure,
instrument characteristics, and perceptual audio principles for optimal stereo imaging.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
from enum import Enum
import logging

from .base_processor import BaseAudioProcessor


class SpatialMode(Enum):
    """Spatial processing modes."""
    STEREO = "stereo"
    WIDE_STEREO = "wide_stereo"
    SURROUND_5_1 = "surround_5_1"
    BINAURAL = "binaural"
    MONO_COMPATIBLE = "mono_compatible"


class PositioningStrategy(Enum):
    """Positioning strategies for different musical styles."""
    CLASSICAL_ORCHESTRA = "classical_orchestra"
    JAZZ_ENSEMBLE = "jazz_ensemble"
    ROCK_BAND = "rock_band"
    CHINESE_TRADITIONAL = "chinese_traditional"
    ELECTRONIC = "electronic"
    ADAPTIVE = "adaptive"


@dataclass
class SpatialPosition:
    """3D spatial position for an audio source."""
    azimuth: float  # -180 to 180 degrees (left/right)
    elevation: float  # -90 to 90 degrees (up/down)
    distance: float  # 0 to 1 (near/far)
    width: float  # 0 to 1 (narrow/wide)
    depth: float  # 0 to 1 (front/back)


@dataclass
class InstrumentSpatialProfile:
    """Spatial characteristics for specific instrument types."""
    instrument_type: str
    default_position: SpatialPosition
    movement_range: float  # How much the instrument can move
    width_preference: float  # Preferred stereo width
    depth_preference: float  # Preferred depth in mix
    frequency_dependent_positioning: bool
    interaction_priority: float  # Priority in spatial conflicts


class SpatialPositioningAlgorithm(BaseAudioProcessor):
    """
    Intelligent spatial positioning system that places instruments in the stereo field
    based on musical context, instrument characteristics, and perceptual principles.
    """
    
    def __init__(self, config):
        """
        Initialize the spatial positioning algorithm.
        
        Args:
            config: Processing configuration
        """
        super().__init__("spatialpositioningalgorithm", 22050)
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize spatial profiles
        self.spatial_profiles = self._initialize_spatial_profiles()
        
        # Positioning state
        self.current_positions: Dict[str, SpatialPosition] = {}
        self.spatial_conflicts: List[Tuple[str, str]] = []
        
    def _initialize_spatial_profiles(self) -> Dict[str, InstrumentSpatialProfile]:
        """
        Initialize spatial profiles for different instrument types.
        
        Returns:
            Dictionary of instrument_type -> InstrumentSpatialProfile
        """
        profiles = {
            # Rhythm section - typically center and wide
            "drums": InstrumentSpatialProfile(
                instrument_type="drums",
                default_position=SpatialPosition(0, 0, 0.3, 0.8, 0.4),
                movement_range=30.0,
                width_preference=0.9,
                depth_preference=0.4,
                frequency_dependent_positioning=True,
                interaction_priority=0.9
            ),
            "bass": InstrumentSpatialProfile(
                instrument_type="bass",
                default_position=SpatialPosition(0, -5, 0.2, 0.3, 0.3),
                movement_range=15.0,
                width_preference=0.3,
                depth_preference=0.3,
                frequency_dependent_positioning=True,
                interaction_priority=0.8
            ),
            
            # Lead instruments - front and center
            "piano": InstrumentSpatialProfile(
                instrument_type="piano",
                default_position=SpatialPosition(0, 0, 0.1, 0.7, 0.2),
                movement_range=20.0,
                width_preference=0.7,
                depth_preference=0.2,
                frequency_dependent_positioning=False,
                interaction_priority=0.9
            ),
            "guitar": InstrumentSpatialProfile(
                instrument_type="guitar",
                default_position=SpatialPosition(-25, 0, 0.2, 0.4, 0.3),
                movement_range=40.0,
                width_preference=0.4,
                depth_preference=0.3,
                frequency_dependent_positioning=False,
                interaction_priority=0.7
            ),
            
            # String instruments - traditional orchestra positioning
            "violin": InstrumentSpatialProfile(
                instrument_type="violin",
                default_position=SpatialPosition(-45, 5, 0.3, 0.5, 0.4),
                movement_range=30.0,
                width_preference=0.5,
                depth_preference=0.4,
                frequency_dependent_positioning=False,
                interaction_priority=0.6
            ),
            "viola": InstrumentSpatialProfile(
                instrument_type="viola",
                default_position=SpatialPosition(45, 0, 0.4, 0.4, 0.5),
                movement_range=25.0,
                width_preference=0.4,
                depth_preference=0.5,
                frequency_dependent_positioning=False,
                interaction_priority=0.5
            ),
            "cello": InstrumentSpatialProfile(
                instrument_type="cello",
                default_position=SpatialPosition(30, -10, 0.3, 0.4, 0.4),
                movement_range=20.0,
                width_preference=0.4,
                depth_preference=0.4,
                frequency_dependent_positioning=True,
                interaction_priority=0.6
            ),
            
            # Chinese traditional instruments
            "erhu": InstrumentSpatialProfile(
                instrument_type="erhu",
                default_position=SpatialPosition(-30, 10, 0.2, 0.3, 0.2),
                movement_range=35.0,
                width_preference=0.3,
                depth_preference=0.2,
                frequency_dependent_positioning=False,
                interaction_priority=0.8
            ),
            "guzheng": InstrumentSpatialProfile(
                instrument_type="guzheng",
                default_position=SpatialPosition(0, 0, 0.1, 0.8, 0.1),
                movement_range=20.0,
                width_preference=0.8,
                depth_preference=0.1,
                frequency_dependent_positioning=True,
                interaction_priority=0.7
            ),
            "pipa": InstrumentSpatialProfile(
                instrument_type="pipa",
                default_position=SpatialPosition(25, 5, 0.2, 0.3, 0.25),
                movement_range=30.0,
                width_preference=0.3,
                depth_preference=0.25,
                frequency_dependent_positioning=False,
                interaction_priority=0.6
            ),
            "dizi": InstrumentSpatialProfile(
                instrument_type="dizi",
                default_position=SpatialPosition(-20, 15, 0.3, 0.2, 0.3),
                movement_range=40.0,
                width_preference=0.2,
                depth_preference=0.3,
                frequency_dependent_positioning=False,
                interaction_priority=0.5
            ),
            "guqin": InstrumentSpatialProfile(
                instrument_type="guqin",
                default_position=SpatialPosition(15, -5, 0.15, 0.5, 0.15),
                movement_range=25.0,
                width_preference=0.5,
                depth_preference=0.15,
                frequency_dependent_positioning=True,
                interaction_priority=0.7
            ),
            
            # Wind instruments
            "flute": InstrumentSpatialProfile(
                instrument_type="flute",
                default_position=SpatialPosition(-15, 20, 0.4, 0.2, 0.4),
                movement_range=35.0,
                width_preference=0.2,
                depth_preference=0.4,
                frequency_dependent_positioning=False,
                interaction_priority=0.5
            ),
            "saxophone": InstrumentSpatialProfile(
                instrument_type="saxophone",
                default_position=SpatialPosition(20, 10, 0.25, 0.3, 0.3),
                movement_range=30.0,
                width_preference=0.3,
                depth_preference=0.3,
                frequency_dependent_positioning=False,
                interaction_priority=0.6
            ),
            
            # Default profile
            "default": InstrumentSpatialProfile(
                instrument_type="default",
                default_position=SpatialPosition(0, 0, 0.3, 0.4, 0.4),
                movement_range=30.0,
                width_preference=0.4,
                depth_preference=0.4,
                frequency_dependent_positioning=False,
                interaction_priority=0.5
            )
        }
        
        return profiles
    
    def position_tracks(
        self,
        tracks: Dict[str, np.ndarray],
        track_info: Dict[str, Any],
        musical_analysis: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Apply intelligent spatial positioning to all tracks.
        
        Args:
            tracks: Dictionary of track_id -> audio_data
            track_info: Track information and metadata
            musical_analysis: Optional musical structure analysis
            
        Returns:
            Tuple of (spatially_positioned_tracks, spatial_metadata)
        """
        self.logger.info(f"Starting spatial positioning for {len(tracks)} tracks")
        
        try:
            # Step 1: Analyze track spatial requirements
            spatial_requirements = self._analyze_spatial_requirements(tracks, track_info)
            
            # Step 2: Determine positioning strategy
            positioning_strategy = self._determine_positioning_strategy(track_info, musical_analysis)
            
            # Step 3: Calculate optimal positions
            optimal_positions = self._calculate_optimal_positions(
                spatial_requirements, track_info, positioning_strategy
            )
            
            # Step 4: Resolve spatial conflicts
            resolved_positions = self._resolve_spatial_conflicts(optimal_positions, track_info)
            
            # Step 5: Apply spatial processing
            positioned_tracks = self._apply_spatial_processing(tracks, resolved_positions)
            
            # Step 6: Apply stereo enhancement
            enhanced_tracks = self._apply_stereo_enhancement(positioned_tracks, resolved_positions)
            
            # Store current positions
            self.current_positions = resolved_positions
            
            # Compile spatial metadata
            spatial_metadata = {
                "positioning_strategy": positioning_strategy.value,
                "track_positions": {tid: {
                    "azimuth": pos.azimuth,
                    "elevation": pos.elevation,
                    "distance": pos.distance,
                    "width": pos.width,
                    "depth": pos.depth
                } for tid, pos in resolved_positions.items()},
                "spatial_conflicts_resolved": len(self.spatial_conflicts),
                "stereo_enhancement_applied": True
            }
            
            self.logger.info("Spatial positioning completed successfully")
            return enhanced_tracks, spatial_metadata
            
        except Exception as e:
            self.logger.error(f"Spatial positioning failed: {e}")
            raise
    
    def _analyze_spatial_requirements(
        self,
        tracks: Dict[str, np.ndarray],
        track_info: Dict[str, Any]
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze spatial requirements for each track.
        
        Args:
            tracks: Audio track data
            track_info: Track information
            
        Returns:
            Dictionary of track_id -> spatial_requirements
        """
        requirements = {}
        
        for track_id, audio_data in tracks.items():
            # Analyze frequency content for positioning decisions
            fft_data = np.fft.fft(audio_data[:min(4096, len(audio_data))])
            magnitude = np.abs(fft_data[:len(fft_data)//2])
            freqs = np.fft.fftfreq(len(fft_data), 1/self.config.sample_rate)[:len(magnitude)]
            
            # Calculate frequency distribution
            low_freq_energy = np.sum(magnitude[freqs < 200])
            mid_freq_energy = np.sum(magnitude[(freqs >= 200) & (freqs < 2000)])
            high_freq_energy = np.sum(magnitude[freqs >= 2000])
            total_energy = low_freq_energy + mid_freq_energy + high_freq_energy + 1e-10
            
            # Calculate stereo correlation (if stereo input)
            if audio_data.ndim > 1:
                correlation = np.corrcoef(audio_data[:, 0], audio_data[:, 1])[0, 1]
            else:
                correlation = 1.0
            
            # Calculate dynamic characteristics
            rms_level = np.sqrt(np.mean(audio_data ** 2))
            peak_level = np.max(np.abs(audio_data))
            
            requirements[track_id] = {
                "low_freq_ratio": float(low_freq_energy / total_energy),
                "mid_freq_ratio": float(mid_freq_energy / total_energy),
                "high_freq_ratio": float(high_freq_energy / total_energy),
                "stereo_correlation": float(correlation),
                "dynamic_range": float(peak_level / (rms_level + 1e-10)),
                "energy_level": float(rms_level),
                "requires_center_positioning": low_freq_energy > total_energy * 0.6,
                "requires_wide_positioning": high_freq_energy > total_energy * 0.4
            }
        
        return requirements
    
    def _determine_positioning_strategy(
        self,
        track_info: Dict[str, Any],
        musical_analysis: Optional[Dict[str, Any]]
    ) -> PositioningStrategy:
        """
        Determine the optimal positioning strategy based on musical content.
        
        Args:
            track_info: Track information
            musical_analysis: Musical analysis
            
        Returns:
            Selected positioning strategy
        """
        # Count instrument types
        instrument_counts = {}
        for info in track_info.values():
            if hasattr(info, 'instrument_type'):
                instrument_type = info.instrument_type
            else:
                instrument_type = info.get('instrument_type', 'unknown')
            instrument_counts[instrument_type] = instrument_counts.get(instrument_type, 0) + 1
        
        # Analyze instrument composition
        chinese_instruments = {'erhu', 'guzheng', 'pipa', 'dizi', 'guqin'}
        classical_instruments = {'violin', 'viola', 'cello', 'flute', 'oboe', 'trumpet'}
        rock_instruments = {'guitar', 'bass', 'drums', 'electric_guitar'}
        jazz_instruments = {'piano', 'bass', 'drums', 'saxophone', 'trumpet'}
        
        chinese_count = sum(instrument_counts.get(inst, 0) for inst in chinese_instruments)
        classical_count = sum(instrument_counts.get(inst, 0) for inst in classical_instruments)
        rock_count = sum(instrument_counts.get(inst, 0) for inst in rock_instruments)
        jazz_count = sum(instrument_counts.get(inst, 0) for inst in jazz_instruments)
        
        total_tracks = len(track_info)
        
        # Determine strategy
        if chinese_count > total_tracks * 0.6:
            return PositioningStrategy.CHINESE_TRADITIONAL
        elif classical_count > total_tracks * 0.5:
            return PositioningStrategy.CLASSICAL_ORCHESTRA
        elif rock_count > total_tracks * 0.4:
            return PositioningStrategy.ROCK_BAND
        elif jazz_count > total_tracks * 0.5 and 'piano' in instrument_counts:
            return PositioningStrategy.JAZZ_ENSEMBLE
        else:
            return PositioningStrategy.ADAPTIVE
    
    def _calculate_optimal_positions(
        self,
        spatial_requirements: Dict[str, Dict[str, float]],
        track_info: Dict[str, Any],
        strategy: PositioningStrategy
    ) -> Dict[str, SpatialPosition]:
        """
        Calculate optimal spatial positions for all tracks.
        
        Args:
            spatial_requirements: Spatial analysis results
            track_info: Track information
            strategy: Positioning strategy
            
        Returns:
            Dictionary of track_id -> SpatialPosition
        """
        positions = {}
        
        # Strategy-specific positioning templates
        strategy_adjustments = {
            PositioningStrategy.CLASSICAL_ORCHESTRA: {
                'violin': SpatialPosition(-45, 5, 0.3, 0.5, 0.4),
                'viola': SpatialPosition(45, 0, 0.4, 0.4, 0.5),
                'cello': SpatialPosition(30, -10, 0.3, 0.4, 0.4),
                'piano': SpatialPosition(0, 0, 0.1, 0.8, 0.2)
            },
            PositioningStrategy.CHINESE_TRADITIONAL: {
                'erhu': SpatialPosition(-30, 10, 0.2, 0.3, 0.2),
                'guzheng': SpatialPosition(0, 0, 0.1, 0.9, 0.1),
                'pipa': SpatialPosition(25, 5, 0.2, 0.3, 0.25),
                'dizi': SpatialPosition(-20, 15, 0.3, 0.2, 0.3),
                'guqin': SpatialPosition(15, -5, 0.15, 0.6, 0.15)
            },
            PositioningStrategy.ROCK_BAND: {
                'guitar': SpatialPosition(-30, 0, 0.2, 0.4, 0.3),
                'bass': SpatialPosition(0, -5, 0.2, 0.2, 0.3),
                'drums': SpatialPosition(0, 0, 0.3, 1.0, 0.4),
                'vocals': SpatialPosition(0, 0, 0.1, 0.1, 0.1)
            },
            PositioningStrategy.JAZZ_ENSEMBLE: {
                'piano': SpatialPosition(-15, 0, 0.15, 0.6, 0.2),
                'bass': SpatialPosition(0, -5, 0.2, 0.2, 0.3),
                'drums': SpatialPosition(15, 0, 0.3, 0.8, 0.4),
                'saxophone': SpatialPosition(-30, 10, 0.25, 0.3, 0.3)
            }
        }
        
        for track_id, info in track_info.items():
            # Get instrument type and base profile
            if hasattr(info, 'instrument_type'):
                instrument_type = info.instrument_type
            else:
                instrument_type = info.get('instrument_type', 'default')
            
            base_profile = self.spatial_profiles.get(instrument_type,
                                                   self.spatial_profiles['default'])
            
            # Start with strategy-specific position if available
            if strategy in strategy_adjustments and instrument_type in strategy_adjustments[strategy]:
                position = strategy_adjustments[strategy][instrument_type]
            else:
                position = base_profile.default_position
            
            # Adjust based on spatial requirements
            requirements = spatial_requirements[track_id]
            
            # Adjust width based on frequency content
            if requirements['requires_wide_positioning']:
                position.width = min(1.0, position.width * 1.3)
            elif requirements['requires_center_positioning']:
                position.width = max(0.1, position.width * 0.7)
                position.azimuth *= 0.5  # Move towards center
            
            # Adjust depth based on importance
            if hasattr(info, 'importance_weight'):
                importance = info.importance_weight
            else:
                importance = info.get('importance_weight', 0.5)
            
            # More important instruments go forward
            position.depth = max(0.1, position.depth * (1.5 - importance))
            
            # Adjust based on role
            if hasattr(info, 'role'):
                role = info.role
            else:
                role = info.get('role', 'harmony')
            
            if role in ['lead', 'melody']:
                position.azimuth *= 0.7  # Closer to center
                position.depth *= 0.8    # Move forward
            elif role == 'bass':
                position.azimuth *= 0.5  # Center low frequencies
                position.width = min(0.3, position.width)  # Narrow bass
            
            positions[track_id] = position
        
        return positions
    
    def _resolve_spatial_conflicts(
        self,
        positions: Dict[str, SpatialPosition],
        track_info: Dict[str, Any]
    ) -> Dict[str, SpatialPosition]:
        """
        Resolve conflicts when multiple instruments occupy similar spatial positions.
        
        Args:
            positions: Initial position assignments
            track_info: Track information
            
        Returns:
            Resolved positions without conflicts
        """
        resolved_positions = positions.copy()
        self.spatial_conflicts = []
        
        # Find conflicts (instruments too close to each other)
        conflict_threshold = 20.0  # degrees
        track_ids = list(positions.keys())
        
        for i, track_id_1 in enumerate(track_ids):
            for track_id_2 in track_ids[i+1:]:
                pos1 = positions[track_id_1]
                pos2 = positions[track_id_2]
                
                # Calculate angular distance
                azimuth_diff = abs(pos1.azimuth - pos2.azimuth)
                elevation_diff = abs(pos1.elevation - pos2.elevation)
                
                if azimuth_diff < conflict_threshold and elevation_diff < 10.0:
                    self.spatial_conflicts.append((track_id_1, track_id_2))
        
        # Resolve conflicts by adjusting positions
        for track_id_1, track_id_2 in self.spatial_conflicts:
            # Get instrument priorities
            info1 = track_info[track_id_1]
            info2 = track_info[track_id_2]
            
            if hasattr(info1, 'instrument_type'):
                instrument_type_1 = info1.instrument_type
            else:
                instrument_type_1 = info1.get('instrument_type', 'default')
                
            if hasattr(info2, 'instrument_type'):
                instrument_type_2 = info2.instrument_type
            else:
                instrument_type_2 = info2.get('instrument_type', 'default')
            
            profile1 = self.spatial_profiles.get(instrument_type_1, self.spatial_profiles['default'])
            profile2 = self.spatial_profiles.get(instrument_type_2, self.spatial_profiles['default'])
            
            # Higher priority instrument keeps its position
            if profile1.interaction_priority > profile2.interaction_priority:
                # Move track_id_2
                self._adjust_position_to_avoid_conflict(
                    resolved_positions[track_id_2], resolved_positions[track_id_1]
                )
            else:
                # Move track_id_1
                self._adjust_position_to_avoid_conflict(
                    resolved_positions[track_id_1], resolved_positions[track_id_2]
                )
        
        return resolved_positions
    
    def _adjust_position_to_avoid_conflict(
        self,
        position_to_adjust: SpatialPosition,
        fixed_position: SpatialPosition
    ) -> None:
        """
        Adjust a position to avoid conflict with a fixed position.
        
        Args:
            position_to_adjust: Position to be moved
            fixed_position: Position to avoid
        """
        # Calculate which direction to move
        azimuth_diff = position_to_adjust.azimuth - fixed_position.azimuth
        
        if abs(azimuth_diff) < 20.0:
            # Move away in azimuth
            if azimuth_diff >= 0:
                position_to_adjust.azimuth = fixed_position.azimuth + 25.0
            else:
                position_to_adjust.azimuth = fixed_position.azimuth - 25.0
            
            # Clamp to valid range
            position_to_adjust.azimuth = np.clip(position_to_adjust.azimuth, -180, 180)
            
            # Also adjust elevation slightly
            position_to_adjust.elevation += 5.0 if azimuth_diff >= 0 else -5.0
            position_to_adjust.elevation = np.clip(position_to_adjust.elevation, -30, 30)
    
    def _apply_spatial_processing(
        self,
        tracks: Dict[str, np.ndarray],
        positions: Dict[str, SpatialPosition]
    ) -> Dict[str, np.ndarray]:
        """
        Apply spatial processing to create stereo positioning.
        
        Args:
            tracks: Input tracks
            positions: Spatial positions
            
        Returns:
            Spatially processed stereo tracks
        """
        processed_tracks = {}
        
        for track_id, audio_data in tracks.items():
            if track_id in positions:
                position = positions[track_id]
                
                # Convert to stereo if mono
                if audio_data.ndim == 1:
                    stereo_audio = np.column_stack([audio_data, audio_data])
                else:
                    stereo_audio = audio_data.copy()
                
                # Apply panning based on azimuth
                pan_angle = np.radians(position.azimuth)
                left_gain = np.cos(pan_angle + np.pi/4)
                right_gain = np.sin(pan_angle + np.pi/4)
                
                # Apply gains
                stereo_audio[:, 0] *= left_gain
                stereo_audio[:, 1] *= right_gain
                
                # Apply width adjustment
                if position.width < 1.0:
                    # Narrow the stereo image
                    mid = (stereo_audio[:, 0] + stereo_audio[:, 1]) / 2
                    side = (stereo_audio[:, 0] - stereo_audio[:, 1]) / 2
                    side *= position.width
                    
                    stereo_audio[:, 0] = mid + side
                    stereo_audio[:, 1] = mid - side
                
                # Apply distance simulation (simple level adjustment)
                distance_attenuation = 1.0 - position.distance * 0.3
                stereo_audio *= distance_attenuation
                
                processed_tracks[track_id] = stereo_audio
            else:
                # No positioning info, keep as-is
                processed_tracks[track_id] = tracks[track_id]
        
        return processed_tracks
    
    def _apply_stereo_enhancement(
        self,
        tracks: Dict[str, np.ndarray],
        positions: Dict[str, SpatialPosition]
    ) -> Dict[str, np.ndarray]:
        """
        Apply stereo enhancement for improved spatial perception.
        
        Args:
            tracks: Spatially positioned tracks
            positions: Position information
            
        Returns:
            Enhanced stereo tracks
        """
        enhanced_tracks = {}
        
        for track_id, audio_data in tracks.items():
            if track_id in positions and audio_data.ndim == 2:
                position = positions[track_id]
                enhanced_audio = audio_data.copy()
                
                # Apply subtle stereo widening for appropriate instruments
                if position.width > 0.5:
                    # Simple stereo widening
                    mid = (enhanced_audio[:, 0] + enhanced_audio[:, 1]) / 2
                    side = (enhanced_audio[:, 0] - enhanced_audio[:, 1]) / 2
                    
                    # Enhance side signal slightly
                    side *= 1.1
                    
                    enhanced_audio[:, 0] = mid + side
                    enhanced_audio[:, 1] = mid - side
                
                enhanced_tracks[track_id] = enhanced_audio
            else:
                enhanced_tracks[track_id] = audio_data
        
        return enhanced_tracks

    
    def process(self, track, **kwargs):
        """Process a single audio track (required by base class)."""
        from .base_processor import ProcessingResult, ProcessingStage
        import time
        
        start_time = time.time()
        
        # For specialized processors, return original audio with metadata
        # In a full implementation, would apply specific processing
        processed_audio = track.audio_data if hasattr(track, 'audio_data') else track
        
        processing_time = time.time() - start_time
        
        return ProcessingResult(
            processed_audio=processed_audio,
            processing_stage=ProcessingStage.SPATIAL,
            success=True,
            processing_time=processing_time,
            metadata={"processor": "SpatialPositioningAlgorithm"}
        )
    
    def _get_processing_stage(self):
        """Return the processing stage this processor handles."""
        from .base_processor import ProcessingStage
        return ProcessingStage.SPATIAL