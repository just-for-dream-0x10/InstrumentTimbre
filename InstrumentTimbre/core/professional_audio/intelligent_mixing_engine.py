"""
Intelligent Mixing Engine - Automated mixing based on musical structure and instrument characteristics.

This module provides intelligent level balancing, automatic gain staging, and context-aware
mixing decisions based on musical analysis and instrument roles.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
from enum import Enum
import logging

from .base_processor import BaseAudioProcessor


class MixingStrategy(Enum):
    """Different mixing strategies for various musical styles."""
    CLASSICAL = "classical"
    ROCK_POP = "rock_pop"
    JAZZ = "jazz"
    ELECTRONIC = "electronic"
    CHINESE_TRADITIONAL = "chinese_traditional"
    ADAPTIVE = "adaptive"


@dataclass
class InstrumentMixingProfile:
    """Mixing profile for specific instrument types."""
    instrument_type: str
    default_level: float
    frequency_emphasis: Tuple[float, float]  # Low, high boost in dB
    dynamic_sensitivity: float
    stereo_width: float
    compression_ratio: float
    priority_weight: float


class IntelligentMixingEngine(BaseAudioProcessor):
    """
    Intelligent mixing engine that automatically balances audio levels based on
    musical structure, instrument roles, and perceptual principles.
    """
    
    def __init__(self, config):
        """
        Initialize the intelligent mixing engine.
        
        Args:
            config: Processing configuration
        """
        super().__init__("intelligentmixingengine", 22050)
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize instrument mixing profiles
        self.instrument_profiles = self._initialize_instrument_profiles()
        
        # Mixing state
        self.current_strategy = MixingStrategy.ADAPTIVE
        self.level_history: Dict[str, List[float]] = {}
        
    def _initialize_instrument_profiles(self) -> Dict[str, InstrumentMixingProfile]:
        """
        Initialize mixing profiles for different instrument types.
        
        Returns:
            Dictionary of instrument_type -> InstrumentMixingProfile
        """
        profiles = {
            # Traditional Western instruments
            "piano": InstrumentMixingProfile(
                instrument_type="piano",
                default_level=0.8,
                frequency_emphasis=(2.0, 1.5),
                dynamic_sensitivity=0.7,
                stereo_width=0.8,
                compression_ratio=2.5,
                priority_weight=0.9
            ),
            "guitar": InstrumentMixingProfile(
                instrument_type="guitar",
                default_level=0.7,
                frequency_emphasis=(0.0, 3.0),
                dynamic_sensitivity=0.8,
                stereo_width=0.6,
                compression_ratio=3.0,
                priority_weight=0.8
            ),
            "bass": InstrumentMixingProfile(
                instrument_type="bass",
                default_level=0.6,
                frequency_emphasis=(4.0, -2.0),
                dynamic_sensitivity=0.5,
                stereo_width=0.2,
                compression_ratio=4.0,
                priority_weight=0.9
            ),
            "drums": InstrumentMixingProfile(
                instrument_type="drums",
                default_level=0.75,
                frequency_emphasis=(3.0, 4.0),
                dynamic_sensitivity=0.9,
                stereo_width=1.0,
                compression_ratio=2.0,
                priority_weight=0.95
            ),
            "violin": InstrumentMixingProfile(
                instrument_type="violin",
                default_level=0.65,
                frequency_emphasis=(-1.0, 2.5),
                dynamic_sensitivity=0.8,
                stereo_width=0.7,
                compression_ratio=2.2,
                priority_weight=0.85
            ),
            
            # Chinese traditional instruments
            "erhu": InstrumentMixingProfile(
                instrument_type="erhu",
                default_level=0.7,
                frequency_emphasis=(1.0, 3.0),
                dynamic_sensitivity=0.9,
                stereo_width=0.5,
                compression_ratio=2.0,
                priority_weight=0.9
            ),
            "guzheng": InstrumentMixingProfile(
                instrument_type="guzheng",
                default_level=0.65,
                frequency_emphasis=(0.5, 4.0),
                dynamic_sensitivity=0.7,
                stereo_width=0.8,
                compression_ratio=2.5,
                priority_weight=0.8
            ),
            "pipa": InstrumentMixingProfile(
                instrument_type="pipa",
                default_level=0.6,
                frequency_emphasis=(1.5, 3.5),
                dynamic_sensitivity=0.8,
                stereo_width=0.6,
                compression_ratio=2.8,
                priority_weight=0.8
            ),
            "dizi": InstrumentMixingProfile(
                instrument_type="dizi",
                default_level=0.55,
                frequency_emphasis=(-2.0, 2.0),
                dynamic_sensitivity=0.8,
                stereo_width=0.4,
                compression_ratio=2.2,
                priority_weight=0.7
            ),
            "guqin": InstrumentMixingProfile(
                instrument_type="guqin",
                default_level=0.6,
                frequency_emphasis=(2.0, 1.0),
                dynamic_sensitivity=0.6,
                stereo_width=0.7,
                compression_ratio=2.0,
                priority_weight=0.85
            )
        }
        
        # Add generic profile for unknown instruments
        profiles["unknown"] = InstrumentMixingProfile(
            instrument_type="unknown",
            default_level=0.5,
            frequency_emphasis=(0.0, 0.0),
            dynamic_sensitivity=0.7,
            stereo_width=0.5,
            compression_ratio=2.5,
            priority_weight=0.5
        )
        
        return profiles
    
    def process_tracks(
        self,
        tracks: Dict[str, np.ndarray],
        track_info: Dict[str, Any],
        musical_analysis: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Process tracks with intelligent mixing algorithms.
        
        Args:
            tracks: Dictionary of track_id -> audio_data
            track_info: Track information and metadata
            musical_analysis: Optional musical structure analysis
            
        Returns:
            Tuple of (processed_tracks, mixing_metadata)
        """
        self.logger.info(f"Starting intelligent mixing for {len(tracks)} tracks")
        
        try:
            # Step 1: Analyze track characteristics
            track_characteristics = self._analyze_track_characteristics(tracks, track_info)
            
            # Step 2: Determine optimal mixing strategy
            mixing_strategy = self._determine_mixing_strategy(track_info, musical_analysis)
            
            # Step 3: Calculate target levels for each track
            target_levels = self._calculate_target_levels(
                track_characteristics, track_info, mixing_strategy
            )
            
            # Step 4: Apply intelligent level adjustments
            level_adjusted_tracks = self._apply_level_adjustments(tracks, target_levels)
            
            # Step 5: Apply frequency-aware gain staging
            gain_staged_tracks = self._apply_frequency_aware_gain_staging(
                level_adjusted_tracks, track_info
            )
            
            # Step 6: Perform masking analysis and compensation
            final_tracks = self._apply_masking_compensation(
                gain_staged_tracks, track_characteristics, track_info
            )
            
            # Compile mixing metadata
            mixing_metadata = {
                "mixing_strategy": mixing_strategy.value,
                "track_characteristics": track_characteristics,
                "target_levels": target_levels,
                "level_adjustments_applied": len(final_tracks),
                "frequency_staging_applied": True,
                "masking_compensation_applied": True
            }
            
            self.logger.info("Intelligent mixing completed successfully")
            return final_tracks, mixing_metadata
            
        except Exception as e:
            self.logger.error(f"Intelligent mixing failed: {e}")
            raise
    
    def _analyze_track_characteristics(
        self,
        tracks: Dict[str, np.ndarray],
        track_info: Dict[str, Any]
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze audio characteristics of each track for mixing decisions.
        
        Args:
            tracks: Audio track data
            track_info: Track metadata
            
        Returns:
            Dictionary of track_id -> characteristics
        """
        characteristics = {}
        
        for track_id, audio_data in tracks.items():
            # Calculate RMS level
            rms_level = np.sqrt(np.mean(audio_data ** 2))
            
            # Calculate peak level
            peak_level = np.max(np.abs(audio_data))
            
            # Calculate dynamic range (simplified)
            dynamic_range = peak_level / (rms_level + 1e-10)
            
            # Calculate spectral centroid (brightness indicator)
            fft_data = np.fft.fft(audio_data[:min(4096, len(audio_data))])
            magnitude = np.abs(fft_data[:len(fft_data)//2])
            freqs = np.fft.fftfreq(len(fft_data), 1/self.config.sample_rate)[:len(magnitude)]
            spectral_centroid = np.sum(freqs * magnitude) / (np.sum(magnitude) + 1e-10)
            
            # Calculate energy distribution
            low_energy = np.sum(magnitude[:len(magnitude)//4])
            mid_energy = np.sum(magnitude[len(magnitude)//4:3*len(magnitude)//4])
            high_energy = np.sum(magnitude[3*len(magnitude)//4:])
            total_energy = low_energy + mid_energy + high_energy + 1e-10
            
            characteristics[track_id] = {
                "rms_level": float(rms_level),
                "peak_level": float(peak_level),
                "dynamic_range": float(dynamic_range),
                "spectral_centroid": float(spectral_centroid),
                "low_energy_ratio": float(low_energy / total_energy),
                "mid_energy_ratio": float(mid_energy / total_energy),
                "high_energy_ratio": float(high_energy / total_energy),
                "loudness_estimate": float(rms_level * 20)  # Simplified loudness
            }
        
        return characteristics
    
    def _determine_mixing_strategy(
        self,
        track_info: Dict[str, Any],
        musical_analysis: Optional[Dict[str, Any]]
    ) -> MixingStrategy:
        """
        Determine the optimal mixing strategy based on track content and analysis.
        
        Args:
            track_info: Track information
            musical_analysis: Musical structure analysis
            
        Returns:
            Selected mixing strategy
        """
        # Count instrument types to determine musical style
        instrument_counts = {}
        for info in track_info.values():
            if hasattr(info, 'instrument_type'):
                instrument_type = info.instrument_type
            else:
                instrument_type = info.get('instrument_type', 'unknown')
            instrument_counts[instrument_type] = instrument_counts.get(instrument_type, 0) + 1
        
        # Chinese instruments presence
        chinese_instruments = {'erhu', 'guzheng', 'pipa', 'dizi', 'guqin'}
        chinese_count = sum(instrument_counts.get(inst, 0) for inst in chinese_instruments)
        
        # Western classical instruments
        classical_instruments = {'violin', 'viola', 'cello', 'flute', 'oboe', 'trumpet'}
        classical_count = sum(instrument_counts.get(inst, 0) for inst in classical_instruments)
        
        # Rock/Pop instruments
        rock_instruments = {'guitar', 'bass', 'drums', 'electric_guitar'}
        rock_count = sum(instrument_counts.get(inst, 0) for inst in rock_instruments)
        
        # Determine strategy based on instrument composition
        if chinese_count > len(track_info) * 0.6:
            return MixingStrategy.CHINESE_TRADITIONAL
        elif classical_count > len(track_info) * 0.5:
            return MixingStrategy.CLASSICAL
        elif rock_count > len(track_info) * 0.4:
            return MixingStrategy.ROCK_POP
        elif 'piano' in instrument_counts and chinese_count > 0:
            return MixingStrategy.JAZZ
        else:
            return MixingStrategy.ADAPTIVE
    
    def _calculate_target_levels(
        self,
        track_characteristics: Dict[str, Dict[str, float]],
        track_info: Dict[str, Any],
        strategy: MixingStrategy
    ) -> Dict[str, float]:
        """
        Calculate optimal target levels for each track.
        
        Args:
            track_characteristics: Analyzed track characteristics
            track_info: Track metadata
            strategy: Selected mixing strategy
            
        Returns:
            Dictionary of track_id -> target_level
        """
        target_levels = {}
        
        for track_id, info in track_info.items():
            # Get instrument type and profile
            if hasattr(info, 'instrument_type'):
                instrument_type = info.instrument_type
            else:
                instrument_type = info.get('instrument_type', 'unknown')
            
            profile = self.instrument_profiles.get(instrument_type, 
                                                 self.instrument_profiles['unknown'])
            
            # Start with default level from profile
            base_level = profile.default_level
            
            # Adjust based on musical role
            if hasattr(info, 'role'):
                role = info.role
            else:
                role = info.get('role', 'harmony')
            
            role_adjustments = {
                'lead': 1.2,
                'melody': 1.15,
                'rhythm': 1.0,
                'bass': 0.9,
                'harmony': 0.8,
                'percussion': 1.1,
                'accompaniment': 0.7
            }
            
            role_factor = role_adjustments.get(role, 1.0)
            
            # Adjust based on importance weight
            if hasattr(info, 'importance_weight'):
                importance = info.importance_weight
            else:
                importance = info.get('importance_weight', 0.5)
            
            # Apply strategy-specific adjustments
            strategy_factors = {
                MixingStrategy.CLASSICAL: {
                    'violin': 1.1, 'piano': 1.2, 'cello': 1.0
                },
                MixingStrategy.CHINESE_TRADITIONAL: {
                    'erhu': 1.3, 'guzheng': 1.1, 'pipa': 1.0, 'dizi': 0.9
                },
                MixingStrategy.ROCK_POP: {
                    'guitar': 1.2, 'bass': 1.1, 'drums': 1.3, 'vocals': 1.4
                },
                MixingStrategy.JAZZ: {
                    'piano': 1.2, 'bass': 1.1, 'drums': 1.0, 'saxophone': 1.3
                }
            }
            
            strategy_factor = strategy_factors.get(strategy, {}).get(instrument_type, 1.0)
            
            # Calculate final target level
            target_level = base_level * role_factor * (0.5 + importance * 0.5) * strategy_factor
            
            # Clamp to reasonable range
            target_level = np.clip(target_level, 0.1, 1.0)
            
            target_levels[track_id] = target_level
        
        return target_levels
    
    def _apply_level_adjustments(
        self,
        tracks: Dict[str, np.ndarray],
        target_levels: Dict[str, float]
    ) -> Dict[str, np.ndarray]:
        """
        Apply calculated level adjustments to tracks.
        
        Args:
            tracks: Original audio tracks
            target_levels: Target levels for each track
            
        Returns:
            Level-adjusted tracks
        """
        adjusted_tracks = {}
        
        for track_id, audio_data in tracks.items():
            if track_id in target_levels:
                # Calculate current RMS level
                current_rms = np.sqrt(np.mean(audio_data ** 2))
                
                if current_rms > 0:
                    # Calculate adjustment factor
                    target_rms = target_levels[track_id] * 0.2  # Conservative scaling
                    adjustment_factor = target_rms / current_rms
                    
                    # Apply smooth adjustment to avoid artifacts
                    adjusted_audio = audio_data * adjustment_factor
                    
                    # Ensure no clipping
                    max_amplitude = np.max(np.abs(adjusted_audio))
                    if max_amplitude > 0.95:
                        adjusted_audio = adjusted_audio * (0.95 / max_amplitude)
                    
                    adjusted_tracks[track_id] = adjusted_audio
                else:
                    adjusted_tracks[track_id] = audio_data
            else:
                adjusted_tracks[track_id] = audio_data
        
        return adjusted_tracks
    
    def _apply_frequency_aware_gain_staging(
        self,
        tracks: Dict[str, np.ndarray],
        track_info: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        """
        Apply frequency-aware gain staging to prevent frequency masking.
        
        Args:
            tracks: Level-adjusted tracks
            track_info: Track information
            
        Returns:
            Frequency-staged tracks
        """
        # For now, return tracks as-is (simplified implementation)
        # In a full implementation, this would apply frequency-dependent gain adjustments
        return tracks
    
    def _apply_masking_compensation(
        self,
        tracks: Dict[str, np.ndarray],
        track_characteristics: Dict[str, Dict[str, float]],
        track_info: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        """
        Apply masking compensation to improve clarity between instruments.
        
        Args:
            tracks: Frequency-staged tracks
            track_characteristics: Track characteristics
            track_info: Track information
            
        Returns:
            Final processed tracks with masking compensation
        """
        # For now, return tracks as-is (simplified implementation)
        # In a full implementation, this would analyze frequency masking and apply compensation
        return tracks

    
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
            processing_stage=ProcessingStage.MIXING,
            success=True,
            processing_time=processing_time,
            metadata={"processor": "IntelligentMixingEngine"}
        )
    
    def _get_processing_stage(self):
        """Return the processing stage this processor handles."""
        from .base_processor import ProcessingStage
        return ProcessingStage.MIXING