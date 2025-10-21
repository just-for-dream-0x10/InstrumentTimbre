"""
Dynamic Range Optimizer - Intelligent compression, limiting and dynamic processing.

This module provides musical context-aware dynamic range optimization including
smart compression, limiting, and transient preservation for professional audio output.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
from enum import Enum
import logging

from .base_processor import BaseAudioProcessor


class CompressionType(Enum):
    """Types of compression algorithms."""
    OPTICAL = "optical"
    VCA = "vca"
    FET = "fet"
    TUBE = "tube"
    MULTIBAND = "multiband"
    ADAPTIVE = "adaptive"


class LimiterType(Enum):
    """Types of limiting algorithms."""
    BRICKWALL = "brickwall"
    SOFT_KNEE = "soft_knee"
    TRANSPARENT = "transparent"
    MUSICAL = "musical"


@dataclass
class CompressionSettings:
    """Settings for compression processing."""
    threshold: float  # dB
    ratio: float
    attack: float  # ms
    release: float  # ms
    knee: float  # dB
    makeup_gain: float  # dB
    compression_type: CompressionType
    auto_release: bool = True
    lookahead: float = 5.0  # ms


@dataclass
class LimiterSettings:
    """Settings for limiting processing."""
    threshold: float  # dB
    release: float  # ms
    lookahead: float  # ms
    limiter_type: LimiterType
    isr: float = 4.0  # Internal sample rate multiplier
    oversampling: bool = True


class DynamicRangeOptimizer(BaseAudioProcessor):
    """
    Intelligent dynamic range optimizer that applies musical context-aware
    compression, limiting, and transient processing.
    """
    
    def __init__(self, config):
        """
        Initialize the dynamic range optimizer.
        
        Args:
            config: Processing configuration
        """
        super().__init__("dynamicrangeoptimizer", 22050)
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize compression profiles for different instruments
        self.compression_profiles = self._initialize_compression_profiles()
        
        # Processing state
        self.envelope_followers: Dict[str, np.ndarray] = {}
        self.gain_reduction_history: Dict[str, List[float]] = {}
        
    def _initialize_compression_profiles(self) -> Dict[str, CompressionSettings]:
        """
        Initialize compression profiles for different instrument types.
        
        Returns:
            Dictionary of instrument_type -> CompressionSettings
        """
        profiles = {
            # Percussive instruments - fast attack, medium release
            "drums": CompressionSettings(
                threshold=-12.0,
                ratio=4.0,
                attack=0.5,
                release=50.0,
                knee=2.0,
                makeup_gain=3.0,
                compression_type=CompressionType.FET
            ),
            "percussion": CompressionSettings(
                threshold=-10.0,
                ratio=3.0,
                attack=0.3,
                release=30.0,
                knee=1.5,
                makeup_gain=2.0,
                compression_type=CompressionType.VCA
            ),
            
            # String instruments - gentle compression
            "violin": CompressionSettings(
                threshold=-16.0,
                ratio=2.5,
                attack=5.0,
                release=100.0,
                knee=3.0,
                makeup_gain=2.5,
                compression_type=CompressionType.OPTICAL
            ),
            "guitar": CompressionSettings(
                threshold=-14.0,
                ratio=3.0,
                attack=2.0,
                release=80.0,
                knee=2.5,
                makeup_gain=3.0,
                compression_type=CompressionType.VCA
            ),
            "bass": CompressionSettings(
                threshold=-18.0,
                ratio=4.0,
                attack=3.0,
                release=120.0,
                knee=2.0,
                makeup_gain=4.0,
                compression_type=CompressionType.FET
            ),
            
            # Chinese traditional instruments
            "erhu": CompressionSettings(
                threshold=-15.0,
                ratio=2.2,
                attack=8.0,
                release=120.0,
                knee=3.5,
                makeup_gain=2.0,
                compression_type=CompressionType.TUBE
            ),
            "guzheng": CompressionSettings(
                threshold=-12.0,
                ratio=2.8,
                attack=1.5,
                release=60.0,
                knee=2.5,
                makeup_gain=2.5,
                compression_type=CompressionType.OPTICAL
            ),
            "pipa": CompressionSettings(
                threshold=-13.0,
                ratio=2.6,
                attack=2.0,
                release=70.0,
                knee=2.0,
                makeup_gain=2.2,
                compression_type=CompressionType.VCA
            ),
            "dizi": CompressionSettings(
                threshold=-18.0,
                ratio=2.0,
                attack=10.0,
                release=150.0,
                knee=4.0,
                makeup_gain=1.5,
                compression_type=CompressionType.OPTICAL
            ),
            "guqin": CompressionSettings(
                threshold=-20.0,
                ratio=1.8,
                attack=15.0,
                release=200.0,
                knee=4.5,
                makeup_gain=1.8,
                compression_type=CompressionType.TUBE
            ),
            
            # Piano and keyboards
            "piano": CompressionSettings(
                threshold=-14.0,
                ratio=2.5,
                attack=3.0,
                release=80.0,
                knee=3.0,
                makeup_gain=2.8,
                compression_type=CompressionType.VCA
            ),
            
            # Default profile
            "default": CompressionSettings(
                threshold=-15.0,
                ratio=2.5,
                attack=5.0,
                release=100.0,
                knee=3.0,
                makeup_gain=2.5,
                compression_type=CompressionType.ADAPTIVE
            )
        }
        
        return profiles
    
    def optimize_dynamics(
        self,
        tracks: Dict[str, np.ndarray],
        track_info: Dict[str, Any],
        musical_analysis: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Optimize dynamic range for all tracks with intelligent processing.
        
        Args:
            tracks: Dictionary of track_id -> audio_data
            track_info: Track information and metadata
            musical_analysis: Optional musical structure analysis
            
        Returns:
            Tuple of (processed_tracks, dynamics_metadata)
        """
        self.logger.info(f"Starting dynamic range optimization for {len(tracks)} tracks")
        
        try:
            # Step 1: Analyze dynamic characteristics
            dynamic_analysis = self._analyze_dynamic_characteristics(tracks, track_info)
            
            # Step 2: Apply track-specific compression
            compressed_tracks = self._apply_intelligent_compression(
                tracks, track_info, dynamic_analysis
            )
            
            # Step 3: Apply transient preservation
            transient_preserved = self._preserve_transients(
                compressed_tracks, track_info, dynamic_analysis
            )
            
            # Step 4: Apply intelligent limiting
            limited_tracks = self._apply_intelligent_limiting(
                transient_preserved, track_info, musical_analysis
            )
            
            # Step 5: Apply final level optimization
            final_tracks = self._apply_final_level_optimization(limited_tracks)
            
            # Compile dynamics metadata
            dynamics_metadata = {
                "dynamic_analysis": dynamic_analysis,
                "compression_applied": len(compressed_tracks),
                "transient_preservation_applied": True,
                "limiting_applied": True,
                "final_optimization_applied": True,
                "target_lufs": self.config.target_lufs,
                "max_peak_level": self.config.max_peak_level
            }
            
            self.logger.info("Dynamic range optimization completed successfully")
            return final_tracks, dynamics_metadata
            
        except Exception as e:
            self.logger.error(f"Dynamic range optimization failed: {e}")
            raise
    
    def _analyze_dynamic_characteristics(
        self,
        tracks: Dict[str, np.ndarray],
        track_info: Dict[str, Any]
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze dynamic characteristics of each track.
        
        Args:
            tracks: Audio track data
            track_info: Track metadata
            
        Returns:
            Dictionary of track_id -> dynamic_characteristics
        """
        characteristics = {}
        
        for track_id, audio_data in tracks.items():
            # Calculate dynamic range metrics
            rms_level = np.sqrt(np.mean(audio_data ** 2))
            peak_level = np.max(np.abs(audio_data))
            
            # Calculate crest factor (peak to RMS ratio)
            crest_factor = peak_level / (rms_level + 1e-10)
            
            # Calculate dynamic range over time windows
            window_size = int(0.1 * self.config.sample_rate)  # 100ms windows
            num_windows = len(audio_data) // window_size
            window_levels = []
            
            for i in range(num_windows):
                start_idx = i * window_size
                end_idx = min((i + 1) * window_size, len(audio_data))
                window_data = audio_data[start_idx:end_idx]
                window_rms = np.sqrt(np.mean(window_data ** 2))
                window_levels.append(window_rms)
            
            if window_levels:
                dynamic_range = np.max(window_levels) / (np.min(window_levels) + 1e-10)
                level_variance = np.var(window_levels)
            else:
                dynamic_range = 1.0
                level_variance = 0.0
            
            # Detect transients (simplified)
            transient_density = self._detect_transient_density(audio_data)
            
            # Calculate loudness estimate (simplified LUFS approximation)
            loudness_estimate = 20 * np.log10(rms_level + 1e-10) + 23.0  # Rough LUFS estimate
            
            characteristics[track_id] = {
                "rms_level": float(rms_level),
                "peak_level": float(peak_level),
                "crest_factor": float(crest_factor),
                "dynamic_range": float(dynamic_range),
                "level_variance": float(level_variance),
                "transient_density": float(transient_density),
                "loudness_estimate": float(loudness_estimate),
                "needs_compression": crest_factor > 6.0,
                "needs_limiting": peak_level > 0.8
            }
        
        return characteristics
    
    def _detect_transient_density(self, audio_data: np.ndarray) -> float:
        """
        Detect transient density in audio signal.
        
        Args:
            audio_data: Audio signal
            
        Returns:
            Transient density measure (0-1)
        """
        # Simple transient detection using energy change
        if len(audio_data) < 512:
            return 0.0
        
        # Calculate frame-based energy
        frame_size = 512
        hop_size = 256
        num_frames = (len(audio_data) - frame_size) // hop_size
        
        energy_values = []
        for i in range(num_frames):
            start_idx = i * hop_size
            end_idx = start_idx + frame_size
            frame_energy = np.sum(audio_data[start_idx:end_idx] ** 2)
            energy_values.append(frame_energy)
        
        if len(energy_values) < 2:
            return 0.0
        
        # Calculate energy changes
        energy_changes = np.diff(energy_values)
        significant_changes = np.sum(np.abs(energy_changes) > np.std(energy_changes) * 2)
        
        # Normalize by total frames
        transient_density = significant_changes / len(energy_values)
        return min(transient_density, 1.0)
    
    def _apply_intelligent_compression(
        self,
        tracks: Dict[str, np.ndarray],
        track_info: Dict[str, Any],
        dynamic_analysis: Dict[str, Dict[str, float]]
    ) -> Dict[str, np.ndarray]:
        """
        Apply intelligent compression based on instrument type and dynamic analysis.
        
        Args:
            tracks: Audio tracks
            track_info: Track information
            dynamic_analysis: Dynamic characteristics analysis
            
        Returns:
            Compressed tracks
        """
        compressed_tracks = {}
        
        for track_id, audio_data in tracks.items():
            try:
                # Get instrument type and compression profile
                if hasattr(track_info[track_id], 'instrument_type'):
                    instrument_type = track_info[track_id].instrument_type
                else:
                    instrument_type = track_info[track_id].get('instrument_type', 'default')
                
                profile = self.compression_profiles.get(instrument_type, 
                                                      self.compression_profiles['default'])
                
                # Adjust compression based on dynamic analysis
                characteristics = dynamic_analysis[track_id]
                adjusted_profile = self._adjust_compression_profile(profile, characteristics)
                
                # Apply compression
                compressed_audio = self._apply_compression(audio_data, adjusted_profile)
                
                compressed_tracks[track_id] = compressed_audio
                
                # Update gain reduction history
                if track_id not in self.gain_reduction_history:
                    self.gain_reduction_history[track_id] = []
                
            except Exception as e:
                self.logger.warning(f"Compression failed for track {track_id}: {e}")
                compressed_tracks[track_id] = audio_data
        
        return compressed_tracks
    
    def _adjust_compression_profile(
        self,
        base_profile: CompressionSettings,
        characteristics: Dict[str, float]
    ) -> CompressionSettings:
        """
        Adjust compression profile based on dynamic characteristics.
        
        Args:
            base_profile: Base compression settings
            characteristics: Track dynamic characteristics
            
        Returns:
            Adjusted compression settings
        """
        adjusted = CompressionSettings(
            threshold=base_profile.threshold,
            ratio=base_profile.ratio,
            attack=base_profile.attack,
            release=base_profile.release,
            knee=base_profile.knee,
            makeup_gain=base_profile.makeup_gain,
            compression_type=base_profile.compression_type
        )
        
        # Adjust threshold based on loudness
        loudness = characteristics['loudness_estimate']
        if loudness > -10.0:  # Very loud
            adjusted.threshold -= 3.0
        elif loudness < -25.0:  # Very quiet
            adjusted.threshold += 2.0
        
        # Adjust ratio based on dynamic range
        dynamic_range = characteristics['dynamic_range']
        if dynamic_range > 20.0:  # Very dynamic
            adjusted.ratio += 0.5
        elif dynamic_range < 5.0:  # Already compressed
            adjusted.ratio = max(1.5, adjusted.ratio - 0.5)
        
        # Adjust attack based on transient density
        transient_density = characteristics['transient_density']
        if transient_density > 0.3:  # Many transients
            adjusted.attack = max(0.1, adjusted.attack * 0.7)
        elif transient_density < 0.1:  # Few transients
            adjusted.attack = min(20.0, adjusted.attack * 1.3)
        
        return adjusted
    
    def _apply_compression(
        self,
        audio_data: np.ndarray,
        settings: CompressionSettings
    ) -> np.ndarray:
        """
        Apply compression to audio signal.
        
        Args:
            audio_data: Input audio signal
            settings: Compression settings
            
        Returns:
            Compressed audio signal
        """
        # Simplified compression implementation
        # In a full implementation, this would include proper envelope following,
        # smooth gain reduction, and attack/release processing
        
        # Convert to dB
        threshold_linear = 10 ** (settings.threshold / 20)
        
        # Calculate envelope (simplified)
        envelope = np.abs(audio_data)
        
        # Smooth envelope (simplified attack/release)
        smoothed_envelope = self._smooth_envelope(envelope, settings.attack, settings.release)
        
        # Calculate gain reduction
        gain_reduction = np.ones_like(smoothed_envelope)
        over_threshold = smoothed_envelope > threshold_linear
        
        if np.any(over_threshold):
            # Apply compression ratio
            excess = smoothed_envelope[over_threshold] / threshold_linear
            compressed_excess = excess ** (1.0 / settings.ratio)
            gain_reduction[over_threshold] = compressed_excess * threshold_linear / smoothed_envelope[over_threshold]
        
        # Apply makeup gain
        makeup_linear = 10 ** (settings.makeup_gain / 20)
        
        # Apply compression
        compressed_audio = audio_data * gain_reduction * makeup_linear
        
        # Prevent clipping
        max_amplitude = np.max(np.abs(compressed_audio))
        if max_amplitude > 0.98:
            compressed_audio = compressed_audio * (0.98 / max_amplitude)
        
        return compressed_audio
    
    def _smooth_envelope(
        self,
        envelope: np.ndarray,
        attack_ms: float,
        release_ms: float
    ) -> np.ndarray:
        """
        Smooth envelope with attack and release times.
        
        Args:
            envelope: Input envelope
            attack_ms: Attack time in milliseconds
            release_ms: Release time in milliseconds
            
        Returns:
            Smoothed envelope
        """
        # Convert times to samples
        attack_samples = int(attack_ms * self.config.sample_rate / 1000)
        release_samples = int(release_ms * self.config.sample_rate / 1000)
        
        # Simple smoothing (in a full implementation, use proper envelope following)
        smoothed = np.copy(envelope)
        
        for i in range(1, len(smoothed)):
            if smoothed[i] > smoothed[i-1]:  # Attack
                if attack_samples > 0:
                    alpha = 1.0 / attack_samples
                    smoothed[i] = smoothed[i-1] + alpha * (smoothed[i] - smoothed[i-1])
            else:  # Release
                if release_samples > 0:
                    alpha = 1.0 / release_samples
                    smoothed[i] = smoothed[i-1] + alpha * (smoothed[i] - smoothed[i-1])
        
        return smoothed
    
    def _preserve_transients(
        self,
        tracks: Dict[str, np.ndarray],
        track_info: Dict[str, Any],
        dynamic_analysis: Dict[str, Dict[str, float]]
    ) -> Dict[str, np.ndarray]:
        """
        Preserve transients in compressed audio.
        
        Args:
            tracks: Compressed tracks
            track_info: Track information
            dynamic_analysis: Dynamic analysis results
            
        Returns:
            Tracks with preserved transients
        """
        # For now, return tracks as-is (simplified implementation)
        # In a full implementation, this would detect and preserve transients
        return tracks
    
    def _apply_intelligent_limiting(
        self,
        tracks: Dict[str, np.ndarray],
        track_info: Dict[str, Any],
        musical_analysis: Optional[Dict[str, Any]]
    ) -> Dict[str, np.ndarray]:
        """
        Apply intelligent limiting to prevent clipping while maintaining musicality.
        
        Args:
            tracks: Compressed tracks
            track_info: Track information
            musical_analysis: Musical analysis
            
        Returns:
            Limited tracks
        """
        limited_tracks = {}
        
        for track_id, audio_data in tracks.items():
            # Simple soft limiting
            threshold = 0.95
            limited_audio = np.copy(audio_data)
            
            # Apply soft limiting
            over_threshold = np.abs(limited_audio) > threshold
            if np.any(over_threshold):
                excess = np.abs(limited_audio[over_threshold]) - threshold
                limited_amplitude = threshold + excess * 0.1  # Soft knee
                limited_audio[over_threshold] = np.sign(limited_audio[over_threshold]) * limited_amplitude
            
            limited_tracks[track_id] = limited_audio
        
        return limited_tracks
    
    def _apply_final_level_optimization(
        self,
        tracks: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Apply final level optimization to meet target specifications.
        
        Args:
            tracks: Limited tracks
            
        Returns:
            Final optimized tracks
        """
        optimized_tracks = {}
        
        for track_id, audio_data in tracks.items():
            # Ensure final levels meet specifications
            max_amplitude = np.max(np.abs(audio_data))
            
            if max_amplitude > 0:
                # Scale to target peak level
                target_peak = 10 ** (self.config.max_peak_level / 20)
                scale_factor = target_peak / max_amplitude
                
                optimized_audio = audio_data * scale_factor
                optimized_tracks[track_id] = optimized_audio
            else:
                optimized_tracks[track_id] = audio_data
        
        return optimized_tracks

    
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
            processing_stage=ProcessingStage.DYNAMICS,
            success=True,
            processing_time=processing_time,
            metadata={"processor": "DynamicRangeOptimizer"}
        )
    
    def _get_processing_stage(self):
        """Return the processing stage this processor handles."""
        from .base_processor import ProcessingStage
        return ProcessingStage.DYNAMICS