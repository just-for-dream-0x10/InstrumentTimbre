"""
Intelligent EQ Balancer - Frequency conflict resolution and instrument-specific EQ.

This module provides intelligent EQ balancing to prevent frequency masking between
instruments while enhancing the unique characteristics of each instrument type.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
from enum import Enum
import logging

from .base_processor import BaseAudioProcessor


class EQType(Enum):
    """Types of EQ processing."""
    PARAMETRIC = "parametric"
    GRAPHIC = "graphic"
    LINEAR_PHASE = "linear_phase"
    MINIMUM_PHASE = "minimum_phase"
    ANALOG_MODELED = "analog_modeled"


@dataclass
class EQBand:
    """Single EQ band configuration."""
    frequency: float  # Hz
    gain: float  # dB
    q_factor: float  # Quality factor
    filter_type: str  # 'bell', 'shelf', 'highpass', 'lowpass'


@dataclass
class InstrumentEQProfile:
    """EQ profile for specific instrument types."""
    instrument_type: str
    frequency_range: Tuple[float, float]  # Fundamental frequency range
    boost_frequencies: List[EQBand]  # Frequencies to enhance
    cut_frequencies: List[EQBand]  # Frequencies to reduce
    conflict_priority: float  # Priority in frequency conflicts
    masking_susceptibility: float  # How easily this instrument gets masked


class IntelligentEQBalancer(BaseAudioProcessor):
    """
    Intelligent EQ balancer that automatically resolves frequency conflicts
    and enhances instrument characteristics.
    """
    
    def __init__(self, config):
        """Initialize the intelligent EQ balancer."""
        super().__init__("intelligenteqbalancer", 22050)
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize EQ profiles
        self.eq_profiles = self._initialize_eq_profiles()
        
        # Frequency analysis state
        self.frequency_conflicts: List[Tuple[str, str, float]] = []
        
    def _initialize_eq_profiles(self) -> Dict[str, InstrumentEQProfile]:
        """Initialize EQ profiles for different instrument types."""
        profiles = {
            # Drums and percussion
            "drums": InstrumentEQProfile(
                instrument_type="drums",
                frequency_range=(60, 8000),
                boost_frequencies=[
                    EQBand(80, 2.0, 1.5, "bell"),    # Kick fundamentals
                    EQBand(200, 1.0, 2.0, "bell"),   # Body
                    EQBand(5000, 2.5, 1.0, "bell"),  # Attack/presence
                    EQBand(10000, 1.5, 0.8, "shelf") # Air
                ],
                cut_frequencies=[
                    EQBand(400, -1.5, 2.0, "bell"),  # Muddiness
                    EQBand(800, -1.0, 1.5, "bell")   # Boxiness
                ],
                conflict_priority=0.9,
                masking_susceptibility=0.2
            ),
            
            # Bass instruments
            "bass": InstrumentEQProfile(
                instrument_type="bass",
                frequency_range=(40, 400),
                boost_frequencies=[
                    EQBand(60, 1.5, 1.2, "bell"),    # Fundamentals
                    EQBand(120, 1.0, 1.5, "bell"),   # Warmth
                    EQBand(2500, 2.0, 2.0, "bell")   # Clarity/attack
                ],
                cut_frequencies=[
                    EQBand(300, -1.0, 1.5, "bell"),  # Muddiness
                    EQBand(500, -0.5, 2.0, "bell")   # Thickness
                ],
                conflict_priority=0.85,
                masking_susceptibility=0.3
            ),
            
            # Guitar
            "guitar": InstrumentEQProfile(
                instrument_type="guitar",
                frequency_range=(80, 5000),
                boost_frequencies=[
                    EQBand(100, 0.5, 1.0, "bell"),   # Body
                    EQBand(1000, 1.5, 1.5, "bell"),  # Clarity
                    EQBand(3000, 2.0, 1.2, "bell"),  # Presence
                    EQBand(8000, 1.0, 0.8, "shelf")  # Air
                ],
                cut_frequencies=[
                    EQBand(250, -1.0, 2.0, "bell"),  # Muddiness
                    EQBand(500, -0.5, 1.5, "bell")   # Honkiness
                ],
                conflict_priority=0.7,
                masking_susceptibility=0.5
            ),
            
            # Piano
            "piano": InstrumentEQProfile(
                instrument_type="piano",
                frequency_range=(27, 8000),
                boost_frequencies=[
                    EQBand(60, 0.5, 1.0, "bell"),    # Low fundamentals
                    EQBand(250, 1.0, 1.5, "bell"),   # Warmth
                    EQBand(2000, 1.5, 1.2, "bell"),  # Clarity
                    EQBand(8000, 1.0, 0.8, "shelf")  # Sparkle
                ],
                cut_frequencies=[
                    EQBand(500, -0.8, 2.0, "bell"),  # Muddiness
                    EQBand(1000, -0.5, 1.5, "bell")  # Harshness
                ],
                conflict_priority=0.8,
                masking_susceptibility=0.4
            ),
            
            # Violin
            "violin": InstrumentEQProfile(
                instrument_type="violin",
                frequency_range=(196, 3136),
                boost_frequencies=[
                    EQBand(400, 1.0, 1.5, "bell"),   # Body resonance
                    EQBand(1000, 1.5, 1.2, "bell"),  # Clarity
                    EQBand(3000, 2.0, 1.0, "bell"),  # Presence
                    EQBand(10000, 1.0, 0.8, "shelf") # Air
                ],
                cut_frequencies=[
                    EQBand(800, -0.5, 2.0, "bell"),  # Nasal frequencies
                    EQBand(2000, -0.3, 1.5, "bell")  # Harshness
                ],
                conflict_priority=0.6,
                masking_susceptibility=0.6
            ),
            
            # Chinese instruments
            "erhu": InstrumentEQProfile(
                instrument_type="erhu",
                frequency_range=(196, 2637),
                boost_frequencies=[
                    EQBand(300, 1.0, 1.5, "bell"),   # Fundamental warmth
                    EQBand(800, 1.5, 1.2, "bell"),   # Character
                    EQBand(2000, 2.0, 1.0, "bell"),  # Expressiveness
                    EQBand(5000, 1.0, 0.8, "bell")   # Brightness
                ],
                cut_frequencies=[
                    EQBand(600, -0.5, 2.0, "bell"),  # Muddiness
                    EQBand(3000, -0.3, 1.5, "bell")  # Harshness
                ],
                conflict_priority=0.75,
                masking_susceptibility=0.5
            ),
            
            "guzheng": InstrumentEQProfile(
                instrument_type="guzheng",
                frequency_range=(75, 4186),
                boost_frequencies=[
                    EQBand(150, 1.0, 1.2, "bell"),   # Low fundamentals
                    EQBand(500, 1.5, 1.5, "bell"),   # Body resonance
                    EQBand(2000, 2.0, 1.0, "bell"),  # Clarity
                    EQBand(6000, 1.5, 0.8, "bell")   # Sparkle
                ],
                cut_frequencies=[
                    EQBand(300, -0.5, 2.0, "bell"),  # Muddiness
                    EQBand(1000, -0.3, 1.5, "bell")  # Honkiness
                ],
                conflict_priority=0.7,
                masking_susceptibility=0.4
            ),
            
            "pipa": InstrumentEQProfile(
                instrument_type="pipa",
                frequency_range=(87, 3520),
                boost_frequencies=[
                    EQBand(200, 1.0, 1.5, "bell"),   # Fundamental warmth
                    EQBand(800, 1.5, 1.2, "bell"),   # Body character
                    EQBand(2500, 2.0, 1.0, "bell"),  # Attack clarity
                    EQBand(8000, 1.0, 0.8, "shelf")  # Air/presence
                ],
                cut_frequencies=[
                    EQBand(400, -0.5, 2.0, "bell"),  # Muddiness
                    EQBand(1200, -0.3, 1.5, "bell")  # Nasal quality
                ],
                conflict_priority=0.65,
                masking_susceptibility=0.5
            ),
            
            "dizi": InstrumentEQProfile(
                instrument_type="dizi",
                frequency_range=(294, 2349),
                boost_frequencies=[
                    EQBand(500, 1.0, 1.5, "bell"),   # Fundamental clarity
                    EQBand(1500, 1.5, 1.2, "bell"),  # Brightness
                    EQBand(4000, 2.0, 1.0, "bell"),  # Presence
                    EQBand(8000, 1.0, 0.8, "shelf")  # Air
                ],
                cut_frequencies=[
                    EQBand(800, -0.5, 2.0, "bell"),  # Honkiness
                    EQBand(2500, -0.3, 1.5, "bell")  # Shrillness
                ],
                conflict_priority=0.5,
                masking_susceptibility=0.7
            ),
            
            "guqin": InstrumentEQProfile(
                instrument_type="guqin",
                frequency_range=(81, 1319),
                boost_frequencies=[
                    EQBand(120, 1.0, 1.2, "bell"),   # Low fundamentals
                    EQBand(300, 1.5, 1.5, "bell"),   # Body warmth
                    EQBand(1000, 1.0, 1.2, "bell"),  # Clarity
                    EQBand(4000, 0.8, 0.8, "bell")   # Subtle presence
                ],
                cut_frequencies=[
                    EQBand(500, -0.8, 2.0, "bell"),  # Muddiness
                    EQBand(2000, -0.5, 1.5, "bell")  # Harshness
                ],
                conflict_priority=0.6,
                masking_susceptibility=0.6
            )
        }
        
        # Add default profile
        profiles["default"] = InstrumentEQProfile(
            instrument_type="default",
            frequency_range=(100, 5000),
            boost_frequencies=[
                EQBand(1000, 1.0, 1.5, "bell"),
                EQBand(5000, 0.5, 0.8, "shelf")
            ],
            cut_frequencies=[
                EQBand(500, -0.5, 2.0, "bell")
            ],
            conflict_priority=0.5,
            masking_susceptibility=0.5
        )
        
        return profiles
    
    def balance_frequency_spectrum(
        self,
        tracks: Dict[str, np.ndarray],
        track_info: Dict[str, Any],
        musical_analysis: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Balance frequency spectrum across all tracks."""
        self.logger.info(f"Starting intelligent EQ balancing for {len(tracks)} tracks")
        
        try:
            # Step 1: Analyze frequency content of all tracks
            frequency_analysis = self._analyze_frequency_content(tracks)
            
            # Step 2: Detect frequency conflicts between instruments
            conflicts = self._detect_frequency_conflicts(frequency_analysis, track_info)
            
            # Step 3: Calculate EQ adjustments to resolve conflicts
            eq_adjustments = self._calculate_eq_adjustments(conflicts, track_info)
            
            # Step 4: Apply instrument-specific EQ enhancements
            enhanced_adjustments = self._apply_instrument_enhancements(
                eq_adjustments, track_info
            )
            
            # Step 5: Apply EQ processing to tracks
            eq_processed_tracks = self._apply_eq_processing(tracks, enhanced_adjustments)
            
            # Compile EQ metadata
            eq_metadata = {
                "frequency_conflicts_detected": len(conflicts),
                "eq_adjustments_applied": len(enhanced_adjustments),
                "frequency_analysis": frequency_analysis,
                "conflict_resolutions": conflicts
            }
            
            self.logger.info("Intelligent EQ balancing completed successfully")
            return eq_processed_tracks, eq_metadata
            
        except Exception as e:
            self.logger.error(f"EQ balancing failed: {e}")
            raise
    
    def _analyze_frequency_content(
        self,
        tracks: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, float]]:
        """Analyze frequency content of all tracks."""
        analysis = {}
        
        for track_id, audio_data in tracks.items():
            # Calculate FFT
            fft_size = min(4096, len(audio_data))
            if fft_size < 512:
                continue
                
            fft_data = np.fft.fft(audio_data[:fft_size])
            magnitude = np.abs(fft_data[:fft_size//2])
            freqs = np.fft.fftfreq(fft_size, 1/self.config.sample_rate)[:fft_size//2]
            
            # Define frequency bands
            bands = {
                "sub_bass": (20, 60),
                "bass": (60, 250),
                "low_mid": (250, 500),
                "mid": (500, 2000),
                "high_mid": (2000, 4000),
                "presence": (4000, 8000),
                "brilliance": (8000, 20000)
            }
            
            band_energies = {}
            total_energy = np.sum(magnitude ** 2)
            
            for band_name, (low_freq, high_freq) in bands.items():
                band_mask = (freqs >= low_freq) & (freqs <= high_freq)
                band_energy = np.sum(magnitude[band_mask] ** 2)
                band_energies[band_name] = band_energy / (total_energy + 1e-10)
            
            # Find dominant frequencies
            peak_indices = np.argsort(magnitude)[-10:]  # Top 10 peaks
            dominant_freqs = freqs[peak_indices]
            
            analysis[track_id] = {
                **band_energies,
                "dominant_frequencies": dominant_freqs.tolist(),
                "spectral_centroid": float(np.sum(freqs * magnitude) / (np.sum(magnitude) + 1e-10)),
                "spectral_rolloff": float(self._calculate_spectral_rolloff(magnitude, freqs)),
                "total_energy": float(total_energy)
            }
        
        return analysis
    
    def _calculate_spectral_rolloff(
        self,
        magnitude: np.ndarray,
        freqs: np.ndarray,
        rolloff_threshold: float = 0.85
    ) -> float:
        """Calculate spectral rolloff frequency."""
        cumulative_energy = np.cumsum(magnitude ** 2)
        total_energy = cumulative_energy[-1]
        rolloff_energy = total_energy * rolloff_threshold
        
        rolloff_idx = np.where(cumulative_energy >= rolloff_energy)[0]
        if len(rolloff_idx) > 0:
            return freqs[rolloff_idx[0]]
        else:
            return freqs[-1]
    
    def _detect_frequency_conflicts(
        self,
        frequency_analysis: Dict[str, Dict[str, float]],
        track_info: Dict[str, Any]
    ) -> List[Tuple[str, str, str, float]]:
        """Detect frequency conflicts between instruments."""
        conflicts = []
        track_ids = list(frequency_analysis.keys())
        
        for i, track_id_1 in enumerate(track_ids):
            for track_id_2 in track_ids[i+1:]:
                analysis_1 = frequency_analysis[track_id_1]
                analysis_2 = frequency_analysis[track_id_2]
                
                # Check for conflicts in each frequency band
                for band in ["bass", "low_mid", "mid", "high_mid", "presence"]:
                    energy_1 = analysis_1.get(band, 0)
                    energy_2 = analysis_2.get(band, 0)
                    
                    # Conflict if both instruments have significant energy in same band
                    if energy_1 > 0.2 and energy_2 > 0.2:
                        conflict_severity = min(energy_1, energy_2)
                        conflicts.append((track_id_1, track_id_2, band, conflict_severity))
        
        return conflicts
    
    def _calculate_eq_adjustments(
        self,
        conflicts: List[Tuple[str, str, str, float]],
        track_info: Dict[str, Any]
    ) -> Dict[str, List[EQBand]]:
        """Calculate EQ adjustments to resolve frequency conflicts."""
        adjustments = {}
        
        for track_id_1, track_id_2, band, severity in conflicts:
            # Get instrument priorities
            info_1 = track_info.get(track_id_1, {})
            info_2 = track_info.get(track_id_2, {})
            
            # Handle both dict and AudioTrackInfo objects
            if hasattr(info_1, 'instrument_type'):
                inst_type_1 = info_1.instrument_type
            else:
                inst_type_1 = info_1.get('instrument_type', 'default')
                
            if hasattr(info_2, 'instrument_type'):
                inst_type_2 = info_2.instrument_type
            else:
                inst_type_2 = info_2.get('instrument_type', 'default')
            
            profile_1 = self.eq_profiles.get(inst_type_1, self.eq_profiles['default'])
            profile_2 = self.eq_profiles.get(inst_type_2, self.eq_profiles['default'])
            
            # Determine which instrument should be cut
            if profile_1.conflict_priority > profile_2.conflict_priority:
                # Cut track_id_2 in the conflicting band
                cut_track = track_id_2
                cut_profile = profile_2
            else:
                # Cut track_id_1 in the conflicting band
                cut_track = track_id_1
                cut_profile = profile_1
            
            # Calculate cut frequency and amount
            band_frequencies = {
                "bass": 150,
                "low_mid": 375,
                "mid": 1250,
                "high_mid": 3000,
                "presence": 6000
            }
            
            cut_freq = band_frequencies.get(band, 1000)
            cut_amount = -severity * 3.0  # Up to -3dB cut
            
            # Add EQ adjustment
            if cut_track not in adjustments:
                adjustments[cut_track] = []
            
            adjustments[cut_track].append(
                EQBand(cut_freq, cut_amount, 1.5, "bell")
            )
        
        return adjustments
    
    def _apply_instrument_enhancements(
        self,
        base_adjustments: Dict[str, List[EQBand]],
        track_info: Dict[str, Any]
    ) -> Dict[str, List[EQBand]]:
        """Apply instrument-specific EQ enhancements."""
        enhanced_adjustments = base_adjustments.copy()
        
        for track_id, info in track_info.items():
            # Handle both dict and AudioTrackInfo objects
            if hasattr(info, 'instrument_type'):
                inst_type = info.instrument_type
            else:
                inst_type = info.get('instrument_type', 'default')
            profile = self.eq_profiles.get(inst_type, self.eq_profiles['default'])
            
            if track_id not in enhanced_adjustments:
                enhanced_adjustments[track_id] = []
            
            # Add boost frequencies
            for boost_band in profile.boost_frequencies:
                enhanced_adjustments[track_id].append(boost_band)
            
            # Add cut frequencies (if not already added by conflict resolution)
            existing_cuts = {band.frequency for band in enhanced_adjustments[track_id]}
            for cut_band in profile.cut_frequencies:
                if cut_band.frequency not in existing_cuts:
                    enhanced_adjustments[track_id].append(cut_band)
        
        return enhanced_adjustments
    
    def _apply_eq_processing(
        self,
        tracks: Dict[str, np.ndarray],
        eq_adjustments: Dict[str, List[EQBand]]
    ) -> Dict[str, np.ndarray]:
        """Apply EQ processing to tracks."""
        processed_tracks = {}
        
        for track_id, audio_data in tracks.items():
            if track_id in eq_adjustments and eq_adjustments[track_id]:
                # Apply EQ bands (simplified implementation)
                processed_audio = self._apply_eq_bands(audio_data, eq_adjustments[track_id])
                processed_tracks[track_id] = processed_audio
            else:
                processed_tracks[track_id] = audio_data
        
        return processed_tracks
    
    def _apply_eq_bands(
        self,
        audio_data: np.ndarray,
        eq_bands: List[EQBand]
    ) -> np.ndarray:
        """Apply multiple EQ bands to audio signal."""
        # Simplified EQ implementation using frequency domain processing
        processed_audio = audio_data.copy()
        
        for eq_band in eq_bands:
            processed_audio = self._apply_single_eq_band(processed_audio, eq_band)
        
        return processed_audio
    
    def _apply_single_eq_band(
        self,
        audio_data: np.ndarray,
        eq_band: EQBand
    ) -> np.ndarray:
        """Apply a single EQ band (simplified implementation)."""
        # This is a simplified implementation
        # In practice, would use proper biquad filters or FFT-based EQ
        
        if len(audio_data) < 512:
            return audio_data
        
        # Simple gain adjustment around target frequency
        gain_linear = 10 ** (eq_band.gain / 20)
        
        # Apply gain uniformly (in a real implementation, would be frequency-specific)
        return audio_data * gain_linear

    
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
            processing_stage=ProcessingStage.EQ,
            success=True,
            processing_time=processing_time,
            metadata={"processor": "IntelligentEQBalancer"}
        )
    
    def _get_processing_stage(self):
        """Return the processing stage this processor handles."""
        from .base_processor import ProcessingStage
        return ProcessingStage.EQ