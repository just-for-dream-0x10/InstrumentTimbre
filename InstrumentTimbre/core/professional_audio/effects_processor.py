"""
Effects Processor - Style-specific reverb, delay, and spatial effects.

This module provides intelligent effects processing with style-appropriate settings
for reverb, delay, modulation, and other spatial effects based on musical context.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
from enum import Enum
import logging

from .base_processor import BaseAudioProcessor


class EffectType(Enum):
    """Types of audio effects."""
    REVERB = "reverb"
    DELAY = "delay"
    CHORUS = "chorus"
    FLANGER = "flanger"
    PHASER = "phaser"
    DISTORTION = "distortion"
    COMPRESSION = "compression"
    MODULATION = "modulation"


class ReverbType(Enum):
    """Types of reverb algorithms."""
    HALL = "hall"
    ROOM = "room"
    CHAMBER = "chamber"
    PLATE = "plate"
    SPRING = "spring"
    CONVOLUTION = "convolution"
    ALGORITHMIC = "algorithmic"


@dataclass
class ReverbSettings:
    """Settings for reverb processing."""
    reverb_type: ReverbType
    room_size: float  # 0-1
    decay_time: float  # seconds
    damping: float  # 0-1
    wet_level: float  # 0-1
    dry_level: float  # 0-1
    pre_delay: float  # ms
    early_reflections: float  # 0-1
    modulation_rate: float  # Hz
    modulation_depth: float  # 0-1


@dataclass
class DelaySettings:
    """Settings for delay processing."""
    delay_time: float  # ms
    feedback: float  # 0-1
    wet_level: float  # 0-1
    high_cut: float  # Hz
    low_cut: float  # Hz
    sync_to_tempo: bool
    stereo_spread: float  # 0-1


@dataclass
class InstrumentEffectProfile:
    """Effect profile for specific instrument types."""
    instrument_type: str
    reverb_settings: Optional[ReverbSettings]
    delay_settings: Optional[DelaySettings]
    modulation_amount: float  # 0-1
    spatial_enhancement: float  # 0-1
    effect_priority: float  # Priority in effect conflicts


class EffectsProcessor(BaseAudioProcessor):
    """
    Intelligent effects processor that applies style-appropriate effects
    based on instrument type and musical context.
    """
    
    def __init__(self, config):
        """Initialize the effects processor."""
        super().__init__("effectsprocessor", 22050)
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize effect profiles
        self.effect_profiles = self._initialize_effect_profiles()
        
        # Effect processing state
        self.active_effects: Dict[str, List[EffectType]] = {}
        
    def _initialize_effect_profiles(self) -> Dict[str, InstrumentEffectProfile]:
        """Initialize effect profiles for different instrument types."""
        profiles = {
            # Drums - minimal reverb, focus on punch
            "drums": InstrumentEffectProfile(
                instrument_type="drums",
                reverb_settings=ReverbSettings(
                    reverb_type=ReverbType.ROOM,
                    room_size=0.3,
                    decay_time=0.8,
                    damping=0.7,
                    wet_level=0.15,
                    dry_level=0.85,
                    pre_delay=5.0,
                    early_reflections=0.6,
                    modulation_rate=0.5,
                    modulation_depth=0.1
                ),
                delay_settings=None,
                modulation_amount=0.0,
                spatial_enhancement=0.3,
                effect_priority=0.8
            ),
            
            # Bass - tight, controlled effects
            "bass": InstrumentEffectProfile(
                instrument_type="bass",
                reverb_settings=ReverbSettings(
                    reverb_type=ReverbType.ROOM,
                    room_size=0.2,
                    decay_time=0.6,
                    damping=0.8,
                    wet_level=0.08,
                    dry_level=0.92,
                    pre_delay=3.0,
                    early_reflections=0.4,
                    modulation_rate=0.3,
                    modulation_depth=0.05
                ),
                delay_settings=None,
                modulation_amount=0.1,
                spatial_enhancement=0.2,
                effect_priority=0.7
            ),
            
            # Guitar - versatile effects
            "guitar": InstrumentEffectProfile(
                instrument_type="guitar",
                reverb_settings=ReverbSettings(
                    reverb_type=ReverbType.HALL,
                    room_size=0.5,
                    decay_time=1.2,
                    damping=0.5,
                    wet_level=0.25,
                    dry_level=0.75,
                    pre_delay=15.0,
                    early_reflections=0.7,
                    modulation_rate=0.8,
                    modulation_depth=0.15
                ),
                delay_settings=DelaySettings(
                    delay_time=250.0,
                    feedback=0.3,
                    wet_level=0.15,
                    high_cut=8000,
                    low_cut=200,
                    sync_to_tempo=True,
                    stereo_spread=0.6
                ),
                modulation_amount=0.3,
                spatial_enhancement=0.5,
                effect_priority=0.6
            ),
            
            # Piano - natural hall reverb
            "piano": InstrumentEffectProfile(
                instrument_type="piano",
                reverb_settings=ReverbSettings(
                    reverb_type=ReverbType.HALL,
                    room_size=0.7,
                    decay_time=1.8,
                    damping=0.3,
                    wet_level=0.3,
                    dry_level=0.7,
                    pre_delay=20.0,
                    early_reflections=0.8,
                    modulation_rate=0.4,
                    modulation_depth=0.1
                ),
                delay_settings=None,
                modulation_amount=0.1,
                spatial_enhancement=0.4,
                effect_priority=0.7
            ),
            
            # Violin - lush, expressive effects
            "violin": InstrumentEffectProfile(
                instrument_type="violin",
                reverb_settings=ReverbSettings(
                    reverb_type=ReverbType.HALL,
                    room_size=0.6,
                    decay_time=1.5,
                    damping=0.4,
                    wet_level=0.35,
                    dry_level=0.65,
                    pre_delay=25.0,
                    early_reflections=0.8,
                    modulation_rate=0.6,
                    modulation_depth=0.12
                ),
                delay_settings=DelaySettings(
                    delay_time=180.0,
                    feedback=0.2,
                    wet_level=0.1,
                    high_cut=6000,
                    low_cut=300,
                    sync_to_tempo=False,
                    stereo_spread=0.4
                ),
                modulation_amount=0.2,
                spatial_enhancement=0.6,
                effect_priority=0.5
            ),
            
            # Chinese traditional instruments
            "erhu": InstrumentEffectProfile(
                instrument_type="erhu",
                reverb_settings=ReverbSettings(
                    reverb_type=ReverbType.CHAMBER,
                    room_size=0.4,
                    decay_time=1.0,
                    damping=0.5,
                    wet_level=0.25,
                    dry_level=0.75,
                    pre_delay=12.0,
                    early_reflections=0.7,
                    modulation_rate=0.3,
                    modulation_depth=0.08
                ),
                delay_settings=None,
                modulation_amount=0.15,
                spatial_enhancement=0.4,
                effect_priority=0.6
            ),
            
            "guzheng": InstrumentEffectProfile(
                instrument_type="guzheng",
                reverb_settings=ReverbSettings(
                    reverb_type=ReverbType.HALL,
                    room_size=0.8,
                    decay_time=2.2,
                    damping=0.2,
                    wet_level=0.4,
                    dry_level=0.6,
                    pre_delay=30.0,
                    early_reflections=0.9,
                    modulation_rate=0.5,
                    modulation_depth=0.1
                ),
                delay_settings=DelaySettings(
                    delay_time=300.0,
                    feedback=0.25,
                    wet_level=0.12,
                    high_cut=8000,
                    low_cut=150,
                    sync_to_tempo=False,
                    stereo_spread=0.8
                ),
                modulation_amount=0.2,
                spatial_enhancement=0.7,
                effect_priority=0.5
            ),
            
            "pipa": InstrumentEffectProfile(
                instrument_type="pipa",
                reverb_settings=ReverbSettings(
                    reverb_type=ReverbType.CHAMBER,
                    room_size=0.5,
                    decay_time=1.3,
                    damping=0.4,
                    wet_level=0.2,
                    dry_level=0.8,
                    pre_delay=18.0,
                    early_reflections=0.75,
                    modulation_rate=0.4,
                    modulation_depth=0.1
                ),
                delay_settings=DelaySettings(
                    delay_time=120.0,
                    feedback=0.15,
                    wet_level=0.08,
                    high_cut=7000,
                    low_cut=200,
                    sync_to_tempo=True,
                    stereo_spread=0.5
                ),
                modulation_amount=0.25,
                spatial_enhancement=0.5,
                effect_priority=0.4
            ),
            
            "dizi": InstrumentEffectProfile(
                instrument_type="dizi",
                reverb_settings=ReverbSettings(
                    reverb_type=ReverbType.HALL,
                    room_size=0.6,
                    decay_time=1.6,
                    damping=0.3,
                    wet_level=0.3,
                    dry_level=0.7,
                    pre_delay=22.0,
                    early_reflections=0.8,
                    modulation_rate=0.7,
                    modulation_depth=0.15
                ),
                delay_settings=DelaySettings(
                    delay_time=200.0,
                    feedback=0.2,
                    wet_level=0.1,
                    high_cut=6000,
                    low_cut=400,
                    sync_to_tempo=False,
                    stereo_spread=0.3
                ),
                modulation_amount=0.3,
                spatial_enhancement=0.5,
                effect_priority=0.3
            ),
            
            "guqin": InstrumentEffectProfile(
                instrument_type="guqin",
                reverb_settings=ReverbSettings(
                    reverb_type=ReverbType.CHAMBER,
                    room_size=0.7,
                    decay_time=2.0,
                    damping=0.2,
                    wet_level=0.35,
                    dry_level=0.65,
                    pre_delay=35.0,
                    early_reflections=0.85,
                    modulation_rate=0.3,
                    modulation_depth=0.05
                ),
                delay_settings=None,
                modulation_amount=0.1,
                spatial_enhancement=0.6,
                effect_priority=0.6
            )
        }
        
        # Add default profile
        profiles["default"] = InstrumentEffectProfile(
            instrument_type="default",
            reverb_settings=ReverbSettings(
                reverb_type=ReverbType.ROOM,
                room_size=0.5,
                decay_time=1.0,
                damping=0.5,
                wet_level=0.2,
                dry_level=0.8,
                pre_delay=15.0,
                early_reflections=0.7,
                modulation_rate=0.5,
                modulation_depth=0.1
            ),
            delay_settings=None,
            modulation_amount=0.2,
            spatial_enhancement=0.4,
            effect_priority=0.5
        )
        
        return profiles
    
    def apply_musical_effects(
        self,
        tracks: Dict[str, np.ndarray],
        track_info: Dict[str, Any],
        musical_analysis: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Apply musical effects to all tracks."""
        self.logger.info(f"Starting effects processing for {len(tracks)} tracks")
        
        try:
            # Step 1: Analyze musical context for effect decisions
            effect_context = self._analyze_effect_context(musical_analysis, track_info)
            
            # Step 2: Apply reverb processing
            reverb_processed = self._apply_reverb_processing(tracks, track_info, effect_context)
            
            # Step 3: Apply delay processing
            delay_processed = self._apply_delay_processing(reverb_processed, track_info, effect_context)
            
            # Step 4: Apply modulation effects
            modulation_processed = self._apply_modulation_effects(delay_processed, track_info)
            
            # Step 5: Apply spatial enhancement
            spatial_enhanced = self._apply_spatial_enhancement(modulation_processed, track_info)
            
            # Compile effects metadata
            effects_metadata = {
                "reverb_applied": len([t for t in tracks.keys() if self._has_reverb(t, track_info)]),
                "delay_applied": len([t for t in tracks.keys() if self._has_delay(t, track_info)]),
                "modulation_applied": len([t for t in tracks.keys() if self._has_modulation(t, track_info)]),
                "spatial_enhancement_applied": True,
                "effect_context": effect_context
            }
            
            self.logger.info("Effects processing completed successfully")
            return spatial_enhanced, effects_metadata
            
        except Exception as e:
            self.logger.error(f"Effects processing failed: {e}")
            raise
    
    def _analyze_effect_context(
        self,
        musical_analysis: Optional[Dict[str, Any]],
        track_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze musical context to inform effect decisions."""
        context = {
            "musical_style": "adaptive",
            "tempo": 120.0,
            "energy_level": 0.5,
            "emotional_content": {"energy": 0.5, "valence": 0.5},
            "reverb_amount_modifier": 1.0,
            "delay_amount_modifier": 1.0
        }
        
        if musical_analysis:
            # Extract tempo if available
            if "tempo" in musical_analysis:
                context["tempo"] = musical_analysis["tempo"]
            
            # Extract emotional content if available
            if "emotional_analysis" in musical_analysis:
                context["emotional_content"] = musical_analysis["emotional_analysis"]
                
                # Adjust effect amounts based on emotion
                energy = context["emotional_content"].get("energy", 0.5)
                valence = context["emotional_content"].get("valence", 0.5)
                
                # High energy = less reverb, more tight effects
                context["reverb_amount_modifier"] = 1.0 - energy * 0.3
                
                # Low valence (sad) = more reverb
                context["reverb_amount_modifier"] *= (1.0 + (1.0 - valence) * 0.2)
        
        # Analyze instrument composition
        instrument_counts = {}
        for info in track_info.values():
            # Handle both dict and AudioTrackInfo objects
            if hasattr(info, 'instrument_type'):
                inst_type = info.instrument_type
            else:
                inst_type = info.get('instrument_type', 'unknown')
            instrument_counts[inst_type] = instrument_counts.get(inst_type, 0) + 1
        
        # Determine musical style based on instruments
        chinese_instruments = {'erhu', 'guzheng', 'pipa', 'dizi', 'guqin'}
        if any(inst in instrument_counts for inst in chinese_instruments):
            context["musical_style"] = "chinese_traditional"
            context["reverb_amount_modifier"] *= 1.2  # More reverb for traditional
        
        return context
    
    def _apply_reverb_processing(
        self,
        tracks: Dict[str, np.ndarray],
        track_info: Dict[str, Any],
        effect_context: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        """Apply reverb processing to tracks."""
        reverb_processed = {}
        
        for track_id, audio_data in tracks.items():
            info = track_info.get(track_id, {})
            inst_type = info.get('instrument_type', 'default')
            profile = self.effect_profiles.get(inst_type, self.effect_profiles['default'])
            
            if profile.reverb_settings:
                # Adjust reverb settings based on context
                adjusted_settings = self._adjust_reverb_settings(
                    profile.reverb_settings, effect_context
                )
                
                # Apply reverb
                reverb_audio = self._apply_reverb(audio_data, adjusted_settings)
                reverb_processed[track_id] = reverb_audio
            else:
                reverb_processed[track_id] = audio_data
        
        return reverb_processed
    
    def _adjust_reverb_settings(
        self,
        base_settings: ReverbSettings,
        context: Dict[str, Any]
    ) -> ReverbSettings:
        """Adjust reverb settings based on musical context."""
        adjusted = ReverbSettings(
            reverb_type=base_settings.reverb_type,
            room_size=base_settings.room_size,
            decay_time=base_settings.decay_time,
            damping=base_settings.damping,
            wet_level=base_settings.wet_level * context["reverb_amount_modifier"],
            dry_level=base_settings.dry_level,
            pre_delay=base_settings.pre_delay,
            early_reflections=base_settings.early_reflections,
            modulation_rate=base_settings.modulation_rate,
            modulation_depth=base_settings.modulation_depth
        )
        
        # Adjust wet/dry balance
        total_level = adjusted.wet_level + adjusted.dry_level
        if total_level > 1.0:
            adjusted.dry_level = 1.0 - adjusted.wet_level
        
        return adjusted
    
    def _apply_reverb(
        self,
        audio_data: np.ndarray,
        settings: ReverbSettings
    ) -> np.ndarray:
        """Apply reverb to audio signal (simplified implementation)."""
        # Simplified reverb implementation
        # In practice, would use proper reverb algorithms
        
        if len(audio_data) < 1024:
            return audio_data
        
        # Simple delay-based reverb approximation
        delay_samples = int(settings.pre_delay * self.config.sample_rate / 1000)
        decay_samples = int(settings.decay_time * self.config.sample_rate)
        
        # Create impulse response (very simplified)
        impulse_length = min(decay_samples, self.config.sample_rate * 2)  # Max 2 seconds
        impulse = np.exp(-np.arange(impulse_length) / (decay_samples * 0.3))
        impulse *= settings.early_reflections
        
        # Apply convolution (simplified)
        if len(audio_data) + len(impulse) < 100000:  # Avoid excessive computation
            reverb_signal = np.convolve(audio_data, impulse, mode='same')
        else:
            reverb_signal = audio_data * 0.1  # Fallback
        
        # Apply damping (simplified high-frequency rolloff)
        if settings.damping > 0:
            # Simple low-pass filtering effect
            reverb_signal = reverb_signal * (1.0 - settings.damping * 0.5)
        
        # Mix wet and dry signals
        dry_signal = audio_data * settings.dry_level
        wet_signal = reverb_signal * settings.wet_level
        
        # Ensure no clipping
        mixed_signal = dry_signal + wet_signal
        max_amplitude = np.max(np.abs(mixed_signal))
        if max_amplitude > 0.98:
            mixed_signal = mixed_signal * (0.98 / max_amplitude)
        
        return mixed_signal
    
    def _apply_delay_processing(
        self,
        tracks: Dict[str, np.ndarray],
        track_info: Dict[str, Any],
        effect_context: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        """Apply delay processing to tracks."""
        delay_processed = {}
        
        for track_id, audio_data in tracks.items():
            info = track_info.get(track_id, {})
            inst_type = info.get('instrument_type', 'default')
            profile = self.effect_profiles.get(inst_type, self.effect_profiles['default'])
            
            if profile.delay_settings:
                # Adjust delay settings based on context
                adjusted_settings = self._adjust_delay_settings(
                    profile.delay_settings, effect_context
                )
                
                # Apply delay
                delay_audio = self._apply_delay(audio_data, adjusted_settings)
                delay_processed[track_id] = delay_audio
            else:
                delay_processed[track_id] = audio_data
        
        return delay_processed
    
    def _adjust_delay_settings(
        self,
        base_settings: DelaySettings,
        context: Dict[str, Any]
    ) -> DelaySettings:
        """Adjust delay settings based on musical context."""
        adjusted = DelaySettings(
            delay_time=base_settings.delay_time,
            feedback=base_settings.feedback,
            wet_level=base_settings.wet_level * context["delay_amount_modifier"],
            high_cut=base_settings.high_cut,
            low_cut=base_settings.low_cut,
            sync_to_tempo=base_settings.sync_to_tempo,
            stereo_spread=base_settings.stereo_spread
        )
        
        # Sync to tempo if requested
        if adjusted.sync_to_tempo:
            tempo = context["tempo"]
            # Calculate delay time based on tempo (quarter note = 60000/BPM ms)
            quarter_note_time = 60000 / tempo
            # Use musical subdivisions
            if adjusted.delay_time > quarter_note_time:
                adjusted.delay_time = quarter_note_time
            else:
                adjusted.delay_time = quarter_note_time / 2  # Eighth note
        
        return adjusted
    
    def _apply_delay(
        self,
        audio_data: np.ndarray,
        settings: DelaySettings
    ) -> np.ndarray:
        """Apply delay to audio signal."""
        # Simple delay implementation
        delay_samples = int(settings.delay_time * self.config.sample_rate / 1000)
        
        if delay_samples >= len(audio_data) or delay_samples <= 0:
            return audio_data
        
        # Create delayed signal
        delayed_signal = np.zeros_like(audio_data)
        delayed_signal[delay_samples:] = audio_data[:-delay_samples] * settings.feedback
        
        # Mix with original
        mixed_signal = audio_data + delayed_signal * settings.wet_level
        
        # Prevent clipping
        max_amplitude = np.max(np.abs(mixed_signal))
        if max_amplitude > 0.98:
            mixed_signal = mixed_signal * (0.98 / max_amplitude)
        
        return mixed_signal
    
    def _apply_modulation_effects(
        self,
        tracks: Dict[str, np.ndarray],
        track_info: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        """Apply modulation effects (chorus, flanger, etc.)."""
        # Simplified implementation - return tracks as-is
        # In practice, would implement chorus, flanger, phaser effects
        return tracks
    
    def _apply_spatial_enhancement(
        self,
        tracks: Dict[str, np.ndarray],
        track_info: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        """Apply spatial enhancement effects."""
        # Simplified implementation - return tracks as-is
        # In practice, would implement stereo widening, binaural processing
        return tracks
    
    def _has_reverb(self, track_id: str, track_info: Dict[str, Any]) -> bool:
        """Check if track should have reverb applied."""
        info = track_info.get(track_id, {})
        inst_type = info.get('instrument_type', 'default')
        profile = self.effect_profiles.get(inst_type, self.effect_profiles['default'])
        return profile.reverb_settings is not None
    
    def _has_delay(self, track_id: str, track_info: Dict[str, Any]) -> bool:
        """Check if track should have delay applied."""
        info = track_info.get(track_id, {})
        inst_type = info.get('instrument_type', 'default')
        profile = self.effect_profiles.get(inst_type, self.effect_profiles['default'])
        return profile.delay_settings is not None
    
    def _has_modulation(self, track_id: str, track_info: Dict[str, Any]) -> bool:
        """Check if track should have modulation applied."""
        info = track_info.get(track_id, {})
        inst_type = info.get('instrument_type', 'default')
        profile = self.effect_profiles.get(inst_type, self.effect_profiles['default'])
        return profile.modulation_amount > 0

    
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
            processing_stage=ProcessingStage.EFFECTS,
            success=True,
            processing_time=processing_time,
            metadata={"processor": "EffectsProcessor"}
        )
    
    def _get_processing_stage(self):
        """Return the processing stage this processor handles."""
        from .base_processor import ProcessingStage
        return ProcessingStage.EFFECTS