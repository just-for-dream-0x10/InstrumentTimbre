"""
Music Arranger - Practical Music Production Tool

This module implements a practical music arrangement system that:
1. Takes a simple melody and creates full multi-track arrangements
2. Adds appropriate instruments based on style
3. Applies style-specific dynamics and accents
4. Maintains melody recognizability while enhancing musical richness
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import soundfile as sf
from pathlib import Path

# Configure logging for the module
logger = logging.getLogger(__name__)


class ArrangementStyle(Enum):
    """Musical arrangement styles"""
    CHINESE_TRADITIONAL = "chinese_traditional"
    WESTERN_POP = "western_pop"
    JAZZ = "jazz"
    CLASSICAL = "classical"
    FOLK = "folk"


@dataclass
class TrackConfig:
    """Configuration for individual track in arrangement"""
    instrument: str
    role: str  # "melody", "harmony", "bass", "percussion", "fill"
    volume_level: float = 0.8
    pan_position: float = 0.0  # -1.0 (left) to 1.0 (right)
    reverb_amount: float = 0.2
    apply_dynamics: bool = True
    accent_pattern: Optional[List[float]] = None


@dataclass
class ArrangementResult:
    """Result of music arrangement process"""
    tracks: Dict[str, np.ndarray]  # track_name -> audio_data
    mixed_audio: np.ndarray
    arrangement_info: Dict[str, Any]
    sample_rate: int = 22050
    success: bool = True
    warnings: List[str] = None


class MusicArranger:
    """
    Practical music arranger for creating full arrangements from simple melodies
    
    This is a production-focused tool that generates actual audio tracks:
    - Melody track (enhanced original)
    - Harmony tracks (chords, counter-melodies)  
    - Bass line
    - Percussion/rhythm
    - Fill instruments
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize music arranger
        
        Args:
            config: Optional configuration for arrangement engine
        """
        self.config = config or {}
        self.sample_rate = self.config.get('sample_rate', 22050)
        
        # Load instrument sound banks and patterns
        self._initialize_instruments()
        self._initialize_style_patterns()
        
        logger.info("MusicArranger initialized for practical music production")
    
    def _initialize_instruments(self) -> None:
        """Initialize basic instrument synthesizers"""
        # Simple synthesized instruments for now
        # In production, this would load actual samples/soundfonts
        
        self.instruments = {
            # Traditional Chinese
            'erhu': self._create_string_synth(fundamental_ratio=1.0, harmonics=[1, 0.3, 0.1]),
            'pipa': self._create_plucked_synth(attack_time=0.01, decay_rate=3.0),
            'guzheng': self._create_plucked_synth(attack_time=0.005, decay_rate=2.0),
            'dizi': self._create_wind_synth(breath_noise=0.1),
            
            # Western instruments
            'piano': self._create_piano_synth(),
            'guitar': self._create_plucked_synth(attack_time=0.02, decay_rate=2.5),
            'violin': self._create_string_synth(fundamental_ratio=1.0, harmonics=[1, 0.4, 0.2, 0.1]),
            'cello': self._create_string_synth(fundamental_ratio=0.5, harmonics=[1, 0.3, 0.1]),
            'flute': self._create_wind_synth(breath_noise=0.05),
            
            # Rhythm section
            'bass': self._create_bass_synth(),
            'drums': self._create_drum_kit(),
        }
    
    def _initialize_style_patterns(self) -> None:
        """Initialize style-specific arrangement patterns"""
        
        self.style_patterns = {
            ArrangementStyle.CHINESE_TRADITIONAL: {
                'instruments': ['erhu', 'pipa', 'guzheng', 'dizi'],
                'rhythm_pattern': [1.0, 0.5, 0.7, 0.5],  # Strong-weak-medium-weak
                'harmony_style': 'pentatonic',
                'bass_style': 'minimal',
                'dynamics_curve': 'gradual_swell',
                'tempo_feel': 'relaxed'
            },
            
            ArrangementStyle.WESTERN_POP: {
                'instruments': ['piano', 'guitar', 'bass', 'drums'],
                'rhythm_pattern': [1.0, 0.3, 0.8, 0.3],  # Strong backbeat
                'harmony_style': 'triad_based',
                'bass_style': 'walking',
                'dynamics_curve': 'build_and_release',
                'tempo_feel': 'steady'
            },
            
            ArrangementStyle.CLASSICAL: {
                'instruments': ['violin', 'cello', 'piano'],
                'rhythm_pattern': [1.0, 0.6, 0.8, 0.6],
                'harmony_style': 'classical_harmony',
                'bass_style': 'alberti_bass',
                'dynamics_curve': 'classical_phrasing',
                'tempo_feel': 'expressive'
            }
        }
    
    def create_arrangement(self, 
                         melody_audio: np.ndarray,
                         style: ArrangementStyle,
                         target_length: Optional[float] = None) -> ArrangementResult:
        """
        Create full musical arrangement from input melody
        
        Args:
            melody_audio: Input melody audio
            style: Target arrangement style
            target_length: Optional target length in seconds
            
        Returns:
            ArrangementResult with all tracks and mixed audio
        """
        logger.info("Creating %s arrangement from melody", style.value)
        
        try:
            # Analyze input melody
            melody_analysis = self._analyze_melody(melody_audio)
            
            # Get style configuration
            style_config = self.style_patterns[style]
            
            # Create arrangement tracks
            tracks = {}
            
            # 1. Enhanced melody track
            tracks['melody'] = self._create_melody_track(
                melody_audio, melody_analysis, style_config
            )
            
            # 2. Harmony tracks
            harmony_tracks = self._create_harmony_tracks(
                melody_analysis, style_config
            )
            tracks.update(harmony_tracks)
            
            # 3. Bass line
            tracks['bass'] = self._create_bass_track(
                melody_analysis, style_config
            )
            
            # 4. Rhythm/percussion
            if 'drums' in style_config['instruments']:
                tracks['drums'] = self._create_rhythm_track(
                    melody_analysis, style_config
                )
            
            # 5. Apply dynamics and effects
            processed_tracks = self._apply_arrangement_dynamics(
                tracks, melody_analysis, style_config
            )
            
            # 6. Mix all tracks
            mixed_audio = self._mix_tracks(processed_tracks)
            
            # Create arrangement info
            arrangement_info = {
                'style': style.value,
                'track_count': len(processed_tracks),
                'total_length': len(mixed_audio) / self.sample_rate,
                'instruments_used': list(processed_tracks.keys()),
                'melody_analysis': melody_analysis
            }
            
            result = ArrangementResult(
                tracks=processed_tracks,
                mixed_audio=mixed_audio,
                arrangement_info=arrangement_info,
                sample_rate=self.sample_rate,
                success=True,
                warnings=[]
            )
            
            logger.info("Arrangement created successfully: %d tracks, %.1f seconds",
                       len(processed_tracks), len(mixed_audio) / self.sample_rate)
            
            return result
            
        except Exception as e:
            logger.error("Arrangement creation failed: %s", e)
            return ArrangementResult(
                tracks={},
                mixed_audio=np.array([]),
                arrangement_info={},
                success=False,
                warnings=[f"Arrangement failed: {str(e)}"]
            )
    
    def _analyze_melody(self, melody_audio: np.ndarray) -> Dict[str, Any]:
        """Analyze melody for arrangement purposes"""
        
        # Extract basic musical features
        analysis = {}
        
        # Tempo estimation (simple beat tracking)
        tempo = self._estimate_tempo(melody_audio)
        analysis['tempo'] = tempo
        
        # Key detection (simplified)
        key = self._estimate_key(melody_audio)
        analysis['key'] = key
        
        # Phrase detection
        phrases = self._detect_phrases(melody_audio)
        analysis['phrases'] = phrases
        
        # Energy analysis for dynamics
        energy_curve = self._analyze_energy(melody_audio)
        analysis['energy_curve'] = energy_curve
        
        # Pitch range
        pitch_range = self._analyze_pitch_range(melody_audio)
        analysis['pitch_range'] = pitch_range
        
        logger.info("Melody analysis: tempo=%.1f BPM, key=%s, %d phrases",
                   tempo, key, len(phrases))
        
        return analysis
    
    def _create_melody_track(self, melody_audio: np.ndarray, 
                           analysis: Dict[str, Any],
                           style_config: Dict[str, Any]) -> np.ndarray:
        """Create enhanced melody track with style-appropriate processing"""
        
        enhanced_melody = melody_audio.copy()
        
        # Apply style-specific melody enhancements
        if style_config.get('dynamics_curve') == 'gradual_swell':
            # Chinese traditional style - gentle dynamics
            swell_curve = self._create_swell_curve(len(enhanced_melody), intensity=0.3)
            enhanced_melody = enhanced_melody * swell_curve
            
        elif style_config.get('dynamics_curve') == 'build_and_release':
            # Pop style - build tension and release
            dynamics_curve = self._create_pop_dynamics(len(enhanced_melody))
            enhanced_melody = enhanced_melody * dynamics_curve
        
        # Add subtle vibrato for expressive styles
        if style_config.get('tempo_feel') == 'expressive':
            enhanced_melody = self._add_vibrato(enhanced_melody, rate=5.0, depth=0.05)
        
        # Normalize and return
        enhanced_melody = enhanced_melody * 0.8  # Leave headroom
        return enhanced_melody
    
    def _create_harmony_tracks(self, analysis: Dict[str, Any],
                             style_config: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Create harmony tracks based on melody analysis"""
        
        harmony_tracks = {}
        tempo = analysis['tempo']
        key = analysis['key']
        duration = len(analysis.get('energy_curve', [1000])) / self.sample_rate * 100  # Rough estimate
        
        # Generate chord progression based on key and style
        chord_progression = self._generate_chord_progression(key, style_config['harmony_style'])
        
        # Create harmony instruments
        harmony_instruments = [inst for inst in style_config['instruments'] 
                             if inst not in ['bass', 'drums']][1:]  # Skip melody instrument
        
        for i, instrument in enumerate(harmony_instruments[:2]):  # Limit to 2 harmony tracks
            if instrument in self.instruments:
                harmony_audio = self._synthesize_harmony_part(
                    chord_progression, tempo, duration, instrument, style_config
                )
                harmony_tracks[f'harmony_{instrument}'] = harmony_audio
        
        return harmony_tracks
    
    def _create_bass_track(self, analysis: Dict[str, Any],
                         style_config: Dict[str, Any]) -> np.ndarray:
        """Create bass line track"""
        
        tempo = analysis['tempo']
        key = analysis['key']
        duration = len(analysis.get('energy_curve', [1000])) / self.sample_rate * 100
        
        # Generate bass line based on chord progression and style
        if style_config['bass_style'] == 'walking':
            bass_audio = self._create_walking_bass(key, tempo, duration)
        elif style_config['bass_style'] == 'minimal':
            bass_audio = self._create_minimal_bass(key, tempo, duration)
        else:
            bass_audio = self._create_simple_bass(key, tempo, duration)
        
        return bass_audio
    
    def _create_rhythm_track(self, analysis: Dict[str, Any],
                           style_config: Dict[str, Any]) -> np.ndarray:
        """Create rhythm/percussion track"""
        
        tempo = analysis['tempo']
        duration = len(analysis.get('energy_curve', [1000])) / self.sample_rate * 100
        rhythm_pattern = style_config['rhythm_pattern']
        
        # Create basic drum pattern
        drums_audio = self._synthesize_drum_pattern(tempo, duration, rhythm_pattern)
        
        return drums_audio
    
    def _apply_arrangement_dynamics(self, tracks: Dict[str, np.ndarray],
                                  analysis: Dict[str, Any],
                                  style_config: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Apply dynamics, panning, and effects to all tracks"""
        
        processed_tracks = {}
        energy_curve = analysis.get('energy_curve', [])
        
        for track_name, audio in tracks.items():
            processed_audio = audio.copy()
            
            # Apply track-specific volume
            if track_name == 'melody':
                processed_audio *= 1.0  # Melody at full volume
            elif track_name.startswith('harmony'):
                processed_audio *= 0.6  # Harmony lower
            elif track_name == 'bass':
                processed_audio *= 0.7  # Bass prominent but not overpowering
            elif track_name == 'drums':
                processed_audio *= 0.5  # Drums as background
            
            # Apply energy-based dynamics if available
            if len(energy_curve) > 0:
                # Resample energy curve to match audio length
                resampled_energy = np.interp(
                    np.linspace(0, 1, len(processed_audio)),
                    np.linspace(0, 1, len(energy_curve)),
                    energy_curve
                )
                # Apply gentle energy-based modulation
                processed_audio *= (0.7 + 0.3 * resampled_energy)
            
            processed_tracks[track_name] = processed_audio
        
        return processed_tracks
    
    def _mix_tracks(self, tracks: Dict[str, np.ndarray]) -> np.ndarray:
        """Mix all tracks into final stereo audio"""
        
        if not tracks:
            return np.array([])
        
        # Find maximum length
        max_length = max(len(audio) for audio in tracks.values())
        
        # Create stereo mix
        mixed_audio = np.zeros(max_length)
        
        for track_name, audio in tracks.items():
            # Pad audio to max length if needed
            if len(audio) < max_length:
                padded_audio = np.pad(audio, (0, max_length - len(audio)))
            else:
                padded_audio = audio[:max_length]
            
            # Simple mixing (in production would include panning, EQ, etc.)
            mixed_audio += padded_audio
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(mixed_audio))
        if max_val > 0.95:
            mixed_audio = mixed_audio * (0.95 / max_val)
        
        return mixed_audio
    
    # Synthesis helper methods
    def _create_string_synth(self, fundamental_ratio: float = 1.0, 
                           harmonics: List[float] = [1.0, 0.3, 0.1]) -> Dict[str, Any]:
        """Create string instrument synthesizer config"""
        return {
            'type': 'string',
            'fundamental_ratio': fundamental_ratio,
            'harmonics': harmonics,
            'attack_time': 0.1,
            'decay_rate': 1.0,
            'sustain_level': 0.7
        }
    
    def _create_plucked_synth(self, attack_time: float = 0.01, 
                            decay_rate: float = 3.0) -> Dict[str, Any]:
        """Create plucked instrument synthesizer config"""
        return {
            'type': 'plucked',
            'attack_time': attack_time,
            'decay_rate': decay_rate,
            'harmonics': [1.0, 0.4, 0.2, 0.1]
        }
    
    def _create_wind_synth(self, breath_noise: float = 0.1) -> Dict[str, Any]:
        """Create wind instrument synthesizer config"""
        return {
            'type': 'wind',
            'breath_noise': breath_noise,
            'harmonics': [1.0, 0.2, 0.1],
            'vibrato_rate': 5.0,
            'vibrato_depth': 0.03
        }
    
    def _create_piano_synth(self) -> Dict[str, Any]:
        """Create piano synthesizer config"""
        return {
            'type': 'piano',
            'harmonics': [1.0, 0.5, 0.3, 0.2, 0.1],
            'attack_time': 0.01,
            'decay_rate': 2.0,
            'sustain_level': 0.3
        }
    
    def _create_bass_synth(self) -> Dict[str, Any]:
        """Create bass synthesizer config"""
        return {
            'type': 'bass',
            'harmonics': [1.0, 0.3, 0.1],
            'attack_time': 0.05,
            'decay_rate': 1.5,
            'sustain_level': 0.8
        }
    
    def _create_drum_kit(self) -> Dict[str, Any]:
        """Create drum kit synthesizer config"""
        return {
            'type': 'drums',
            'kick': {'freq': 60, 'decay': 0.5},
            'snare': {'freq': 200, 'noise': 0.7, 'decay': 0.2},
            'hihat': {'freq': 8000, 'noise': 0.9, 'decay': 0.1}
        }
    
    # Audio synthesis methods
    def _synthesize_note(self, frequency: float, duration: float, 
                        instrument_config: Dict[str, Any]) -> np.ndarray:
        """Synthesize a single note with given instrument"""
        
        sample_count = int(duration * self.sample_rate)
        t = np.linspace(0, duration, sample_count)
        
        # Generate basic waveform
        if instrument_config['type'] == 'string':
            audio = self._generate_string_sound(frequency, t, instrument_config)
        elif instrument_config['type'] == 'plucked':
            audio = self._generate_plucked_sound(frequency, t, instrument_config)
        elif instrument_config['type'] == 'wind':
            audio = self._generate_wind_sound(frequency, t, instrument_config)
        elif instrument_config['type'] == 'piano':
            audio = self._generate_piano_sound(frequency, t, instrument_config)
        elif instrument_config['type'] == 'bass':
            audio = self._generate_bass_sound(frequency, t, instrument_config)
        else:
            # Default sine wave
            audio = np.sin(2 * np.pi * frequency * t)
        
        return audio
    
    def _generate_string_sound(self, freq: float, t: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """Generate string instrument sound"""
        harmonics = config.get('harmonics', [1.0, 0.3, 0.1])
        audio = np.zeros_like(t)
        
        for i, amplitude in enumerate(harmonics):
            audio += amplitude * np.sin(2 * np.pi * freq * (i + 1) * t)
        
        # Apply envelope
        attack_time = config.get('attack_time', 0.1)
        decay_rate = config.get('decay_rate', 1.0)
        
        envelope = self._create_adsr_envelope(len(t), attack_time, decay_rate, 0.7, 0.3)
        return audio * envelope
    
    def _generate_plucked_sound(self, freq: float, t: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """Generate plucked instrument sound"""
        harmonics = config.get('harmonics', [1.0, 0.4, 0.2])
        audio = np.zeros_like(t)
        
        for i, amplitude in enumerate(harmonics):
            audio += amplitude * np.sin(2 * np.pi * freq * (i + 1) * t)
        
        # Sharp attack, exponential decay
        decay_rate = config.get('decay_rate', 3.0)
        envelope = np.exp(-decay_rate * t)
        
        return audio * envelope
    
    def _generate_wind_sound(self, freq: float, t: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """Generate wind instrument sound"""
        harmonics = config.get('harmonics', [1.0, 0.2, 0.1])
        audio = np.zeros_like(t)
        
        for i, amplitude in enumerate(harmonics):
            audio += amplitude * np.sin(2 * np.pi * freq * (i + 1) * t)
        
        # Add breath noise
        breath_noise = config.get('breath_noise', 0.1)
        noise = np.random.normal(0, breath_noise, len(t))
        audio += noise
        
        # Add vibrato
        vibrato_rate = config.get('vibrato_rate', 5.0)
        vibrato_depth = config.get('vibrato_depth', 0.03)
        vibrato = vibrato_depth * np.sin(2 * np.pi * vibrato_rate * t)
        audio *= (1 + vibrato)
        
        return audio
    
    def _generate_piano_sound(self, freq: float, t: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """Generate piano sound"""
        harmonics = config.get('harmonics', [1.0, 0.5, 0.3, 0.2])
        audio = np.zeros_like(t)
        
        for i, amplitude in enumerate(harmonics):
            audio += amplitude * np.sin(2 * np.pi * freq * (i + 1) * t)
        
        # Piano-like envelope
        attack_time = config.get('attack_time', 0.01)
        decay_rate = config.get('decay_rate', 2.0)
        sustain_level = config.get('sustain_level', 0.3)
        
        envelope = self._create_adsr_envelope(len(t), attack_time, decay_rate, sustain_level, 0.5)
        return audio * envelope
    
    def _generate_bass_sound(self, freq: float, t: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """Generate bass sound"""
        harmonics = config.get('harmonics', [1.0, 0.3, 0.1])
        audio = np.zeros_like(t)
        
        for i, amplitude in enumerate(harmonics):
            audio += amplitude * np.sin(2 * np.pi * freq * (i + 1) * t)
        
        # Bass envelope - strong attack, sustained
        envelope = self._create_adsr_envelope(len(t), 0.05, 1.5, 0.8, 0.3)
        return audio * envelope