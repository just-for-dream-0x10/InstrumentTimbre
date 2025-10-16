"""
Chinese music theory utilities for instrument analysis.

Provides traditional Chinese music theory knowledge for
culturally-aware instrument recognition and analysis.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from enum import Enum

from ..core import InstrumentType


class ChineseMode(Enum):
    """Traditional Chinese musical modes (调式)."""
    GONG = "gong"      # 宫调 - Major-like mode
    SHANG = "shang"    # 商调 - Dorian-like mode  
    JUE = "jue"        # 角调 - Phrygian-like mode
    ZHI = "zhi"        # 徵调 - Mixolydian-like mode
    YU = "yu"          # 羽调 - Minor-like mode


class PlayingTechnique(Enum):
    """Traditional Chinese playing techniques."""
    HUA_YIN = "hua_yin"           # 滑音 - Sliding notes
    CHAN_YIN = "chan_yin"         # 颤音 - Vibrato
    DUN_YIN = "dun_yin"           # 顿音 - Staccato
    LIAN_YIN = "lian_yin"         # 连音 - Legato
    ZHUANG_YIN = "zhuang_yin"     # 装音 - Grace notes
    LUN_ZHI = "lun_zhi"           # 轮指 - Tremolo (pipa)
    GUA_ZOU = "gua_zou"           # 刮奏 - Glissando (guzheng)


class ChineseMusicTheory:
    """
    Chinese traditional music theory knowledge base.
    
    Provides cultural context and theoretical framework
    for analyzing Chinese traditional instruments.
    """
    
    def __init__(self):
        """Initialize Chinese music theory knowledge."""
        
        # Traditional five-note scale (五声音阶)
        self.wu_sheng_scale = {
            'gong': 0,    # 宫 - Tonic
            'shang': 2,   # 商 - Supertonic  
            'jue': 4,     # 角 - Mediant
            'zhi': 7,     # 徵 - Dominant
            'yu': 9       # 羽 - Submediant
        }
        
        # Seven-note scale with additional notes (七声音阶)
        self.qi_sheng_scale = {
            'gong': 0,     # 宫
            'shang': 2,    # 商
            'jue': 4,      # 角
            'bian_zhi': 6, # 变徵 - Augmented fourth
            'zhi': 7,      # 徵
            'yu': 9,       # 羽
            'bian_gong': 11 # 变宫 - Leading tone
        }
        
        # Traditional modes and their intervals
        self.traditional_modes = {
            ChineseMode.GONG: [0, 2, 4, 7, 9],      # Like major pentatonic
            ChineseMode.SHANG: [0, 2, 5, 7, 10],    # Dorian pentatonic
            ChineseMode.JUE: [0, 3, 5, 8, 10],      # Phrygian pentatonic
            ChineseMode.ZHI: [0, 2, 5, 7, 9],       # Mixolydian pentatonic
            ChineseMode.YU: [0, 3, 5, 8, 10]        # Minor pentatonic
        }
        
        # Instrument family characteristics
        self.instrument_families = {
            'silk_bamboo': {  # 丝竹乐器
                'instruments': [InstrumentType.ERHU, InstrumentType.DIZI, InstrumentType.XIAO],
                'characteristics': {
                    'delicate_expression': True,
                    'frequent_ornamentation': True,
                    'subtle_dynamics': True
                }
            },
            'plucked_strings': {  # 弹拨乐器
                'instruments': [InstrumentType.PIPA, InstrumentType.GUZHENG, InstrumentType.GUQIN],
                'characteristics': {
                    'attack_emphasis': True,
                    'resonance_decay': True,
                    'harmonic_richness': True
                }
            },
            'wind_instruments': {  # 管乐器
                'instruments': [InstrumentType.DIZI, InstrumentType.XIAO, InstrumentType.SUONA],
                'characteristics': {
                    'breath_expression': True,
                    'microtonal_flexibility': True,
                    'dynamic_range': True
                }
            }
        }
        
        # Traditional playing techniques by instrument
        self.instrument_techniques = {
            InstrumentType.ERHU: {
                PlayingTechnique.HUA_YIN: {
                    'frequency': 'very_common',
                    'typical_range_cents': (50, 200),
                    'detection_method': 'pitch_glide'
                },
                PlayingTechnique.CHAN_YIN: {
                    'frequency': 'common', 
                    'typical_rate_hz': (4, 8),
                    'detection_method': 'frequency_modulation'
                }
            },
            InstrumentType.PIPA: {
                PlayingTechnique.LUN_ZHI: {
                    'frequency': 'signature',
                    'typical_rate_hz': (15, 25),
                    'detection_method': 'rapid_attack_pattern'
                },
                PlayingTechnique.ZHUANG_YIN: {
                    'frequency': 'common',
                    'duration_ms': (10, 50),
                    'detection_method': 'grace_note_pattern'
                }
            },
            InstrumentType.GUZHENG: {
                PlayingTechnique.GUA_ZOU: {
                    'frequency': 'common',
                    'pitch_range_semitones': (5, 24),
                    'detection_method': 'continuous_glissando'
                },
                PlayingTechnique.CHAN_YIN: {
                    'frequency': 'common',
                    'typical_rate_hz': (3, 6),
                    'detection_method': 'string_vibrato'
                }
            },
            InstrumentType.DIZI: {
                PlayingTechnique.HUA_YIN: {
                    'frequency': 'very_common',
                    'breath_control': True,
                    'detection_method': 'breath_modulated_glide'
                },
                PlayingTechnique.CHAN_YIN: {
                    'frequency': 'signature',
                    'finger_hole_technique': True,
                    'detection_method': 'rapid_pitch_oscillation'
                }
            }
        }
    
    def get_expected_scale(self, instrument: InstrumentType) -> List[int]:
        """
        Get expected scale intervals for instrument.
        
        Args:
            instrument: Chinese instrument type
            
        Returns:
            List of expected scale intervals (semitones from root)
        """
        # Most Chinese instruments favor pentatonic scales
        if instrument in [InstrumentType.ERHU, InstrumentType.DIZI]:
            return self.traditional_modes[ChineseMode.GONG]  # Gong mode is most common
        elif instrument in [InstrumentType.PIPA, InstrumentType.GUZHENG]:
            return self.traditional_modes[ChineseMode.YU]    # Yu mode common for plucked
        else:
            return list(self.wu_sheng_scale.values())        # Default pentatonic
    
    def get_technique_expectations(self, instrument: InstrumentType) -> Dict:
        """
        Get expected playing techniques for instrument.
        
        Args:
            instrument: Chinese instrument type
            
        Returns:
            Dictionary of expected techniques and their characteristics
        """
        return self.instrument_techniques.get(instrument, {})
    
    def analyze_pentatonic_adherence(self, pitch_sequence: np.ndarray) -> float:
        """
        Analyze how well a pitch sequence adheres to pentatonic scales.
        
        Args:
            pitch_sequence: Sequence of pitches in MIDI note numbers
            
        Returns:
            Adherence score (0.0-1.0, higher = more pentatonic)
        """
        if len(pitch_sequence) == 0:
            return 0.0
        
        # Remove invalid pitches
        valid_pitches = pitch_sequence[~np.isnan(pitch_sequence)]
        if len(valid_pitches) == 0:
            return 0.0
        
        # Convert to semitones within octave
        semitones = valid_pitches % 12
        
        # Count how many notes fall on pentatonic scale degrees
        pentatonic_notes = set(self.wu_sheng_scale.values())
        adherence_count = sum(1 for note in semitones if note in pentatonic_notes)
        
        return adherence_count / len(valid_pitches)
    
    def detect_traditional_mode(self, pitch_histogram: np.ndarray) -> Tuple[ChineseMode, float]:
        """
        Detect which traditional Chinese mode best fits the pitch distribution.
        
        Args:
            pitch_histogram: Histogram of pitch classes (12 bins for chromatic scale)
            
        Returns:
            Tuple of (detected_mode, confidence_score)
        """
        best_mode = ChineseMode.GONG
        best_score = 0.0
        
        # Normalize histogram
        if np.sum(pitch_histogram) > 0:
            pitch_histogram = pitch_histogram / np.sum(pitch_histogram)
        
        # Test each traditional mode
        for mode, intervals in self.traditional_modes.items():
            # Create expected distribution for this mode
            expected = np.zeros(12)
            for interval in intervals:
                expected[interval % 12] = 1.0
            expected = expected / np.sum(expected)
            
            # Calculate correlation with actual distribution
            correlation = np.corrcoef(pitch_histogram, expected)[0, 1]
            if not np.isnan(correlation) and correlation > best_score:
                best_score = correlation
                best_mode = mode
        
        return best_mode, best_score
    
    def get_cultural_weight(self, instrument: InstrumentType) -> float:
        """
        Get cultural importance weight for Chinese traditional instruments.
        
        Args:
            instrument: Instrument type
            
        Returns:
            Weight factor (higher for more traditional instruments)
        """
        chinese_instruments = InstrumentType.get_chinese_instruments()
        
        if instrument not in chinese_instruments:
            return 0.1  # Low weight for non-Chinese instruments
        
        # Higher weights for most culturally significant instruments
        cultural_weights = {
            InstrumentType.GUQIN: 1.0,     # Highest cultural significance
            InstrumentType.ERHU: 0.9,      # Very important
            InstrumentType.PIPA: 0.9,      # Very important
            InstrumentType.GUZHENG: 0.8,   # Important
            InstrumentType.DIZI: 0.8,      # Important
            InstrumentType.XIAO: 0.7,      # Traditional
            InstrumentType.SUONA: 0.7,     # Traditional
            InstrumentType.SHENG: 0.6,     # Traditional
            InstrumentType.RUAN: 0.6,      # Traditional
            InstrumentType.LIUQIN: 0.5,    # Less common
            InstrumentType.GONG: 0.4,      # Percussion
            InstrumentType.DRUM: 0.3,      # Percussion
            InstrumentType.BELL: 0.3       # Percussion
        }
        
        return cultural_weights.get(instrument, 0.5)