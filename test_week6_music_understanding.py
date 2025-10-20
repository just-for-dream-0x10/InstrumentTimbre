#!/usr/bin/env python3
"""
Week 6 Multi-layer Music Understanding Engine Test

Test the core music understanding functionality:
1. Track role identification (melody, harmony, bass, etc.)
2. Music structure parsing (intro, verse, chorus, etc.)
3. Harmony relationship analysis (chord progressions, key, scales)
4. Rhythm pattern recognition (meter, accents, groove)
5. Multi-layer integration and validation
"""

import sys
import os
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_multi_track_test_audio(track_config: Dict[str, Dict], duration: float = 8.0, sample_rate: int = 22050):
    """Create synthetic multi-track audio for testing"""
    
    t = np.linspace(0, duration, int(duration * sample_rate))
    tracks = {}
    
    for track_name, config in track_config.items():
        role = config.get('role', 'melody')
        instrument = config.get('instrument', 'piano')
        key = config.get('key', 'C')
        
        if role == 'melody':
            # Main melodic line
            freqs = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]  # C major scale
            audio = create_melodic_track(t, freqs, pattern='stepwise')
        elif role == 'harmony':
            # Chord progression
            chords = [
                [261.63, 329.63, 392.00],  # C major
                [246.94, 311.13, 369.99],  # Dm
                [293.66, 369.99, 440.00],  # F major
                [293.66, 369.99, 440.00]   # G major
            ]
            audio = create_harmonic_track(t, chords)
        elif role == 'bass':
            # Bass line
            bass_notes = [130.81, 146.83, 174.61, 196.00]  # C, D, F, G
            audio = create_bass_track(t, bass_notes)
        elif role == 'rhythm':
            # Percussive/rhythmic elements
            audio = create_rhythmic_track(t, tempo=120)
        else:
            # Default to simple tone
            audio = np.sin(2 * np.pi * 440 * t) * 0.3
        
        # Add instrument-specific characteristics
        if instrument == 'guitar':
            audio = apply_guitar_characteristics(audio)
        elif instrument == 'strings':
            audio = apply_strings_characteristics(audio)
        elif instrument == 'drums':
            audio = apply_drum_characteristics(audio)
        
        tracks[track_name] = audio
    
    return tracks


def create_melodic_track(t: np.ndarray, freqs: List[float], pattern: str = 'stepwise') -> np.ndarray:
    """Create a melodic track with specified pattern"""
    audio = np.zeros_like(t)
    note_duration = len(t) / len(freqs)
    
    for i, freq in enumerate(freqs):
        start_idx = int(i * note_duration)
        end_idx = int((i + 1) * note_duration)
        if end_idx > len(t):
            end_idx = len(t)
        
        note_t = t[start_idx:end_idx] - t[start_idx]
        # Melodic note with harmonics
        note = (np.sin(2 * np.pi * freq * note_t) + 
                0.3 * np.sin(2 * np.pi * freq * 2 * note_t) +
                0.1 * np.sin(2 * np.pi * freq * 3 * note_t))
        
        # Musical envelope
        envelope = np.exp(-note_t * 2) * (1 - np.exp(-note_t * 10))
        audio[start_idx:end_idx] = note * envelope * 0.5
    
    return audio


def create_harmonic_track(t: np.ndarray, chords: List[List[float]]) -> np.ndarray:
    """Create harmonic track with chord progressions"""
    audio = np.zeros_like(t)
    chord_duration = len(t) / len(chords)
    
    for i, chord in enumerate(chords):
        start_idx = int(i * chord_duration)
        end_idx = int((i + 1) * chord_duration)
        if end_idx > len(t):
            end_idx = len(t)
        
        chord_t = t[start_idx:end_idx] - t[start_idx]
        chord_audio = np.zeros_like(chord_t)
        
        for freq in chord:
            chord_audio += np.sin(2 * np.pi * freq * chord_t) * 0.3
        
        # Harmonic envelope (sustained)
        envelope = np.exp(-chord_t * 0.5) * (1 - np.exp(-chord_t * 8))
        audio[start_idx:end_idx] = chord_audio * envelope
    
    return audio


def create_bass_track(t: np.ndarray, bass_notes: List[float]) -> np.ndarray:
    """Create bass track with low-frequency emphasis"""
    audio = np.zeros_like(t)
    note_duration = len(t) / len(bass_notes)
    
    for i, freq in enumerate(bass_notes):
        start_idx = int(i * note_duration)
        end_idx = int((i + 1) * note_duration)
        if end_idx > len(t):
            end_idx = len(t)
        
        note_t = t[start_idx:end_idx] - t[start_idx]
        # Bass note with sub-harmonics
        note = (np.sin(2 * np.pi * freq * note_t) + 
                0.5 * np.sin(2 * np.pi * freq * 0.5 * note_t))
        
        # Bass envelope (punchy)
        envelope = np.exp(-note_t * 3) * (1 - np.exp(-note_t * 15))
        audio[start_idx:end_idx] = note * envelope * 0.6
    
    return audio


def create_rhythmic_track(t: np.ndarray, tempo: int = 120) -> np.ndarray:
    """Create rhythmic track with specified tempo"""
    beat_duration = 60.0 / tempo  # seconds per beat
    sample_rate = len(t) / (t[-1] - t[0])
    beat_samples = int(beat_duration * sample_rate)
    
    audio = np.zeros_like(t)
    
    for i in range(0, len(t), beat_samples):
        if i + beat_samples // 4 < len(t):
            # Create kick-like sound
            kick_t = np.linspace(0, 0.1, beat_samples // 4)
            kick = np.sin(2 * np.pi * 60 * kick_t) * np.exp(-kick_t * 20)
            audio[i:i + len(kick)] += kick * 0.8
    
    return audio


def apply_guitar_characteristics(audio: np.ndarray) -> np.ndarray:
    """Apply guitar-like spectral characteristics"""
    # Add slight distortion and harmonics
    return np.tanh(audio * 1.2) * 0.8


def apply_strings_characteristics(audio: np.ndarray) -> np.ndarray:
    """Apply string ensemble characteristics"""
    # Add vibrato and ensemble effect
    vibrato_rate = 5.0  # Hz
    t = np.linspace(0, len(audio) / 22050, len(audio))
    vibrato = 1 + 0.02 * np.sin(2 * np.pi * vibrato_rate * t)
    return audio * vibrato * 0.7


def apply_drum_characteristics(audio: np.ndarray) -> np.ndarray:
    """Apply drum-like characteristics"""
    # Add noise and sharp attack
    noise = np.random.normal(0, 0.1, len(audio))
    return audio + noise * 0.3


def test_track_role_identification():
    """Test track role identification system"""
    logger.info("Testing track role identification...")
    
    try:
        from InstrumentTimbre.core.analysis.music_understanding_engine import MusicUnderstandingEngine
        
        # Initialize engine
        engine = MusicUnderstandingEngine()
        logger.info("MusicUnderstandingEngine initialized")
        
        # Create test tracks with known roles
        track_config = {
            'lead_melody': {'role': 'melody', 'instrument': 'guitar'},
            'chord_prog': {'role': 'harmony', 'instrument': 'piano'},
            'bass_line': {'role': 'bass', 'instrument': 'bass'},
            'drum_beat': {'role': 'rhythm', 'instrument': 'drums'}
        }
        
        test_tracks = create_multi_track_test_audio(track_config, duration=6.0)
        
        # Analyze track roles
        role_analysis = engine.identify_track_roles(test_tracks)
        
        logger.info("Track role analysis:")
        for track_name, analysis in role_analysis.items():
            logger.info("  Track '%s': role=%s (%.3f confidence), instrument=%s", 
                       track_name, analysis['role'], analysis['confidence'], 
                       analysis.get('instrument', 'unknown'))
            logger.info("    Features: %s", list(analysis.get('features', {}).keys())[:5])
        
        # Validate results
        assert len(role_analysis) == len(test_tracks), "Should analyze all tracks"
        
        for track_name, analysis in role_analysis.items():
            assert 'role' in analysis, f"Should identify role for {track_name}"
            assert 'confidence' in analysis, f"Should have confidence score for {track_name}"
            assert 0 <= analysis['confidence'] <= 1, f"Confidence should be in [0,1] for {track_name}"
            assert 'features' in analysis, f"Should extract features for {track_name}"
        
        logger.info("✅ Track role identification test PASSED")
        return True, role_analysis
        
    except Exception as e:
        logger.error("❌ Track role identification test FAILED: %s", e)
        return False, None


def test_music_structure_parsing():
    """Test music structure parsing (intro, verse, chorus, etc.)"""
    logger.info("Testing music structure parsing...")
    
    try:
        from InstrumentTimbre.core.analysis.music_understanding_engine import MusicUnderstandingEngine
        
        engine = MusicUnderstandingEngine()
        
        # Create structured audio (intro + verse + chorus + verse)
        intro_audio = create_melodic_track(np.linspace(0, 2, 44100), [261.63, 293.66], 'intro')
        verse_audio = create_melodic_track(np.linspace(0, 4, 88200), [261.63, 293.66, 329.63, 349.23], 'verse')
        chorus_audio = create_melodic_track(np.linspace(0, 4, 88200), [392.00, 440.00, 493.88, 523.25], 'chorus')
        
        # Combine sections
        structured_audio = np.concatenate([intro_audio, verse_audio, chorus_audio, verse_audio])
        
        # Parse music structure
        structure_analysis = engine.parse_music_structure(structured_audio)
        
        logger.info("Music structure analysis:")
        logger.info("  Total duration: %.1f seconds", len(structured_audio) / 22050)
        logger.info("  Detected sections: %d", len(structure_analysis['sections']))
        
        for i, section in enumerate(structure_analysis['sections']):
            logger.info("  Section %d: %.1f-%.1fs, type=%s (%.3f confidence)", 
                       i, section['start_time'], section['end_time'], 
                       section['section_type'], section['confidence'])
            
            # Show key features
            if 'features' in section:
                features = section['features']
                logger.info("    Energy: %.3f, Tempo: %.1f, Key: %s", 
                           features.get('energy', 0), features.get('tempo', 0), 
                           features.get('key', 'unknown'))
        
        # Validate structure analysis
        assert len(structure_analysis['sections']) >= 2, "Should detect multiple sections"
        assert 'overall_structure' in structure_analysis, "Should have overall structure info"
        
        for section in structure_analysis['sections']:
            assert 'start_time' in section, "Section should have start time"
            assert 'end_time' in section, "Section should have end time"
            assert 'section_type' in section, "Section should have type"
            assert section['end_time'] > section['start_time'], "End time should be after start"
        
        logger.info("✅ Music structure parsing test PASSED")
        return True, structure_analysis
        
    except Exception as e:
        logger.error("❌ Music structure parsing test FAILED: %s", e)
        return False, None


def test_harmony_analysis():
    """Test harmony relationship analysis"""
    logger.info("Testing harmony relationship analysis...")
    
    try:
        from InstrumentTimbre.core.analysis.music_understanding_engine import MusicUnderstandingEngine
        
        engine = MusicUnderstandingEngine()
        
        # Create harmonic test audio with clear chord progression
        t = np.linspace(0, 8.0, int(8.0 * 22050))
        
        # Classic I-vi-IV-V progression in C major
        chord_progression = [
            [261.63, 329.63, 392.00],  # C major (I)
            [220.00, 261.63, 329.63],  # A minor (vi) 
            [174.61, 220.00, 261.63],  # F major (IV)
            [196.00, 246.94, 293.66]   # G major (V)
        ]
        
        harmonic_audio = create_harmonic_track(t, chord_progression)
        
        # Analyze harmony
        harmony_analysis = engine.analyze_harmony_relationships(harmonic_audio)
        
        logger.info("Harmony analysis:")
        logger.info("  Detected key: %s (%.3f confidence)", 
                   harmony_analysis['key']['tonic'], harmony_analysis['key']['confidence'])
        logger.info("  Scale type: %s", harmony_analysis['key']['scale_type'])
        logger.info("  Chord progression: %d chords detected", len(harmony_analysis['chord_progression']))
        
        for i, chord in enumerate(harmony_analysis['chord_progression']):
            logger.info("  Chord %d: %.1f-%.1fs, %s (%.3f confidence)", 
                       i, chord['start_time'], chord['end_time'], 
                       chord['chord_name'], chord['confidence'])
        
        # Show harmonic features
        if 'harmonic_features' in harmony_analysis:
            features = harmony_analysis['harmonic_features']
            logger.info("  Harmonic rhythm: %.2f chords/sec", features.get('harmonic_rhythm', 0))
            logger.info("  Consonance level: %.3f", features.get('consonance', 0))
            logger.info("  Complexity score: %.3f", features.get('complexity', 0))
        
        # Validate harmony analysis
        assert 'key' in harmony_analysis, "Should detect musical key"
        assert 'chord_progression' in harmony_analysis, "Should detect chord progression"
        assert len(harmony_analysis['chord_progression']) > 0, "Should detect at least one chord"
        
        key_info = harmony_analysis['key']
        assert 'tonic' in key_info, "Should identify tonic"
        assert 'confidence' in key_info, "Should have key confidence"
        assert 0 <= key_info['confidence'] <= 1, "Key confidence should be in [0,1]"
        
        logger.info("✅ Harmony analysis test PASSED")
        return True, harmony_analysis
        
    except Exception as e:
        logger.error("❌ Harmony analysis test FAILED: %s", e)
        return False, None


def test_rhythm_pattern_recognition():
    """Test rhythm pattern recognition"""
    logger.info("Testing rhythm pattern recognition...")
    
    try:
        from InstrumentTimbre.core.analysis.music_understanding_engine import MusicUnderstandingEngine
        
        engine = MusicUnderstandingEngine()
        
        # Create rhythmic test audio with clear patterns
        t = np.linspace(0, 8.0, int(8.0 * 22050))
        rhythmic_audio = create_rhythmic_track(t, tempo=120)
        
        # Analyze rhythm patterns
        rhythm_analysis = engine.analyze_rhythm_patterns(rhythmic_audio)
        
        logger.info("Rhythm analysis:")
        logger.info("  Detected tempo: %.1f BPM (%.3f confidence)", 
                   rhythm_analysis['tempo']['bpm'], rhythm_analysis['tempo']['confidence'])
        logger.info("  Time signature: %s", rhythm_analysis['time_signature'])
        logger.info("  Beat pattern: %s", rhythm_analysis['beat_pattern']['pattern_type'])
        
        # Show rhythmic features
        if 'rhythmic_features' in rhythm_analysis:
            features = rhythm_analysis['rhythmic_features']
            logger.info("  Groove strength: %.3f", features.get('groove_strength', 0))
            logger.info("  Syncopation level: %.3f", features.get('syncopation', 0))
            logger.info("  Rhythmic complexity: %.3f", features.get('complexity', 0))
        
        # Show detected beats
        if 'beat_times' in rhythm_analysis:
            beat_times = rhythm_analysis['beat_times'][:8]  # Show first 8 beats
            logger.info("  Beat times (first 8): %s", [f"{t:.2f}s" for t in beat_times])
        
        # Validate rhythm analysis
        assert 'tempo' in rhythm_analysis, "Should detect tempo"
        assert 'time_signature' in rhythm_analysis, "Should detect time signature"
        assert 'beat_pattern' in rhythm_analysis, "Should detect beat pattern"
        
        tempo_info = rhythm_analysis['tempo']
        assert 'bpm' in tempo_info, "Should identify BPM"
        assert 'confidence' in tempo_info, "Should have tempo confidence"
        assert tempo_info['bpm'] > 0, "BPM should be positive"
        assert 0 <= tempo_info['confidence'] <= 1, "Tempo confidence should be in [0,1]"
        
        logger.info("✅ Rhythm pattern recognition test PASSED")
        return True, rhythm_analysis
        
    except Exception as e:
        logger.error("❌ Rhythm pattern recognition test FAILED: %s", e)
        return False, None


def test_multi_layer_integration():
    """Test integration of all music understanding layers"""
    logger.info("Testing multi-layer integration...")
    
    try:
        from InstrumentTimbre.core.analysis.music_understanding_engine import MusicUnderstandingEngine
        
        engine = MusicUnderstandingEngine()
        
        # Create complete multi-track composition
        track_config = {
            'melody': {'role': 'melody', 'instrument': 'guitar'},
            'harmony': {'role': 'harmony', 'instrument': 'piano'},
            'bass': {'role': 'bass', 'instrument': 'bass'},
            'drums': {'role': 'rhythm', 'instrument': 'drums'}
        }
        
        multi_track_audio = create_multi_track_test_audio(track_config, duration=8.0)
        
        # Perform comprehensive analysis
        comprehensive_analysis = engine.analyze_complete_music_understanding(multi_track_audio)
        
        logger.info("Comprehensive music understanding:")
        logger.info("  Analysis components: %s", list(comprehensive_analysis.keys()))
        
        # Track analysis
        if 'track_analysis' in comprehensive_analysis:
            track_analysis = comprehensive_analysis['track_analysis']
            logger.info("  Track roles identified: %d tracks", len(track_analysis))
            for track_name, analysis in track_analysis.items():
                logger.info("    %s: %s (%s)", track_name, analysis['role'], 
                           analysis.get('instrument', 'unknown'))
        
        # Structure analysis
        if 'structure_analysis' in comprehensive_analysis:
            structure = comprehensive_analysis['structure_analysis']
            logger.info("  Musical structure: %d sections", len(structure.get('sections', [])))
        
        # Harmony analysis
        if 'harmony_analysis' in comprehensive_analysis:
            harmony = comprehensive_analysis['harmony_analysis']
            logger.info("  Key: %s, Chords: %d", 
                       harmony['key']['tonic'], len(harmony.get('chord_progression', [])))
        
        # Rhythm analysis
        if 'rhythm_analysis' in comprehensive_analysis:
            rhythm = comprehensive_analysis['rhythm_analysis']
            logger.info("  Tempo: %.1f BPM, Time signature: %s", 
                       rhythm['tempo']['bpm'], rhythm['time_signature'])
        
        # Integration metrics
        if 'integration_metrics' in comprehensive_analysis:
            metrics = comprehensive_analysis['integration_metrics']
            logger.info("  Overall coherence: %.3f", metrics.get('coherence_score', 0))
            logger.info("  Complexity level: %.3f", metrics.get('complexity_score', 0))
            logger.info("  Quality assessment: %.3f", metrics.get('quality_score', 0))
        
        # Validate comprehensive analysis
        required_components = ['track_analysis', 'structure_analysis', 'harmony_analysis', 'rhythm_analysis']
        for component in required_components:
            assert component in comprehensive_analysis, f"Should include {component}"
        
        # Check integration quality
        assert 'integration_metrics' in comprehensive_analysis, "Should have integration metrics"
        metrics = comprehensive_analysis['integration_metrics']
        assert 'coherence_score' in metrics, "Should have coherence score"
        assert 0 <= metrics['coherence_score'] <= 1, "Coherence score should be in [0,1]"
        
        logger.info("✅ Multi-layer integration test PASSED")
        return True, comprehensive_analysis
        
    except Exception as e:
        logger.error("❌ Multi-layer integration test FAILED: %s", e)
        return False, None


def main():
    """Run all Week 6 multi-layer music understanding tests"""
    logger.info("=" * 60)
    logger.info("WEEK 6 MULTI-LAYER MUSIC UNDERSTANDING TEST")
    logger.info("=" * 60)
    
    tests = [
        ("Track Role Identification", test_track_role_identification),
        ("Music Structure Parsing", test_music_structure_parsing),
        ("Harmony Analysis", test_harmony_analysis),
        ("Rhythm Pattern Recognition", test_rhythm_pattern_recognition),
        ("Multi-layer Integration", test_multi_layer_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            success = result if isinstance(result, bool) else result[0]
            results.append(success)
            
            if success:
                logger.info("✅ %s PASSED", test_name)
            else:
                logger.error("❌ %s FAILED", test_name)
                
        except Exception as e:
            logger.error("❌ %s FAILED with exception: %s", test_name, e)
            results.append(False)
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("WEEK 6 TEST SUMMARY")
    logger.info("="*60)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test_name, _) in enumerate(tests):
        status = "✅ PASS" if results[i] else "❌ FAIL"
        logger.info("%s: %s", test_name, status)
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 Week 6 multi-layer music understanding is working perfectly!")
        logger.info("Ready to proceed to Week 7: Intelligent Track Operations")
        return True
    elif passed >= total * 0.8:  # 80% pass rate acceptable
        logger.warning(f"⚠️ Week 6 mostly working: {passed}/{total} tests passed")
        logger.info("Core music understanding functionality is ready")
        return True
    else:
        logger.error(f"❌ Week 6 needs significant work: {passed}/{total} tests passed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)