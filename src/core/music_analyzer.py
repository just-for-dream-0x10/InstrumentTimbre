"""
 - System
Music Structure Analyzer - Core module for System

：、、
"""

import numpy as np
import librosa
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from enum import Enum
import scipy.signal as signal

class TrackRole(Enum):
    """Role"""
    MELODY = "melody"           # 
    HARMONY = "harmony"         # 
    BASS = "bass"              # 
    ACCOMPANIMENT = "accompaniment"  # 
    DECORATION = "decoration"   # 
    RHYTHM = "rhythm"          # 

class MusicSection(Enum):
    """"""
    INTRO = "intro"            # 
    VERSE = "verse"            # 
    CHORUS = "chorus"          # 
    BRIDGE = "bridge"          # 
    OUTRO = "outro"            # 
    INSTRUMENTAL = "instrumental"  # 

@dataclass
class StructureSegment:
    """"""
    section_type: MusicSection
    start_time: float
    end_time: float
    confidence: float
    characteristics: Dict[str, float]

@dataclass
class HarmonyAnalysis:
    """harmonyAnalysis results"""
    key_signature: str
    chord_progression: List[str]
    modulation_points: List[float]
    harmonic_rhythm: float
    consonance_score: float

@dataclass
class MusicStructureResult:
    """Music structureAnalysis results"""
    track_roles: Dict[int, TrackRole]
    structure_segments: List[StructureSegment]
    harmony_analysis: HarmonyAnalysis
    rhythm_pattern: Dict[str, float]
    overall_form: str

class MusicStructureAnalyzer:
    """
    
    
    ：
    1. 
    2.  (、、)
    3. 
    4. 
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.structure_model = self._load_structure_model(model_path)
        self.harmony_analyzer = HarmonyAnalyzer()
        self.rhythm_analyzer = RhythmAnalyzer()
        
    def _load_structure_model(self, model_path: Optional[str]) -> nn.Module:
        """load"""
        if model_path:
            model = torch.load(model_path, map_location=self.device)
        else:
            model = StructureClassifier()
        
        model.to(self.device)
        model.eval()
        return model
    
    def analyze(self, audio_data: np.ndarray, sr: int = 22050) -> MusicStructureResult:
        """
        
        
        Args:
            audio_data: 
            sr: 
            
        Returns:
            MusicStructureResult: 
        """
        # 1. Role
        track_roles = self._analyze_track_roles(audio_data, sr)
        
        # 2. 
        structure_segments = self._analyze_structure_segments(audio_data, sr)
        
        # 3. harmony
        harmony_analysis = self.harmony_analyzer.analyze(audio_data, sr)
        
        # 4. rhythm
        rhythm_pattern = self.rhythm_analyzer.analyze(audio_data, sr)
        
        # 5. Musical form
        overall_form = self._identify_overall_form(structure_segments)
        
        return MusicStructureResult(
            track_roles=track_roles,
            structure_segments=structure_segments,
            harmony_analysis=harmony_analysis,
            rhythm_pattern=rhythm_pattern,
            overall_form=overall_form
        )
    
    def _analyze_track_roles(self, audio_data: np.ndarray, sr: int) -> Dict[int, TrackRole]:
        """Role"""
        # 
        roles = {}
        
        # 1. bass
        bass_energy = self._calculate_bass_energy(audio_data, sr)
        if bass_energy > 0.3:
            roles[0] = TrackRole.BASS
        
        # 2. melody
        melody_confidence = self._detect_melody_line(audio_data, sr)
        if melody_confidence > 0.5:
            roles[1] = TrackRole.MELODY
        
        # 3. harmony
        harmony_presence = self._detect_harmony(audio_data, sr)
        if harmony_presence > 0.4:
            roles[2] = TrackRole.HARMONY
        
        # 4. rhythm
        rhythm_strength = self._detect_rhythm_elements(audio_data, sr)
        if rhythm_strength > 0.6:
            roles[3] = TrackRole.RHYTHM
        
        return roles
    
    def _analyze_structure_segments(self, audio_data: np.ndarray, sr: int) -> List[StructureSegment]:
        """"""
        segments = []
        
        # 1. 
        chroma = librosa.feature.chroma(y=audio_data, sr=sr)
        recurrence_matrix = librosa.segment.recurrence_matrix(chroma)
        
        # 2. 
        boundaries = self._detect_section_boundaries(audio_data, sr)
        
        # 3. 
        for i, boundary in enumerate(boundaries[:-1]):
            start_time = boundary
            end_time = boundaries[i + 1]
            
            # 
            segment_audio = audio_data[int(start_time * sr):int(end_time * sr)]
            section_type, confidence = self._classify_section(segment_audio, sr)
            
            # 
            characteristics = self._analyze_section_characteristics(segment_audio, sr)
            
            segment = StructureSegment(
                section_type=section_type,
                start_time=start_time,
                end_time=end_time,
                confidence=confidence,
                characteristics=characteristics
            )
            segments.append(segment)
        
        return segments
    
    def _detect_section_boundaries(self, audio_data: np.ndarray, sr: int) -> List[float]:
        """"""
        # 
        hop_length = 512
        
        # 1. 
        chroma = librosa.feature.chroma(y=audio_data, sr=sr, hop_length=hop_length)
        chroma_novelty = librosa.onset.onset_strength(S=chroma, sr=sr, hop_length=hop_length)
        
        # 2. MFCC
        mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, hop_length=hop_length)
        mfcc_novelty = librosa.onset.onset_strength(S=mfcc, sr=sr, hop_length=hop_length)
        
        # 3. 
        combined_novelty = chroma_novelty + mfcc_novelty
        
        # 4. 
        peaks = librosa.onset.onset_detect(
            onset_envelope=combined_novelty,
            sr=sr,
            hop_length=hop_length,
            units='time'
        )
        
        # 
        boundaries = [0.0] + list(peaks) + [len(audio_data) / sr]
        return sorted(set(boundaries))
    
    def _classify_section(self, segment_audio: np.ndarray, sr: int) -> Tuple[MusicSection, float]:
        """"""
        # 
        features = self._extract_section_features(segment_audio, sr)
        
        # 
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            predictions = self.structure_model(features_tensor)
            probabilities = torch.softmax(predictions, dim=-1)
        
        # 
        section_idx = torch.argmax(probabilities, dim=-1).item()
        confidence = float(torch.max(probabilities))
        
        sections = list(MusicSection)
        return sections[section_idx], confidence
    
    def _extract_section_features(self, audio_data: np.ndarray, sr: int) -> np.ndarray:
        """"""
        features = []
        
        # 1. 
        rms = librosa.feature.rms(y=audio_data)
        features.extend([np.mean(rms), np.std(rms)])
        
        # 2. 
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sr)
        features.extend([np.mean(spectral_centroids), np.std(spectral_centroids)])
        
        # 3. 
        chroma = librosa.feature.chroma(y=audio_data, sr=sr)
        features.extend(np.mean(chroma, axis=1))
        
        # 4. rhythm
        tempo, beats = librosa.beat.beat_track(y=audio_data, sr=sr)
        features.append(tempo)
        
        return np.array(features)
    
    def _analyze_section_characteristics(self, audio_data: np.ndarray, sr: int) -> Dict[str, float]:
        """"""
        characteristics = {}
        
        # dynamic_range
        rms = librosa.feature.rms(y=audio_data)
        characteristics['dynamic_range'] = float(np.max(rms) - np.min(rms))
        
        # timbre
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr)
        characteristics['brightness'] = float(np.mean(spectral_centroid) / sr * 2)
        
        # rhythm
        onset_strength = librosa.onset.onset_strength(y=audio_data, sr=sr)
        characteristics['rhythmic_intensity'] = float(np.mean(onset_strength))
        
        # harmony
        chroma = librosa.feature.chroma(y=audio_data, sr=sr)
        chroma_std = np.std(chroma, axis=1)
        characteristics['harmonic_complexity'] = float(np.mean(chroma_std))
        
        return characteristics
    
    def _calculate_bass_energy(self, audio_data: np.ndarray, sr: int) -> float:
        """bass"""
        # 
        nyquist = sr // 2
        low_cutoff = 200  # 200Hz
        b, a = signal.butter(4, low_cutoff / nyquist, btype='low')
        bass_signal = signal.filtfilt(b, a, audio_data)
        
        # 
        total_energy = np.sum(audio_data ** 2)
        bass_energy = np.sum(bass_signal ** 2)
        
        return bass_energy / (total_energy + 1e-8)
    
    def _detect_melody_line(self, audio_data: np.ndarray, sr: int) -> float:
        """melodyConfidence"""
        # melody
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio_data, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr
        )
        
        # melody
        melody_confidence = np.mean(voiced_probs[~np.isnan(voiced_probs)])
        return float(melody_confidence)
    
    def _detect_harmony(self, audio_data: np.ndarray, sr: int) -> float:
        """harmony"""
        # harmony
        chroma = librosa.feature.chroma(y=audio_data, sr=sr)
        
        # pitch
        active_pitches = np.sum(chroma > 0.1, axis=0)
        harmony_presence = np.mean(active_pitches > 2)  # 2
        
        return float(harmony_presence)
    
    def _detect_rhythm_elements(self, audio_data: np.ndarray, sr: int) -> float:
        """rhythm"""
        # 
        onset_strength = librosa.onset.onset_strength(y=audio_data, sr=sr)
        rhythm_strength = np.mean(onset_strength)
        
        return float(rhythm_strength)
    
    def _identify_overall_form(self, segments: List[StructureSegment]) -> str:
        """Musical form"""
        section_sequence = [seg.section_type.value for seg in segments]
        
        # Musical form
        if self._matches_pattern(section_sequence, ['intro', 'verse', 'chorus', 'verse', 'chorus']):
            return "Verse-Chorus Form"
        elif self._matches_pattern(section_sequence, ['intro', 'verse', 'bridge', 'verse']):
            return "AABA Form"
        elif len([s for s in section_sequence if s == 'verse']) >= 3:
            return "Verse Form"
        else:
            return "Free Form"
    
    def _matches_pattern(self, sequence: List[str], pattern: List[str]) -> bool:
        """"""
        if len(sequence) < len(pattern):
            return False
        
        for i in range(len(sequence) - len(pattern) + 1):
            if sequence[i:i+len(pattern)] == pattern:
                return True
        return False


class HarmonyAnalyzer:
    """harmony"""
    
    def analyze(self, audio_data: np.ndarray, sr: int) -> HarmonyAnalysis:
        """harmony"""
        # 1. Key signature
        key_signature = self._detect_key(audio_data, sr)
        
        # 2. chord
        chord_progression = self._detect_chord_progression(audio_data, sr)
        
        # 3. 
        modulation_points = self._detect_modulations(audio_data, sr)
        
        # 4. harmonyrhythm
        harmonic_rhythm = self._analyze_harmonic_rhythm(audio_data, sr)
        
        # 5. 
        consonance_score = self._calculate_consonance_score(audio_data, sr)
        
        return HarmonyAnalysis(
            key_signature=key_signature,
            chord_progression=chord_progression,
            modulation_points=modulation_points,
            harmonic_rhythm=harmonic_rhythm,
            consonance_score=consonance_score
        )
    
    def _detect_key(self, audio_data: np.ndarray, sr: int) -> str:
        """Key signature"""
        chroma = librosa.feature.chroma(y=audio_data, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        
        # Krumhansl-Schmuckler
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        best_correlation = -1
        best_key = 'C major'
        
        for i in range(12):
            # 
            major_corr = np.corrcoef(chroma_mean, np.roll(major_profile, i))[0, 1]
            if major_corr > best_correlation:
                best_correlation = major_corr
                best_key = f"{keys[i]} major"
            
            # 
            minor_corr = np.corrcoef(chroma_mean, np.roll(minor_profile, i))[0, 1]
            if minor_corr > best_correlation:
                best_correlation = minor_corr
                best_key = f"{keys[i]} minor"
        
        return best_key
    
    def _detect_chord_progression(self, audio_data: np.ndarray, sr: int) -> List[str]:
        """chord"""
        # chord
        hop_length = 2048
        chroma = librosa.feature.chroma(y=audio_data, sr=sr, hop_length=hop_length)
        
        # chord
        chord_templates = {
            'C': [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
            'Dm': [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
            'Em': [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
            'F': [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
            'G': [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            'Am': [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
        }
        
        progression = []
        for i in range(chroma.shape[1]):
            frame_chroma = chroma[:, i]
            best_chord = 'C'
            best_score = -1
            
            for chord, template in chord_templates.items():
                score = np.dot(frame_chroma, template)
                if score > best_score:
                    best_score = score
                    best_chord = chord
            
            progression.append(best_chord)
        
        # chord
        simplified_progression = []
        for chord in progression:
            if not simplified_progression or chord != simplified_progression[-1]:
                simplified_progression.append(chord)
        
        return simplified_progression
    
    def _detect_modulations(self, audio_data: np.ndarray, sr: int) -> List[float]:
        """"""
        # Key signature
        window_size = sr * 4  # 4
        hop_size = sr * 1     # 1
        
        modulation_points = []
        prev_key = None
        
        for start in range(0, len(audio_data) - window_size, hop_size):
            window_audio = audio_data[start:start + window_size]
            current_key = self._detect_key(window_audio, sr)
            
            if prev_key and current_key != prev_key:
                modulation_points.append(start / sr)
            
            prev_key = current_key
        
        return modulation_points
    
    def _analyze_harmonic_rhythm(self, audio_data: np.ndarray, sr: int) -> float:
        """harmonyrhythm"""
        chord_progression = self._detect_chord_progression(audio_data, sr)
        
        # chord
        total_time = len(audio_data) / sr
        chord_changes = len(chord_progression)
        
        harmonic_rhythm = chord_changes / total_time  # 
        return float(harmonic_rhythm)
    
    def _calculate_consonance_score(self, audio_data: np.ndarray, sr: int) -> float:
        """"""
        chroma = librosa.feature.chroma(y=audio_data, sr=sr)
        
        # ：pitch
        consonance_scores = []
        
        for i in range(chroma.shape[1]):
            frame_chroma = chroma[:, i]
            active_pitches = np.where(frame_chroma > 0.1)[0]
            
            if len(active_pitches) >= 2:
                # 
                intervals = []
                for j in range(len(active_pitches)):
                    for k in range(j + 1, len(active_pitches)):
                        interval = (active_pitches[k] - active_pitches[j]) % 12
                        intervals.append(interval)
                
                #  ()
                consonance_weights = {0: 1.0, 7: 0.9, 5: 0.8, 4: 0.7, 3: 0.6, 8: 0.6, 9: 0.5}
                frame_consonance = np.mean([consonance_weights.get(interval, 0.3) for interval in intervals])
                consonance_scores.append(frame_consonance)
        
        return float(np.mean(consonance_scores)) if consonance_scores else 0.5


class RhythmAnalyzer:
    """rhythm"""
    
    def analyze(self, audio_data: np.ndarray, sr: int) -> Dict[str, float]:
        """rhythm"""
        # 1. Tempo
        tempo, beats = librosa.beat.beat_track(y=audio_data, sr=sr)
        
        # 2. 
        time_signature = self._detect_time_signature(audio_data, sr, beats)
        
        # 3. rhythm
        rhythmic_complexity = self._calculate_rhythmic_complexity(audio_data, sr)
        
        # 4. 
        beat_consistency = self._analyze_beat_consistency(beats)
        
        return {
            'tempo': float(tempo),
            'time_signature': time_signature,
            'rhythmic_complexity': rhythmic_complexity,
            'beat_consistency': beat_consistency
        }
    
    def _detect_time_signature(self, audio_data: np.ndarray, sr: int, beats: np.ndarray) -> float:
        """"""
        if len(beats) < 8:
            return 4.0  # 4/4
        
        # 
        beat_intervals = np.diff(beats)
        median_interval = np.median(beat_intervals)
        
        # 
        onset_strength = librosa.onset.onset_strength(y=audio_data, sr=sr)
        onset_beats = librosa.frames_to_time(
            np.arange(len(onset_strength)), sr=sr, hop_length=512
        )
        
        # 
        if median_interval > 0.8:  # ，2/43/4
            return 3.0
        else:  # ，4/4
            return 4.0
    
    def _calculate_rhythmic_complexity(self, audio_data: np.ndarray, sr: int) -> float:
        """rhythm"""
        onset_strength = librosa.onset.onset_strength(y=audio_data, sr=sr)
        
        # 
        if np.mean(onset_strength) == 0:
            return 0.0
        
        complexity = np.std(onset_strength) / np.mean(onset_strength)
        return float(min(complexity, 1.0))  # [0,1]
    
    def _analyze_beat_consistency(self, beats: np.ndarray) -> float:
        """Tempo"""
        if len(beats) < 3:
            return 0.0
        
        beat_intervals = np.diff(beats)
        consistency = 1.0 - (np.std(beat_intervals) / (np.mean(beat_intervals) + 1e-8))
        return float(max(0.0, consistency))


class StructureClassifier(nn.Module):
    """"""
    
    def __init__(self, input_dim: int = 16, num_sections: int = 6):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(32, 16),
            nn.ReLU(),
            
            nn.Linear(16, num_sections)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


# 
if __name__ == "__main__":
    analyzer = MusicStructureAnalyzer()
    
    # 
    try:
        audio_data, sr = librosa.load("test_audio.wav", sr=22050)
        result = analyzer.analyze(audio_data, sr)
        
        print(":")
        print(f": {result.track_roles}")
        print(f": {result.overall_form}")
        print(f": {result.harmony_analysis.key_signature}")
        print(f": {result.rhythm_pattern}")
        
    except Exception as e:
        print(f": {e}")
        
        # generate
        test_audio = np.random.randn(22050 * 10)  # 10
        result = analyzer.analyze(test_audio, 22050)
        print(f" - : {result.overall_form}")