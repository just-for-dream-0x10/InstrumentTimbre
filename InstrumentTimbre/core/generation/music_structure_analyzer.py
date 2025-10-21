"""
Music Structure Analyzer - System Development Task

This module implements advanced music structure analysis to support
melody preservation by understanding the hierarchical structure of music.
"""

import numpy as np
import librosa
from typing import Dict, List, Tuple, Optional, Any
import logging
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d


class MusicStructureAnalyzer:
    """
    Advanced music structure analyzer for melody preservation
    
    Analyzes musical structure at multiple levels:
    1. Note level - individual pitch events
    2. Motif level - short melodic patterns (2-4 notes)  
    3. Phrase level - musical sentences (4-8 measures)
    4. Section level - larger structural units (verse, chorus)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Analysis parameters
        self.sample_rate = self.config.get('sample_rate', 22050)
        self.hop_length = self.config.get('hop_length', 512)
        self.frame_length = self.config.get('frame_length', 2048)
        
        # Structure detection parameters
        self.motif_min_length = self.config.get('motif_min_length', 3)
        self.motif_max_length = self.config.get('motif_max_length', 8)
        self.phrase_min_duration = self.config.get('phrase_min_duration', 2.0)
        self.phrase_max_duration = self.config.get('phrase_max_duration', 8.0)
        
        self.logger.info("MusicStructureAnalyzer initialized")
    
    def analyze_structure(self, audio_data: np.ndarray, melody_dna: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive music structure analysis
        
        Args:
            audio_data: Input audio signal
            melody_dna: Melody DNA from MelodyPreservationEngine
            
        Returns:
            Detailed structure analysis
        """
        structure = {}
        
        # Extract basic elements from melody DNA
        f0_track = melody_dna['f0_track']
        time_track = melody_dna['time_track']
        
        # 1. Note-level analysis
        structure['notes'] = self._analyze_notes(f0_track, time_track)
        
        # 2. Motif detection and analysis
        structure['motifs'] = self._detect_motifs(f0_track, time_track)
        
        # 3. Enhanced phrase analysis
        structure['phrases'] = self._analyze_phrases(
            f0_track, time_track, melody_dna['phrase_boundaries']
        )
        
        # 4. Section-level analysis
        structure['sections'] = self._detect_sections(audio_data, structure['phrases'])
        
        # 5. Hierarchical relationships
        structure['hierarchy'] = self._build_hierarchy(structure)
        
        # 6. Structural importance weights
        structure['importance_weights'] = self._compute_importance_weights(structure)
        
        self.logger.info(f"Structure analysis complete: {len(structure['notes'])} notes, "
                        f"{len(structure['motifs'])} motifs, {len(structure['phrases'])} phrases, "
                        f"{len(structure['sections'])} sections")
        
        return structure
    
    def _analyze_notes(self, f0_track: np.ndarray, time_track: np.ndarray) -> List[Dict[str, Any]]:
        """Analyze individual notes with detailed characteristics"""
        notes = []
        
        if len(f0_track) == 0:
            return notes
        
        # Convert to semitones for analysis
        semitones = 12 * np.log2(f0_track)
        
        # Detect note boundaries using pitch stability
        note_boundaries = self._detect_note_boundaries(semitones, time_track)
        
        for i in range(len(note_boundaries) - 1):
            start_idx = note_boundaries[i]
            end_idx = note_boundaries[i + 1]
            
            if end_idx <= start_idx:
                continue
                
            # Extract note characteristics
            note_f0 = f0_track[start_idx:end_idx]
            note_time = time_track[start_idx:end_idx]
            note_semitones = semitones[start_idx:end_idx]
            
            # Compute note properties
            duration = note_time[-1] - note_time[0]
            avg_pitch = np.mean(note_f0)
            pitch_stability = 1.0 - (np.std(note_semitones) / 12.0)  # Normalize by octave
            
            # Detect vibrato and ornaments
            vibrato_info = self._detect_vibrato(note_f0, note_time)
            
            note = {
                'start_time': note_time[0],
                'end_time': note_time[-1],
                'duration': duration,
                'pitch_hz': avg_pitch,
                'pitch_semitone': np.mean(note_semitones),
                'pitch_stability': max(0, min(1, pitch_stability)),
                'vibrato': vibrato_info,
                'importance': self._compute_note_importance(
                    duration, pitch_stability, vibrato_info
                )
            }
            
            notes.append(note)
        
        return notes
    
    def _detect_note_boundaries(self, semitones: np.ndarray, time_track: np.ndarray) -> List[int]:
        """Detect note boundaries based on pitch stability"""
        boundaries = [0]  # Start boundary
        
        if len(semitones) < 5:
            boundaries.append(len(semitones) - 1)
            return boundaries
        
        # Smooth the pitch track
        smoothed = gaussian_filter1d(semitones, sigma=2.0)
        
        # Detect large pitch changes
        pitch_diff = np.abs(np.diff(smoothed))
        threshold = np.percentile(pitch_diff, 75)  # Top 25% of changes
        
        change_points = np.where(pitch_diff > threshold)[0] + 1
        
        # Filter change points that are too close together
        min_note_duration = 0.1  # 100ms minimum note duration
        min_frames = int(min_note_duration * len(time_track) / (time_track[-1] - time_track[0]))
        
        filtered_boundaries = [0]
        for point in change_points:
            if point - filtered_boundaries[-1] >= min_frames:
                filtered_boundaries.append(point)
        
        filtered_boundaries.append(len(semitones) - 1)  # End boundary
        
        return filtered_boundaries
    
    def _detect_vibrato(self, f0_values: np.ndarray, time_values: np.ndarray) -> Dict[str, Any]:
        """Detect vibrato characteristics in a note"""
        if len(f0_values) < 10:
            return {'present': False, 'rate': 0, 'extent': 0}
        
        # Convert to semitones for analysis
        semitones = 12 * np.log2(f0_values)
        
        # Remove linear trend
        detrended = semitones - np.linspace(semitones[0], semitones[-1], len(semitones))
        
        # Analyze frequency content
        fft = np.fft.fft(detrended)
        freqs = np.fft.fftfreq(len(detrended), d=np.mean(np.diff(time_values)))
        
        # Look for periodic content in vibrato range (4-8 Hz)
        vibrato_mask = (freqs >= 4) & (freqs <= 8)
        if np.any(vibrato_mask):
            vibrato_power = np.max(np.abs(fft[vibrato_mask]))
            total_power = np.sum(np.abs(fft))
            
            vibrato_present = vibrato_power / total_power > 0.1  # 10% threshold
            
            if vibrato_present:
                vibrato_freq_idx = np.argmax(np.abs(fft[vibrato_mask]))
                vibrato_rate = freqs[vibrato_mask][vibrato_freq_idx]
                vibrato_extent = np.std(detrended)  # Semitones
            else:
                vibrato_rate = 0
                vibrato_extent = 0
        else:
            vibrato_present = False
            vibrato_rate = 0
            vibrato_extent = 0
        
        return {
            'present': vibrato_present,
            'rate': vibrato_rate,
            'extent': vibrato_extent
        }
    
    def _compute_note_importance(self, duration: float, stability: float, vibrato: Dict[str, Any]) -> float:
        """Compute the structural importance of a note"""
        importance = 0.0
        
        # Duration importance (longer notes are more important)
        duration_score = min(1.0, duration / 2.0)  # Normalize to 2 seconds
        importance += 0.4 * duration_score
        
        # Stability importance (stable notes are more important)
        importance += 0.3 * stability
        
        # Vibrato importance (expressive notes are more important)
        vibrato_score = 0.5 if vibrato['present'] else 0.0
        importance += 0.2 * vibrato_score
        
        # Position importance (to be added by caller based on phrase position)
        importance += 0.1  # Base importance
        
        return min(1.0, importance)
    
    def _detect_motifs(self, f0_track: np.ndarray, time_track: np.ndarray) -> List[Dict[str, Any]]:
        """Detect melodic motifs (short repeated patterns)"""
        motifs = []
        
        if len(f0_track) < self.motif_min_length:
            return motifs
        
        # Convert to interval sequence for pattern matching
        semitones = 12 * np.log2(f0_track)
        intervals = np.diff(semitones)
        
        # Find repeated patterns
        for length in range(self.motif_min_length, min(self.motif_max_length, len(intervals))):
            for start in range(len(intervals) - length):
                pattern = intervals[start:start+length]
                
                # Look for similar patterns elsewhere
                for search_start in range(start + length, len(intervals) - length):
                    candidate = intervals[search_start:search_start+length]
                    
                    # Check similarity (allow small variations)
                    similarity = 1.0 - np.mean(np.abs(pattern - candidate)) / 12.0
                    
                    if similarity > 0.8:  # 80% similarity threshold
                        motif = {
                            'pattern': pattern.tolist(),
                            'occurrences': [
                                {'start_time': time_track[start], 'end_time': time_track[start+length]},
                                {'start_time': time_track[search_start], 'end_time': time_track[search_start+length]}
                            ],
                            'similarity': similarity,
                            'length': length
                        }
                        motifs.append(motif)
        
        return motifs
    
    def _analyze_phrases(self, f0_track: np.ndarray, time_track: np.ndarray, 
                        phrase_boundaries: List[float]) -> List[Dict[str, Any]]:
        """Enhanced phrase analysis"""
        phrases = []
        
        for i in range(len(phrase_boundaries) - 1):
            start_time = phrase_boundaries[i]
            end_time = phrase_boundaries[i + 1]
            
            # Find corresponding indices in the tracks
            start_idx = np.argmin(np.abs(time_track - start_time))
            end_idx = np.argmin(np.abs(time_track - end_time))
            
            if end_idx <= start_idx:
                continue
                
            phrase_f0 = f0_track[start_idx:end_idx]
            phrase_time = time_track[start_idx:end_idx]
            
            if len(phrase_f0) > 0:
                phrase_semitones = 12 * np.log2(phrase_f0)
                
                phrase = {
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': end_time - start_time,
                    'pitch_range': np.max(phrase_semitones) - np.min(phrase_semitones),
                    'pitch_mean': np.mean(phrase_semitones),
                    'pitch_trend': np.polyfit(range(len(phrase_semitones)), phrase_semitones, 1)[0] if len(phrase_semitones) > 1 else 0,
                    'num_notes': len(phrase_f0),
                    'importance': self._compute_phrase_importance(phrase_f0, end_time - start_time)
                }
                
                phrases.append(phrase)
        
        return phrases
    
    def _compute_phrase_importance(self, phrase_f0: np.ndarray, duration: float) -> float:
        """Compute phrase importance based on various factors"""
        importance = 0.0
        
        # Duration importance
        duration_score = min(1.0, duration / self.phrase_max_duration)
        importance += 0.4 * duration_score
        
        # Pitch range importance
        if len(phrase_f0) > 1:
            semitones = 12 * np.log2(phrase_f0)
            pitch_range = np.max(semitones) - np.min(semitones)
            range_score = min(1.0, pitch_range / 12.0)  # Normalize by octave
            importance += 0.3 * range_score
        
        # Complexity importance (more notes = more complex)
        complexity_score = min(1.0, len(phrase_f0) / 20.0)  # Normalize by 20 notes
        importance += 0.2 * complexity_score
        
        # Base importance
        importance += 0.1
        
        return min(1.0, importance)
    
    def _detect_sections(self, audio_data: np.ndarray, phrases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect larger structural sections"""
        sections = []
        
        if len(phrases) < 2:
            # Single section for the whole piece
            sections.append({
                'start_time': 0.0,
                'end_time': len(audio_data) / self.sample_rate,
                'type': 'single_section',
                'phrases': phrases,
                'characteristics': {}
            })
            return sections
        
        # Group phrases into sections based on similarity
        current_section_start = 0
        current_section_phrases = [phrases[0]]
        
        for i in range(1, len(phrases)):
            # Simple sectioning based on phrase gaps
            gap = phrases[i]['start_time'] - phrases[i-1]['end_time']
            
            if gap > 1.0:  # 1 second gap indicates section boundary
                # Close current section
                sections.append({
                    'start_time': phrases[current_section_start]['start_time'],
                    'end_time': phrases[i-1]['end_time'],
                    'type': f'section_{len(sections)+1}',
                    'phrases': current_section_phrases,
                    'characteristics': self._analyze_section_characteristics(current_section_phrases)
                })
                
                # Start new section
                current_section_start = i
                current_section_phrases = [phrases[i]]
            else:
                current_section_phrases.append(phrases[i])
        
        # Add final section
        if current_section_phrases:
            sections.append({
                'start_time': phrases[current_section_start]['start_time'],
                'end_time': phrases[-1]['end_time'],
                'type': f'section_{len(sections)+1}',
                'phrases': current_section_phrases,
                'characteristics': self._analyze_section_characteristics(current_section_phrases)
            })
        
        return sections
    
    def _analyze_section_characteristics(self, phrases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze characteristics of a section"""
        if not phrases:
            return {}
        
        # Aggregate phrase characteristics
        durations = [p['duration'] for p in phrases]
        pitch_ranges = [p['pitch_range'] for p in phrases]
        pitch_means = [p['pitch_mean'] for p in phrases]
        
        return {
            'num_phrases': len(phrases),
            'total_duration': sum(durations),
            'avg_phrase_duration': np.mean(durations),
            'avg_pitch_range': np.mean(pitch_ranges),
            'avg_pitch_level': np.mean(pitch_means),
            'pitch_stability': 1.0 - np.std(pitch_means) / 12.0 if len(pitch_means) > 1 else 1.0
        }
    
    def _build_hierarchy(self, structure: Dict[str, Any]) -> Dict[str, Any]:
        """Build hierarchical relationships between structural elements"""
        hierarchy = {
            'levels': ['notes', 'motifs', 'phrases', 'sections'],
            'relationships': {}
        }
        
        # Map notes to phrases
        notes = structure.get('notes', [])
        phrases = structure.get('phrases', [])
        
        for phrase in phrases:
            phrase_notes = []
            for note in notes:
                if (phrase['start_time'] <= note['start_time'] <= phrase['end_time'] or
                    phrase['start_time'] <= note['end_time'] <= phrase['end_time']):
                    phrase_notes.append(note)
            phrase['notes'] = phrase_notes
        
        # Map phrases to sections
        sections = structure.get('sections', [])
        for section in sections:
            section_phrases = []
            for phrase in phrases:
                if (section['start_time'] <= phrase['start_time'] <= section['end_time']):
                    section_phrases.append(phrase)
            section['phrases'] = section_phrases
        
        return hierarchy
    
    def _compute_importance_weights(self, structure: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Compute importance weights for all structural elements"""
        weights = {}
        
        # Note importance weights
        notes = structure.get('notes', [])
        if notes:
            note_weights = np.array([note.get('importance', 0.5) for note in notes])
            weights['notes'] = note_weights
        
        # Phrase importance weights
        phrases = structure.get('phrases', [])
        if phrases:
            phrase_weights = np.array([phrase.get('importance', 0.5) for phrase in phrases])
            weights['phrases'] = phrase_weights
        
        # Section importance weights (uniform for now)
        sections = structure.get('sections', [])
        if sections:
            section_weights = np.ones(len(sections)) / len(sections)
            weights['sections'] = section_weights
        
        return weights