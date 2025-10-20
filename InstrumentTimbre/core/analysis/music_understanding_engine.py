"""
Music Understanding Engine - Week 6 Development Task

Multi-layer music understanding system for comprehensive music analysis.
Implements track role identification, structure parsing, harmony analysis, and rhythm recognition.
"""

import numpy as np
import torch
import torch.nn as nn
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import librosa
from scipy import signal
from scipy.stats import mode
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logger = logging.getLogger(__name__)


class TrackRole(Enum):
    """Track role categories"""
    MELODY = "melody"
    HARMONY = "harmony"
    BASS = "bass"
    RHYTHM = "rhythm"
    ACCOMPANIMENT = "accompaniment"
    LEAD = "lead"
    PAD = "pad"


class StructureType(Enum):
    """Music structure section types"""
    INTRO = "intro"
    VERSE = "verse"
    CHORUS = "chorus"
    BRIDGE = "bridge"
    OUTRO = "outro"
    SOLO = "solo"
    BREAKDOWN = "breakdown"


@dataclass
class TrackAnalysis:
    """Track role analysis result"""
    role: TrackRole
    instrument: str
    confidence: float
    features: Dict[str, float]
    spectral_characteristics: Dict[str, float]


@dataclass
class StructureSection:
    """Music structure section"""
    start_time: float
    end_time: float
    section_type: StructureType
    confidence: float
    features: Dict[str, Any]


@dataclass
class HarmonyAnalysis:
    """Harmony analysis result"""
    key: Dict[str, Any]
    chord_progression: List[Dict[str, Any]]
    harmonic_features: Dict[str, float]
    modulations: List[Dict[str, Any]]


@dataclass
class RhythmAnalysis:
    """Rhythm analysis result"""
    tempo: Dict[str, float]
    time_signature: str
    beat_pattern: Dict[str, Any]
    beat_times: List[float]
    rhythmic_features: Dict[str, float]


class MusicUnderstandingEngine:
    """
    Multi-layer music understanding engine
    
    Provides comprehensive analysis of musical content including:
    - Track role identification
    - Music structure parsing
    - Harmony relationship analysis  
    - Rhythm pattern recognition
    - Multi-layer integration
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize music understanding engine
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Audio processing parameters
        self.sample_rate = self.config.get('sample_rate', 22050)
        self.hop_length = self.config.get('hop_length', 512)
        self.n_fft = self.config.get('n_fft', 2048)
        self.n_mels = self.config.get('n_mels', 128)
        
        # Analysis parameters
        self.window_size = self.config.get('analysis_window', 4.0)  # 4 second windows
        self.overlap_ratio = self.config.get('overlap_ratio', 0.5)
        
        # Initialize analysis modules
        self._init_analysis_modules()
        
        logger.info("MusicUnderstandingEngine initialized on device: %s", self.device)
    
    def _init_analysis_modules(self) -> None:
        """Initialize internal analysis modules"""
        
        # Track role classifier
        self.role_classifier = self._create_role_classifier()
        
        # Structure analyzer
        self.structure_analyzer = self._create_structure_analyzer()
        
        # Harmony analyzer  
        self.harmony_analyzer = self._create_harmony_analyzer()
        
        # Rhythm analyzer
        self.rhythm_analyzer = self._create_rhythm_analyzer()
        
        logger.info("Analysis modules initialized")
    
    def _create_role_classifier(self) -> nn.Module:
        """Create track role classification network"""
        
        class RoleClassifier(nn.Module):
            def __init__(self, input_dim: int = 128, num_roles: int = 7):
                super(RoleClassifier, self).__init__()
                
                self.feature_extractor = nn.Sequential(
                    nn.Linear(input_dim, 256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, 256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, 128),
                    nn.ReLU()
                )
                
                self.role_classifier = nn.Linear(128, num_roles)
                
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                features = self.feature_extractor(x)
                role_logits = self.role_classifier(features)
                return role_logits
        
        classifier = RoleClassifier(input_dim=self.n_mels)
        classifier.to(self.device)
        return classifier
    
    def _create_structure_analyzer(self) -> nn.Module:
        """Create music structure analysis network"""
        
        class StructureAnalyzer(nn.Module):
            def __init__(self, input_dim: int = 128, num_sections: int = 7):
                super(StructureAnalyzer, self).__init__()
                
                self.feature_extractor = nn.Sequential(
                    nn.Linear(input_dim, 256),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(256, 256),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(256, 128),
                    nn.ReLU()
                )
                
                self.section_classifier = nn.Linear(128, num_sections)
                
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                features = self.feature_extractor(x)
                section_logits = self.section_classifier(features)
                return section_logits
        
        analyzer = StructureAnalyzer(input_dim=self.n_mels)
        analyzer.to(self.device)
        return analyzer
    
    def _create_harmony_analyzer(self) -> Dict[str, Any]:
        """Create harmony analysis components"""
        
        # Chroma-based harmony analysis
        return {
            'chroma_processor': self._setup_chroma_processor(),
            'key_detector': self._setup_key_detector(),
            'chord_detector': self._setup_chord_detector()
        }
    
    def _create_rhythm_analyzer(self) -> Dict[str, Any]:
        """Create rhythm analysis components"""
        
        # Onset and tempo analysis
        return {
            'onset_detector': self._setup_onset_detector(),
            'tempo_estimator': self._setup_tempo_estimator(),
            'beat_tracker': self._setup_beat_tracker()
        }
    
    def _setup_chroma_processor(self) -> Dict[str, Any]:
        """Setup chroma feature processing"""
        return {
            'n_chroma': 12,
            'tuning': 0.0,
            'norm': 2
        }
    
    def _setup_key_detector(self) -> Dict[str, Any]:
        """Setup key detection algorithm"""
        
        # Krumhansl-Schmuckler key profiles
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        
        return {
            'major_profile': major_profile / np.sum(major_profile),
            'minor_profile': minor_profile / np.sum(minor_profile),
            'keys': ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        }
    
    def _setup_chord_detector(self) -> Dict[str, Any]:
        """Setup chord detection templates"""
        
        # Basic chord templates (major, minor, diminished, augmented)
        chord_templates = {
            'major': np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]),
            'minor': np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]),
            'diminished': np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0]),
            'augmented': np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0])
        }
        
        return {
            'templates': chord_templates,
            'chord_names': ['major', 'minor', 'diminished', 'augmented']
        }
    
    def _setup_onset_detector(self) -> Dict[str, Any]:
        """Setup onset detection parameters"""
        return {
            'method': 'superflux',
            'lag': 2,
            'max_size': 3,
            'wait': 5,
            'threshold': 0.7
        }
    
    def _setup_tempo_estimator(self) -> Dict[str, Any]:
        """Setup tempo estimation parameters"""
        return {
            'method': 'degara',
            'sr': self.sample_rate,
            'hop_length': self.hop_length
        }
    
    def _setup_beat_tracker(self) -> Dict[str, Any]:
        """Setup beat tracking parameters"""
        return {
            'method': 'ellis',
            'start_bpm': 120.0,
            'std_bpm': 1.0,
            'ac_size': 8.0
        }
    
    def identify_track_roles(self, tracks: Dict[str, np.ndarray]) -> Dict[str, Dict[str, Any]]:
        """
        Identify the role of each track in the mix
        
        Args:
            tracks: Dictionary mapping track names to audio arrays
            
        Returns:
            Dictionary mapping track names to role analysis results
        """
        logger.info("Analyzing track roles for %d tracks", len(tracks))
        
        role_analysis = {}
        
        for track_name, audio in tracks.items():
            try:
                # Extract features for role classification
                features = self._extract_track_features(audio)
                
                # Classify track role
                role_probs = self._classify_track_role(features)
                
                # Determine primary role
                role_idx = np.argmax(role_probs)
                roles = list(TrackRole)
                primary_role = roles[role_idx]
                confidence = float(role_probs[role_idx])
                
                # Extract spectral characteristics
                spectral_chars = self._extract_spectral_characteristics(audio)
                
                # Estimate instrument type
                instrument = self._estimate_instrument_type(features, spectral_chars)
                
                role_analysis[track_name] = {
                    'role': primary_role.value,
                    'instrument': instrument,
                    'confidence': confidence,
                    'features': features,
                    'spectral_characteristics': spectral_chars,
                    'role_probabilities': dict(zip([r.value for r in roles], role_probs))
                }
                
                logger.info("Track '%s': %s (%s, %.3f confidence)", 
                           track_name, primary_role.value, instrument, confidence)
                
            except Exception as e:
                logger.error("Failed to analyze track '%s': %s", track_name, e)
                # Default analysis
                role_analysis[track_name] = {
                    'role': TrackRole.ACCOMPANIMENT.value,
                    'instrument': 'unknown',
                    'confidence': 0.1,
                    'features': {},
                    'spectral_characteristics': {}
                }
        
        return role_analysis
    
    def parse_music_structure(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Parse music structure into sections (intro, verse, chorus, etc.)
        
        Args:
            audio: Input audio signal
            
        Returns:
            Dictionary containing structure analysis results
        """
        logger.info("Parsing music structure for audio: %d samples", len(audio))
        
        try:
            # Extract features for structure analysis
            features = self._extract_structure_features(audio)
            
            # Detect section boundaries using novelty detection
            boundaries = self._detect_section_boundaries(audio, features)
            
            # Classify sections
            sections = self._classify_sections(audio, boundaries, features)
            
            # Analyze overall structure
            overall_structure = self._analyze_overall_structure(sections)
            
            structure_analysis = {
                'sections': sections,
                'boundaries': boundaries,
                'overall_structure': overall_structure,
                'total_duration': len(audio) / self.sample_rate,
                'analysis_confidence': self._calculate_structure_confidence(sections)
            }
            
            logger.info("Structure analysis complete: %d sections detected", len(sections))
            return structure_analysis
            
        except Exception as e:
            logger.error("Structure parsing failed: %s", e)
            return self._create_default_structure(audio)
    
    def analyze_harmony_relationships(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Analyze harmony relationships including key, chord progressions
        
        Args:
            audio: Input audio signal
            
        Returns:
            Dictionary containing harmony analysis results
        """
        logger.info("Analyzing harmony relationships for audio: %d samples", len(audio))
        
        try:
            # Extract chroma features
            chroma = self._extract_chroma_features(audio)
            
            # Detect key
            key_analysis = self._detect_musical_key(chroma)
            
            # Detect chord progression
            chord_progression = self._detect_chord_progression(chroma)
            
            # Analyze harmonic features
            harmonic_features = self._extract_harmonic_features(chroma, chord_progression)
            
            # Detect modulations
            modulations = self._detect_modulations(chroma, key_analysis)
            
            harmony_analysis = {
                'key': key_analysis,
                'chord_progression': chord_progression,
                'harmonic_features': harmonic_features,
                'modulations': modulations,
                'chroma_features': chroma.tolist() if chroma.size < 1000 else chroma[:, :100].tolist()
            }
            
            logger.info("Harmony analysis complete: key=%s, %d chords", 
                       key_analysis['tonic'], len(chord_progression))
            return harmony_analysis
            
        except Exception as e:
            logger.error("Harmony analysis failed: %s", e)
            return self._create_default_harmony()
    
    def analyze_rhythm_patterns(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Analyze rhythm patterns including tempo, time signature, beats
        
        Args:
            audio: Input audio signal
            
        Returns:
            Dictionary containing rhythm analysis results
        """
        logger.info("Analyzing rhythm patterns for audio: %d samples", len(audio))
        
        try:
            # Detect tempo
            tempo_analysis = self._detect_tempo(audio)
            
            # Detect time signature
            time_signature = self._detect_time_signature(audio, tempo_analysis)
            
            # Track beats
            beat_times = self._track_beats(audio, tempo_analysis)
            
            # Analyze beat patterns
            beat_pattern = self._analyze_beat_pattern(audio, beat_times)
            
            # Extract rhythmic features
            rhythmic_features = self._extract_rhythmic_features(audio, beat_times, tempo_analysis)
            
            rhythm_analysis = {
                'tempo': tempo_analysis,
                'time_signature': time_signature,
                'beat_pattern': beat_pattern,
                'beat_times': beat_times,
                'rhythmic_features': rhythmic_features,
                'onset_times': self._detect_onset_times(audio)
            }
            
            logger.info("Rhythm analysis complete: %.1f BPM, %s time", 
                       tempo_analysis['bpm'], time_signature)
            return rhythm_analysis
            
        except Exception as e:
            logger.error("Rhythm analysis failed: %s", e)
            return self._create_default_rhythm()
    
    def analyze_complete_music_understanding(self, tracks: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Perform comprehensive multi-layer music understanding analysis
        
        Args:
            tracks: Dictionary mapping track names to audio arrays
            
        Returns:
            Dictionary containing complete analysis results
        """
        logger.info("Performing comprehensive music understanding analysis")
        
        try:
            # Individual track analysis
            track_analysis = self.identify_track_roles(tracks)
            
            # Create combined audio for structure/harmony/rhythm analysis
            combined_audio = self._create_combined_audio(tracks)
            
            # Structure analysis
            structure_analysis = self.parse_music_structure(combined_audio)
            
            # Harmony analysis
            harmony_analysis = self.analyze_harmony_relationships(combined_audio)
            
            # Rhythm analysis
            rhythm_analysis = self.analyze_rhythm_patterns(combined_audio)
            
            # Integration analysis
            integration_metrics = self._analyze_integration_quality(
                track_analysis, structure_analysis, harmony_analysis, rhythm_analysis
            )
            
            comprehensive_analysis = {
                'track_analysis': track_analysis,
                'structure_analysis': structure_analysis,
                'harmony_analysis': harmony_analysis,
                'rhythm_analysis': rhythm_analysis,
                'integration_metrics': integration_metrics,
                'overall_assessment': self._create_overall_assessment(integration_metrics)
            }
            
            logger.info("Comprehensive analysis complete")
            return comprehensive_analysis
            
        except Exception as e:
            logger.error("Comprehensive analysis failed: %s", e)
            return self._create_default_comprehensive_analysis(tracks)
    
    # Helper methods for feature extraction and analysis
    
    def _extract_track_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract features for track role classification"""
        
        if len(audio) == 0:
            return {}
        
        try:
            # Basic spectral features
            mel_spec = librosa.feature.melspectrogram(
                y=audio, sr=self.sample_rate, hop_length=self.hop_length, n_mels=self.n_mels
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Statistical features
            features = {
                'spectral_centroid': float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate))),
                'spectral_bandwidth': float(np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate))),
                'spectral_rolloff': float(np.mean(librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate))),
                'zero_crossing_rate': float(np.mean(librosa.feature.zero_crossing_rate(audio))),
                'rms_energy': float(np.mean(librosa.feature.rms(y=audio))),
                'tempo': float(librosa.beat.tempo(y=audio, sr=self.sample_rate)[0]),
                'pitch_range': float(np.max(mel_spec_db) - np.min(mel_spec_db)),
                'spectral_contrast': float(np.mean(librosa.feature.spectral_contrast(y=audio, sr=self.sample_rate))),
            }
            
            # MFCC features (first 13 coefficients)
            mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
            for i in range(13):
                features[f'mfcc_{i}'] = float(np.mean(mfcc[i]))
            
            return features
            
        except Exception as e:
            logger.warning("Feature extraction failed: %s", e)
            return {}
    
    def _classify_track_role(self, features: Dict[str, float]) -> np.ndarray:
        """Classify track role using extracted features"""
        
        if not features:
            # Default uniform distribution
            return np.ones(len(TrackRole)) / len(TrackRole)
        
        try:
            # Simple heuristic-based classification
            spectral_centroid = features.get('spectral_centroid', 1000)
            rms_energy = features.get('rms_energy', 0.1)
            zero_crossing_rate = features.get('zero_crossing_rate', 0.1)
            spectral_bandwidth = features.get('spectral_bandwidth', 1000)
            
            # Role probabilities based on acoustic characteristics
            role_scores = np.zeros(len(TrackRole))
            
            # Melody: high spectral centroid, moderate energy
            if spectral_centroid > 2000 and rms_energy > 0.05:
                role_scores[0] = 0.8  # MELODY
            
            # Harmony: moderate spectral features, sustained energy
            if 1000 < spectral_centroid < 3000 and rms_energy > 0.03:
                role_scores[1] = 0.7  # HARMONY
            
            # Bass: low spectral centroid, high energy
            if spectral_centroid < 1000 and rms_energy > 0.1:
                role_scores[2] = 0.9  # BASS
            
            # Rhythm: high zero crossing rate, variable energy
            if zero_crossing_rate > 0.15:
                role_scores[3] = 0.6  # RHYTHM
            
            # Accompaniment: moderate all features
            if 500 < spectral_centroid < 2000:
                role_scores[4] = 0.5  # ACCOMPANIMENT
            
            # Lead: high centroid and energy
            if spectral_centroid > 2500 and rms_energy > 0.08:
                role_scores[5] = 0.7  # LEAD
            
            # Pad: low energy, sustained
            if rms_energy < 0.05 and spectral_bandwidth > 1500:
                role_scores[6] = 0.6  # PAD
            
            # Normalize and add noise to prevent all-zero
            role_scores += np.random.random(len(TrackRole)) * 0.1
            role_scores = role_scores / np.sum(role_scores)
            
            return role_scores
            
        except Exception as e:
            logger.warning("Role classification failed: %s", e)
            return np.ones(len(TrackRole)) / len(TrackRole)
    
    def _extract_spectral_characteristics(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract spectral characteristics for instrument identification"""
        
        if len(audio) == 0:
            return {}
        
        try:
            # Spectral features
            stft = librosa.stft(audio, hop_length=self.hop_length, n_fft=self.n_fft)
            magnitude = np.abs(stft)
            
            characteristics = {
                'fundamental_freq': float(np.argmax(np.mean(magnitude, axis=1)) * self.sample_rate / self.n_fft),
                'harmonic_ratio': float(np.sum(magnitude[1::2]) / (np.sum(magnitude[::2]) + 1e-8)),
                'noise_ratio': float(np.std(magnitude) / (np.mean(magnitude) + 1e-8)),
                'attack_time': self._estimate_attack_time(audio),
                'decay_rate': self._estimate_decay_rate(audio),
                'brightness': float(np.sum(magnitude[magnitude.shape[0]//2:]) / np.sum(magnitude))
            }
            
            return characteristics
            
        except Exception as e:
            logger.warning("Spectral characteristics extraction failed: %s", e)
            return {}
    
    def _estimate_instrument_type(self, features: Dict[str, float], 
                                spectral_chars: Dict[str, float]) -> str:
        """Estimate instrument type based on features"""
        
        if not features or not spectral_chars:
            return 'unknown'
        
        try:
            # Simple heuristic instrument classification
            spectral_centroid = features.get('spectral_centroid', 1000)
            brightness = spectral_chars.get('brightness', 0.5)
            attack_time = spectral_chars.get('attack_time', 0.1)
            harmonic_ratio = spectral_chars.get('harmonic_ratio', 1.0)
            
            # Classification rules
            if spectral_centroid < 500 and attack_time > 0.1:
                return 'bass'
            elif brightness > 0.6 and attack_time < 0.05:
                return 'guitar'
            elif harmonic_ratio > 1.5 and attack_time > 0.05:
                return 'piano'
            elif brightness < 0.3 and harmonic_ratio < 1.0:
                return 'strings'
            elif attack_time < 0.01:
                return 'drums'
            else:
                return 'synth'
                
        except Exception as e:
            logger.warning("Instrument estimation failed: %s", e)
            return 'unknown'
    
    def _estimate_attack_time(self, audio: np.ndarray) -> float:
        """Estimate attack time of audio signal"""
        
        try:
            # Find onset and measure rise time
            onset_envelope = librosa.onset.onset_strength(y=audio, sr=self.sample_rate)
            if len(onset_envelope) == 0:
                return 0.1
            
            # Find first significant onset
            threshold = np.max(onset_envelope) * 0.1
            onset_idx = np.where(onset_envelope > threshold)[0]
            
            if len(onset_idx) == 0:
                return 0.1
            
            # Estimate attack time (rough approximation)
            attack_samples = min(1024, len(audio) // 4)  # Max 1024 samples
            attack_time = attack_samples / self.sample_rate
            
            return float(attack_time)
            
        except Exception:
            return 0.1
    
    def _estimate_decay_rate(self, audio: np.ndarray) -> float:
        """Estimate decay rate of audio signal"""
        
        try:
            # Simple envelope follower
            envelope = np.abs(audio)
            
            # Smooth the envelope
            window_size = min(1024, len(envelope) // 10)
            if window_size < 3:
                return 1.0
            
            smoothed = np.convolve(envelope, np.ones(window_size) / window_size, mode='same')
            
            # Find peak and measure decay
            peak_idx = np.argmax(smoothed)
            if peak_idx > len(smoothed) - 100:
                return 1.0
            
            # Measure decay from peak
            decay_section = smoothed[peak_idx:peak_idx + min(1000, len(smoothed) - peak_idx)]
            if len(decay_section) < 2:
                return 1.0
            
            # Linear fit to log envelope (exponential decay)
            try:
                log_envelope = np.log(decay_section + 1e-8)
                decay_rate = -np.polyfit(range(len(log_envelope)), log_envelope, 1)[0]
                return float(max(0.1, min(10.0, decay_rate)))  # Clamp to reasonable range
            except:
                return 1.0
                
        except Exception:
            return 1.0
    
    # Structure analysis helper methods
    
    def _extract_structure_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract features for structure analysis"""
        
        try:
            # Extract chroma and MFCC features for structure analysis
            chroma = librosa.feature.chroma_stft(y=audio, sr=self.sample_rate, hop_length=self.hop_length)
            mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13, hop_length=self.hop_length)
            
            # Combine features
            features = np.vstack([chroma, mfcc])
            return features
            
        except Exception as e:
            logger.warning("Structure feature extraction failed: %s", e)
            # Return dummy features
            n_frames = max(1, len(audio) // self.hop_length)
            return np.random.random((25, n_frames)) * 0.1
    
    def _detect_section_boundaries(self, audio: np.ndarray, features: np.ndarray) -> List[float]:
        """Detect section boundaries using novelty detection"""
        
        try:
            # Compute self-similarity matrix
            similarity_matrix = np.corrcoef(features.T)
            
            # Apply novelty function
            novelty_curve = np.zeros(similarity_matrix.shape[0])
            kernel_size = min(16, similarity_matrix.shape[0] // 4)
            
            for i in range(kernel_size, len(novelty_curve) - kernel_size):
                # Local novelty based on self-similarity
                local_sim = similarity_matrix[i-kernel_size:i+kernel_size, i-kernel_size:i+kernel_size]
                novelty_curve[i] = 1.0 - np.mean(local_sim)
            
            # Find peaks in novelty curve
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(novelty_curve, height=np.percentile(novelty_curve, 75))
            
            # Convert frame indices to time
            boundary_times = [0.0]  # Always start with 0
            for peak in peaks:
                time_sec = peak * self.hop_length / self.sample_rate
                if time_sec > 1.0:  # Minimum section length of 1 second
                    boundary_times.append(time_sec)
            
            # Add end time
            boundary_times.append(len(audio) / self.sample_rate)
            
            return sorted(list(set(boundary_times)))
            
        except Exception as e:
            logger.warning("Boundary detection failed: %s", e)
            # Default to simple time-based sections
            duration = len(audio) / self.sample_rate
            return [0.0, duration / 3, 2 * duration / 3, duration]
    
    def _classify_sections(self, audio: np.ndarray, boundaries: List[float], 
                          features: np.ndarray) -> List[Dict[str, Any]]:
        """Classify each section type"""
        
        sections = []
        section_types = list(StructureType)
        
        for i in range(len(boundaries) - 1):
            start_time = boundaries[i]
            end_time = boundaries[i + 1]
            
            # Extract section audio
            start_sample = int(start_time * self.sample_rate)
            end_sample = int(end_time * self.sample_rate)
            section_audio = audio[start_sample:end_sample]
            
            # Simple heuristic classification
            if i == 0:
                section_type = StructureType.INTRO
            elif i == len(boundaries) - 2:
                section_type = StructureType.OUTRO
            elif i % 2 == 1:
                section_type = StructureType.VERSE
            else:
                section_type = StructureType.CHORUS
            
            # Calculate features for this section
            section_features = self._calculate_section_features(section_audio)
            
            sections.append({
                'start_time': start_time,
                'end_time': end_time,
                'section_type': section_type.value,
                'confidence': 0.7,  # Placeholder confidence
                'features': section_features
            })
        
        return sections
    
    def _calculate_section_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Calculate features for a specific section"""
        
        if len(audio) == 0:
            return {}
        
        try:
            features = {
                'energy': float(np.mean(librosa.feature.rms(y=audio))),
                'tempo': float(librosa.beat.tempo(y=audio, sr=self.sample_rate)[0]),
                'spectral_centroid': float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate))),
                'duration': len(audio) / self.sample_rate
            }
            
            # Simple key estimation
            chroma = librosa.feature.chroma_stft(y=audio, sr=self.sample_rate)
            key_idx = np.argmax(np.mean(chroma, axis=1))
            keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            features['key'] = keys[key_idx]
            
            return features
            
        except Exception as e:
            logger.warning("Section feature calculation failed: %s", e)
            return {'energy': 0.1, 'tempo': 120.0, 'spectral_centroid': 1000.0}
    
    def _analyze_overall_structure(self, sections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze overall structure pattern"""
        
        section_sequence = [s['section_type'] for s in sections]
        
        return {
            'section_sequence': section_sequence,
            'total_sections': len(sections),
            'structure_pattern': self._identify_structure_pattern(section_sequence),
            'average_section_length': np.mean([s['end_time'] - s['start_time'] for s in sections])
        }
    
    def _identify_structure_pattern(self, sequence: List[str]) -> str:
        """Identify common structure patterns"""
        
        sequence_str = '-'.join(sequence)
        
        # Common patterns
        if 'intro' in sequence_str and 'outro' in sequence_str:
            if 'verse' in sequence_str and 'chorus' in sequence_str:
                return 'verse-chorus'
            else:
                return 'simple'
        else:
            return 'free-form'
    
    def _calculate_structure_confidence(self, sections: List[Dict[str, Any]]) -> float:
        """Calculate overall structure analysis confidence"""
        
        if not sections:
            return 0.0
        
        confidences = [s.get('confidence', 0.5) for s in sections]
        return float(np.mean(confidences))
    
    def _create_default_structure(self, audio: np.ndarray) -> Dict[str, Any]:
        """Create default structure analysis"""
        
        duration = len(audio) / self.sample_rate
        
        return {
            'sections': [{
                'start_time': 0.0,
                'end_time': duration,
                'section_type': StructureType.VERSE.value,
                'confidence': 0.1,
                'features': {'energy': 0.1, 'tempo': 120.0}
            }],
            'boundaries': [0.0, duration],
            'overall_structure': {
                'section_sequence': ['verse'],
                'total_sections': 1,
                'structure_pattern': 'simple'
            },
            'total_duration': duration,
            'analysis_confidence': 0.1
        }
    
    # Harmony analysis helper methods
    
    def _extract_chroma_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract chroma features for harmony analysis"""
        
        try:
            chroma = librosa.feature.chroma_stft(
                y=audio, sr=self.sample_rate, hop_length=self.hop_length,
                tuning=self.harmony_analyzer['chroma_processor']['tuning'],
                norm=self.harmony_analyzer['chroma_processor']['norm']
            )
            return chroma
            
        except Exception as e:
            logger.warning("Chroma extraction failed: %s", e)
            n_frames = max(1, len(audio) // self.hop_length)
            return np.random.random((12, n_frames)) * 0.1
    
    def _detect_musical_key(self, chroma: np.ndarray) -> Dict[str, Any]:
        """Detect musical key using Krumhansl-Schmuckler algorithm"""
        
        try:
            # Average chroma over time
            chroma_mean = np.mean(chroma, axis=1)
            
            # Key detection profiles
            major_profile = self.harmony_analyzer['key_detector']['major_profile']
            minor_profile = self.harmony_analyzer['key_detector']['minor_profile']
            keys = self.harmony_analyzer['key_detector']['keys']
            
            # Calculate correlations for all keys
            major_correlations = []
            minor_correlations = []
            
            for i in range(12):
                # Rotate chroma to test each key
                rotated_chroma = np.roll(chroma_mean, i)
                
                # Correlate with profiles
                major_corr = np.corrcoef(rotated_chroma, major_profile)[0, 1]
                minor_corr = np.corrcoef(rotated_chroma, minor_profile)[0, 1]
                
                major_correlations.append(major_corr)
                minor_correlations.append(minor_corr)
            
            # Find best match
            best_major_idx = np.argmax(major_correlations)
            best_minor_idx = np.argmax(minor_correlations)
            
            best_major_corr = major_correlations[best_major_idx]
            best_minor_corr = minor_correlations[best_minor_idx]
            
            if best_major_corr > best_minor_corr:
                tonic = keys[best_major_idx]
                scale_type = 'major'
                confidence = float(best_major_corr)
            else:
                tonic = keys[best_minor_idx]
                scale_type = 'minor'
                confidence = float(best_minor_corr)
            
            # Normalize confidence to [0, 1]
            confidence = max(0.0, min(1.0, (confidence + 1) / 2))
            
            return {
                'tonic': tonic,
                'scale_type': scale_type,
                'confidence': confidence,
                'major_correlations': major_correlations,
                'minor_correlations': minor_correlations
            }
            
        except Exception as e:
            logger.warning("Key detection failed: %s", e)
            return {
                'tonic': 'C',
                'scale_type': 'major',
                'confidence': 0.1,
                'major_correlations': [0.1] * 12,
                'minor_correlations': [0.1] * 12
            }
    
    def _detect_chord_progression(self, chroma: np.ndarray) -> List[Dict[str, Any]]:
        """Detect chord progression from chroma features"""
        
        try:
            # Segment chroma into chord-length windows
            window_size = int(2.0 * self.sample_rate / self.hop_length)  # 2-second windows
            chord_progression = []
            
            for i in range(0, chroma.shape[1], window_size // 2):  # 50% overlap
                end_idx = min(i + window_size, chroma.shape[1])
                if end_idx - i < window_size // 2:  # Skip short segments
                    break
                
                window_chroma = chroma[:, i:end_idx]
                avg_chroma = np.mean(window_chroma, axis=1)
                
                # Simple chord detection
                chord_info = self._classify_chord(avg_chroma)
                
                start_time = i * self.hop_length / self.sample_rate
                end_time = end_idx * self.hop_length / self.sample_rate
                
                chord_progression.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'chord_name': chord_info['name'],
                    'confidence': chord_info['confidence'],
                    'chroma_vector': avg_chroma.tolist()
                })
            
            return chord_progression
            
        except Exception as e:
            logger.warning("Chord progression detection failed: %s", e)
            # Default single chord
            duration = chroma.shape[1] * self.hop_length / self.sample_rate
            return [{
                'start_time': 0.0,
                'end_time': duration,
                'chord_name': 'C major',
                'confidence': 0.1,
                'chroma_vector': [1.0] + [0.0] * 11
            }]
    
    def _classify_chord(self, chroma_vector: np.ndarray) -> Dict[str, Any]:
        """Classify a chord from chroma vector"""
        
        try:
            # Find root note
            root_idx = np.argmax(chroma_vector)
            keys = self.harmony_analyzer['key_detector']['keys']
            root_note = keys[root_idx]
            
            # Chord templates
            templates = self.harmony_analyzer['chord_detector']['templates']
            chord_names = self.harmony_analyzer['chord_detector']['chord_names']
            
            best_match = 'major'
            best_correlation = -1
            
            # Test each chord type
            for chord_type, template in templates.items():
                # Rotate template to root
                rotated_template = np.roll(template, root_idx)
                correlation = np.corrcoef(chroma_vector, rotated_template)[0, 1]
                
                if correlation > best_correlation:
                    best_correlation = correlation
                    best_match = chord_type
            
            chord_name = f"{root_note} {best_match}"
            confidence = max(0.0, min(1.0, (best_correlation + 1) / 2))
            
            return {
                'name': chord_name,
                'confidence': confidence,
                'root': root_note,
                'type': best_match
            }
            
        except Exception as e:
            logger.warning("Chord classification failed: %s", e)
            return {
                'name': 'C major',
                'confidence': 0.1,
                'root': 'C',
                'type': 'major'
            }
    
    def _extract_harmonic_features(self, chroma: np.ndarray, 
                                 chord_progression: List[Dict[str, Any]]) -> Dict[str, float]:
        """Extract harmonic features"""
        
        try:
            features = {}
            
            # Harmonic rhythm (chords per second)
            total_duration = len(chord_progression) * 2.0  # Approximate
            features['harmonic_rhythm'] = len(chord_progression) / max(1, total_duration)
            
            # Consonance level
            consonance_scores = []
            for chord in chord_progression:
                chroma_vec = np.array(chord['chroma_vector'])
                # Simple consonance measure based on perfect fifths and thirds
                consonance = (chroma_vec[0] + chroma_vec[4] + chroma_vec[7]) / 3  # Root, third, fifth
                consonance_scores.append(consonance)
            
            features['consonance'] = float(np.mean(consonance_scores)) if consonance_scores else 0.5
            
            # Complexity (number of unique chords)
            unique_chords = len(set([c['chord_name'] for c in chord_progression]))
            features['complexity'] = unique_chords / max(1, len(chord_progression))
            
            # Tonal stability
            features['tonal_stability'] = float(np.std(chroma, axis=1).mean())
            
            return features
            
        except Exception as e:
            logger.warning("Harmonic feature extraction failed: %s", e)
            return {
                'harmonic_rhythm': 0.5,
                'consonance': 0.5,
                'complexity': 0.5,
                'tonal_stability': 0.5
            }
    
    def _detect_modulations(self, chroma: np.ndarray, key_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect key modulations"""
        
        try:
            modulations = []
            
            # Simple modulation detection by analyzing key in segments
            segment_length = chroma.shape[1] // 4  # 4 segments
            
            for i in range(4):
                start_idx = i * segment_length
                end_idx = min((i + 1) * segment_length, chroma.shape[1])
                
                if end_idx - start_idx < segment_length // 2:
                    continue
                
                segment_chroma = chroma[:, start_idx:end_idx]
                segment_key = self._detect_musical_key(segment_chroma)
                
                if segment_key['tonic'] != key_analysis['tonic']:
                    start_time = start_idx * self.hop_length / self.sample_rate
                    modulations.append({
                        'time': start_time,
                        'from_key': key_analysis['tonic'],
                        'to_key': segment_key['tonic'],
                        'confidence': segment_key['confidence']
                    })
            
            return modulations
            
        except Exception as e:
            logger.warning("Modulation detection failed: %s", e)
            return []
    
    def _create_default_harmony(self) -> Dict[str, Any]:
        """Create default harmony analysis"""
        
        return {
            'key': {
                'tonic': 'C',
                'scale_type': 'major',
                'confidence': 0.1
            },
            'chord_progression': [{
                'start_time': 0.0,
                'end_time': 4.0,
                'chord_name': 'C major',
                'confidence': 0.1
            }],
            'harmonic_features': {
                'harmonic_rhythm': 0.25,
                'consonance': 0.5,
                'complexity': 0.1
            },
            'modulations': []
        }
    
    # Rhythm analysis helper methods
    
    def _detect_tempo(self, audio: np.ndarray) -> Dict[str, float]:
        """Detect tempo using librosa"""
        
        try:
            tempo, beats = librosa.beat.beat_track(y=audio, sr=self.sample_rate)
            
            # Calculate tempo confidence based on beat consistency
            if len(beats) > 1:
                beat_intervals = np.diff(beats) * self.hop_length / self.sample_rate
                tempo_confidence = 1.0 - min(1.0, np.std(beat_intervals) / np.mean(beat_intervals))
            else:
                tempo_confidence = 0.1
            
            return {
                'bpm': float(tempo),
                'confidence': float(max(0.1, tempo_confidence))
            }
            
        except Exception as e:
            logger.warning("Tempo detection failed: %s", e)
            return {'bpm': 120.0, 'confidence': 0.1}
    
    def _detect_time_signature(self, audio: np.ndarray, tempo_analysis: Dict[str, float]) -> str:
        """Detect time signature"""
        
        try:
            # Simple time signature detection based on beat patterns
            tempo = tempo_analysis['bpm']
            
            # Analyze onset patterns
            onset_envelope = librosa.onset.onset_strength(y=audio, sr=self.sample_rate)
            
            # Look for downbeat patterns (simplified)
            if tempo > 140:
                return "4/4"  # Fast tempo typically 4/4
            elif tempo < 80:
                return "3/4"  # Slow tempo might be waltz
            else:
                return "4/4"  # Default to 4/4
                
        except Exception as e:
            logger.warning("Time signature detection failed: %s", e)
            return "4/4"
    
    def _track_beats(self, audio: np.ndarray, tempo_analysis: Dict[str, float]) -> List[float]:
        """Track beat positions"""
        
        try:
            tempo, beats = librosa.beat.beat_track(y=audio, sr=self.sample_rate)
            
            # Convert beat frames to time
            beat_times = librosa.frames_to_time(beats, sr=self.sample_rate, hop_length=self.hop_length)
            
            return beat_times.tolist()
            
        except Exception as e:
            logger.warning("Beat tracking failed: %s", e)
            # Generate regular beat grid
            bpm = tempo_analysis['bpm']
            duration = len(audio) / self.sample_rate
            beat_interval = 60.0 / bpm
            return [i * beat_interval for i in range(int(duration / beat_interval) + 1)]
    
    def _analyze_beat_pattern(self, audio: np.ndarray, beat_times: List[float]) -> Dict[str, Any]:
        """Analyze beat pattern characteristics"""
        
        try:
            # Analyze beat intervals
            if len(beat_times) < 2:
                return {'pattern_type': 'irregular', 'consistency': 0.1}
            
            intervals = np.diff(beat_times)
            
            # Calculate consistency
            mean_interval = np.mean(intervals)
            std_interval = np.std(intervals)
            consistency = 1.0 - min(1.0, std_interval / mean_interval) if mean_interval > 0 else 0.1
            
            # Classify pattern type
            if consistency > 0.8:
                pattern_type = 'regular'
            elif consistency > 0.5:
                pattern_type = 'mostly_regular'
            else:
                pattern_type = 'irregular'
            
            return {
                'pattern_type': pattern_type,
                'consistency': float(consistency),
                'mean_interval': float(mean_interval),
                'interval_std': float(std_interval)
            }
            
        except Exception as e:
            logger.warning("Beat pattern analysis failed: %s", e)
            return {'pattern_type': 'regular', 'consistency': 0.5}
    
    def _extract_rhythmic_features(self, audio: np.ndarray, beat_times: List[float], 
                                 tempo_analysis: Dict[str, float]) -> Dict[str, float]:
        """Extract rhythmic features"""
        
        try:
            features = {}
            
            # Groove strength (beat consistency)
            if len(beat_times) > 1:
                intervals = np.diff(beat_times)
                groove_strength = 1.0 - min(1.0, np.std(intervals) / np.mean(intervals))
            else:
                groove_strength = 0.1
            
            features['groove_strength'] = float(groove_strength)
            
            # Syncopation level (simplified)
            onset_envelope = librosa.onset.onset_strength(y=audio, sr=self.sample_rate)
            onset_times = librosa.frames_to_time(
                librosa.onset.onset_detect(onset_strength=onset_envelope, sr=self.sample_rate),
                sr=self.sample_rate
            )
            
            # Count onsets between beats (simplified syncopation measure)
            syncopation_count = 0
            for i in range(len(beat_times) - 1):
                beat_start = beat_times[i]
                beat_end = beat_times[i + 1]
                
                # Count onsets in off-beat positions
                off_beat_onsets = [t for t in onset_times 
                                 if beat_start + (beat_end - beat_start) * 0.25 < t < beat_start + (beat_end - beat_start) * 0.75]
                syncopation_count += len(off_beat_onsets)
            
            features['syncopation'] = min(1.0, syncopation_count / max(1, len(beat_times)))
            
            # Rhythmic complexity
            features['complexity'] = min(1.0, len(onset_times) / max(1, len(beat_times)) / 4)
            
            # Tempo stability
            features['tempo_stability'] = tempo_analysis['confidence']
            
            return features
            
        except Exception as e:
            logger.warning("Rhythmic feature extraction failed: %s", e)
            return {
                'groove_strength': 0.5,
                'syncopation': 0.3,
                'complexity': 0.4,
                'tempo_stability': 0.5
            }
    
    def _detect_onset_times(self, audio: np.ndarray) -> List[float]:
        """Detect onset times"""
        
        try:
            onset_envelope = librosa.onset.onset_strength(y=audio, sr=self.sample_rate)
            onset_frames = librosa.onset.onset_detect(onset_strength=onset_envelope, sr=self.sample_rate)
            onset_times = librosa.frames_to_time(onset_frames, sr=self.sample_rate)
            
            return onset_times.tolist()
            
        except Exception as e:
            logger.warning("Onset detection failed: %s", e)
            return []
    
    def _create_default_rhythm(self) -> Dict[str, Any]:
        """Create default rhythm analysis"""
        
        return {
            'tempo': {'bpm': 120.0, 'confidence': 0.1},
            'time_signature': '4/4',
            'beat_pattern': {'pattern_type': 'regular', 'consistency': 0.5},
            'beat_times': [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
            'rhythmic_features': {
                'groove_strength': 0.5,
                'syncopation': 0.3,
                'complexity': 0.4
            },
            'onset_times': []
        }
    
    # Integration and utility methods
    
    def _create_combined_audio(self, tracks: Dict[str, np.ndarray]) -> np.ndarray:
        """Create combined audio from multiple tracks"""
        
        if not tracks:
            return np.array([])
        
        # Find maximum length
        max_length = max(len(audio) for audio in tracks.values())
        
        # Combine tracks
        combined = np.zeros(max_length)
        for audio in tracks.values():
            if len(audio) <= max_length:
                combined[:len(audio)] += audio
            else:
                combined += audio[:max_length]
        
        # Normalize
        if len(tracks) > 1:
            combined = combined / len(tracks)
        
        return combined
    
    def _analyze_integration_quality(self, track_analysis: Dict[str, Any], 
                                   structure_analysis: Dict[str, Any],
                                   harmony_analysis: Dict[str, Any], 
                                   rhythm_analysis: Dict[str, Any]) -> Dict[str, float]:
        """Analyze quality of multi-layer integration"""
        
        try:
            metrics = {}
            
            # Coherence score based on analysis consistency
            coherence_factors = []
            
            # Track role diversity
            roles = [analysis['role'] for analysis in track_analysis.values()]
            role_diversity = len(set(roles)) / max(1, len(roles))
            coherence_factors.append(role_diversity)
            
            # Structure consistency
            structure_confidence = structure_analysis.get('analysis_confidence', 0.5)
            coherence_factors.append(structure_confidence)
            
            # Harmonic consistency
            harmony_confidence = harmony_analysis['key']['confidence']
            coherence_factors.append(harmony_confidence)
            
            # Rhythmic consistency
            rhythm_confidence = rhythm_analysis['tempo']['confidence']
            coherence_factors.append(rhythm_confidence)
            
            metrics['coherence_score'] = float(np.mean(coherence_factors))
            
            # Complexity score
            complexity_factors = []
            complexity_factors.append(len(track_analysis) / 10.0)  # Number of tracks
            complexity_factors.append(len(structure_analysis.get('sections', [])) / 10.0)  # Structural complexity
            complexity_factors.append(harmony_analysis['harmonic_features']['complexity'])  # Harmonic complexity
            complexity_factors.append(rhythm_analysis['rhythmic_features']['complexity'])  # Rhythmic complexity
            
            metrics['complexity_score'] = float(np.mean(complexity_factors))
            
            # Quality assessment
            quality_factors = []
            
            # Average track confidence
            track_confidences = [analysis['confidence'] for analysis in track_analysis.values()]
            quality_factors.append(np.mean(track_confidences))
            
            # Integration consistency
            quality_factors.append(metrics['coherence_score'])
            
            # Feature richness
            quality_factors.append(min(1.0, metrics['complexity_score']))
            
            metrics['quality_score'] = float(np.mean(quality_factors))
            
            return metrics
            
        except Exception as e:
            logger.warning("Integration quality analysis failed: %s", e)
            return {
                'coherence_score': 0.5,
                'complexity_score': 0.5,
                'quality_score': 0.5
            }
    
    def _create_overall_assessment(self, integration_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Create overall assessment of music understanding"""
        
        quality_score = integration_metrics['quality_score']
        
        if quality_score > 0.8:
            assessment = "excellent"
        elif quality_score > 0.6:
            assessment = "good"
        elif quality_score > 0.4:
            assessment = "fair"
        else:
            assessment = "poor"
        
        return {
            'overall_quality': assessment,
            'confidence': quality_score,
            'recommendations': self._generate_recommendations(integration_metrics)
        }
    
    def _generate_recommendations(self, metrics: Dict[str, float]) -> List[str]:
        """Generate recommendations based on analysis"""
        
        recommendations = []
        
        if metrics['coherence_score'] < 0.5:
            recommendations.append("Consider improving track role separation")
        
        if metrics['complexity_score'] < 0.3:
            recommendations.append("Music structure could be more varied")
        
        if metrics['quality_score'] < 0.6:
            recommendations.append("Overall music understanding confidence is low")
        
        if not recommendations:
            recommendations.append("Music understanding analysis is satisfactory")
        
        return recommendations
    
    def _create_default_comprehensive_analysis(self, tracks: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Create default comprehensive analysis"""
        
        # Default track analysis
        track_analysis = {}
        for track_name in tracks.keys():
            track_analysis[track_name] = {
                'role': TrackRole.ACCOMPANIMENT.value,
                'instrument': 'unknown',
                'confidence': 0.1,
                'features': {},
                'spectral_characteristics': {}
            }
        
        return {
            'track_analysis': track_analysis,
            'structure_analysis': self._create_default_structure(np.array([])),
            'harmony_analysis': self._create_default_harmony(),
            'rhythm_analysis': self._create_default_rhythm(),
            'integration_metrics': {
                'coherence_score': 0.1,
                'complexity_score': 0.1,
                'quality_score': 0.1
            },
            'overall_assessment': {
                'overall_quality': 'poor',
                'confidence': 0.1,
                'recommendations': ['Analysis failed - using defaults']
            }
        }