"""
Melody Preservation Algorithm - Core Generation Component

This module implements the core algorithm that ensures generated music maintains
the recognizable characteristics of the original melody while allowing for
style transformations, instrumentation changes, and rhythmic modifications.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
from typing import Dict, List, Tuple, Optional, Any
import logging
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr


class MelodyPreservationEngine:
    """
    Core engine for preserving melody characteristics during music generation
    
    This engine extracts and preserves the essential melodic DNA:
    1. Pitch contour (音高轮廓) - the shape of the melody
    2. Interval relationships (音程关系) - the gaps between notes  
    3. Rhythmic skeleton (节奏骨架) - the timing structure
    4. Phrase boundaries (乐句边界) - the musical sentences
    5. Characteristic notes (特征音符) - the memorable peaks and valleys
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize melody preservation engine
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Core preservation parameters
        self.min_similarity_threshold = self.config.get('min_similarity', 0.8)
        self.contour_weight = self.config.get('contour_weight', 0.4)
        self.interval_weight = self.config.get('interval_weight', 0.3)
        self.rhythm_weight = self.config.get('rhythm_weight', 0.2)
        self.phrase_weight = self.config.get('phrase_weight', 0.1)
        
        # Audio processing parameters
        self.sample_rate = self.config.get('sample_rate', 22050)
        self.hop_length = self.config.get('hop_length', 512)
        self.frame_length = self.config.get('frame_length', 2048)
        
        # Pitch tracking parameters
        self.f0_min = self.config.get('f0_min', 80)
        self.f0_max = self.config.get('f0_max', 2000)
        
        self.logger.info("MelodyPreservationEngine initialized")
        self.logger.info(f"Similarity threshold: {self.min_similarity_threshold}")
    
    def extract_melody_dna(self, audio_data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract the essential DNA of a melody for preservation
        
        Args:
            audio_data: Input audio signal
            
        Returns:
            Dictionary containing melody DNA components
        """
        try:
            # 1. Extract fundamental frequency (F0) track
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio_data,
                fmin=self.f0_min,
                fmax=self.f0_max,
                sr=self.sample_rate,
                frame_length=self.frame_length,
                hop_length=self.hop_length
            )
            
            # Filter out unvoiced frames
            f0_voiced = f0[voiced_flag]
            time_voiced = librosa.frames_to_time(
                np.where(voiced_flag)[0], 
                sr=self.sample_rate, 
                hop_length=self.hop_length
            )
            
            if len(f0_voiced) < 10:
                raise ValueError("Insufficient voiced content for melody extraction")
            
            # 2. Extract pitch contour (normalized)
            pitch_contour = self._extract_pitch_contour(f0_voiced, time_voiced)
            
            # 3. Extract interval sequence
            interval_sequence = self._extract_interval_sequence(f0_voiced)
            
            # 4. Extract rhythmic skeleton
            rhythmic_skeleton = self._extract_rhythmic_skeleton(audio_data, time_voiced)
            
            # 5. Identify phrase boundaries
            phrase_boundaries = self._identify_phrase_boundaries(f0_voiced, time_voiced)
            
            # 6. Extract characteristic notes
            characteristic_notes = self._extract_characteristic_notes(f0_voiced, time_voiced)
            
            # 7. Compute melodic statistics
            melodic_stats = self._compute_melodic_statistics(f0_voiced)
            
            melody_dna = {
                'pitch_contour': pitch_contour,
                'interval_sequence': interval_sequence,
                'rhythmic_skeleton': rhythmic_skeleton,
                'phrase_boundaries': phrase_boundaries,
                'characteristic_notes': characteristic_notes,
                'melodic_stats': melodic_stats,
                'f0_track': f0_voiced,
                'time_track': time_voiced,
                'voiced_flag': voiced_flag,
                'original_length': len(audio_data) / self.sample_rate
            }
            
            self.logger.info(f"Melody DNA extracted: {len(f0_voiced)} voiced frames, "
                           f"{len(phrase_boundaries)} phrases, "
                           f"{len(characteristic_notes)} characteristic notes")
            
            return melody_dna
            
        except Exception as e:
            self.logger.error(f"Failed to extract melody DNA: {e}")
            raise
    
    def _extract_pitch_contour(self, f0_voiced: np.ndarray, time_voiced: np.ndarray) -> np.ndarray:
        """Extract normalized pitch contour"""
        
        # Convert to log scale for perceptual relevance
        log_f0 = np.log2(f0_voiced)
        
        # Normalize to [0, 1] range
        min_pitch = np.min(log_f0)
        max_pitch = np.max(log_f0)
        pitch_range = max_pitch - min_pitch
        
        if pitch_range > 0:
            normalized_contour = (log_f0 - min_pitch) / pitch_range
        else:
            normalized_contour = np.ones_like(log_f0) * 0.5
        
        # Smooth the contour to reduce noise
        from scipy.ndimage import gaussian_filter1d
        smoothed_contour = gaussian_filter1d(normalized_contour, sigma=1.0)
        
        return smoothed_contour
    
    def _extract_interval_sequence(self, f0_voiced: np.ndarray) -> np.ndarray:
        """Extract sequence of melodic intervals (in semitones)"""
        
        if len(f0_voiced) < 2:
            return np.array([])
        
        # Convert to semitones
        semitones = 12 * np.log2(f0_voiced)
        
        # Compute intervals between consecutive notes
        intervals = np.diff(semitones)
        
        # Quantize to nearest semitone for stability
        quantized_intervals = np.round(intervals)
        
        # Clip extreme intervals (likely noise)
        clipped_intervals = np.clip(quantized_intervals, -24, 24)
        
        return clipped_intervals
    
    def _extract_rhythmic_skeleton(self, audio_data: np.ndarray, time_voiced: np.ndarray) -> Dict[str, Any]:
        """Extract rhythmic skeleton information"""
        
        # Extract onset times
        onset_frames = librosa.onset.onset_detect(
            y=audio_data,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            units='frames'
        )
        
        onset_times = librosa.frames_to_time(
            onset_frames, sr=self.sample_rate, hop_length=self.hop_length
        )
        
        # Extract tempo and beat
        tempo, beat_frames = librosa.beat.beat_track(
            y=audio_data,
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        
        beat_times = librosa.frames_to_time(
            beat_frames, sr=self.sample_rate, hop_length=self.hop_length
        )
        
        # Compute inter-onset intervals
        if len(onset_times) > 1:
            ioi = np.diff(onset_times)  # Inter-Onset Intervals
        else:
            ioi = np.array([])
        
        # Compute rhythm pattern relative to beats
        rhythm_pattern = self._quantize_to_beats(onset_times, beat_times)
        
        return {
            'tempo': tempo,
            'onset_times': onset_times,
            'beat_times': beat_times,
            'inter_onset_intervals': ioi,
            'rhythm_pattern': rhythm_pattern,
            'note_density': len(onset_times) / (len(audio_data) / self.sample_rate)
        }
    
    def _identify_phrase_boundaries(self, f0_voiced: np.ndarray, time_voiced: np.ndarray) -> List[float]:
        """Identify phrase boundaries in the melody"""
        
        if len(f0_voiced) < 20:
            return [time_voiced[0], time_voiced[-1]]
        
        # Method 1: Detect large pitch jumps
        semitones = 12 * np.log2(f0_voiced)
        pitch_diff = np.abs(np.diff(semitones))
        jump_threshold = np.percentile(pitch_diff, 85)  # Top 15% of jumps
        jump_indices = np.where(pitch_diff > jump_threshold)[0] + 1
        
        # Method 2: Detect long pauses (from rhythmic analysis)
        # This would need the full audio, simplified for now
        
        # Method 3: Detect contour direction changes
        contour_diff = np.diff(semitones)
        direction_changes = np.where(np.diff(np.sign(contour_diff)) != 0)[0] + 1
        
        # Combine methods and filter
        boundary_candidates = np.concatenate([jump_indices, direction_changes])
        boundary_candidates = np.unique(boundary_candidates)
        
        # Convert to time
        boundary_times = [time_voiced[0]]  # Start boundary
        for idx in boundary_candidates:
            if 0 < idx < len(time_voiced):
                boundary_times.append(time_voiced[idx])
        boundary_times.append(time_voiced[-1])  # End boundary
        
        # Remove boundaries that are too close together (< 1 second)
        filtered_boundaries = [boundary_times[0]]
        for t in boundary_times[1:]:
            if t - filtered_boundaries[-1] > 1.0:
                filtered_boundaries.append(t)
        
        return filtered_boundaries
    
    def _extract_characteristic_notes(self, f0_voiced: np.ndarray, time_voiced: np.ndarray) -> List[Dict[str, float]]:
        """Extract characteristic notes (peaks, valleys, and sustained notes)"""
        
        characteristic_notes = []
        
        if len(f0_voiced) < 5:
            return characteristic_notes
        
        semitones = 12 * np.log2(f0_voiced)
        
        # Find local maxima (peaks)
        from scipy.signal import find_peaks
        peak_indices, peak_props = find_peaks(
            semitones, 
            height=np.percentile(semitones, 60),  # Above 60th percentile
            distance=5  # At least 5 frames apart
        )
        
        # Find local minima (valleys)
        valley_indices, valley_props = find_peaks(
            -semitones,
            height=-np.percentile(semitones, 40),  # Below 40th percentile
            distance=5
        )
        
        # Add peaks
        for idx in peak_indices:
            characteristic_notes.append({
                'type': 'peak',
                'time': time_voiced[idx],
                'pitch': f0_voiced[idx],
                'semitone': semitones[idx],
                'prominence': peak_props['prominences'][list(peak_indices).index(idx)] if 'prominences' in peak_props else 0
            })
        
        # Add valleys
        for idx in valley_indices:
            characteristic_notes.append({
                'type': 'valley',
                'time': time_voiced[idx],
                'pitch': f0_voiced[idx],
                'semitone': semitones[idx],
                'prominence': valley_props['prominences'][list(valley_indices).index(idx)] if 'prominences' in valley_props else 0
            })
        
        # Find sustained notes (notes held for longer duration)
        sustained_threshold = 0.5  # 0.5 seconds
        current_pitch = None
        start_time = None
        
        for i, (pitch, time) in enumerate(zip(f0_voiced, time_voiced)):
            if current_pitch is None:
                current_pitch = pitch
                start_time = time
            elif abs(12 * np.log2(pitch / current_pitch)) < 0.5:  # Within half semitone
                continue
            else:
                # Pitch changed, check if previous was sustained
                duration = time - start_time
                if duration > sustained_threshold:
                    characteristic_notes.append({
                        'type': 'sustained',
                        'time': start_time,
                        'pitch': current_pitch,
                        'semitone': 12 * np.log2(current_pitch),
                        'duration': duration
                    })
                current_pitch = pitch
                start_time = time
        
        # Sort by time
        characteristic_notes.sort(key=lambda x: x['time'])
        
        return characteristic_notes
    
    def _compute_melodic_statistics(self, f0_voiced: np.ndarray) -> Dict[str, float]:
        """Compute statistical characteristics of the melody"""
        
        if len(f0_voiced) == 0:
            return {}
        
        semitones = 12 * np.log2(f0_voiced)
        
        return {
            'pitch_range': np.max(semitones) - np.min(semitones),
            'pitch_mean': np.mean(semitones),
            'pitch_std': np.std(semitones),
            'pitch_median': np.median(semitones),
            'pitch_iqr': np.percentile(semitones, 75) - np.percentile(semitones, 25),
            'pitch_skewness': self._compute_skewness(semitones),
            'pitch_kurtosis': self._compute_kurtosis(semitones),
            'num_voiced_frames': len(f0_voiced)
        }
    
    def _compute_skewness(self, data: np.ndarray) -> float:
        """Compute skewness of data"""
        if len(data) < 3:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    def _compute_kurtosis(self, data: np.ndarray) -> float:
        """Compute kurtosis of data"""
        if len(data) < 4:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _quantize_to_beats(self, onset_times: np.ndarray, beat_times: np.ndarray) -> np.ndarray:
        """Quantize onset times to beat grid"""
        if len(beat_times) < 2:
            return onset_times
        
        beat_interval = np.mean(np.diff(beat_times))
        quantized = []
        
        for onset in onset_times:
            # Find closest beat
            distances = np.abs(beat_times - onset)
            closest_beat_idx = np.argmin(distances)
            closest_beat_time = beat_times[closest_beat_idx]
            
            # Quantize to beat subdivision
            subdivision = (onset - closest_beat_time) / beat_interval
            quantized_subdivision = np.round(subdivision * 4) / 4  # Quantize to 16th notes
            quantized_time = closest_beat_time + quantized_subdivision * beat_interval
            
            quantized.append(quantized_time)
        
        return np.array(quantized)
    
    def compute_melody_similarity(self, dna1: Dict[str, Any], dna2: Dict[str, Any]) -> float:
        """
        Compute similarity between two melody DNAs
        
        Args:
            dna1: First melody DNA
            dna2: Second melody DNA
            
        Returns:
            Similarity score [0, 1] where 1 = identical
        """
        similarities = {}
        
        # 1. Pitch contour similarity
        similarities['contour'] = self._compute_contour_similarity(
            dna1['pitch_contour'], dna2['pitch_contour']
        )
        
        # 2. Interval sequence similarity
        similarities['intervals'] = self._compute_interval_similarity(
            dna1['interval_sequence'], dna2['interval_sequence']
        )
        
        # 3. Rhythmic similarity
        similarities['rhythm'] = self._compute_rhythm_similarity(
            dna1['rhythmic_skeleton'], dna2['rhythmic_skeleton']
        )
        
        # 4. Phrase structure similarity
        similarities['phrases'] = self._compute_phrase_similarity(
            dna1['phrase_boundaries'], dna2['phrase_boundaries']
        )
        
        # Weighted combination
        total_similarity = (
            self.contour_weight * similarities['contour'] +
            self.interval_weight * similarities['intervals'] +
            self.rhythm_weight * similarities['rhythm'] +
            self.phrase_weight * similarities['phrases']
        )
        
        # Ensure scalar output
        if hasattr(total_similarity, 'item'):
            total_similarity = total_similarity.item()
        elif isinstance(total_similarity, np.ndarray):
            total_similarity = float(total_similarity)
        
        self.logger.debug(f"Similarity components: {similarities}")
        self.logger.debug(f"Total similarity: {total_similarity:.3f}")
        
        return float(total_similarity)
    
    def _compute_contour_similarity(self, contour1: np.ndarray, contour2: np.ndarray) -> float:
        """Compute pitch contour similarity using multiple metrics"""
        
        if len(contour1) == 0 or len(contour2) == 0:
            return 0.0
        
        # Resample to same length for comparison
        if len(contour1) != len(contour2):
            from scipy.interpolate import interp1d
            target_length = min(len(contour1), len(contour2), 100)
            
            if len(contour1) > 1:
                f1 = interp1d(np.linspace(0, 1, len(contour1)), contour1)
                contour1_resampled = f1(np.linspace(0, 1, target_length))
            else:
                contour1_resampled = np.full(target_length, contour1[0])
            
            if len(contour2) > 1:
                f2 = interp1d(np.linspace(0, 1, len(contour2)), contour2)
                contour2_resampled = f2(np.linspace(0, 1, target_length))
            else:
                contour2_resampled = np.full(target_length, contour2[0])
        else:
            contour1_resampled = contour1
            contour2_resampled = contour2
        
        # Multi-metric similarity computation
        similarities = []
        
        # 1. Correlation-based similarity
        try:
            correlation, _ = pearsonr(contour1_resampled, contour2_resampled)
            corr_sim = (correlation + 1) / 2 if not np.isnan(correlation) else 0.0
            similarities.append(corr_sim)
        except:
            similarities.append(0.0)
        
        # 2. Dynamic Time Warping distance
        try:
            from scipy.spatial.distance import euclidean
            # Simplified DTW using euclidean distance
            dtw_distance = euclidean(contour1_resampled, contour2_resampled)
            # Normalize by contour length and convert to similarity
            dtw_sim = 1.0 / (1.0 + dtw_distance / len(contour1_resampled))
            similarities.append(dtw_sim)
        except:
            similarities.append(0.0)
        
        # 3. Shape-based similarity (direction changes)
        try:
            # Compute first derivatives (direction)
            dir1 = np.diff(contour1_resampled)
            dir2 = np.diff(contour2_resampled)
            
            # Compare direction patterns
            if len(dir1) > 0 and len(dir2) > 0:
                dir_corr, _ = pearsonr(dir1, dir2)
                dir_sim = (dir_corr + 1) / 2 if not np.isnan(dir_corr) else 0.0
            else:
                dir_sim = 0.0
            similarities.append(dir_sim)
        except:
            similarities.append(0.0)
        
        # 4. Range and variation similarity
        try:
            range1 = np.max(contour1_resampled) - np.min(contour1_resampled)
            range2 = np.max(contour2_resampled) - np.min(contour2_resampled)
            range_sim = 1.0 - abs(range1 - range2) / (max(range1, range2) + 1e-8)
            
            var1 = np.var(contour1_resampled)
            var2 = np.var(contour2_resampled)
            var_sim = 1.0 - abs(var1 - var2) / (max(var1, var2) + 1e-8)
            
            stat_sim = (range_sim + var_sim) / 2
            similarities.append(stat_sim)
        except:
            similarities.append(0.0)
        
        # Weighted combination with emphasis on correlation and DTW
        weights = [0.4, 0.3, 0.2, 0.1]  # Emphasize correlation and DTW
        final_similarity = sum(w * s for w, s in zip(weights, similarities))
        
        return max(0.0, min(1.0, final_similarity))
    
    def _compute_interval_similarity(self, intervals1: np.ndarray, intervals2: np.ndarray) -> float:
        """Compute interval sequence similarity with multiple metrics"""
        
        if len(intervals1) == 0 or len(intervals2) == 0:
            return 0.0
        
        similarities = []
        
        # 1. Histogram-based similarity (distribution of intervals)
        try:
            interval_range = np.arange(-24, 26)  # -2 to +2 octaves, bins
            hist1 = np.histogram(intervals1, bins=interval_range)[0]
            hist2 = np.histogram(intervals2, bins=interval_range)[0]
            
            # Normalize histograms
            hist1 = hist1 / (np.sum(hist1) + 1e-8)
            hist2 = hist2 / (np.sum(hist2) + 1e-8)
            
            # Compute histogram intersection
            intersection = np.sum(np.minimum(hist1, hist2))
            similarities.append(intersection)
        except:
            similarities.append(0.0)
        
        # 2. Sequential pattern similarity (order matters)
        try:
            # Use longest common subsequence approach
            min_len = min(len(intervals1), len(intervals2))
            if min_len > 0:
                # Compare first min_len intervals directly
                comparison_len = min(min_len, 50)  # Limit for performance
                seq1 = intervals1[:comparison_len]
                seq2 = intervals2[:comparison_len]
                
                # Count exact matches in sequence
                exact_matches = np.sum(np.abs(seq1 - seq2) < 0.5)  # Within 0.5 semitone
                sequence_sim = exact_matches / comparison_len
                similarities.append(sequence_sim)
            else:
                similarities.append(0.0)
        except:
            similarities.append(0.0)
        
        # 3. Statistical similarity (mean, variance of intervals)
        try:
            mean1, mean2 = np.mean(intervals1), np.mean(intervals2)
            var1, var2 = np.var(intervals1), np.var(intervals2)
            
            # Mean interval similarity
            mean_diff = abs(mean1 - mean2)
            mean_sim = 1.0 / (1.0 + mean_diff / 12.0)  # Normalize by octave
            
            # Variance similarity
            var_diff = abs(var1 - var2)
            var_sim = 1.0 / (1.0 + var_diff / 144.0)  # Normalize by octave^2
            
            stat_sim = (mean_sim + var_sim) / 2
            similarities.append(stat_sim)
        except:
            similarities.append(0.0)
        
        # 4. Contour direction similarity (up/down patterns)
        try:
            dir1 = np.sign(intervals1)  # +1 for up, -1 for down, 0 for same
            dir2 = np.sign(intervals2)
            
            min_len = min(len(dir1), len(dir2))
            if min_len > 0:
                dir_matches = np.sum(dir1[:min_len] == dir2[:min_len])
                dir_sim = dir_matches / min_len
                similarities.append(dir_sim)
            else:
                similarities.append(0.0)
        except:
            similarities.append(0.0)
        
        # Weighted combination
        weights = [0.4, 0.3, 0.2, 0.1]  # Histogram, sequence, stats, direction
        final_similarity = sum(w * s for w, s in zip(weights, similarities))
        
        return max(0.0, min(1.0, final_similarity))
    
    def _compute_rhythm_similarity(self, rhythm1: Dict[str, Any], rhythm2: Dict[str, Any]) -> float:
        """Compute rhythmic similarity"""
        
        # Compare tempo
        tempo1 = rhythm1.get('tempo', 120)
        tempo2 = rhythm2.get('tempo', 120)
        tempo_sim = 1.0 - abs(tempo1 - tempo2) / max(tempo1, tempo2)
        tempo_sim = max(0, tempo_sim)
        
        # Compare note density
        density1 = rhythm1.get('note_density', 1.0)
        density2 = rhythm2.get('note_density', 1.0)
        density_sim = 1.0 - abs(density1 - density2) / max(density1, density2)
        density_sim = max(0, density_sim)
        
        # Compare rhythm patterns (simplified)
        pattern_sim = 0.5  # Placeholder for more complex rhythm pattern comparison
        
        # Weighted combination
        rhythm_similarity = 0.5 * tempo_sim + 0.3 * density_sim + 0.2 * pattern_sim
        
        # Ensure scalar output
        if hasattr(rhythm_similarity, 'item'):
            rhythm_similarity = rhythm_similarity.item()
        elif isinstance(rhythm_similarity, np.ndarray):
            rhythm_similarity = float(rhythm_similarity)
        
        return float(rhythm_similarity)
    
    def _compute_phrase_similarity(self, phrases1: List[float], phrases2: List[float]) -> float:
        """Compute phrase structure similarity"""
        
        if len(phrases1) < 2 or len(phrases2) < 2:
            return 0.5  # Neutral similarity for undefined phrase structure
        
        # Normalize phrase lengths
        lengths1 = np.diff(phrases1)
        lengths2 = np.diff(phrases2)
        
        # Compare number of phrases
        num_phrases_sim = 1.0 - abs(len(lengths1) - len(lengths2)) / max(len(lengths1), len(lengths2))
        
        # Compare average phrase length
        avg_length_sim = 1.0 - abs(np.mean(lengths1) - np.mean(lengths2)) / max(np.mean(lengths1), np.mean(lengths2))
        
        # Combine
        phrase_similarity = 0.6 * num_phrases_sim + 0.4 * avg_length_sim
        
        return phrase_similarity
    
    def validate_preservation(self, original_audio: np.ndarray, generated_audio: np.ndarray) -> Dict[str, Any]:
        """
        Validate that generated audio preserves original melody
        
        Args:
            original_audio: Original audio signal
            generated_audio: Generated audio signal
            
        Returns:
            Dictionary with validation results
        """
        try:
            # Extract DNA from both
            original_dna = self.extract_melody_dna(original_audio)
            generated_dna = self.extract_melody_dna(generated_audio)
            
            # Compute similarity
            similarity = self.compute_melody_similarity(original_dna, generated_dna)
            
            # Check if meets threshold
            preservation_passed = similarity >= self.min_similarity_threshold
            
            # Detailed component analysis
            component_analysis = {
                'contour_similarity': self._compute_contour_similarity(
                    original_dna['pitch_contour'], generated_dna['pitch_contour']
                ),
                'interval_similarity': self._compute_interval_similarity(
                    original_dna['interval_sequence'], generated_dna['interval_sequence']
                ),
                'rhythm_similarity': self._compute_rhythm_similarity(
                    original_dna['rhythmic_skeleton'], generated_dna['rhythmic_skeleton']
                ),
                'phrase_similarity': self._compute_phrase_similarity(
                    original_dna['phrase_boundaries'], generated_dna['phrase_boundaries']
                )
            }
            
            return {
                'overall_similarity': similarity,
                'preservation_passed': preservation_passed,
                'threshold': self.min_similarity_threshold,
                'component_analysis': component_analysis,
                'original_dna': original_dna,
                'generated_dna': generated_dna,
                'recommendations': self._generate_preservation_recommendations(component_analysis)
            }
            
        except Exception as e:
            self.logger.error(f"Preservation validation failed: {e}")
            return {
                'overall_similarity': 0.0,
                'preservation_passed': False,
                'error': str(e),
                'component_analysis': {},
                'recommendations': ['Error occurred during validation']
            }
    
    def _generate_preservation_recommendations(self, component_analysis: Dict[str, float]) -> List[str]:
        """Generate recommendations for improving melody preservation"""
        
        recommendations = []
        threshold = 0.7
        
        if component_analysis['contour_similarity'] < threshold:
            recommendations.append("Improve pitch contour preservation - focus on melodic shape")
        
        if component_analysis['interval_similarity'] < threshold:
            recommendations.append("Preserve interval relationships - maintain melodic jumps and steps")
        
        if component_analysis['rhythm_similarity'] < threshold:
            recommendations.append("Maintain rhythmic characteristics - preserve tempo and note timing")
        
        if component_analysis['phrase_similarity'] < threshold:
            recommendations.append("Preserve phrase structure - maintain musical sentence boundaries")
        
        if not recommendations:
            recommendations.append("Melody preservation is excellent!")
        
        return recommendations