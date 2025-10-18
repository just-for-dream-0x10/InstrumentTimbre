"""
Enhanced Melody Similarity Calculator - Week 3 Development Task

This module provides improved similarity calculation methods that better
distinguish between melodically similar and different music pieces.
"""

import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
from scipy.signal import correlate
from typing import Dict, List, Tuple, Optional, Any
import logging


class EnhancedSimilarityCalculator:
    """
    Enhanced similarity calculator with multiple advanced metrics
    
    Improves upon basic similarity by incorporating:
    1. Multi-scale contour analysis
    2. Interval pattern matching with weights
    3. Rhythmic micro-timing analysis
    4. Perceptual distance metrics
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Similarity calculation parameters
        self.contour_scales = self.config.get('contour_scales', [1, 2, 4, 8])  # Multi-scale analysis
        self.interval_weight_decay = self.config.get('interval_weight_decay', 0.9)  # Recent intervals more important
        self.min_pattern_length = self.config.get('min_pattern_length', 3)
        self.max_pattern_length = self.config.get('max_pattern_length', 8)
        
        self.logger.info("EnhancedSimilarityCalculator initialized")
    
    def compute_enhanced_melody_similarity(self, dna1: Dict[str, Any], dna2: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute enhanced melody similarity with detailed component analysis
        
        Args:
            dna1: First melody DNA
            dna2: Second melody DNA
            
        Returns:
            Dictionary with similarity scores and components
        """
        similarities = {}
        
        # 1. Multi-scale contour similarity
        similarities.update(self._compute_multiscale_contour_similarity(
            dna1['pitch_contour'], dna2['pitch_contour']
        ))
        
        # 2. Weighted interval similarity
        similarities.update(self._compute_weighted_interval_similarity(
            dna1['interval_sequence'], dna2['interval_sequence']
        ))
        
        # 3. Pattern-based similarity
        similarities.update(self._compute_pattern_similarity(
            dna1['interval_sequence'], dna2['interval_sequence']
        ))
        
        # 4. Rhythmic micro-similarity
        similarities.update(self._compute_rhythmic_microsimilarity(
            dna1['rhythmic_skeleton'], dna2['rhythmic_skeleton']
        ))
        
        # 5. Structural similarity
        similarities.update(self._compute_structural_similarity(dna1, dna2))
        
        # 6. Perceptual distance
        similarities['perceptual_distance'] = self._compute_perceptual_distance(dna1, dna2)
        
        # 7. Overall enhanced similarity
        similarities['enhanced_overall'] = self._compute_weighted_overall_similarity(similarities)
        
        return similarities
    
    def _compute_multiscale_contour_similarity(self, contour1: np.ndarray, contour2: np.ndarray) -> Dict[str, float]:
        """Compute contour similarity at multiple time scales"""
        
        if len(contour1) == 0 or len(contour2) == 0:
            return {'contour_fine': 0.0, 'contour_coarse': 0.0, 'contour_multiscale': 0.0}
        
        scale_similarities = []
        
        for scale in self.contour_scales:
            # Downsample contours by averaging
            if len(contour1) >= scale and len(contour2) >= scale:
                # Reshape and average over scale-sized windows
                len1_scaled = (len(contour1) // scale) * scale
                len2_scaled = (len(contour2) // scale) * scale
                
                c1_scaled = contour1[:len1_scaled].reshape(-1, scale).mean(axis=1)
                c2_scaled = contour2[:len2_scaled].reshape(-1, scale).mean(axis=1)
                
                # Resample to same length
                if len(c1_scaled) != len(c2_scaled):
                    from scipy.interpolate import interp1d
                    target_len = min(len(c1_scaled), len(c2_scaled))
                    
                    if len(c1_scaled) > 1:
                        f1 = interp1d(np.linspace(0, 1, len(c1_scaled)), c1_scaled)
                        c1_scaled = f1(np.linspace(0, 1, target_len))
                    
                    if len(c2_scaled) > 1:
                        f2 = interp1d(np.linspace(0, 1, len(c2_scaled)), c2_scaled)
                        c2_scaled = f2(np.linspace(0, 1, target_len))
                
                # Compute correlation at this scale
                if len(c1_scaled) > 1 and len(c2_scaled) > 1:
                    try:
                        correlation, _ = pearsonr(c1_scaled, c2_scaled)
                        scale_sim = (correlation + 1) / 2 if not np.isnan(correlation) else 0.0
                    except:
                        scale_sim = 0.0
                else:
                    scale_sim = 0.0
                
                scale_similarities.append(scale_sim)
            else:
                scale_similarities.append(0.0)
        
        # Weight scales (fine details more important)
        weights = np.array([1.0, 0.8, 0.6, 0.4])[:len(scale_similarities)]
        weights = weights / np.sum(weights)
        
        multiscale_sim = np.sum(np.array(scale_similarities) * weights)
        
        return {
            'contour_fine': scale_similarities[0] if len(scale_similarities) > 0 else 0.0,
            'contour_coarse': scale_similarities[-1] if len(scale_similarities) > 0 else 0.0,
            'contour_multiscale': multiscale_sim
        }
    
    def _compute_weighted_interval_similarity(self, intervals1: np.ndarray, intervals2: np.ndarray) -> Dict[str, float]:
        """Compute interval similarity with temporal weighting"""
        
        if len(intervals1) == 0 or len(intervals2) == 0:
            return {'interval_weighted': 0.0, 'interval_recent': 0.0, 'interval_pattern': 0.0}
        
        # 1. Recent intervals are more important (recency bias)
        min_len = min(len(intervals1), len(intervals2))
        if min_len > 0:
            # Create exponential decay weights (recent = more important)
            weights = np.power(self.interval_weight_decay, np.arange(min_len))[::-1]  # Reverse for recent emphasis
            weights = weights / np.sum(weights)  # Normalize
            
            # Compare intervals with weights
            int1_subset = intervals1[:min_len]
            int2_subset = intervals2[:min_len]
            
            # Weighted similarity computation
            diff = np.abs(int1_subset - int2_subset)
            weighted_diff = np.sum(diff * weights)
            weighted_sim = 1.0 / (1.0 + weighted_diff / 12.0)  # Normalize by octave
            
            # Recent intervals only (last 25%)
            recent_len = max(1, min_len // 4)
            recent_diff = np.mean(np.abs(int1_subset[-recent_len:] - int2_subset[-recent_len:]))
            recent_sim = 1.0 / (1.0 + recent_diff / 12.0)
        else:
            weighted_sim = 0.0
            recent_sim = 0.0
        
        # 2. Pattern-based similarity (longer sequences)
        pattern_sim = self._compute_interval_pattern_similarity(intervals1, intervals2)
        
        return {
            'interval_weighted': weighted_sim,
            'interval_recent': recent_sim,
            'interval_pattern': pattern_sim
        }
    
    def _compute_interval_pattern_similarity(self, intervals1: np.ndarray, intervals2: np.ndarray) -> float:
        """Compute similarity based on interval patterns"""
        
        if len(intervals1) < self.min_pattern_length or len(intervals2) < self.min_pattern_length:
            return 0.0
        
        similarities = []
        
        # Look for matching patterns of different lengths
        for pattern_len in range(self.min_pattern_length, min(self.max_pattern_length, len(intervals1), len(intervals2))):
            # Extract all patterns of this length from both sequences
            patterns1 = []
            patterns2 = []
            
            for i in range(len(intervals1) - pattern_len + 1):
                patterns1.append(intervals1[i:i+pattern_len])
            
            for i in range(len(intervals2) - pattern_len + 1):
                patterns2.append(intervals2[i:i+pattern_len])
            
            # Find best matches between patterns
            if patterns1 and patterns2:
                max_similarity = 0.0
                for p1 in patterns1:
                    for p2 in patterns2:
                        # Compute pattern similarity
                        diff = np.mean(np.abs(p1 - p2))
                        sim = 1.0 / (1.0 + diff / 12.0)
                        max_similarity = max(max_similarity, sim)
                
                similarities.append(max_similarity)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _compute_pattern_similarity(self, intervals1: np.ndarray, intervals2: np.ndarray) -> Dict[str, float]:
        """Compute pattern-based similarity metrics"""
        
        # 1. N-gram similarity (sequential patterns)
        ngram_sim = self._compute_ngram_similarity(intervals1, intervals2)
        
        # 2. Motif similarity (repeated patterns)
        motif_sim = self._compute_motif_similarity(intervals1, intervals2)
        
        # 3. Contour pattern similarity (up/down patterns)
        contour_pattern_sim = self._compute_contour_pattern_similarity(intervals1, intervals2)
        
        return {
            'pattern_ngram': ngram_sim,
            'pattern_motif': motif_sim,
            'pattern_contour': contour_pattern_sim
        }
    
    def _compute_ngram_similarity(self, intervals1: np.ndarray, intervals2: np.ndarray) -> float:
        """Compute similarity based on n-gram patterns"""
        
        if len(intervals1) < 3 or len(intervals2) < 3:
            return 0.0
        
        # Extract trigrams (3-note patterns)
        trigrams1 = set()
        trigrams2 = set()
        
        for i in range(len(intervals1) - 2):
            # Quantize intervals to semitones for exact matching
            trigram = tuple(np.round(intervals1[i:i+3]).astype(int).tolist())
            trigrams1.add(trigram)
        
        for i in range(len(intervals2) - 2):
            trigram = tuple(np.round(intervals2[i:i+3]).astype(int).tolist())
            trigrams2.add(trigram)
        
        # Compute Jaccard similarity
        if len(trigrams1) == 0 and len(trigrams2) == 0:
            return 1.0
        elif len(trigrams1) == 0 or len(trigrams2) == 0:
            return 0.0
        else:
            intersection = len(trigrams1.intersection(trigrams2))
            union = len(trigrams1.union(trigrams2))
            return intersection / union
    
    def _compute_motif_similarity(self, intervals1: np.ndarray, intervals2: np.ndarray) -> float:
        """Compute similarity based on repeated motifs"""
        
        # Find repeated patterns in each sequence
        motifs1 = self._find_repeated_patterns(intervals1)
        motifs2 = self._find_repeated_patterns(intervals2)
        
        if not motifs1 or not motifs2:
            return 0.0
        
        # Compare motifs between sequences
        similarities = []
        for m1 in motifs1:
            for m2 in motifs2:
                if len(m1) == len(m2):
                    diff = np.mean(np.abs(np.array(m1) - np.array(m2)))
                    sim = 1.0 / (1.0 + diff / 12.0)
                    similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _find_repeated_patterns(self, intervals: np.ndarray, min_repeats: int = 2) -> List[List[float]]:
        """Find patterns that repeat in the interval sequence"""
        
        repeated_patterns = []
        
        for length in range(2, min(8, len(intervals) // 2)):
            for start in range(len(intervals) - length):
                pattern = intervals[start:start+length]
                
                # Look for repetitions
                repetitions = 0
                for search_start in range(start + length, len(intervals) - length + 1):
                    candidate = intervals[search_start:search_start+length]
                    
                    # Check if similar (within 1 semitone)
                    if np.all(np.abs(pattern - candidate) <= 1.0):
                        repetitions += 1
                
                if repetitions >= min_repeats - 1:  # Pattern + repetitions
                    repeated_patterns.append(pattern.tolist())
        
        return repeated_patterns
    
    def _compute_contour_pattern_similarity(self, intervals1: np.ndarray, intervals2: np.ndarray) -> float:
        """Compute similarity of contour patterns (direction changes)"""
        
        if len(intervals1) < 3 or len(intervals2) < 3:
            return 0.0
        
        # Convert to direction patterns
        directions1 = np.sign(intervals1)  # +1 up, -1 down, 0 same
        directions2 = np.sign(intervals2)
        
        # Extract direction change patterns
        changes1 = []
        changes2 = []
        
        for i in range(len(directions1) - 1):
            if directions1[i] != directions1[i+1]:
                changes1.append((directions1[i], directions1[i+1]))
        
        for i in range(len(directions2) - 1):
            if directions2[i] != directions2[i+1]:
                changes2.append((directions2[i], directions2[i+1]))
        
        # Compare direction change patterns
        if not changes1 or not changes2:
            return 0.5  # Neutral similarity
        
        # Count matching direction changes
        matches = 0
        total_comparisons = min(len(changes1), len(changes2))
        
        for i in range(total_comparisons):
            if changes1[i] == changes2[i]:
                matches += 1
        
        return matches / total_comparisons if total_comparisons > 0 else 0.0
    
    def _compute_rhythmic_microsimilarity(self, rhythm1: Dict[str, Any], rhythm2: Dict[str, Any]) -> Dict[str, float]:
        """Compute detailed rhythmic similarity"""
        
        # 1. Micro-timing similarity (onset precision)
        timing_sim = self._compute_timing_similarity(rhythm1.get('onset_times', []), rhythm2.get('onset_times', []))
        
        # 2. Beat alignment similarity
        beat_sim = self._compute_beat_similarity(rhythm1.get('beat_times', []), rhythm2.get('beat_times', []))
        
        # 3. Groove similarity (rhythmic feel)
        groove_sim = self._compute_groove_similarity(rhythm1, rhythm2)
        
        return {
            'rhythm_timing': timing_sim,
            'rhythm_beat': beat_sim,
            'rhythm_groove': groove_sim
        }
    
    def _compute_timing_similarity(self, onsets1: List[float], onsets2: List[float]) -> float:
        """Compare micro-timing of onsets"""
        
        if len(onsets1) < 2 or len(onsets2) < 2:
            return 0.0
        
        # Normalize timing to same duration
        duration1 = onsets1[-1] - onsets1[0]
        duration2 = onsets2[-1] - onsets2[0]
        
        if duration1 <= 0 or duration2 <= 0:
            return 0.0
        
        # Normalize to [0, 1] range
        norm_onsets1 = [(t - onsets1[0]) / duration1 for t in onsets1]
        norm_onsets2 = [(t - onsets2[0]) / duration2 for t in onsets2]
        
        # Compare inter-onset intervals
        ioi1 = np.diff(norm_onsets1)
        ioi2 = np.diff(norm_onsets2)
        
        # Compute similarity of timing patterns
        min_len = min(len(ioi1), len(ioi2))
        if min_len > 0:
            timing_diff = np.mean(np.abs(ioi1[:min_len] - ioi2[:min_len]))
            timing_sim = 1.0 / (1.0 + timing_diff * 10)  # Scale factor for timing sensitivity
        else:
            timing_sim = 0.0
        
        return timing_sim
    
    def _compute_beat_similarity(self, beats1: List[float], beats2: List[float]) -> float:
        """Compare beat structure"""
        
        if len(beats1) < 2 or len(beats2) < 2:
            return 0.0
        
        # Compare beat intervals (tempo stability)
        intervals1 = np.diff(beats1)
        intervals2 = np.diff(beats2)
        
        # Coefficient of variation for tempo stability
        cv1 = np.std(intervals1) / np.mean(intervals1) if np.mean(intervals1) > 0 else 1.0
        cv2 = np.std(intervals2) / np.mean(intervals2) if np.mean(intervals2) > 0 else 1.0
        
        # Similarity based on tempo stability
        stability_sim = 1.0 - abs(cv1 - cv2)
        
        return max(0.0, stability_sim)
    
    def _compute_groove_similarity(self, rhythm1: Dict[str, Any], rhythm2: Dict[str, Any]) -> float:
        """Compare rhythmic groove characteristics"""
        
        # Simple groove similarity based on note density and tempo
        density1 = rhythm1.get('note_density', 0)
        density2 = rhythm2.get('note_density', 0)
        
        tempo1 = rhythm1.get('tempo', 120)
        tempo2 = rhythm2.get('tempo', 120)
        
        # Density similarity
        max_density = max(density1, density2, 1.0)
        density_sim = 1.0 - abs(density1 - density2) / max_density
        
        # Tempo similarity
        max_tempo = max(tempo1, tempo2, 1.0)
        tempo_sim = 1.0 - abs(tempo1 - tempo2) / max_tempo
        
        return (density_sim + tempo_sim) / 2
    
    def _compute_structural_similarity(self, dna1: Dict[str, Any], dna2: Dict[str, Any]) -> Dict[str, float]:
        """Compute structural similarity metrics"""
        
        # 1. Phrase structure similarity
        phrases1 = dna1.get('phrase_boundaries', [])
        phrases2 = dna2.get('phrase_boundaries', [])
        phrase_sim = self._compare_phrase_structure(phrases1, phrases2)
        
        # 2. Characteristic notes similarity
        char1 = dna1.get('characteristic_notes', [])
        char2 = dna2.get('characteristic_notes', [])
        char_sim = self._compare_characteristic_notes(char1, char2)
        
        # 3. Statistical similarity
        stats1 = dna1.get('melodic_stats', {})
        stats2 = dna2.get('melodic_stats', {})
        stats_sim = self._compare_melodic_stats(stats1, stats2)
        
        return {
            'structure_phrases': phrase_sim,
            'structure_characteristic': char_sim,
            'structure_statistics': stats_sim
        }
    
    def _compare_phrase_structure(self, phrases1: List[float], phrases2: List[float]) -> float:
        """Compare phrase boundary structures"""
        
        if len(phrases1) < 2 or len(phrases2) < 2:
            return 0.5
        
        # Compare number of phrases
        count_sim = 1.0 - abs(len(phrases1) - len(phrases2)) / max(len(phrases1), len(phrases2))
        
        # Compare phrase length distributions
        lengths1 = np.diff(phrases1)
        lengths2 = np.diff(phrases2)
        
        if len(lengths1) > 0 and len(lengths2) > 0:
            # Compare average phrase length
            avg_sim = 1.0 - abs(np.mean(lengths1) - np.mean(lengths2)) / max(np.mean(lengths1), np.mean(lengths2))
            
            # Compare phrase length variability
            var_sim = 1.0 - abs(np.std(lengths1) - np.std(lengths2)) / max(np.std(lengths1), np.std(lengths2), 1.0)
            
            length_sim = (avg_sim + var_sim) / 2
        else:
            length_sim = 0.0
        
        return (count_sim + length_sim) / 2
    
    def _compare_characteristic_notes(self, char1: List[Dict], char2: List[Dict]) -> float:
        """Compare characteristic notes"""
        
        if not char1 or not char2:
            return 0.5
        
        # Compare counts by type
        types1 = {}
        types2 = {}
        
        for note in char1:
            note_type = note.get('type', 'unknown')
            types1[note_type] = types1.get(note_type, 0) + 1
        
        for note in char2:
            note_type = note.get('type', 'unknown')
            types2[note_type] = types2.get(note_type, 0) + 1
        
        # Compare type distributions
        all_types = set(types1.keys()).union(set(types2.keys()))
        
        if not all_types:
            return 1.0
        
        similarities = []
        for note_type in all_types:
            count1 = types1.get(note_type, 0)
            count2 = types2.get(note_type, 0)
            max_count = max(count1, count2, 1)
            sim = 1.0 - abs(count1 - count2) / max_count
            similarities.append(sim)
        
        return np.mean(similarities)
    
    def _compare_melodic_stats(self, stats1: Dict[str, float], stats2: Dict[str, float]) -> float:
        """Compare melodic statistics"""
        
        if not stats1 or not stats2:
            return 0.5
        
        # Compare key statistics
        key_stats = ['pitch_range', 'pitch_mean', 'pitch_std', 'pitch_median']
        similarities = []
        
        for stat in key_stats:
            if stat in stats1 and stat in stats2:
                val1 = stats1[stat]
                val2 = stats2[stat]
                max_val = max(abs(val1), abs(val2), 1.0)
                sim = 1.0 - abs(val1 - val2) / max_val
                similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.5
    
    def _compute_perceptual_distance(self, dna1: Dict[str, Any], dna2: Dict[str, Any]) -> float:
        """Compute perceptual distance between melodies"""
        
        # Combine multiple perceptual factors
        factors = []
        
        # 1. Pitch height difference
        stats1 = dna1.get('melodic_stats', {})
        stats2 = dna2.get('melodic_stats', {})
        
        if 'pitch_mean' in stats1 and 'pitch_mean' in stats2:
            pitch_diff = abs(stats1['pitch_mean'] - stats2['pitch_mean'])
            pitch_distance = pitch_diff / 12.0  # Normalize by octave
            factors.append(pitch_distance)
        
        # 2. Contour shape difference
        contour1 = dna1.get('pitch_contour', np.array([]))
        contour2 = dna2.get('pitch_contour', np.array([]))
        
        if len(contour1) > 0 and len(contour2) > 0:
            # Resample to same length
            if len(contour1) != len(contour2):
                from scipy.interpolate import interp1d
                target_len = min(len(contour1), len(contour2), 50)
                
                if len(contour1) > 1:
                    f1 = interp1d(np.linspace(0, 1, len(contour1)), contour1)
                    contour1_resampled = f1(np.linspace(0, 1, target_len))
                else:
                    contour1_resampled = np.full(target_len, contour1[0])
                
                if len(contour2) > 1:
                    f2 = interp1d(np.linspace(0, 1, len(contour2)), contour2)
                    contour2_resampled = f2(np.linspace(0, 1, target_len))
                else:
                    contour2_resampled = np.full(target_len, contour2[0])
            else:
                contour1_resampled = contour1
                contour2_resampled = contour2
            
            contour_distance = np.mean(np.abs(contour1_resampled - contour2_resampled))
            factors.append(contour_distance)
        
        # 3. Rhythmic complexity difference
        rhythm1 = dna1.get('rhythmic_skeleton', {})
        rhythm2 = dna2.get('rhythmic_skeleton', {})
        
        density1 = rhythm1.get('note_density', 0)
        density2 = rhythm2.get('note_density', 0)
        density_distance = abs(density1 - density2) / max(density1, density2, 1.0)
        factors.append(density_distance)
        
        # Average perceptual distance
        return np.mean(factors) if factors else 1.0
    
    def _compute_weighted_overall_similarity(self, similarities: Dict[str, float]) -> float:
        """Compute weighted overall similarity from components"""
        
        # Define weights for different similarity components
        weights = {
            'contour_multiscale': 0.25,
            'interval_weighted': 0.20,
            'pattern_ngram': 0.15,
            'rhythm_timing': 0.15,
            'structure_phrases': 0.10,
            'pattern_motif': 0.10,
            'structure_statistics': 0.05
        }
        
        # Compute weighted sum
        total_weight = 0.0
        weighted_sum = 0.0
        
        for component, weight in weights.items():
            if component in similarities:
                weighted_sum += similarities[component] * weight
                total_weight += weight
        
        # Normalize by actual total weight
        if total_weight > 0:
            overall_similarity = weighted_sum / total_weight
        else:
            # Fallback to simple average
            available_similarities = [v for k, v in similarities.items() if isinstance(v, (int, float))]
            overall_similarity = np.mean(available_similarities) if available_similarities else 0.0
        
        return overall_similarity