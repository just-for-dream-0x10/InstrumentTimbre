"""
Melody Constraint Optimizer - System Development Task

This module implements the constraint optimization system that ensures
generated music preserves the essential melodic characteristics while
allowing controlled variations for style transfer and generation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging


class MelodyConstraintOptimizer:
    """
    Constraint optimizer for melody preservation during music generation
    
    This optimizer ensures generated music satisfies multiple constraints:
    1. Melodic similarity constraints (contour, intervals)
    2. Rhythmic preservation constraints  
    3. Structural coherence constraints
    4. Style-specific constraints (optional)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Optimization parameters
        self.learning_rate = self.config.get('learning_rate', 0.01)
        self.max_iterations = self.config.get('max_iterations', 100)
        self.convergence_threshold = self.config.get('convergence_threshold', 1e-4)
        
        # Constraint weights
        self.melody_weight = self.config.get('melody_weight', 0.4)
        self.rhythm_weight = self.config.get('rhythm_weight', 0.3)
        self.structure_weight = self.config.get('structure_weight', 0.2)
        self.style_weight = self.config.get('style_weight', 0.1)
        
        # Constraint thresholds
        self.min_melody_similarity = self.config.get('min_melody_similarity', 0.8)
        self.min_rhythm_similarity = self.config.get('min_rhythm_similarity', 0.7)
        self.min_structure_coherence = self.config.get('min_structure_coherence', 0.6)
        
        self.logger.info("MelodyConstraintOptimizer initialized")
        self.logger.info(f"Constraint weights: melody={self.melody_weight}, "
                        f"rhythm={self.rhythm_weight}, structure={self.structure_weight}")
    
    def optimize_generation(self, 
                          original_dna: Dict[str, Any],
                          target_parameters: Dict[str, Any],
                          generation_function: Callable,
                          **generation_kwargs) -> Dict[str, Any]:
        """
        Optimize music generation to satisfy melody preservation constraints
        
        Args:
            original_dna: Original melody DNA to preserve
            target_parameters: Target generation parameters (style, tempo, etc.)
            generation_function: Function that generates music given parameters
            **generation_kwargs: Additional arguments for generation function
            
        Returns:
            Optimized generation result with constraint satisfaction info
        """
        self.logger.info("Starting constraint optimization...")
        
        # Initialize optimization variables
        current_params = target_parameters.copy()
        best_params = None
        best_score = float('-inf')
        iteration_history = []
        
        # Convert parameters to torch tensors for optimization
        param_tensors = self._params_to_tensors(current_params)
        
        # Setup optimizer
        optimizer = optim.Adam(param_tensors.values(), lr=self.learning_rate)
        
        for iteration in range(self.max_iterations):
            optimizer.zero_grad()
            
            # Convert tensors back to parameters
            current_params = self._tensors_to_params(param_tensors, target_parameters)
            
            # Generate music with current parameters
            try:
                generation_result = generation_function(current_params, **generation_kwargs)
                generated_audio = generation_result['audio']
                generated_dna = generation_result.get('dna', None)
                
                # If DNA not provided, extract it
                if generated_dna is None:
                    from InstrumentTimbre.core.generation.melody_preservation import MelodyPreservationEngine
                    engine = MelodyPreservationEngine()
                    generated_dna = engine.extract_melody_dna(generated_audio)
                
            except Exception as e:
                self.logger.warning(f"Generation failed at iteration {iteration}: {e}")
                continue
            
            # Compute constraint violations and objective
            constraints = self._evaluate_constraints(original_dna, generated_dna, current_params)
            objective = self._compute_objective(constraints)
            
            # Track best result
            if objective > best_score:
                best_score = objective
                best_params = current_params.copy()
            
            # Record iteration
            iteration_info = {
                'iteration': iteration,
                'objective': objective,
                'constraints': constraints,
                'parameters': current_params.copy()
            }
            iteration_history.append(iteration_info)
            
            # Check convergence
            if iteration > 0:
                improvement = objective - iteration_history[-2]['objective']
                if abs(improvement) < self.convergence_threshold:
                    self.logger.info(f"Converged after {iteration + 1} iterations")
                    break
            
            # Compute gradients and update
            loss = -objective  # Minimize negative objective
            if hasattr(loss, 'backward'):
                loss.backward()
                optimizer.step()
            
            # Log progress
            if iteration % 10 == 0:
                self.logger.info(f"Iteration {iteration}: objective={objective:.4f}, "
                               f"melody_sim={constraints['melody_similarity']:.3f}, "
                               f"rhythm_sim={constraints['rhythm_similarity']:.3f}")
        
        # Final generation with best parameters
        if best_params is not None:
            final_result = generation_function(best_params, **generation_kwargs)
        else:
            final_result = generation_result  # Use last result if no best found
        
        return {
            'generated_audio': final_result['audio'],
            'optimized_parameters': best_params or current_params,
            'constraint_satisfaction': constraints,
            'objective_score': best_score,
            'optimization_history': iteration_history,
            'converged': iteration < self.max_iterations - 1
        }
    
    def _params_to_tensors(self, params: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Convert parameter dictionary to optimizable tensors"""
        tensors = {}
        
        for key, value in params.items():
            if isinstance(value, (int, float)):
                # Convert numeric parameters to tensors with gradients
                tensors[key] = torch.tensor(float(value), requires_grad=True)
            elif isinstance(value, (list, tuple)) and all(isinstance(x, (int, float)) for x in value):
                # Convert numeric arrays to tensors
                tensors[key] = torch.tensor([float(x) for x in value], requires_grad=True)
            # Skip non-numeric parameters
        
        return tensors
    
    def _tensors_to_params(self, tensors: Dict[str, torch.Tensor], template: Dict[str, Any]) -> Dict[str, Any]:
        """Convert tensors back to parameter dictionary"""
        params = template.copy()
        
        for key, tensor in tensors.items():
            if key in params:
                if tensor.dim() == 0:  # Scalar
                    params[key] = float(tensor.item())
                else:  # Array
                    params[key] = tensor.detach().numpy().tolist()
        
        return params
    
    def _evaluate_constraints(self, original_dna: Dict[str, Any], 
                            generated_dna: Dict[str, Any],
                            params: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate all constraint violations"""
        constraints = {}
        
        # Import here to avoid circular imports
        from InstrumentTimbre.core.generation.melody_preservation import MelodyPreservationEngine
        
        # 1. Melody similarity constraint
        engine = MelodyPreservationEngine()
        melody_similarity = engine.compute_melody_similarity(original_dna, generated_dna)
        constraints['melody_similarity'] = melody_similarity
        constraints['melody_violation'] = max(0, self.min_melody_similarity - melody_similarity)
        
        # 2. Rhythm similarity constraint
        rhythm_similarity = engine._compute_rhythm_similarity(
            original_dna['rhythmic_skeleton'], 
            generated_dna['rhythmic_skeleton']
        )
        if hasattr(rhythm_similarity, 'item'):
            rhythm_similarity = rhythm_similarity.item()
        constraints['rhythm_similarity'] = rhythm_similarity
        constraints['rhythm_violation'] = max(0, self.min_rhythm_similarity - rhythm_similarity)
        
        # 3. Structure coherence constraint
        structure_coherence = self._compute_structure_coherence(original_dna, generated_dna)
        constraints['structure_coherence'] = structure_coherence
        constraints['structure_violation'] = max(0, self.min_structure_coherence - structure_coherence)
        
        # 4. Style consistency (if style parameters provided)
        if 'target_style' in params:
            style_consistency = self._compute_style_consistency(generated_dna, params)
            constraints['style_consistency'] = style_consistency
        else:
            constraints['style_consistency'] = 1.0  # No style constraint
        
        return constraints
    
    def _compute_objective(self, constraints: Dict[str, float]) -> float:
        """Compute optimization objective from constraints"""
        
        # Reward high similarity scores
        melody_reward = constraints['melody_similarity'] * self.melody_weight
        rhythm_reward = constraints['rhythm_similarity'] * self.rhythm_weight
        structure_reward = constraints['structure_coherence'] * self.structure_weight
        style_reward = constraints.get('style_consistency', 1.0) * self.style_weight
        
        # Penalize constraint violations
        melody_penalty = constraints['melody_violation'] * self.melody_weight * 2
        rhythm_penalty = constraints['rhythm_violation'] * self.rhythm_weight * 2
        structure_penalty = constraints['structure_violation'] * self.structure_weight * 2
        
        # Total objective (maximize)
        objective = (melody_reward + rhythm_reward + structure_reward + style_reward - 
                    melody_penalty - rhythm_penalty - structure_penalty)
        
        return objective
    
    def _compute_structure_coherence(self, original_dna: Dict[str, Any], 
                                   generated_dna: Dict[str, Any]) -> float:
        """Compute structural coherence between original and generated"""
        
        coherence_scores = []
        
        # 1. Phrase structure similarity
        orig_phrases = original_dna.get('phrase_boundaries', [])
        gen_phrases = generated_dna.get('phrase_boundaries', [])
        
        if len(orig_phrases) > 1 and len(gen_phrases) > 1:
            # Compare number of phrases
            phrase_count_sim = 1.0 - abs(len(orig_phrases) - len(gen_phrases)) / max(len(orig_phrases), len(gen_phrases))
            coherence_scores.append(phrase_count_sim)
            
            # Compare phrase length distribution
            orig_lengths = np.diff(orig_phrases)
            gen_lengths = np.diff(gen_phrases)
            if len(orig_lengths) > 0 and len(gen_lengths) > 0:
                length_sim = 1.0 - abs(np.mean(orig_lengths) - np.mean(gen_lengths)) / max(np.mean(orig_lengths), np.mean(gen_lengths))
                coherence_scores.append(length_sim)
        
        # 2. Pitch range coherence
        orig_stats = original_dna.get('melodic_stats', {})
        gen_stats = generated_dna.get('melodic_stats', {})
        
        if 'pitch_range' in orig_stats and 'pitch_range' in gen_stats:
            range_sim = 1.0 - abs(orig_stats['pitch_range'] - gen_stats['pitch_range']) / max(orig_stats['pitch_range'], gen_stats['pitch_range'])
            coherence_scores.append(range_sim)
        
        # 3. Characteristic notes preservation
        orig_char = original_dna.get('characteristic_notes', [])
        gen_char = generated_dna.get('characteristic_notes', [])
        
        if len(orig_char) > 0 and len(gen_char) > 0:
            # Simple count-based similarity
            count_sim = 1.0 - abs(len(orig_char) - len(gen_char)) / max(len(orig_char), len(gen_char))
            coherence_scores.append(count_sim)
        
        # Return average coherence or default
        return np.mean(coherence_scores) if coherence_scores else 0.5
    
    def _compute_style_consistency(self, generated_dna: Dict[str, Any], 
                                 params: Dict[str, Any]) -> float:
        """Compute how well the generated music matches target style"""
        
        target_style = params.get('target_style', 'neutral')
        consistency = 1.0  # Base consistency
        
        # Style-specific checks
        if target_style == 'chinese_traditional':
            # Check for pentatonic scale usage, specific intervals, etc.
            intervals = generated_dna.get('interval_sequence', [])
            if len(intervals) > 0:
                # Prefer smaller intervals (steps) over large jumps
                small_intervals = np.sum(np.abs(intervals) <= 2) / len(intervals)
                consistency *= (0.5 + 0.5 * small_intervals)
        
        elif target_style == 'western_classical':
            # Check for diatonic harmony, structured phrases
            pass  # Placeholder for classical style checks
        
        elif target_style == 'modern_pop':
            # Check for repetitive patterns, moderate pitch range
            pass  # Placeholder for pop style checks
        
        return consistency
    
    def validate_constraints(self, original_dna: Dict[str, Any], 
                           generated_dna: Dict[str, Any]) -> Dict[str, bool]:
        """Validate that all constraints are satisfied"""
        
        constraints = self._evaluate_constraints(original_dna, generated_dna, {})
        
        validation = {
            'melody_preserved': constraints['melody_similarity'] >= self.min_melody_similarity,
            'rhythm_preserved': constraints['rhythm_similarity'] >= self.min_rhythm_similarity,
            'structure_coherent': constraints['structure_coherence'] >= self.min_structure_coherence,
            'all_satisfied': True
        }
        
        # Check if all constraints are satisfied
        validation['all_satisfied'] = all([
            validation['melody_preserved'],
            validation['rhythm_preserved'], 
            validation['structure_coherent']
        ])
        
        return validation