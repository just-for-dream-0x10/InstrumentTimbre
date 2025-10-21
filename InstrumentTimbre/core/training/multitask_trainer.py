"""
Multi-task Joint Training System - System Development Task

This module implements joint training for the unified model that combines
emotion analysis, instrument recognition, and music generation tasks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import time
from pathlib import Path

# Configure logging for the module
logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Types of tasks in multi-task learning"""
    EMOTION_ANALYSIS = "emotion_analysis"
    INSTRUMENT_RECOGNITION = "instrument_recognition"
    MUSIC_GENERATION = "music_generation"
    STYLE_TRANSFER = "style_transfer"


@dataclass
class TaskConfig:
    """Configuration for individual task"""
    task_type: TaskType
    loss_weight: float = 1.0
    learning_rate: float = 1e-4
    warmup_steps: int = 1000
    gradient_clipping: float = 1.0
    evaluation_metric: str = "accuracy"
    target_performance: float = 0.9


@dataclass
class TrainingConfig:
    """Configuration for multi-task training"""
    batch_size: int = 32
    num_epochs: int = 100
    warmup_epochs: int = 5
    save_interval: int = 10
    evaluation_interval: int = 5
    early_stopping_patience: int = 20
    gradient_accumulation_steps: int = 1
    mixed_precision: bool = True
    checkpoint_dir: str = "checkpoints"
    log_interval: int = 100


@dataclass
class TrainingMetrics:
    """Training metrics tracking"""
    epoch: int
    step: int
    task_losses: Dict[str, float]
    task_metrics: Dict[str, float]
    total_loss: float
    learning_rates: Dict[str, float]
    training_time: float
    memory_usage: float


class MultiTaskTrainer:
    """
    Multi-task trainer for unified emotion analysis and music generation model
    
    Provides:
    - Joint training of multiple tasks
    - Task-specific loss weighting and scheduling
    - Gradient balancing across tasks
    - Performance monitoring and early stopping
    - Checkpoint management
    """
    
    def __init__(self, 
                 model: nn.Module,
                 task_configs: Dict[TaskType, TaskConfig],
                 training_config: TrainingConfig):
        """
        Initialize multi-task trainer
        
        Args:
            model: Unified model with multiple task heads
            task_configs: Configuration for each task
            training_config: Overall training configuration
        """
        self.model = model
        self.task_configs = task_configs
        self.training_config = training_config
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Setup optimizers and schedulers
        self.optimizers = self._setup_optimizers()
        self.schedulers = self._setup_schedulers()
        
        # Setup loss functions
        self.loss_functions = self._setup_loss_functions()
        
        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.best_metrics = {task.value: 0.0 for task in TaskType}
        self.training_history = []
        
        # Early stopping
        self.patience_counters = {task.value: 0 for task in TaskType}
        
        # Mixed precision setup
        if training_config.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        # Create checkpoint directory
        Path(training_config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info("MultiTaskTrainer initialized with %d tasks", len(task_configs))
    
    def _setup_optimizers(self) -> Dict[str, optim.Optimizer]:
        """Setup task-specific optimizers"""
        optimizers = {}
        
        for task_type, config in self.task_configs.items():
            # Get task-specific parameters
            if hasattr(self.model, f'{task_type.value}_head'):
                task_head = getattr(self.model, f'{task_type.value}_head')
                task_params = list(task_head.parameters())
            else:
                # If no specific head, use all model parameters
                task_params = list(self.model.parameters())
            
            # Create optimizer
            optimizer = optim.AdamW(
                task_params,
                lr=config.learning_rate,
                weight_decay=1e-5,
                betas=(0.9, 0.999)
            )
            
            optimizers[task_type.value] = optimizer
        
        return optimizers
    
    def _setup_schedulers(self) -> Dict[str, Any]:
        """Setup learning rate schedulers"""
        schedulers = {}
        
        for task_type, config in self.task_configs.items():
            optimizer = self.optimizers[task_type.value]
            
            # Cosine annealing with warmup
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=config.warmup_steps,
                T_mult=2,
                eta_min=config.learning_rate * 0.01
            )
            
            schedulers[task_type.value] = scheduler
        
        return schedulers
    
    def _setup_loss_functions(self) -> Dict[str, nn.Module]:
        """Setup task-specific loss functions"""
        loss_functions = {
            TaskType.EMOTION_ANALYSIS.value: nn.CrossEntropyLoss(label_smoothing=0.1),
            TaskType.INSTRUMENT_RECOGNITION.value: nn.CrossEntropyLoss(label_smoothing=0.1),
            TaskType.MUSIC_GENERATION.value: nn.MSELoss(),
            TaskType.STYLE_TRANSFER.value: nn.MSELoss()
        }
        
        return loss_functions
    
    def train_epoch(self, dataloaders: Dict[str, DataLoader]) -> TrainingMetrics:
        """Train one epoch across all tasks"""
        
        self.model.train()
        epoch_start_time = time.time()
        
        # Initialize metrics
        epoch_losses = {task.value: 0.0 for task in TaskType}
        epoch_metrics = {task.value: 0.0 for task in TaskType}
        total_steps = {task.value: 0 for task in TaskType}
        
        # Create task iterators
        task_iterators = {}
        for task_name, dataloader in dataloaders.items():
            task_iterators[task_name] = iter(dataloader)
        
        # Training loop
        max_steps = max(len(dl) for dl in dataloaders.values())
        
        for step in range(max_steps):
            step_losses = {}
            
            # Process each task
            for task_name, iterator in task_iterators.items():
                try:
                    batch = next(iterator)
                except StopIteration:
                    # Restart iterator if exhausted
                    task_iterators[task_name] = iter(dataloaders[task_name])
                    batch = next(task_iterators[task_name])
                
                # Process batch for this task
                task_loss, task_metric = self._process_task_batch(task_name, batch)
                
                if task_loss is not None:
                    step_losses[task_name] = task_loss
                    epoch_losses[task_name] += task_loss
                    epoch_metrics[task_name] += task_metric
                    total_steps[task_name] += 1
            
            # Combined backward pass and optimization
            if step_losses:
                self._combined_backward_pass(step_losses)
            
            # Update learning rates
            for scheduler in self.schedulers.values():
                scheduler.step()
            
            # Logging
            if step % self.training_config.log_interval == 0:
                self._log_training_step(step, step_losses)
            
            self.current_step += 1
        
        # Average metrics over epoch
        for task_name in epoch_losses:
            if total_steps[task_name] > 0:
                epoch_losses[task_name] /= total_steps[task_name]
                epoch_metrics[task_name] /= total_steps[task_name]
        
        # Calculate total loss
        total_loss = sum(
            loss * self.task_configs[TaskType(task_name)].loss_weight
            for task_name, loss in epoch_losses.items()
        )
        
        # Get current learning rates
        current_lrs = {
            task_name: optimizer.param_groups[0]['lr']
            for task_name, optimizer in self.optimizers.items()
        }
        
        # Create metrics object
        metrics = TrainingMetrics(
            epoch=self.current_epoch,
            step=self.current_step,
            task_losses=epoch_losses,
            task_metrics=epoch_metrics,
            total_loss=total_loss,
            learning_rates=current_lrs,
            training_time=time.time() - epoch_start_time,
            memory_usage=self._get_memory_usage()
        )
        
        self.training_history.append(metrics)
        self.current_epoch += 1
        
        return metrics
    
    def _process_task_batch(self, task_name: str, batch: Dict[str, torch.Tensor]) -> Tuple[Optional[float], float]:
        """Process a batch for a specific task"""
        
        try:
            # Move batch to device
            inputs = batch['features'].to(self.device)
            targets = batch['targets'].to(self.device) if 'targets' in batch else None
            
            # Get task type
            task_type = TaskType(task_name)
            
            # Forward pass with mixed precision
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self._forward_task(inputs, task_type, batch)
                    loss = self._calculate_task_loss(outputs, targets, task_type)
            else:
                outputs = self._forward_task(inputs, task_type, batch)
                loss = self._calculate_task_loss(outputs, targets, task_type)
            
            # Calculate metric
            metric = self._calculate_task_metric(outputs, targets, task_type)
            
            return loss.item() if loss is not None else None, metric
            
        except Exception as e:
            logger.warning("Error processing batch for task %s: %s", task_name, e)
            return None, 0.0
    
    def _forward_task(self, inputs: torch.Tensor, task_type: TaskType, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass for specific task"""
        
        # Get shared features from backbone
        if hasattr(self.model, 'shared_backbone'):
            shared_features = self.model.shared_backbone(inputs)
        else:
            shared_features = inputs
        
        # Task-specific forward pass
        if task_type == TaskType.EMOTION_ANALYSIS:
            if hasattr(self.model, 'emotion_head'):
                outputs = self.model.emotion_head(shared_features)
            else:
                outputs = shared_features
                
        elif task_type == TaskType.INSTRUMENT_RECOGNITION:
            if hasattr(self.model, 'instrument_head'):
                outputs = self.model.instrument_head(shared_features)
            else:
                outputs = shared_features
                
        elif task_type == TaskType.MUSIC_GENERATION:
            if hasattr(self.model, 'generation_head'):
                conditioning = batch.get('conditioning', None)
                generation_result = self.model.generation_head(
                    shared_features, conditioning=conditioning
                )
                outputs = generation_result.generated_features
            else:
                outputs = shared_features
                
        elif task_type == TaskType.STYLE_TRANSFER:
            if hasattr(self.model, 'generation_head'):
                from InstrumentTimbre.core.generation.music_generation_head import GenerationMode
                conditioning = batch.get('conditioning', None)
                generation_result = self.model.generation_head(
                    shared_features, 
                    conditioning=conditioning,
                    generation_mode=GenerationMode.STYLE_TRANSFER
                )
                outputs = generation_result.generated_features
            else:
                outputs = shared_features
        else:
            outputs = shared_features
        
        return outputs
    
    def _calculate_task_loss(self, outputs: torch.Tensor, targets: torch.Tensor, task_type: TaskType) -> Optional[torch.Tensor]:
        """Calculate loss for specific task"""
        
        if targets is None:
            return None
        
        loss_fn = self.loss_functions[task_type.value]
        
        # Reshape if necessary
        if task_type in [TaskType.EMOTION_ANALYSIS, TaskType.INSTRUMENT_RECOGNITION]:
            # Classification tasks
            if outputs.dim() > 2:
                outputs = outputs.view(-1, outputs.size(-1))
            if targets.dim() > 1:
                targets = targets.view(-1)
                
        elif task_type in [TaskType.MUSIC_GENERATION, TaskType.STYLE_TRANSFER]:
            # Regression tasks - ensure same shape
            if outputs.shape != targets.shape:
                # Truncate or pad to match target shape
                min_len = min(outputs.size(1), targets.size(1))
                outputs = outputs[:, :min_len, :]
                targets = targets[:, :min_len, :]
        
        try:
            loss = loss_fn(outputs, targets)
            return loss
        except Exception as e:
            logger.warning("Error calculating loss for task %s: %s", task_type.value, e)
            return None
    
    def _calculate_task_metric(self, outputs: torch.Tensor, targets: torch.Tensor, task_type: TaskType) -> float:
        """Calculate evaluation metric for specific task"""
        
        if targets is None:
            return 0.0
        
        try:
            if task_type in [TaskType.EMOTION_ANALYSIS, TaskType.INSTRUMENT_RECOGNITION]:
                # Accuracy for classification
                if outputs.dim() > 2:
                    outputs = outputs.view(-1, outputs.size(-1))
                if targets.dim() > 1:
                    targets = targets.view(-1)
                
                predictions = torch.argmax(outputs, dim=-1)
                accuracy = (predictions == targets).float().mean()
                return accuracy.item()
                
            elif task_type in [TaskType.MUSIC_GENERATION, TaskType.STYLE_TRANSFER]:
                # MSE for generation tasks (lower is better, so return negative)
                if outputs.shape != targets.shape:
                    min_len = min(outputs.size(1), targets.size(1))
                    outputs = outputs[:, :min_len, :]
                    targets = targets[:, :min_len, :]
                
                mse = torch.mean((outputs - targets) ** 2)
                return -mse.item()  # Negative because lower MSE is better
                
        except Exception as e:
            logger.warning("Error calculating metric for task %s: %s", task_type.value, e)
            
        return 0.0
    
    def _combined_backward_pass(self, step_losses: Dict[str, float]) -> None:
        """Perform combined backward pass for all tasks"""
        
        # Zero gradients
        for optimizer in self.optimizers.values():
            optimizer.zero_grad()
        
        # Calculate weighted total loss
        total_loss = 0.0
        for task_name, loss in step_losses.items():
            task_type = TaskType(task_name)
            weight = self.task_configs[task_type].loss_weight
            total_loss += loss * weight
        
        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(torch.tensor(total_loss, requires_grad=True)).backward()
            
            # Gradient clipping and optimization
            for task_name, optimizer in self.optimizers.items():
                task_type = TaskType(task_name)
                clip_value = self.task_configs[task_type].gradient_clipping
                
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    optimizer.param_groups[0]['params'], clip_value
                )
                self.scaler.step(optimizer)
            
            self.scaler.update()
        else:
            torch.tensor(total_loss, requires_grad=True).backward()
            
            # Gradient clipping and optimization
            for task_name, optimizer in self.optimizers.items():
                task_type = TaskType(task_name)
                clip_value = self.task_configs[task_type].gradient_clipping
                
                torch.nn.utils.clip_grad_norm_(
                    optimizer.param_groups[0]['params'], clip_value
                )
                optimizer.step()
    
    def evaluate(self, dataloaders: Dict[str, DataLoader]) -> Dict[str, float]:
        """Evaluate model on all tasks"""
        
        self.model.eval()
        eval_metrics = {}
        
        with torch.no_grad():
            for task_name, dataloader in dataloaders.items():
                task_metrics = []
                
                for batch in dataloader:
                    _, metric = self._process_task_batch(task_name, batch)
                    task_metrics.append(metric)
                
                eval_metrics[task_name] = np.mean(task_metrics) if task_metrics else 0.0
        
        return eval_metrics
    
    def save_checkpoint(self, filepath: str, include_optimizer: bool = True) -> None:
        """Save training checkpoint"""
        
        checkpoint = {
            'epoch': self.current_epoch,
            'step': self.current_step,
            'model_state_dict': self.model.state_dict(),
            'best_metrics': self.best_metrics,
            'task_configs': self.task_configs,
            'training_config': self.training_config,
            'training_history': self.training_history
        }
        
        if include_optimizer:
            checkpoint['optimizers'] = {
                name: optimizer.state_dict() 
                for name, optimizer in self.optimizers.items()
            }
            checkpoint['schedulers'] = {
                name: scheduler.state_dict()
                for name, scheduler in self.schedulers.items()
            }
            
            if self.scaler is not None:
                checkpoint['scaler'] = self.scaler.state_dict()
        
        torch.save(checkpoint, filepath)
        logger.info("Checkpoint saved to %s", filepath)
    
    def load_checkpoint(self, filepath: str, load_optimizer: bool = True) -> None:
        """Load training checkpoint"""
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.current_epoch = checkpoint['epoch']
        self.current_step = checkpoint['step']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_metrics = checkpoint['best_metrics']
        self.training_history = checkpoint.get('training_history', [])
        
        if load_optimizer and 'optimizers' in checkpoint:
            for name, optimizer in self.optimizers.items():
                if name in checkpoint['optimizers']:
                    optimizer.load_state_dict(checkpoint['optimizers'][name])
            
            for name, scheduler in self.schedulers.items():
                if name in checkpoint['schedulers']:
                    scheduler.load_state_dict(checkpoint['schedulers'][name])
            
            if self.scaler is not None and 'scaler' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler'])
        
        logger.info("Checkpoint loaded from %s", filepath)
    
    def _log_training_step(self, step: int, step_losses: Dict[str, float]) -> None:
        """Log training step information"""
        
        loss_str = ", ".join([f"{task}: {loss:.4f}" for task, loss in step_losses.items()])
        memory_usage = self._get_memory_usage()
        
        logger.info("Epoch %d, Step %d - Losses: %s, Memory: %.1f MB",
                   self.current_epoch, step, loss_str, memory_usage)
    
    def _get_memory_usage(self) -> float:
        """Get current GPU memory usage in MB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0.0
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training progress"""
        
        if not self.training_history:
            return {}
        
        latest_metrics = self.training_history[-1]
        
        return {
            'current_epoch': self.current_epoch,
            'total_steps': self.current_step,
            'best_metrics': self.best_metrics,
            'latest_losses': latest_metrics.task_losses,
            'latest_metrics': latest_metrics.task_metrics,
            'memory_usage': latest_metrics.memory_usage,
            'training_time_total': sum(m.training_time for m in self.training_history)
        }