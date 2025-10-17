"""
Main training orchestrator for InstrumentTimbre models
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    # Fallback if tensorboard not available
    SummaryWriter = None
    TENSORBOARD_AVAILABLE = False
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Callable
import logging
import time
from tqdm import tqdm

from .metrics import MetricsCalculator
from .losses import get_loss_function
from .optimizers import get_optimizer
from .schedulers import get_scheduler
from ..models.base import BaseModel

class Trainer:
    """
    Main trainer class for InstrumentTimbre models
    """
    
    def __init__(self, 
                 model: BaseModel,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 config: Dict[str, Any],
                 device: str = 'auto'):
        """
        Initialize trainer
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader  
            config: Training configuration
            device: Device to use ('auto', 'cpu', 'cuda')
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Setup device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        self.logger.info(f"Using device: {self.device}")
        
        # Training parameters
        self.epochs = config.get('epochs', 100)
        self.save_dir = Path(config.get('save_dir', 'checkpoints'))
        self.save_dir.mkdir(exist_ok=True)
        
        # Initialize training components
        self.criterion = get_loss_function(config.get('loss', 'crossentropy'))
        self.optimizer = get_optimizer(model.parameters(), config.get('optimizer', {}))
        self.scheduler = get_scheduler(self.optimizer, config.get('scheduler', {}))
        
        # Metrics and logging
        self.metrics_calc = MetricsCalculator(config.get('metrics', {}))
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        
        # TensorBoard logging
        if config.get('use_tensorboard', True) and TENSORBOARD_AVAILABLE:
            log_dir = config.get('log_dir', 'runs')
            self.writer = SummaryWriter(log_dir)
        else:
            self.writer = None
            if config.get('use_tensorboard', True) and not TENSORBOARD_AVAILABLE:
                self.logger.warning("TensorBoard not available, logging disabled")
        
        # Early stopping
        self.patience = config.get('patience', 10)
        self.patience_counter = 0
        
        # Training state
        self.current_epoch = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
    
    def train(self) -> Dict[str, Any]:
        """
        Main training loop
        
        Returns:
            Training history and final metrics
        """
        self.logger.info(f"Starting training for {self.epochs} epochs")
        self.logger.info(f"Model: {self.model.__class__.__name__}")
        self.logger.info(f"Train samples: {len(self.train_loader.dataset)}")
        self.logger.info(f"Val samples: {len(self.val_loader.dataset)}")
        
        start_time = time.time()
        
        try:
            for epoch in range(self.epochs):
                self.current_epoch = epoch
                
                # Training phase
                train_metrics = self._train_epoch()
                
                # Validation phase
                val_metrics = self._validate_epoch()
                
                # Update learning rate
                if self.scheduler:
                    if hasattr(self.scheduler, 'step'):
                        if 'ReduceLROnPlateau' in str(type(self.scheduler)):
                            self.scheduler.step(val_metrics['loss'])
                        else:
                            self.scheduler.step()
                
                # Log metrics
                self._log_metrics(train_metrics, val_metrics, epoch)
                
                # Save checkpoint
                self._save_checkpoint(val_metrics, epoch)
                
                # Early stopping check
                if self._check_early_stopping(val_metrics):
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
                
                # Progress update
                self.logger.info(
                    f"Epoch {epoch+1}/{self.epochs} - "
                    f"Train Loss: {train_metrics['loss']:.4f}, "
                    f"Train Acc: {train_metrics['accuracy']:.4f}, "
                    f"Val Loss: {val_metrics['loss']:.4f}, "
                    f"Val Acc: {val_metrics['accuracy']:.4f}"
                )
        
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
        
        finally:
            if self.writer:
                self.writer.close()
        
        training_time = time.time() - start_time
        self.logger.info(f"Training completed in {training_time:.2f} seconds")
        
        return {
            'history': self.training_history,
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
            'training_time': training_time,
            'final_epoch': self.current_epoch
        }
    
    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        running_loss = 0.0
        all_predictions = []
        all_targets = []
        
        pbar = tqdm(self.train_loader, desc=f"Train Epoch {self.current_epoch+1}")
        
        for batch_idx, (data, targets) in enumerate(pbar):
            data, targets = data.to(self.device), targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['grad_clip']
                )
            
            self.optimizer.step()
            
            # Collect metrics
            running_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        # Calculate epoch metrics
        avg_loss = running_loss / len(self.train_loader)
        metrics = self.metrics_calc.calculate_metrics(all_targets, all_predictions)
        metrics['loss'] = avg_loss
        
        return metrics
    
    def _validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        
        running_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Val Epoch {self.current_epoch+1}")
            
            for data, targets in pbar:
                data, targets = data.to(self.device), targets.to(self.device)
                
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                
                running_loss += loss.item()
                predictions = torch.argmax(outputs, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
                pbar.set_postfix({'loss': loss.item()})
        
        # Calculate epoch metrics
        avg_loss = running_loss / len(self.val_loader)
        metrics = self.metrics_calc.calculate_metrics(all_targets, all_predictions)
        metrics['loss'] = avg_loss
        
        return metrics
    
    def _log_metrics(self, train_metrics: Dict[str, float], 
                    val_metrics: Dict[str, float], epoch: int):
        """Log metrics to history and TensorBoard"""
        # Update history
        self.training_history['train_loss'].append(train_metrics['loss'])
        self.training_history['val_loss'].append(val_metrics['loss'])
        self.training_history['train_acc'].append(train_metrics['accuracy'])
        self.training_history['val_acc'].append(val_metrics['accuracy'])
        
        # Log to TensorBoard
        if self.writer:
            self.writer.add_scalar('Loss/Train', train_metrics['loss'], epoch)
            self.writer.add_scalar('Loss/Val', val_metrics['loss'], epoch)
            self.writer.add_scalar('Accuracy/Train', train_metrics['accuracy'], epoch)
            self.writer.add_scalar('Accuracy/Val', val_metrics['accuracy'], epoch)
            
            # Log learning rate
            if self.scheduler:
                current_lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('Learning_Rate', current_lr, epoch)
    
    def _save_checkpoint(self, val_metrics: Dict[str, float], epoch: int):
        """Save model checkpoint if validation improves"""
        val_acc = val_metrics['accuracy']
        val_loss = val_metrics['loss']
        
        # Save best accuracy model
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.model.save_checkpoint(
                self.save_dir / 'best_acc_model.pth',
                epoch,
                self.optimizer.state_dict(),
                val_metrics
            )
        
        # Save best loss model
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.model.save_checkpoint(
                self.save_dir / 'best_loss_model.pth',
                epoch,
                self.optimizer.state_dict(),
                val_metrics
            )
        
        # Save latest model
        if epoch % self.config.get('save_frequency', 10) == 0:
            self.model.save_checkpoint(
                self.save_dir / f'model_epoch_{epoch}.pth',
                epoch,
                self.optimizer.state_dict(),
                val_metrics
            )
    
    def _check_early_stopping(self, val_metrics: Dict[str, float]) -> bool:
        """Check if early stopping criteria is met"""
        if val_metrics['accuracy'] > self.best_val_acc:
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.patience
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, Any]:
        """
        Evaluate model on test set
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Test metrics
        """
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        test_loss = 0.0
        
        with torch.no_grad():
            for data, targets in tqdm(test_loader, desc="Testing"):
                data, targets = data.to(self.device), targets.to(self.device)
                
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                
                test_loss += loss.item()
                predictions = torch.argmax(outputs, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        avg_loss = test_loss / len(test_loader)
        metrics = self.metrics_calc.calculate_metrics(all_targets, all_predictions)
        metrics['loss'] = avg_loss
        
        self.logger.info(f"Test Results - Loss: {avg_loss:.4f}, Acc: {metrics['accuracy']:.4f}")
        
        return metrics