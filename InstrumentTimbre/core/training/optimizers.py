"""
Optimizers configuration for InstrumentTimbre training
"""

import torch.optim as optim
from torch.nn.parameter import Parameter
from typing import Dict, Any, Iterator

def get_optimizer(parameters: Iterator[Parameter], 
                 optimizer_config: Dict[str, Any]) -> optim.Optimizer:
    """
    Get optimizer based on configuration
    
    Args:
        parameters: Model parameters
        optimizer_config: Optimizer configuration
        
    Returns:
        PyTorch optimizer
    """
    optimizer_name = optimizer_config.get('name', 'adam').lower()
    lr = optimizer_config.get('lr', 0.001)
    weight_decay = optimizer_config.get('weight_decay', 0.0)
    
    if optimizer_name == 'adam':
        return optim.Adam(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            betas=optimizer_config.get('betas', (0.9, 0.999)),
            eps=optimizer_config.get('eps', 1e-8)
        )
    elif optimizer_name == 'adamw':
        return optim.AdamW(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            betas=optimizer_config.get('betas', (0.9, 0.999)),
            eps=optimizer_config.get('eps', 1e-8)
        )
    elif optimizer_name == 'sgd':
        return optim.SGD(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            momentum=optimizer_config.get('momentum', 0.9),
            nesterov=optimizer_config.get('nesterov', False)
        )
    elif optimizer_name == 'rmsprop':
        return optim.RMSprop(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            momentum=optimizer_config.get('momentum', 0.0),
            alpha=optimizer_config.get('alpha', 0.99)
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

def get_scheduler(optimizer: optim.Optimizer, 
                 scheduler_config: Dict[str, Any]) -> optim.lr_scheduler._LRScheduler:
    """
    Get learning rate scheduler
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_config: Scheduler configuration
        
    Returns:
        PyTorch learning rate scheduler
    """
    if not scheduler_config or scheduler_config.get('name') is None:
        return None
    
    scheduler_name = scheduler_config.get('name', '').lower()
    
    if scheduler_name == 'step':
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config.get('step_size', 30),
            gamma=scheduler_config.get('gamma', 0.1)
        )
    elif scheduler_name == 'multistep':
        return optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=scheduler_config.get('milestones', [30, 60, 90]),
            gamma=scheduler_config.get('gamma', 0.1)
        )
    elif scheduler_name == 'exponential':
        return optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=scheduler_config.get('gamma', 0.95)
        )
    elif scheduler_name == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_config.get('T_max', 100),
            eta_min=scheduler_config.get('eta_min', 0)
        )
    elif scheduler_name == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_config.get('mode', 'min'),
            factor=scheduler_config.get('factor', 0.1),
            patience=scheduler_config.get('patience', 10),
            verbose=scheduler_config.get('verbose', True)
        )
    elif scheduler_name == 'warmup_cosine':
        return WarmupCosineScheduler(
            optimizer,
            warmup_epochs=scheduler_config.get('warmup_epochs', 10),
            max_epochs=scheduler_config.get('max_epochs', 100)
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

class WarmupCosineScheduler(optim.lr_scheduler._LRScheduler):
    """
    Warmup + Cosine Annealing Learning Rate Scheduler
    """
    
    def __init__(self, optimizer, warmup_epochs: int, max_epochs: int, 
                 last_epoch: int = -1):
        """
        Initialize warmup cosine scheduler
        
        Args:
            optimizer: PyTorch optimizer
            warmup_epochs: Number of warmup epochs
            max_epochs: Total number of epochs
            last_epoch: Last epoch index
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """Calculate learning rate"""
        if self.last_epoch < self.warmup_epochs:
            # Warmup phase: linear increase
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs 
                    for base_lr in self.base_lrs]
        else:
            # Cosine annealing phase
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            return [base_lr * 0.5 * (1 + torch.cos(torch.pi * progress)) 
                    for base_lr in self.base_lrs]