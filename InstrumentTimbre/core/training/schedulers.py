"""
Learning rate schedulers for InstrumentTimbre training
This module is imported by optimizers.py for get_scheduler function
"""

# This file serves as a placeholder for additional custom schedulers
# The main scheduler logic is implemented in optimizers.py

from .optimizers import get_scheduler, WarmupCosineScheduler

__all__ = ['get_scheduler', 'WarmupCosineScheduler']