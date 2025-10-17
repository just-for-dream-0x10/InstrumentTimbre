"""
Inference modules for InstrumentTimbre
"""

from .predictor import InstrumentPredictor
from .batch_predictor import BatchPredictor

__all__ = [
    "InstrumentPredictor",
    "BatchPredictor"
]