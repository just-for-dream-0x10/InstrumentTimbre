"""
Model architectures for InstrumentTimbre
"""

from .base import BaseModel
from .cnn import CNNClassifier
from .transformer import TransformerClassifier
from .hybrid import HybridModel

__all__ = [
    "BaseModel",
    "CNNClassifier", 
    "TransformerClassifier",
    "HybridModel"
]