"""
InstrumentTimbre - A framework for analyzing and synthesizing traditional Chinese instrument timbres
"""

# Import submodules
from . import audio
from . import models
from . import utils

# Import main classes directly from models.model
from .models.model import InstrumentTimbreModel

# Define package exports
__all__ = [
    # Main modules
    "audio",
    "models",
    "utils",
    # Core functionality
    "InstrumentTimbreModel",
]

__version__ = "1.0.0"
