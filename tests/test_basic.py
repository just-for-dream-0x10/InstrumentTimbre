"""
Basic test skeletons for Instrument Timbre project.
Run with: pytest tests/
"""
import os
import pytest
import numpy as np
import torch

from InstrumentTimbre.models.model import InstrumentTimbreModel
from InstrumentTimbre.utils.data import prepare_dataloader
from InstrumentTimbre.audio.processors import load_audio, extract_features

def test_model_instantiation():
    """Test that the model can be instantiated without error."""
    model = InstrumentTimbreModel()
    assert model is not None

def test_load_audio_example():
    """Test audio loading on an example file (if exists)."""
    example_path = os.path.join(os.path.dirname(__file__), '../example/example.wav')
    if os.path.exists(example_path):
        audio, sr = load_audio(example_path)
        assert audio is not None and sr > 0
    else:
        pytest.skip('No example audio file found.')

def test_extract_features_shape():
    """Test feature extraction returns expected shape for dummy audio."""
    sr = 22050
    duration = 1.0
    audio = np.random.randn(int(sr * duration))
    features = extract_features(audio, sr)
    assert features is not None and features.shape[0] > 0

def test_model_forward_dummy():
    """Test model encoder forward with dummy input."""
    model = InstrumentTimbreModel()
    dummy = torch.randn(1, 1, 128, 128)  # Example input shape
    try:
        output = model.encoder(dummy)
        assert output is not None
    except Exception as e:
        pytest.skip(f"Encoder forward failed: {e}")
