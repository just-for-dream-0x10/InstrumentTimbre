# InstrumentTimbre

A deep learning system for analyzing and converting Chinese traditional instrument timbres.

## Features

- Instrument timbre feature extraction
- Timbre transfer and conversion
- Support for multiple Chinese traditional instruments (erhu, guzheng, pipa, etc.)
- Deep learning models with attention mechanisms
- Multi-scale feature analysis
- Audio enhancement and processing

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Marky-Shi/music_features.git
cd InstrumentTimbre
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Development mode installation (optional):
```bash
pip install -e .
```

## Project Structure

```
InstrumentTimbre/
├── models/           # Model definitions and implementations
│   ├── __init__.py   # Model component exports
│   ├── model.py      # Main model class
│   ├── encoders.py   # Encoder implementations
│   ├── decoders.py   # Decoder implementations
│   └── attention.py  # Attention mechanism modules
├── audio/            # Audio processing functionality
│   ├── __init__.py   # Audio function exports
│   └── processors.py # Audio processing functions
├── utils/            # Utility functions
│   ├── __init__.py   # Utility function exports
│   ├── export.py     # Model export tools
│   ├── cache.py      # Feature caching
│   └── data.py       # Data processing
├── app.py            # Command-line application
├── train.py          # Training script
├── run.sh            # Quick run script
└── setup.py          # Installation configuration
```

## Quick Start

### 1. Command-line Application

```bash
# Extract timbre features
python app.py extract --model-path saved_models/timbre_model.pt --input-file audio/erhu.wav --output-dir output

# Apply erhu timbre to piano
python app.py apply --model-path saved_models/timbre_model.pt --target-file audio/piano.wav --timbre-file output/erhu_timbre.npz

# Quick run using script
./run.sh extract --model-path saved_models/timbre_model.pt --input-file audio/erhu.wav
```

### 2. Using as a Python Library

```python
from InstrumentTimbre.models import InstrumentTimbreModel
from InstrumentTimbre.audio import load_audio, save_audio

# Initialize model
model = InstrumentTimbreModel(use_pretrained=True)

# Extract timbre features
features = model.extract_timbre("path/to/audio.wav", output_dir="output")

# Timbre conversion
model.replace_instrument(
    audio_file="target.wav",
    target_instrument_file="source.wav",
    output_dir="output",
    intensity=0.8
)
```

### 3. Batch Processing

```python
import os
from InstrumentTimbre.utils import FeatureCache
from InstrumentTimbre.models import InstrumentTimbreModel

# Initialize model and feature cache
model = InstrumentTimbreModel(use_pretrained=True)
cache = FeatureCache()

# Batch process audio files
for audio_file in os.listdir("input_dir"):
    if audio_file.endswith(".wav"):
        # Check if features are already in cache
        features = cache.get(audio_file, "mel")
        if features is None:
            # Extract new features and cache them
            features = model.extract_timbre(f"input_dir/{audio_file}")
            cache.put(audio_file, "mel", features)
```

## API Documentation

### InstrumentTimbreModel

The main model class, providing the following functions:

- `extract_timbre(audio_file, output_dir=None)`: Extract timbre features from audio
- `apply_timbre(target_file, timbre_features, output_dir=None, intensity=1.0)`: Apply timbre features
- `replace_instrument(audio_file, target_instrument_file, output_dir=None, intensity=0.8)`: Replace instrument timbre
- `separate_audio_sources(audio_file, output_dir=None)`: Separate audio sources
- `train(train_data, epochs=2, learning_rate=0.001)`: Train the model

### Audio Processing Tools

```python
from InstrumentTimbre.audio import (
    load_audio,
    save_audio,
    extract_features,
    extract_chinese_instrument_features
)

# Load audio
audio, sr = load_audio("input.wav")

# Extract features
features = extract_chinese_instrument_features(audio, sr, instrument_category="bowed_string")
```

### Utility Functions

```python
from InstrumentTimbre.utils import ModelExporter, FeatureCache

# Export model to different formats
exporter = ModelExporter()
exporter.to_onnx(model, "model.onnx")

# Use feature cache
cache = FeatureCache()
cache.put("audio.wav", "mel", features)
```

## Training Your Own Model

```bash
# Prepare dataset
# Data structure: data/instrument_name/audio_files.wav

# Train model
python train.py

# Or use command-line application for training
python app.py train --data-dir data --model-path saved_models/my_model.pt --chinese-instruments
```

## Notes

1. Ensure input audio has a sample rate of at least 22050Hz
2. WAV format is recommended for audio files
3. For large files, enabling feature caching is recommended
4. GPU acceleration requires the CUDA version of PyTorch

## License

MIT License

## Citation

If you use this project in your research, please cite:

```bibtex
@software{instrumenttimbre,
  title = {InstrumentTimbre: Chinese Traditional Instrument Timbre Analysis},
  author = {TimWood0x10},
  year = {2025},
  url = {}
}
```

