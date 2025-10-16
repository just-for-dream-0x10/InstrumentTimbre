# InstrumentTimbre

A comprehensive machine learning project for musical instrument timbre analysis, feature extraction, and transformation with enhanced support for Chinese traditional instruments.

## 🎵 Project Overview

This project provides advanced timbre analysis capabilities with specialized features for Chinese traditional instruments including Erhu, Pipa, Guzheng, Dizi, and Guqin. The system combines modern deep learning techniques with cultural-aware feature extraction algorithms.

## 🏗️ Architecture

```
InstrumentTimbre/
├── modules/
│   ├── core/                    # Core system components
│   │   ├── config.py           # Configuration management
│   │   ├── exceptions.py       # Custom exceptions
│   │   ├── logger.py          # Logging utilities
│   │   └── models.py          # Data models and types
│   ├── services/               # Business logic services
│   │   ├── base_timbre_service.py
│   │   ├── timbre_analysis_service.py
│   │   ├── timbre_conversion_service.py
│   │   └── timbre_training_service.py
│   └── utils/                  # Enhanced utility modules
│       ├── chinese_instrument_features.py  # Enhanced Chinese instrument analysis
│       └── chinese_music_theory.py        # Chinese music theory
├── models/                     # Neural network architectures
│   ├── attention.py           # Attention mechanisms
│   ├── decoders.py           # Decoder networks
│   ├── encoders.py           # Encoder networks
│   └── model.py              # Main model architecture
├── audio/                     # Audio processing
│   └── processors.py         # Audio signal processing
├── utils/                     # General utilities
│   ├── cache.py              # Caching utilities
│   ├── data.py               # Data handling
│   ├── export.py             # Export functions
│   └── prepare_data.py       # Data preparation
└── example/                   # Examples and demos
    ├── timbre_extraction_visualization.py
    ├── extract_timbre_features.py
    └── visualizations/        # Generated visualizations
```

## 🚀 Key Technologies

### Core Technologies
- **Python 3.8+** - Primary programming language
- **PyTorch** - Deep learning framework
- **Librosa** - Audio analysis and feature extraction
- **NumPy/SciPy** - Scientific computing
- **Matplotlib/Seaborn** - Visualization

### Advanced Audio Processing
- **MFCC** - Mel-Frequency Cepstral Coefficients
- **STFT** - Short-Time Fourier Transform  
- **PYIN** - Advanced pitch estimation
- **Harmonic-Percussive Separation** - Audio component analysis

### Chinese Instrument Analysis
- **Enhanced F0 Analysis** - Fundamental frequency tracking
- **Traditional Technique Detection** - Hua Yin (sliding), Chan Yin (vibrato)
- **Cultural Feature Extraction** - Wu Sheng (pentatonic) scale analysis
- **Ornament Analysis** - Zhuang Shi Yin (decorative notes)

## 🎼 Timbre Extraction Methodology

### 1. Multi-Level Feature Extraction

```
Audio Input → Preprocessing → Feature Extraction → Analysis → Visualization
     ↓              ↓              ↓              ↓          ↓
  WAV/MP3    Normalization   Basic Features   Enhanced     Charts &
             Resampling      MFCC, Spectral   Chinese      Plots
             Windowing       Chroma, ZCR      Features
```

### 2. Chinese Instrument Enhancement Pipeline

```
Raw Audio
    ↓
F0 Extraction (PYIN Algorithm)
    ↓
Traditional Technique Analysis
├── Hua Yin Detection (Sliding Analysis)
├── Chan Yin Analysis (Vibrato Detection)  
├── Wu Sheng Adherence (Pentatonic Scale)
└── Zhuang Shi Yin (Ornament Density)
    ↓
Cultural Feature Quantification
    ↓
Comprehensive Visualization
```

### 3. Feature Categories

#### Basic Audio Features
- **Spectral Features**: Centroid, Bandwidth, Rolloff
- **Temporal Features**: Zero Crossing Rate, RMS Energy
- **Harmonic Features**: Harmonic-to-Noise Ratio, Pitch Stability

#### Enhanced Chinese Features  
- **Pentatonic Adherence**: Conformity to Wu Sheng scale (0-1)
- **Sliding Presence**: Hua Yin technique usage frequency
- **Vibrato Analysis**: Chan Yin rate, extent, and regularity
- **Ornament Density**: Decorative note frequency
- **Cultural Authenticity**: Traditional performance style metrics

#### Advanced Techniques
- **Sliding Velocity Analysis**: Real-time pitch change rates
- **Vibrato Onset Detection**: Automatic vibrato start identification
- **Grace Note Recognition**: Short decorative note detection
- **Portamento Analysis**: Smooth pitch transition measurement

## 📊 Visualization Capabilities

### Comprehensive Analysis Charts
1. **Audio Waveform** - Time domain representation
2. **Spectrogram** - Time-frequency analysis
3. **F0 Contour** - Pitch tracking over time
4. **Sliding Velocity** - Hua Yin technique visualization
5. **Vibrato Pattern** - Chan Yin analysis
6. **Feature Radar Chart** - Multi-dimensional comparison
7. **MFCC Heatmap** - Cepstral coefficient patterns
8. **Spectral Features** - Frequency domain characteristics
9. **Feature Summary** - Quantitative analysis report

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

## 🚀 Quick Start

### Enhanced Visualization (New!)

```bash
# Single file analysis with enhanced Chinese features
cd example
python enhanced_chinese_visualization.py --input erhu1.wav --output ../visualizations

# Process entire directory
python enhanced_chinese_visualization.py --input audio_directory --output ../visualizations --recursive

# Using existing visualization
python timbre_extraction_visualization.py
```

### Generated Visualizations
- **Audio Waveform** - Time domain signal analysis
- **Spectrogram** - Frequency content over time
- **F0 Contour** - Pitch tracking with statistics
- **Sliding Analysis** - Hua Yin (sliding) technique detection
- **Vibrato Analysis** - Chan Yin (vibrato) pattern analysis
- **Feature Radar Chart** - Multi-dimensional feature comparison
- **MFCC Heatmap** - Mel-frequency cepstral coefficients
- **Spectral Features** - Centroid and rolloff analysis
- **Summary Report** - Quantitative feature analysis

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

## 📈 Enhanced Feature Analysis Results

Our enhanced Chinese instrument analysis provides detailed quantitative metrics:

### Sample Analysis Results (Erhu)

| Feature | Erhu1.wav | Erhu2.wav | Description |
|---------|-----------|-----------|-------------|
| **Pentatonic Adherence** | 0.539 | 0.695 | Wu Sheng scale conformity (0-1) |
| **Sliding Presence** | 0.233 | 0.509 | Hua Yin technique frequency |
| **Vibrato Rate** | 2.1 Hz | 2.3 Hz | Chan Yin oscillation frequency |
| **Ornament Density** | 0.056 | 0.193 | Zhuang Shi Yin decorative notes |
| **F0 Mean** | 440.2 Hz | 523.8 Hz | Average fundamental frequency |
| **F0 Range** | 892.1 Hz | 1047.2 Hz | Pitch range span |

### Traditional Technique Terms

- **Hua Yin (滑音)** - Sliding/glissando technique
- **Chan Yin (颤音)** - Vibrato technique  
- **Wu Sheng (五声)** - Pentatonic scale system
- **Zhuang Shi Yin (装饰音)** - Ornamental/decorative notes

## 📚 Documentation

Comprehensive documentation is available in the [docs/](docs/) directory:

### Getting Started
- [📦 Installation Guide](docs/installation.md) - Setup and dependencies
- [🚀 Quick Start Tutorial](docs/quick-start.md) - Basic usage examples
- [🏗️ Architecture Overview](docs/architecture.md) - System design and components

### Core Features
- [🎵 Chinese Instrument Analysis](docs/chinese-instruments.md) - Traditional instrument features
- [📊 Visualization System](docs/visualization.md) - Advanced plotting and charts
- [🔧 Feature Extraction](docs/feature-extraction.md) - Comprehensive audio analysis

### Advanced Topics
- [🎼 Traditional Techniques](docs/traditional-techniques.md) - Hua Yin, Chan Yin, Wu Sheng
- [🎨 Cultural Features](docs/cultural-features.md) - Pentatonic scale and ornaments
- [⚡ Performance Guide](docs/performance.md) - Optimization tips

### API Reference
- [📖 API Documentation](docs/api-reference.md) - Complete function reference
- [⚙️ Configuration Guide](docs/configuration.md) - Settings and parameters
- [🧠 Model Architecture](docs/model-architecture.md) - Neural network designs

## 🎯 Use Cases

1. **Music Education** - Analyze traditional playing techniques
2. **Performance Analysis** - Quantify artistic expression
3. **Instrument Recognition** - Automated classification
4. **Cultural Preservation** - Document traditional methods
5. **Comparative Studies** - Cross-cultural music analysis

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

