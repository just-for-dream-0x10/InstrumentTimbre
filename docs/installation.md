# Installation Guide

## System Requirements

### Python Version
- Python 3.8 or higher
- Recommended: Python 3.9+

### Operating System
- Windows 10/11
- macOS 10.15+
- Linux (Ubuntu 18.04+, CentOS 7+)

## Installation Methods

### Method 1: Clone from Repository

```bash
# Clone the repository
git clone https://github.com/your-repo/InstrumentTimbre.git
cd InstrumentTimbre

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Method 2: Conda Environment

```bash
# Create conda environment
conda create -n instrument-timbre python=3.9
conda activate instrument-timbre

# Install PyTorch (adjust for your CUDA version)
conda install pytorch torchvision torchaudio -c pytorch

# Install other dependencies
pip install -r requirements.txt
```

## Dependencies

### Core Dependencies
```
torch>=1.9.0
librosa>=0.9.0
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.5.0
seaborn>=0.11.0
pandas>=1.3.0
```

### Audio Processing
```
soundfile>=0.10.0
resampy>=0.2.2
```

### Visualization
```
plotly>=5.0.0
ipywidgets>=7.6.0  # For Jupyter notebooks
```

### Optional Dependencies
```
jupyter>=1.0.0     # For notebook examples
pytest>=6.0.0      # For running tests
```

## Verification

### Quick Test
```bash
# Run the demo
python demo.py

# Test enhanced visualization
cd example
python enhanced_chinese_visualization.py --input erhu1.wav --output test_output
```

### Expected Output
```
✅ Found 2 example files
🎨 Running Enhanced Visualization...
📊 Processing: erhu1.wav
Creating visualization for: erhu1.wav
Enhanced visualization saved: test_output/erhu1_enhanced_analysis.png
✅ Successfully processed erhu1.wav
```

## Common Issues

### 1. Audio Loading Errors
**Problem**: `librosa.load()` fails to load audio files
**Solution**: Install additional audio codecs
```bash
# Install FFmpeg for additional audio format support
# Windows (using chocolatey):
choco install ffmpeg
# macOS:
brew install ffmpeg
# Ubuntu:
sudo apt install ffmpeg
```

### 2. Font Rendering Issues
**Problem**: Chinese characters not displaying correctly
**Solution**: The system now uses English-only labels to avoid font issues. No additional setup required.

### 3. CUDA/GPU Issues
**Problem**: PyTorch not using GPU acceleration
**Solution**: Install CUDA-compatible PyTorch
```bash
# Check CUDA version
nvidia-smi

# Install appropriate PyTorch version
# For CUDA 11.6:
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```

### 4. Memory Issues
**Problem**: Out of memory errors during processing
**Solution**: Reduce audio file size or batch size
```python
# In your code, reduce sample rate
audio_data, sr = librosa.load(audio_file, sr=16000)  # Instead of 22050
```

## Development Setup

### For Contributors
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/

# Run linting
flake8 InstrumentTimbre/
black InstrumentTimbre/
```

### IDE Configuration
- **VS Code**: Install Python and Jupyter extensions
- **PyCharm**: Configure Python interpreter to use virtual environment
- **Jupyter**: Install ipywidgets for interactive plots

## Performance Optimization

### Audio Processing
```bash
# Install optimized BLAS libraries
conda install mkl

# For faster audio loading
pip install soundfile
```

### Visualization
```bash
# For faster plotting
pip install kaleido  # For plotly static image export
```

## Next Steps

After successful installation:
1. Read the [Quick Start Guide](quick-start.md)
2. Try the [Basic Examples](examples/basic.md)
3. Explore [Chinese Instrument Analysis](chinese-instruments.md)