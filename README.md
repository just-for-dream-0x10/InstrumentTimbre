# 🎵 InstrumentTimbre - Chinese Traditional Instrument Classification

A comprehensive AI system for analyzing and classifying Chinese traditional instruments using advanced machine learning and audio processing techniques.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train a model
python train.py --data ./data/instruments

# Make predictions
python main.py predict --model model.pth --input audio.wav

# Create visualizations
python main.py visualize --input audio.wav --output ./plots
```

## 🎭 Supported Instruments

- **二胡 (Erhu)** - Two-stringed bowed instrument
- **琵琶 (Pipa)** - Four-stringed plucked instrument  
- **古筝 (Guzheng)** - Plucked zither with movable bridges
- **笛子 (Dizi)** - Transverse bamboo flute
- **古琴 (Guqin)** - Seven-stringed plucked instrument

## 🎯 Main Commands

```bash
# Training
python train.py --data ./data --model enhanced_cnn --epochs 100

# Prediction
python main.py predict -m model.pth -i audio.wav

# Visualization  
python main.py visualize -i audio.wav --style enhanced --instrument erhu

# System info
python main.py info
```

## 📁 Project Structure

```
├── main.py                    # Main CLI entry point
├── train.py                   # Simplified training script
├── config.yaml               # Default configuration
├── requirements.txt          # Dependencies
├── InstrumentTimbre/         # Core package
├── aim/                      # Documentation & plans
├── scripts/                  # Utility scripts
└── data/                     # Training data
```

## 📚 Documentation

For detailed documentation, see `./aim/` directory:
- Development plans
- Architecture overview  
- Visualization guides
- API documentation

## ⚙️ Configuration

Edit `config.yaml` for custom settings:
- Model architecture
- Training parameters
- Feature extraction options
- Output formats

## 🏆 Features

- **28+ Audio Features** specialized for Chinese instruments
- **4 Model Architectures** from CNN to Transformer
- **Professional Visualizations** with cultural analysis
- **Batch Processing** for efficient workflows
- **Modern Training Pipeline** with early stopping & scheduling

## 📞 Support

For questions and issues, check the documentation in `./aim/` or open an issue.