# 🚀 InstrumentTimbre Quick Start Guide

## 🎯 What This Project Does

InstrumentTimbre is an AI system that **automatically identifies and analyzes traditional Chinese musical instruments** from audio recordings. It can distinguish between different instruments like Erhu, Pipa, Guzheng, and others with 97.86% accuracy.

## 📋 Real-World Applications

- **Music Education**: Automatically identify instruments in recordings for teaching
- **Digital Music Libraries**: Catalog and tag audio files by instrument type
- **Cultural Preservation**: Document and analyze traditional Chinese music performances
- **Audio Processing**: Extract detailed acoustic features from instrument recordings
- **Research**: Study acoustic characteristics of traditional Chinese instruments

## 🚀 5-Minute Setup

### 1️⃣ Prepare Your Audio Data
```bash
# Create data directory and add your audio files
mkdir -p data/samples
# Copy your .wav, .mp3, or .flac files to data/samples/
```

### 2️⃣ Train the Model
```bash
# Process data and train the model
bash scripts/prepare_data.sh
bash scripts/standard_train.sh
```

### 3️⃣ Test Predictions
```bash
# Test the trained model
bash scripts/test_predict.sh
```

### 4️⃣ Generate Analysis Charts
```bash
# Create professional audio analysis visualizations
bash scripts/create_visualizations.sh
```

## 🎵 Supported Instruments

The system can identify these traditional Chinese instruments:

- **Erhu (二胡)** - Two-stringed bowed instrument
- **Pipa (琵琶)** - Four-stringed plucked instrument
- **Guzheng (古筝)** - Plucked zither with movable bridges
- **Dizi (笛子)** - Transverse bamboo flute
- **Guqin (古琴)** - Seven-stringed plucked instrument

Plus general categories: vocals, piano, bass, drums, mixed music.

## 📊 What You Get

### Training Performance
- ✅ **97.86% accuracy** on validation data
- ✅ **~15 minutes** training time on typical datasets
- ✅ **Automatic early stopping** to prevent overfitting

### Output Features
- 🎨 **Professional visualizations**: Spectrograms, MFCC analysis, F0 tracking
- 🎯 **Real-time predictions**: Single file or batch processing
- 📊 **Detailed analysis**: 34-dimensional audio feature extraction
- 🔍 **Confidence scores**: Top-k predictions with probability scores

## 🔧 Basic Commands

```bash
# Train a model
python train.py --data ./data/clips --epochs 20

# Predict instrument type
python main.py predict --model outputs/best_acc_model.pth --input audio.wav

# Create audio analysis charts
python main.py visualize --input audio.wav --instrument erhu

# Get help
python main.py --help
```

## 📁 Project Structure

```
InstrumentTimbre/
├── main.py              # Main CLI interface
├── train.py             # Training script
├── config.yaml          # Configuration file
├── scripts/             # Automation scripts
├── data/                # Data directory
├── outputs/             # Model outputs
└── visualizations/      # Generated charts
```

## 🎯 Success Indicators

After training, you should see:
```
🎉 Training Completed!
Best Validation Accuracy: 0.9786
Best Validation Loss: 0.0600
Results saved to: outputs
```

## 💼 Advanced Usage

### Custom Training
```bash
python train.py \
    --data ./your_data \
    --model transformer \
    --epochs 50 \
    --batch-size 64
```

### Batch Prediction
```bash
python main.py predict \
    --model outputs/best_acc_model.pth \
    --input ./audio_directory/ \
    --output results.json \
    --format json
```

### High-Quality Visualizations
```bash
python main.py visualize \
    --input audio.wav \
    --style enhanced \
    --instrument erhu \
    --dpi 300
```

## 🎵 Start analyzing traditional Chinese instruments with AI today!