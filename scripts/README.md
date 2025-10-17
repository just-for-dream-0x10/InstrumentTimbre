# 🎵 InstrumentTimbre Automation Scripts

## 📁 Script Overview

This directory contains automation scripts for the complete InstrumentTimbre workflow, from data preparation to model deployment.

### 🔧 Data Processing
- `prepare_data.sh` - Convert long audio files into training clips

### 🚀 Model Training  
- `standard_train.sh` - Proven training pipeline (97.86% validation accuracy)

### 🎯 Model Testing
- `test_predict.sh` - Prediction testing and performance validation

### 🎨 Visualization Generation
- `create_visualizations.sh` - Professional audio analysis charts

## 🎯 Complete Workflow

### 1️⃣ Data Preparation
```bash
# Place audio files in ./data/samples/
# Then run the segmentation script
bash scripts/prepare_data.sh
```
**What it does**: Converts long audio recordings into 3-second training clips organized by instrument type.

### 2️⃣ Model Training
```bash
# Train using proven configuration (97.86% validation accuracy)
bash scripts/standard_train.sh
```
**What it does**: Trains a CNN model to classify Chinese instruments with optimized hyperparameters.

### 3️⃣ Model Testing
```bash
# Test the trained model
bash scripts/test_predict.sh
```
**What it does**: Validates model performance on test data and generates prediction reports.

### 4️⃣ Visualization Analysis
```bash
# Generate professional audio analysis charts
bash scripts/create_visualizations.sh
```
**What it does**: Creates spectrograms, MFCC plots, and instrument-specific analysis charts.

## 📊 Expected Results

### Training Performance
- **Validation Accuracy**: ~97.86%
- **Training Time**: ~15 minutes
- **Supported Classes**: 7 types (erhu, pipa, guzheng, drums, bass, vocals, mixed)

### Visualization Outputs
- **English-style**: Standard audio analysis charts
- **Enhanced**: Chinese instrument-specific analysis
- **Format**: High-quality PNG (300 DPI)

### Prediction Capabilities
- **Single File**: Real-time audio classification
- **Batch Processing**: Process entire directories
- **Confidence Scores**: Top-3 predictions with probabilities

## 🛠️ Customization

### Modify Training Parameters
Edit `standard_train.sh`:
```bash
BATCH_SIZE=32      # Batch size
EPOCHS=20          # Number of epochs
LEARNING_RATE=0.01 # Learning rate
```

### Modify Visualization Settings
Edit `create_visualizations.sh`:
```bash
STYLE="both"       # both, english, enhanced
DPI=300           # Image quality
```

## 📂 Directory Structure

```
scripts/
├── prepare_data.sh           # Audio segmentation
├── standard_train.sh         # Model training
├── test_predict.sh          # Prediction testing
├── create_visualizations.sh # Chart generation
└── README.md               # This guide
```

## 🎉 One-Click Complete Pipeline

```bash
# Full end-to-end pipeline
bash scripts/prepare_data.sh        # Prepare data
bash scripts/standard_train.sh      # Train model
bash scripts/test_predict.sh        # Test predictions
bash scripts/create_visualizations.sh # Generate visualizations
```

## 💼 Production Use Cases

### Music Education
- Automatically identify instruments in student recordings
- Generate visual feedback for performance analysis

### Digital Libraries
- Catalog audio collections by instrument type
- Extract metadata from traditional music recordings

### Research Applications
- Analyze acoustic characteristics of traditional instruments
- Study performance techniques and regional variations

## ⚠️ Requirements & Limitations

### System Requirements
- **Audio Formats**: .wav, .mp3, .flac supported
- **Memory**: Minimum 8GB RAM recommended
- **Time**: Complete pipeline takes 30-60 minutes
- **Storage**: Visualizations require ~100MB per instrument

### Data Requirements
- **Minimum**: 10+ audio files per instrument class
- **Recommended**: 100+ clips per class for best accuracy
- **Quality**: Clean recordings without background noise preferred

## 🔍 Troubleshooting

### Common Issues
- **"Data directory not found"**: Ensure audio files are in correct location
- **"Out of memory"**: Reduce batch_size parameter
- **"Training interrupted"**: Use resume functionality to continue

### Diagnostic Commands
```bash
# Check data availability
find ./data -name "*.wav" | wc -l

# Check model outputs
ls -la ./outputs/

# Validate prediction results
cat ./predictions_results.json | head -20
```

### Performance Optimization
- Use SSD storage for faster data loading
- Close other applications during training
- Consider reducing batch size on low-memory systems

## 📈 Success Metrics

A successful run should achieve:
- **>95% validation accuracy** on instrument classification
- **<0.1 validation loss** indicating good convergence
- **Balanced predictions** across all instrument classes
- **High-quality visualizations** showing clear acoustic differences