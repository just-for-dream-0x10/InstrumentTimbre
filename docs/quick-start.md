# Quick Start Guide

## 5-Minute Setup

### 1. Installation
```bash
git clone https://github.com/your-repo/InstrumentTimbre.git
cd InstrumentTimbre
pip install -r requirements.txt
```

### 2. Run Demo
```bash
python demo.py
```

### 3. View Results
Check the generated visualizations in `demo_output/` folder.

## Basic Usage

### Enhanced Chinese Instrument Analysis

```bash
# Analyze a single audio file
cd example
python enhanced_chinese_visualization.py --input erhu1.wav --output ../results

# Process multiple files
python enhanced_chinese_visualization.py --input audio_folder --output ../results --recursive
```

### Traditional Feature Extraction

```bash
cd example
python extract_timbre_features.py
python timbre_extraction_visualization.py
```

## Understanding the Output

### Visualization Files
- `*_enhanced_analysis.png` - Complete 9-chart analysis
- `*_basic_analysis.png` - Traditional audio features
- `*_f0_analysis.png` - Pitch and technique analysis

### Key Metrics

| Metric | Range | Description |
|--------|-------|-------------|
| **Pentatonic Adherence** | 0.0-1.0 | Wu Sheng scale conformity |
| **Sliding Presence** | 0.0-1.0 | Hua Yin technique frequency |
| **Vibrato Rate** | 0-15 Hz | Chan Yin oscillation speed |
| **Ornament Density** | 0.0-1.0 | Decorative note frequency |

## Example Analysis Results

### Erhu Performance Comparison

```
Feature                 | Erhu1.wav | Erhu2.wav | Analysis
------------------------|-----------|-----------|----------
Pentatonic Adherence    | 0.539     | 0.695     | Erhu2 more traditional
Sliding Presence        | 0.233     | 0.509     | Erhu2 uses more Hua Yin
Vibrato Rate           | 2.1 Hz    | 2.3 Hz    | Similar Chan Yin
Ornament Density       | 0.056     | 0.193     | Erhu2 more decorative
```

## Common Workflows

### 1. Single File Analysis
```python
from example.enhanced_chinese_visualization import EnhancedChineseInstrumentVisualizer

visualizer = EnhancedChineseInstrumentVisualizer()
features, output_file = visualizer.create_comprehensive_visualization(
    'my_audio.wav', 
    output_dir='my_analysis'
)

print(f"Pentatonic adherence: {features['pentatonic_adherence']:.3f}")
print(f"Sliding presence: {features['sliding_presence']:.3f}")
```

### 2. Batch Processing
```python
import os
from pathlib import Path

visualizer = EnhancedChineseInstrumentVisualizer()
results = visualizer.process_directory('audio_folder', 'analysis_results')

for result in results:
    print(f"{result['audio_file']}: {result['features']['pentatonic_adherence']:.3f}")
```

### 3. Feature Extraction Only
```python
import librosa
from example.enhanced_chinese_visualization import EnhancedChineseInstrumentVisualizer

visualizer = EnhancedChineseInstrumentVisualizer()
audio_data, sr = librosa.load('audio.wav')
features, f0 = visualizer.extract_chinese_features(audio_data, sr)

print("Enhanced Chinese Features:")
for key, value in features.items():
    print(f"  {key}: {value}")
```

## Troubleshooting

### No Visualizations Generated
**Issue**: Script runs but no PNG files created
**Solution**: Check file permissions and output directory
```bash
mkdir -p results
chmod 755 results
```

### Audio Loading Errors
**Issue**: `librosa.load()` fails
**Solution**: Install FFmpeg for additional format support
```bash
# macOS
brew install ffmpeg
# Ubuntu
sudo apt install ffmpeg
```

### Low Feature Values
**Issue**: All features show near-zero values
**Solution**: Check audio quality and instrument type
- Ensure audio contains the expected instrument
- Check for sufficient signal-to-noise ratio
- Verify audio is not heavily processed/compressed

## Next Steps

### For Musicians
1. Analyze your own recordings
2. Compare different performance styles
3. Track improvement over time

### For Researchers
1. Read [Chinese Instruments Guide](chinese-instruments.md)
2. Explore [Traditional Techniques](traditional-techniques.md)
3. Check [API Documentation](api-reference.md)

### For Developers
1. Review [Architecture Overview](architecture.md)
2. Study [Feature Extraction Details](feature-extraction.md)
3. Contribute via [Development Guide](../CONTRIBUTING.md)

## Sample Commands

```bash
# Quick demo
python demo.py

# Analyze traditional instruments
python enhanced_chinese_visualization.py --input instruments/ --output analysis/

# Extract features to JSON
python extract_timbre_features.py --output features.json

# Create visualizations
python timbre_extraction_visualization.py --input audio.wav

# Help and options
python enhanced_chinese_visualization.py --help
```