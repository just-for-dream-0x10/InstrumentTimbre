# Visualization System

## Overview

The InstrumentTimbre visualization system provides comprehensive visual analysis of Chinese traditional instruments through multiple chart types and interactive displays.

## Visualization Types

### 1. Audio Waveform
**Purpose**: Time-domain signal representation
**Shows**: Amplitude variations over time
**Useful for**: Identifying attack patterns, dynamics, silent regions

```python
# Generated automatically in comprehensive analysis
axes[0,0].plot(time, audio_data, alpha=0.7, color='blue')
axes[0,0].set_title('Audio Waveform')
```

### 2. Spectrogram
**Purpose**: Time-frequency analysis
**Shows**: Frequency content evolution over time
**Useful for**: Harmonic structure, formants, spectral changes

```python
D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
librosa.display.specshow(D, y_axis='hz', x_axis='time', sr=sr)
```

### 3. F0 Contour
**Purpose**: Fundamental frequency tracking
**Shows**: Pitch variations over time
**Useful for**: Melody analysis, intonation, pitch stability

Features:
- Mean frequency line
- Valid/invalid pitch regions
- Statistical annotations

### 4. Sliding Analysis (Hua Yin)
**Purpose**: Traditional sliding technique visualization
**Shows**: Pitch velocity and sliding regions
**Useful for**: Quantifying traditional performance techniques

```python
# Velocity calculation in cents per frame
velocity = np.gradient(log_f0)
velocity_smooth = signal.savgol_filter(velocity, 5, 2)

# Sliding detection
sliding_mask = np.abs(velocity_smooth) > sliding_threshold
```

**Color Coding**:
- Purple line: Pitch velocity
- Red regions: Active sliding (Hua Yin)
- Threshold: 20 cents/frame

### 5. Vibrato Analysis (Chan Yin)
**Purpose**: Vibrato pattern visualization
**Shows**: Detrended pitch oscillations
**Useful for**: Analyzing vibrato rate, depth, regularity

**Metrics Displayed**:
- Vibrato rate (Hz)
- Extent (cents)
- Regularity score

### 6. Feature Radar Chart
**Purpose**: Multi-dimensional feature comparison
**Shows**: Six key characteristics on 0-1 scale
**Useful for**: Comparative analysis, style profiling

**Categories**:
1. **Pentatonic Adherence** - Wu Sheng scale conformity
2. **Sliding Presence** - Hua Yin technique usage
3. **Vibrato Rate** - Chan Yin normalized frequency
4. **Ornament Density** - Decorative note frequency
5. **F0 Stability** - Pitch consistency
6. **Spectral Richness** - Harmonic content

### 7. MFCC Heatmap
**Purpose**: Mel-frequency cepstral coefficient analysis
**Shows**: Spectral envelope characteristics over time
**Useful for**: Timbre analysis, instrument identification

**Features**:
- 13 MFCC coefficients
- Time progression (horizontal axis)
- Color intensity represents coefficient values

### 8. Spectral Features
**Purpose**: Frequency domain characteristics
**Shows**: Spectral centroid and rolloff over time
**Useful for**: Brightness analysis, spectral evolution

**Metrics**:
- **Spectral Centroid**: "Brightness" of sound
- **Spectral Rolloff**: High-frequency content measure

### 9. Feature Summary
**Purpose**: Quantitative analysis report
**Shows**: All extracted features with cultural context
**Useful for**: Documentation, comparative studies

## Usage Examples

### Basic Visualization
```python
from example.enhanced_chinese_visualization import EnhancedChineseInstrumentVisualizer

visualizer = EnhancedChineseInstrumentVisualizer()
features, output_file = visualizer.create_comprehensive_visualization(
    'erhu_performance.wav',
    output_dir='visualizations'
)
```

### Batch Processing
```python
# Process entire directory
results = visualizer.process_directory(
    input_dir='audio_files',
    output_dir='batch_analysis'
)

print(f"Processed {len(results)} files")
```

### Command Line Usage
```bash
# Single file
python enhanced_chinese_visualization.py --input audio.wav --output results/

# Directory processing
python enhanced_chinese_visualization.py --input audio_dir/ --output results/ --recursive

# Help
python enhanced_chinese_visualization.py --help
```

## Customization

### Color Schemes
```python
# Modify color palette
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['#1f77b4', '#ff7f0e', '#2ca02c'])

# Custom colormap for heatmaps
sns.heatmap(mfccs, cmap='plasma', ax=axes[2,0])
```

### Chart Layout
```python
# Adjust figure size
fig, axes = plt.subplots(3, 3, figsize=(20, 14))

# Modify spacing
plt.tight_layout(pad=3.0)
```

### Export Options
```python
# High-resolution export
plt.savefig(output_file, dpi=300, bbox_inches='tight')

# Different formats
plt.savefig('analysis.pdf')  # PDF
plt.savefig('analysis.svg')  # SVG for vector graphics
```

## Interactive Features

### Jupyter Notebook Integration
```python
%matplotlib notebook
import ipywidgets as widgets

# Interactive parameter adjustment
@widgets.interact(threshold=(10, 50, 5))
def update_sliding_analysis(threshold=20):
    # Recalculate sliding detection with new threshold
    sliding_mask = np.abs(velocity_smooth) > threshold
    # Update visualization
```

### Plotly Integration (Future Enhancement)
```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Interactive plots with zoom, pan, hover
fig = go.Figure()
fig.add_trace(go.Scatter(x=times, y=f0, mode='lines', name='F0'))
fig.show()
```

## Performance Optimization

### Large Files
```python
# Reduce sample rate for faster processing
audio_data, sr = librosa.load(audio_file, sr=16000)

# Limit analysis duration
max_duration = 60  # seconds
if len(audio_data) > max_duration * sr:
    audio_data = audio_data[:max_duration * sr]
```

### Batch Processing
```python
# Parallel processing
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

def process_file(audio_file):
    return visualizer.create_comprehensive_visualization(audio_file)

with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
    results = list(executor.map(process_file, audio_files))
```

## Error Handling

### Common Issues and Solutions

1. **Font Rendering Problems**
   ```python
   # Use safe fonts
   plt.rcParams['font.family'] = 'DejaVu Sans'
   ```

2. **Memory Issues**
   ```python
   # Clear figures after saving
   plt.close('all')
   
   # Reduce figure DPI for large batches
   plt.savefig(output_file, dpi=150)
   ```

3. **Audio Loading Errors**
   ```python
   try:
       audio_data, sr = librosa.load(audio_file)
   except Exception as e:
       print(f"Error loading {audio_file}: {e}")
       return None
   ```

## Output Formats

### Standard Output
- **PNG**: High-quality raster images (default)
- **PDF**: Vector format for publications
- **SVG**: Scalable vector graphics

### File Naming Convention
```
{filename}_enhanced_analysis.png    # Complete 9-chart analysis
{filename}_basic_analysis.png       # Traditional features only
{filename}_f0_analysis.png         # F0 and techniques focus
```

### Metadata Export
```python
# Save analysis metadata
metadata = {
    'filename': audio_file,
    'features': features,
    'processing_time': time.time() - start_time,
    'parameters': analysis_params
}

with open(f'{base_name}_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
```

## Integration with Other Tools

### Export to Scientific Software
```python
# MATLAB format
import scipy.io
scipy.io.savemat('features.mat', features)

# R format
import pandas as pd
df = pd.DataFrame(features)
df.to_csv('features.csv')
```

### Web Applications
```python
# Base64 encoding for web display
import base64
from io import BytesIO

buf = BytesIO()
plt.savefig(buf, format='png')
buf.seek(0)
img_base64 = base64.b64encode(buf.read()).decode()
```