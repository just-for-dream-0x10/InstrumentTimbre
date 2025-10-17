# Basic Examples

## Example 1: Single File Analysis

### Simple Feature Extraction

```python
import librosa
from example.enhanced_chinese_visualization import EnhancedChineseInstrumentVisualizer

# Initialize visualizer
visualizer = EnhancedChineseInstrumentVisualizer()

# Load and analyze audio
audio_data, sr = librosa.load('erhu_performance.wav')
features, f0 = visualizer.extract_chinese_features(audio_data, sr)

# Print key features
print("Chinese Instrument Features:")
print(f"  Pentatonic Adherence: {features['pentatonic_adherence']:.3f}")
print(f"  Sliding Presence: {features['sliding_presence']:.3f}")
print(f"  Vibrato Rate: {features['vibrato_rate']:.1f} Hz")
print(f"  Ornament Density: {features['ornament_density']:.3f}")
```

### Expected Output
```
Chinese Instrument Features:
  Pentatonic Adherence: 0.695
  Sliding Presence: 0.509
  Vibrato Rate: 2.3 Hz
  Ornament Density: 0.193
```

## Example 2: Complete Visualization

### Generate Comprehensive Charts

```python
from example.enhanced_chinese_visualization import EnhancedChineseInstrumentVisualizer

# Create visualizer
visualizer = EnhancedChineseInstrumentVisualizer()

# Generate complete analysis
features, output_file = visualizer.create_comprehensive_visualization(
    audio_file='examples/erhu_sample.wav',
    output_dir='my_analysis'
)

print(f"Analysis saved to: {output_file}")
print(f"Features extracted: {len(features)} metrics")
```

### Generated Files
- `my_analysis/erhu_sample_enhanced_analysis.png` - Complete 9-chart visualization
- Includes: waveform, spectrogram, F0, sliding, vibrato, radar chart, MFCC, spectral features, summary

## Example 3: Batch Processing

### Process Multiple Files

```python
import os
from pathlib import Path
from example.enhanced_chinese_visualization import EnhancedChineseInstrumentVisualizer

# Initialize
visualizer = EnhancedChineseInstrumentVisualizer()

# Find all audio files
audio_dir = Path('audio_samples')
audio_files = list(audio_dir.glob('*.wav'))

# Process each file
results = []
for audio_file in audio_files:
    try:
        features, output_file = visualizer.create_comprehensive_visualization(
            str(audio_file),
            output_dir='batch_analysis'
        )
        results.append({
            'file': audio_file.name,
            'features': features,
            'visualization': output_file
        })
        print(f"✓ Processed: {audio_file.name}")
    except Exception as e:
        print(f"✗ Failed: {audio_file.name} - {e}")

print(f"\nSuccessfully processed {len(results)} files")
```

## Example 4: Feature Comparison

### Compare Different Performances

```python
import pandas as pd
import matplotlib.pyplot as plt

# Extract features from multiple files
files = ['erhu1.wav', 'erhu2.wav', 'erhu3.wav']
all_features = []

visualizer = EnhancedChineseInstrumentVisualizer()

for file in files:
    audio_data, sr = librosa.load(file)
    features, f0 = visualizer.extract_chinese_features(audio_data, sr)
    features['filename'] = file
    all_features.append(features)

# Create comparison dataframe
df = pd.DataFrame(all_features)

# Plot comparison
key_features = ['pentatonic_adherence', 'sliding_presence', 'vibrato_rate', 'ornament_density']

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.ravel()

for i, feature in enumerate(key_features):
    axes[i].bar(df['filename'], df[feature])
    axes[i].set_title(feature.replace('_', ' ').title())
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('feature_comparison.png')
print("Comparison chart saved as 'feature_comparison.png'")
```

## Example 5: Traditional Technique Focus

### Analyze Specific Techniques

```python
import numpy as np
import librosa
from scipy import signal

def analyze_hua_yin_detail(audio_file):
    """Detailed Hua Yin (sliding) analysis"""
    
    # Load audio
    audio_data, sr = librosa.load(audio_file)
    
    # Extract F0
    f0, _, _ = librosa.pyin(audio_data, fmin=librosa.note_to_hz('C2'), 
                           fmax=librosa.note_to_hz('C7'), sr=sr)
    
    # Calculate sliding metrics
    valid_f0 = f0[~np.isnan(f0)]
    if len(valid_f0) > 10:
        log_f0 = np.log2(valid_f0 + 1e-10) * 1200
        velocity = np.gradient(log_f0)
        velocity_smooth = signal.savgol_filter(velocity, 5, 2)
        
        # Detailed sliding analysis
        sliding_threshold = 20
        sliding_mask = np.abs(velocity_smooth) > sliding_threshold
        
        # Find sliding segments
        diff = np.diff(np.concatenate(([False], sliding_mask, [False])).astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        segments = []
        for start, end in zip(starts, ends):
            duration = (end - start) * (512 / sr)  # Convert to seconds
            direction = 'up' if np.mean(velocity_smooth[start:end]) > 0 else 'down'
            max_velocity = np.max(np.abs(velocity_smooth[start:end]))
            
            segments.append({
                'start_time': start * (512 / sr),
                'duration': duration,
                'direction': direction,
                'max_velocity': max_velocity
            })
        
        return {
            'total_segments': len(segments),
            'segments': segments,
            'sliding_percentage': np.sum(sliding_mask) / len(sliding_mask),
            'avg_velocity': np.mean(np.abs(velocity_smooth))
        }
    
    return None

# Analyze Hua Yin in detail
hua_yin_analysis = analyze_hua_yin_detail('erhu_with_slides.wav')

if hua_yin_analysis:
    print("Detailed Hua Yin Analysis:")
    print(f"  Total sliding segments: {hua_yin_analysis['total_segments']}")
    print(f"  Sliding percentage: {hua_yin_analysis['sliding_percentage']:.1%}")
    print(f"  Average velocity: {hua_yin_analysis['avg_velocity']:.1f} cents/frame")
    
    print("\nSegment Details:")
    for i, segment in enumerate(hua_yin_analysis['segments'][:5]):  # Show first 5
        print(f"  Segment {i+1}: {segment['direction']} slide, "
              f"{segment['duration']:.2f}s, max velocity: {segment['max_velocity']:.1f}")
```

## Example 6: Export Results

### Save Analysis to Different Formats

```python
import json
import csv
from datetime import datetime

def export_analysis_results(features, output_format='json'):
    """Export analysis results to various formats"""
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if output_format == 'json':
        # Export to JSON
        output_file = f'analysis_{timestamp}.json'
        with open(output_file, 'w') as f:
            json.dump(features, f, indent=2, default=str)
        print(f"Results exported to {output_file}")
    
    elif output_format == 'csv':
        # Export to CSV
        output_file = f'analysis_{timestamp}.csv'
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Feature', 'Value'])
            for key, value in features.items():
                writer.writerow([key, value])
        print(f"Results exported to {output_file}")
    
    elif output_format == 'txt':
        # Export to text report
        output_file = f'analysis_{timestamp}.txt'
        with open(output_file, 'w') as f:
            f.write("Chinese Instrument Analysis Report\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("Traditional Techniques:\n")
            f.write(f"  Pentatonic Adherence: {features.get('pentatonic_adherence', 0):.3f}\n")
            f.write(f"  Sliding Presence: {features.get('sliding_presence', 0):.3f}\n")
            f.write(f"  Vibrato Rate: {features.get('vibrato_rate', 0):.1f} Hz\n")
            f.write(f"  Ornament Density: {features.get('ornament_density', 0):.3f}\n")
        print(f"Report exported to {output_file}")

# Example usage
visualizer = EnhancedChineseInstrumentVisualizer()
audio_data, sr = librosa.load('sample.wav')
features, f0 = visualizer.extract_chinese_features(audio_data, sr)

# Export in different formats
export_analysis_results(features, 'json')
export_analysis_results(features, 'csv')
export_analysis_results(features, 'txt')
```

## Example 7: Real-time Analysis

### Stream Processing (Conceptual)

```python
import sounddevice as sd
import numpy as np
from collections import deque

class RealTimeAnalyzer:
    def __init__(self, sample_rate=22050, chunk_size=1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.buffer = deque(maxlen=sample_rate * 5)  # 5-second buffer
        self.visualizer = EnhancedChineseInstrumentVisualizer()
    
    def audio_callback(self, indata, frames, time, status):
        """Callback for real-time audio processing"""
        if status:
            print(status)
        
        # Add to buffer
        self.buffer.extend(indata[:, 0])
        
        # Analyze when buffer is full
        if len(self.buffer) == self.buffer.maxlen:
            audio_data = np.array(self.buffer)
            features, f0 = self.visualizer.extract_chinese_features(audio_data, self.sample_rate)
            
            # Print real-time features
            print(f"\rPentatonic: {features['pentatonic_adherence']:.3f} | "
                  f"Sliding: {features['sliding_presence']:.3f} | "
                  f"Vibrato: {features['vibrato_rate']:.1f} Hz", end='')
    
    def start_analysis(self):
        """Start real-time analysis"""
        print("Starting real-time analysis... Press Ctrl+C to stop")
        try:
            with sd.InputStream(callback=self.audio_callback,
                              channels=1, samplerate=self.sample_rate):
                sd.sleep(10000)  # Run for a long time
        except KeyboardInterrupt:
            print("\nAnalysis stopped")

# Usage (requires microphone)
# analyzer = RealTimeAnalyzer()
# analyzer.start_analysis()
```

## Running the Examples

### Setup
```bash
# Ensure you're in the project directory
cd InstrumentTimbre

# Install dependencies if not already done
pip install -r requirements.txt

# Run examples
python -c "exec(open('docs/examples/example1.py').read())"
```

### Sample Data
Download sample audio files from:
- [Chinese Instrument Samples](https://example.com/samples)
- Or use the provided `example/erhu1.wav` and `example/erhu2.wav`