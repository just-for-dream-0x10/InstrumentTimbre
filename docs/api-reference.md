# API Reference

## Core Modules

### ChineseInstrumentAnalyzer

Main class for enhanced Chinese instrument analysis.

```python
from InstrumentTimbre.modules.utils.chinese_instrument_features import ChineseInstrumentAnalyzer

analyzer = ChineseInstrumentAnalyzer()
```

#### Methods

##### `extract_chinese_features(audio_data, sample_rate, instrument_hint=None)`

Extract comprehensive Chinese instrument features.

**Parameters:**
- `audio_data` (np.ndarray): Audio signal data
- `sample_rate` (int): Audio sample rate
- `instrument_hint` (InstrumentType, optional): Expected instrument type

**Returns:**
- `ChineseInstrumentFeatures`: Feature object with traditional techniques

**Example:**
```python
import librosa
from InstrumentTimbre.modules.core.models import InstrumentType

audio_data, sr = librosa.load('erhu.wav')
features = analyzer.extract_chinese_features(audio_data, sr, InstrumentType.ERHU)

print(f"Pentatonic adherence: {features.pentatonic_adherence}")
print(f"Sliding detection: {features.sliding_detection}")
print(f"Vibrato analysis: {features.vibrato_analysis}")
```

### EnhancedChineseInstrumentVisualizer

Advanced visualization with cultural-aware analysis.

```python
from example.enhanced_chinese_visualization import EnhancedChineseInstrumentVisualizer

visualizer = EnhancedChineseInstrumentVisualizer()
```

#### Methods

##### `create_comprehensive_visualization(audio_file, output_dir="visualizations")`

Create complete 9-chart analysis visualization.

**Parameters:**
- `audio_file` (str): Path to audio file
- `output_dir` (str): Output directory for visualizations

**Returns:**
- `tuple`: (features_dict, output_file_path)

**Example:**
```python
features, output_file = visualizer.create_comprehensive_visualization(
    'erhu_performance.wav',
    output_dir='analysis_results'
)
```

##### `extract_chinese_features(audio_data, sr)`

Extract features with fallback implementation.

**Parameters:**
- `audio_data` (np.ndarray): Audio signal
- `sr` (int): Sample rate

**Returns:**
- `tuple`: (features_dict, f0_array)

##### `process_directory(input_dir, output_dir="visualizations")`

Batch process all audio files in directory.

**Parameters:**
- `input_dir` (str): Input directory path
- `output_dir` (str): Output directory path

**Returns:**
- `list`: Results for each processed file

## Data Models

### ChineseInstrumentFeatures

Feature container for Chinese instrument analysis.

```python
@dataclass
class ChineseInstrumentFeatures:
    # Basic features
    pentatonic_adherence: Optional[float] = None
    ornament_density: Optional[float] = None
    rhythmic_complexity: Optional[float] = None
    
    # Traditional techniques
    sliding_detection: Optional[np.ndarray] = None
    vibrato_analysis: Optional[Dict[str, float]] = None
    pitch_bending: Optional[Dict[str, float]] = None
    
    # Enhanced features
    sliding_velocity: Optional[np.ndarray] = None
    sliding_curvature: Optional[np.ndarray] = None
    portamento_detection: Optional[Dict[str, float]] = None
    
    # Cultural elements
    modal_characteristics: Optional[Dict[str, float]] = None
```

### InstrumentType

Enumeration of supported Chinese instruments.

```python
class InstrumentType(Enum):
    ERHU = "erhu"
    PIPA = "pipa"
    GUZHENG = "guzheng"
    DIZI = "dizi"
    GUQIN = "guqin"
    XIAO = "xiao"
    SUONA = "suona"
```

## Utility Functions

### Feature Extraction

##### `_detect_sliding_techniques(f0, sample_rate)`

Detect Hua Yin (sliding) techniques.

**Parameters:**
- `f0` (np.ndarray): Fundamental frequency array
- `sample_rate` (int): Audio sample rate

**Returns:**
- `np.ndarray`: Binary array indicating sliding regions

##### `_analyze_vibrato_patterns(f0, sample_rate)`

Analyze Chan Yin (vibrato) patterns using FFT.

**Parameters:**
- `f0` (np.ndarray): Fundamental frequency array
- `sample_rate` (int): Audio sample rate

**Returns:**
- `dict`: Vibrato characteristics (rate, extent, regularity)

##### `_calculate_pentatonic_adherence(f0)`

Calculate Wu Sheng (pentatonic) scale adherence.

**Parameters:**
- `f0` (np.ndarray): Fundamental frequency array

**Returns:**
- `float`: Adherence score (0.0-1.0)

### Audio Processing

##### `librosa.pyin()`

Enhanced pitch estimation for Chinese instruments.

```python
f0, voiced_flag, voiced_probs = librosa.pyin(
    audio_data,
    fmin=librosa.note_to_hz('C2'),
    fmax=librosa.note_to_hz('C7'),
    sr=sample_rate
)
```

##### `signal.savgol_filter()`

Savitzky-Golay filter for noise reduction.

```python
velocity_smooth = signal.savgol_filter(velocity, window_length=5, polyorder=2)
```

## Configuration

### Instrument Parameters

Each instrument type has specific analysis parameters:

```python
instrument_params = {
    InstrumentType.ERHU: {
        'fundamental_range': (196, 1568),    # G3 to G6
        'expected_vibrato_rate': (4, 8),     # Hz
        'sliding_sensitivity': 0.7,
        'bow_noise_detection': True
    },
    InstrumentType.PIPA: {
        'fundamental_range': (220, 2093),    # A3 to C7
        'attack_time_range': (0.01, 0.1),
        'tremolo_detection': True,
        'plucking_angle_analysis': True
    }
}
```

### Analysis Thresholds

```python
sliding_params = {
    'min_duration_ms': 50,
    'max_duration_ms': 2000,
    'velocity_threshold': 20,        # cents per frame
    'curvature_sensitivity': 0.1
}

vibrato_params = {
    'min_rate_hz': 2.0,
    'max_rate_hz': 15.0,
    'depth_threshold_cents': 10,
    'regularity_threshold': 0.3
}
```

## Error Handling

### Common Exceptions

```python
try:
    features = analyzer.extract_chinese_features(audio_data, sr)
except AudioProcessingError as e:
    print(f"Audio processing failed: {e}")
except InstrumentDetectionError as e:
    print(f"Instrument detection failed: {e}")
except FeatureExtractionError as e:
    print(f"Feature extraction failed: {e}")
```

### Validation

```python
def validate_audio_input(audio_data, sample_rate):
    """Validate audio input parameters"""
    if len(audio_data) == 0:
        raise ValueError("Empty audio data")
    if sample_rate <= 0:
        raise ValueError("Invalid sample rate")
    if len(audio_data) < sample_rate:  # Less than 1 second
        raise ValueError("Audio too short for analysis")
```

## Performance Tips

### Memory Optimization

```python
# Process in chunks for large files
chunk_size = sample_rate * 30  # 30 seconds
for i in range(0, len(audio_data), chunk_size):
    chunk = audio_data[i:i+chunk_size]
    chunk_features = extract_features(chunk, sample_rate)
```

### Parallel Processing

```python
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

def process_audio_parallel(audio_files):
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        results = list(executor.map(extract_features, audio_files))
    return results
```

## Integration Examples

### With Jupyter Notebooks

```python
import matplotlib.pyplot as plt
%matplotlib inline

# Display visualizations inline
features, _ = visualizer.create_comprehensive_visualization('audio.wav')
plt.show()
```

### With Web Applications

```python
import json
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/analyze/<filename>')
def analyze_audio(filename):
    features = analyzer.extract_chinese_features(filename)
    return jsonify({
        'pentatonic_adherence': features.pentatonic_adherence,
        'sliding_presence': float(np.mean(features.sliding_detection)),
        'vibrato_rate': features.vibrato_analysis.get('rate_hz', 0)
    })
```

### Command Line Tools

```python
import argparse

def main():
    parser = argparse.ArgumentParser(description='Chinese Instrument Analysis')
    parser.add_argument('--input', required=True, help='Input audio file')
    parser.add_argument('--output', default='.', help='Output directory')
    parser.add_argument('--instrument', choices=['erhu', 'pipa', 'guzheng'])
    
    args = parser.parse_args()
    
    # Process audio
    analyzer = ChineseInstrumentAnalyzer()
    # ... analysis code
```