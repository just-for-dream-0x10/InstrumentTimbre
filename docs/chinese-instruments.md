# Chinese Instrument Analysis

## Overview

The InstrumentTimbre system provides specialized analysis capabilities for Chinese traditional instruments, incorporating cultural knowledge and traditional performance techniques into modern audio analysis.

## Supported Instruments

### 1. Erhu (二胡) - Two-Stringed Bowed Instrument

**Characteristics:**
- Frequency range: G3 to G6 (196-1568 Hz)
- Primary techniques: Hua Yin (sliding), Chan Yin (vibrato)
- Cultural significance: Expressive, vocal-like qualities

**Enhanced Features:**
```python
erhu_features = {
    'sliding_sensitivity': 0.7,
    'expected_vibrato_rate': (4, 8),  # Hz
    'bow_noise_detection': True,
    'string_crossing_analysis': True
}
```

### 2. Pipa (琵琶) - Four-Stringed Plucked Instrument

**Characteristics:**
- Frequency range: A3 to C7 (220-2093 Hz)
- Primary techniques: Tremolo (Lunzhi), plucking variations
- Cultural significance: Percussive attacks, rapid ornaments

**Enhanced Features:**
```python
pipa_features = {
    'attack_time_range': (0.01, 0.1),  # Fast attack
    'tremolo_detection': True,
    'plucking_angle_analysis': True,
    'nail_vs_flesh_detection': True
}
```

### 3. Guzheng (古筝) - Plucked Zither

**Characteristics:**
- Frequency range: G3 to C7 (196-2093 Hz)
- Primary techniques: Glissando, harmonics, tremolo
- Cultural significance: Flowing, cascading sounds

**Enhanced Features:**
```python
guzheng_features = {
    'glissando_types': ['up', 'down', 'bidirectional'],
    'resonance_decay': (2.0, 8.0),  # Long resonance
    'string_damping_analysis': True,
    'harmonic_plucking': True
}
```

### 4. Dizi (笛子) - Transverse Bamboo Flute

**Characteristics:**
- Frequency range: D4 to D7 (293-2349 Hz)
- Primary techniques: Breath control, hole shading, trills
- Cultural significance: Breathy, expressive timbre

**Enhanced Features:**
```python
dizi_features = {
    'breath_noise_expected': True,
    'hole_coverage_analysis': True,
    'breath_articulation': ['single', 'double', 'triple'],
    'bamboo_resonance_factor': 1.2
}
```

### 5. Guqin (古琴) - Seven-Stringed Zither

**Characteristics:**
- Frequency range: E2 to C6 (82-1046 Hz)
- Primary techniques: Harmonics, slides, stops
- Cultural significance: Contemplative, scholarly tradition

**Enhanced Features:**
```python
guqin_features = {
    'harmonics_emphasis': True,
    'string_stopping_detection': True,
    'slide_techniques': ['yin', 'nuo', 'chuo', 'zhu'],
    'dynamic_range_db': 60
}
```

## Traditional Techniques Analysis

### Hua Yin (滑音) - Sliding Techniques

**Detection Algorithm:**
```python
def detect_hua_yin(f0, sr):
    # Convert to cents for sensitivity
    log_f0 = np.log2(f0 + 1e-10) * 1200
    
    # Calculate sliding velocity
    velocity = np.gradient(log_f0)
    velocity_smooth = signal.savgol_filter(velocity, 5, 2)
    
    # Detect sustained pitch movement
    sliding_threshold = 20  # cents per frame
    sliding_regions = np.abs(velocity_smooth) > sliding_threshold
    
    return {
        'presence': np.sum(sliding_regions) / len(sliding_regions),
        'velocity_mean': np.mean(np.abs(velocity_smooth)),
        'velocity_max': np.max(np.abs(velocity_smooth))
    }
```

**Types of Hua Yin:**
- **Shang Hua (上滑)** - Upward slides
- **Xia Hua (下滑)** - Downward slides
- **Fu Hua (复滑)** - Complex multi-directional slides

### Chan Yin (颤音) - Vibrato Techniques

**Detection Algorithm:**
```python
def detect_chan_yin(f0, sr):
    # FFT-based vibrato detection
    log_f0 = np.log2(f0 + 1e-10) * 1200
    detrended = signal.detrend(log_f0)
    
    fft = np.fft.fft(detrended)
    freqs = np.fft.fftfreq(len(detrended), d=512/sr)
    
    # Vibrato frequency range: 2-15 Hz
    vibrato_mask = (freqs >= 2.0) & (freqs <= 15.0)
    vibrato_spectrum = np.abs(fft[vibrato_mask])
    
    peak_idx = np.argmax(vibrato_spectrum)
    vibrato_freq = freqs[vibrato_mask][peak_idx]
    
    return {
        'rate_hz': vibrato_freq,
        'extent_cents': np.std(detrended),
        'regularity': np.max(vibrato_spectrum) / np.mean(vibrato_spectrum)
    }
```

**Types of Chan Yin:**
- **Zhi Chan (指颤)** - Finger vibrato
- **Gong Chan (弓颤)** - Bow vibrato (for Erhu)
- **Qi Chan (气颤)** - Breath vibrato (for wind instruments)

### Wu Sheng (五声) - Pentatonic Scale Analysis

**Scale Detection:**
```python
def analyze_wu_sheng(f0):
    # Traditional Chinese pentatonic scale
    wu_sheng_scale = {0, 2, 4, 7, 9}  # Gong, Shang, Jue, Zhi, Yu
    
    midi_notes = librosa.hz_to_midi(f0[~np.isnan(f0)])
    semitones = midi_notes % 12
    
    adherence_count = 0
    for note in semitones:
        if any(abs(note - scale_note) < 0.5 for scale_note in wu_sheng_scale):
            adherence_count += 1
    
    return adherence_count / len(semitones)
```

**Scale Modes:**
- **Gong Mode (宫调)** - C-based pentatonic
- **Shang Mode (商调)** - D-based pentatonic
- **Jue Mode (角调)** - E-based pentatonic
- **Zhi Mode (徵调)** - G-based pentatonic
- **Yu Mode (羽调)** - A-based pentatonic

### Zhuang Shi Yin (装饰音) - Ornamental Notes

**Detection Methods:**
```python
def detect_ornaments(f0, sr):
    midi_notes = librosa.hz_to_midi(f0[~np.isnan(f0)])
    pitch_diff = np.abs(np.diff(midi_notes))
    
    # Grace notes: quick up-down patterns
    grace_notes = 0
    for i in range(1, len(pitch_diff) - 1):
        if (pitch_diff[i] > 1.0 and pitch_diff[i+1] > 1.0 and
            abs(midi_notes[i-1] - midi_notes[i+2]) < 0.5):
            grace_notes += 1
    
    # Ornament density
    ornament_threshold = 0.5  # semitones
    ornaments = np.sum(pitch_diff > ornament_threshold)
    density = ornaments / len(pitch_diff)
    
    return {
        'grace_notes': grace_notes,
        'density': density,
        'total_ornaments': ornaments
    }
```

## Cultural Context

### Performance Aesthetics
- **Shen Yun (神韵)** - Spiritual resonance and expression
- **Qi Xi (气息)** - Breath and life force in performance
- **Yin Yang (阴阳)** - Balance between soft and strong techniques

### Regional Styles
- **Bei Pai (北派)** - Northern style: Bold, vigorous
- **Nan Pai (南派)** - Southern style: Delicate, refined
- **Min Jian (民间)** - Folk style: Simple, direct

## Usage Examples

### Basic Analysis
```python
from InstrumentTimbre.modules.utils.chinese_instrument_features import ChineseInstrumentAnalyzer

analyzer = ChineseInstrumentAnalyzer()
audio_data, sr = librosa.load('erhu_performance.wav')

features = analyzer.extract_chinese_features(audio_data, sr, InstrumentType.ERHU)

print(f"Pentatonic adherence: {features.pentatonic_adherence:.3f}")
print(f"Sliding presence: {features.sliding_detection}")
print(f"Vibrato analysis: {features.vibrato_analysis}")
```

### Advanced Visualization
```python
from example.enhanced_chinese_visualization import EnhancedChineseInstrumentVisualizer

visualizer = EnhancedChineseInstrumentVisualizer()
features, output_file = visualizer.create_comprehensive_visualization(
    'erhu_performance.wav',
    output_dir='analysis_results'
)
```

## References

1. **Traditional Chinese Music Theory**
   - Liang, M. (1985). *Music of the Billion: An Introduction to Chinese Musical Culture*
   - Liu, J. (2010). *The Acoustic Analysis of Chinese Traditional Instruments*

2. **Digital Signal Processing**
   - Müller, M. (2015). *Fundamentals of Music Processing*
   - Peeters, G. (2004). *A large set of audio features for sound description*

3. **Cultural Studies**
   - Stock, J. (1996). *Musical Creativity in Twentieth-Century China*
   - Witzleben, J. L. (1995). *Silk and Bamboo Music in Shanghai*