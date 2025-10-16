# Architecture Overview

## System Design

The InstrumentTimbre system follows a modular architecture with clear separation of concerns, designed for extensibility and maintainability.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    InstrumentTimbre System                     │
├─────────────────────────────────────────────────────────────────┤
│  User Interface Layer                                          │
│  ├── CLI Tools (demo.py, enhanced_visualization.py)           │
│  ├── Example Scripts (examples/)                              │
│  └── Jupyter Notebooks                                        │
├─────────────────────────────────────────────────────────────────┤
│  Application Layer                                             │
│  ├── Visualization Services                                   │
│  ├── Analysis Orchestration                                   │
│  └── Batch Processing                                         │
├─────────────────────────────────────────────────────────────────┤
│  Business Logic Layer                                         │
│  ├── Chinese Instrument Analyzer                             │
│  ├── Traditional Technique Detection                         │
│  ├── Cultural Feature Extraction                             │
│  └── Enhanced Visualization Engine                           │
├─────────────────────────────────────────────────────────────────┤
│  Core Services Layer                                          │
│  ├── Timbre Analysis Service                                 │
│  ├── Timbre Conversion Service                               │
│  ├── Timbre Training Service                                 │
│  └── Base Timbre Service                                     │
├─────────────────────────────────────────────────────────────────┤
│  Foundation Layer                                             │
│  ├── Audio Processing (Librosa, PyTorch)                     │
│  ├── Signal Processing (SciPy, NumPy)                        │
│  ├── Machine Learning (PyTorch, Scikit-learn)                │
│  └── Visualization (Matplotlib, Seaborn)                     │
└─────────────────────────────────────────────────────────────────┘
```

## Module Structure

### Core Modules (`InstrumentTimbre/modules/`)

#### 1. Core (`core/`)
- **Purpose**: Fundamental system components
- **Components**:
  - `config.py` - Configuration management
  - `models.py` - Data models and enums
  - `logger.py` - Logging utilities
  - `exceptions.py` - Custom exception classes

#### 2. Services (`services/`)
- **Purpose**: Business logic and orchestration
- **Components**:
  - `base_timbre_service.py` - Abstract base service
  - `timbre_analysis_service.py` - Core analysis logic
  - `timbre_conversion_service.py` - Audio transformation
  - `timbre_training_service.py` - Model training workflows

#### 3. Utils (`utils/`)
- **Purpose**: Enhanced Chinese instrument analysis
- **Components**:
  - `chinese_instrument_features.py` - Traditional technique detection
  - `chinese_music_theory.py` - Cultural knowledge base

### Model Architecture (`models/`)

#### Neural Network Components
- `encoders.py` - Feature encoding networks
- `decoders.py` - Signal reconstruction networks
- `attention.py` - Attention mechanisms for temporal modeling
- `model.py` - Main architecture integration

### Audio Processing (`audio/`)

#### Signal Processing Pipeline
- `processors.py` - Audio preprocessing and feature extraction

### Utilities (`utils/`)

#### General Purpose Tools
- `data.py` - Data loading and preprocessing
- `cache.py` - Caching mechanisms
- `export.py` - Result export utilities
- `prepare_data.py` - Dataset preparation

## Data Flow

### 1. Audio Input Processing

```
Audio File (WAV/MP3/FLAC)
    ↓
[Librosa Loading & Preprocessing]
    ↓
[Normalization & Resampling to 22050Hz]
    ↓
[Windowing & Frame-based Processing]
    ↓
Audio Data Array (NumPy)
```

### 2. Feature Extraction Pipeline

```
Audio Data
    ↓
┌─────────────────────┬─────────────────────┬─────────────────────┐
│   Basic Features    │  Enhanced Features  │  Cultural Features  │
│                     │                     │                     │
│ • MFCC (13 coeffs)  │ • F0 Tracking      │ • Pentatonic Scale  │
│ • Spectral Centroid │ • Sliding Analysis  │ • Traditional Tech  │
│ • Spectral Rolloff  │ • Vibrato Detection │ • Ornament Analysis │
│ • Zero Crossing Rate│ • Harmonic Analysis │ • Modal Analysis    │
│ • RMS Energy        │ • Onset Detection   │ • Style Recognition │
└─────────────────────┴─────────────────────┴─────────────────────┘
    ↓
Feature Vector (50-dimensional)
```

### 3. Chinese Instrument Enhancement

```
Raw Features
    ↓
[Instrument Type Detection]
    ↓
[Traditional Technique Analysis]
├── Hua Yin (Sliding) Detection
├── Chan Yin (Vibrato) Analysis
├── Wu Sheng (Pentatonic) Adherence
└── Zhuang Shi Yin (Ornament) Density
    ↓
[Cultural Context Integration]
    ↓
Enhanced Feature Set
```

### 4. Visualization Generation

```
Enhanced Features
    ↓
[Chart Generation Pipeline]
├── Audio Waveform
├── Spectrogram
├── F0 Contour
├── Sliding Analysis
├── Vibrato Patterns
├── Feature Radar Chart
├── MFCC Heatmap
├── Spectral Features
└── Summary Report
    ↓
Multi-Panel Visualization (PNG/PDF/SVG)
```

## Component Interactions

### 1. Chinese Instrument Analyzer

```python
class ChineseInstrumentAnalyzer:
    """Main orchestrator for Chinese instrument analysis"""
    
    def __init__(self):
        self.music_theory = ChineseMusicTheory()
        self.processors = AudioProcessors()
        self.detectors = TechniqueDetectors()
    
    def extract_chinese_features(self, audio_data, sr, instrument_hint):
        # Coordinate feature extraction pipeline
        basic_features = self.processors.extract_basic_features(audio_data, sr)
        f0_features = self.processors.extract_f0_features(audio_data, sr)
        traditional_features = self.detectors.detect_techniques(f0_features)
        cultural_features = self.music_theory.analyze_cultural_aspects(f0_features)
        
        return self.combine_features(basic_features, traditional_features, cultural_features)
```

### 2. Enhanced Visualizer

```python
class EnhancedChineseInstrumentVisualizer:
    """Visualization engine with cultural awareness"""
    
    def create_comprehensive_visualization(self, audio_file, output_dir):
        # Load and analyze
        audio_data, sr = self.load_audio(audio_file)
        features, f0 = self.extract_chinese_features(audio_data, sr)
        
        # Generate visualizations
        charts = self.generate_charts(audio_data, features, f0, sr)
        
        # Compose final visualization
        return self.compose_visualization(charts, features, output_dir)
```

## Design Patterns

### 1. Strategy Pattern
Used for different instrument analysis strategies:

```python
class InstrumentAnalysisStrategy:
    def analyze(self, audio_data, sr):
        raise NotImplementedError

class ErhuAnalysisStrategy(InstrumentAnalysisStrategy):
    def analyze(self, audio_data, sr):
        # Erhu-specific analysis
        pass

class PipaAnalysisStrategy(InstrumentAnalysisStrategy):
    def analyze(self, audio_data, sr):
        # Pipa-specific analysis
        pass
```

### 2. Factory Pattern
For creating appropriate analyzers:

```python
class AnalyzerFactory:
    @staticmethod
    def create_analyzer(instrument_type):
        if instrument_type == InstrumentType.ERHU:
            return ErhuAnalyzer()
        elif instrument_type == InstrumentType.PIPA:
            return PipaAnalyzer()
        # ... other instruments
```

### 3. Observer Pattern
For real-time analysis updates:

```python
class AnalysisObserver:
    def update(self, features):
        pass

class VisualizationObserver(AnalysisObserver):
    def update(self, features):
        self.update_plots(features)
```

## Performance Considerations

### 1. Memory Management
- Streaming audio processing for large files
- Chunked analysis to prevent memory overflow
- Efficient NumPy array operations

### 2. Computational Optimization
- Vectorized operations using NumPy/SciPy
- GPU acceleration for neural network components
- Caching of intermediate results

### 3. Scalability
- Modular design allows horizontal scaling
- Batch processing capabilities
- Parallel processing support

## Error Handling Strategy

### 1. Graceful Degradation
```python
try:
    enhanced_features = extract_enhanced_features(audio_data)
except FeatureExtractionError:
    # Fall back to basic features
    enhanced_features = extract_basic_features(audio_data)
```

### 2. Comprehensive Logging
- Different log levels for development and production
- Structured logging for analysis tracking
- Performance metrics collection

### 3. Input Validation
- Audio format validation
- Parameter range checking
- Graceful handling of edge cases

## Extensibility

### 1. Plugin Architecture
New instruments can be added by implementing:
- Instrument-specific parameter sets
- Custom analysis strategies
- Specialized visualization components

### 2. Configuration-Driven Behavior
- YAML/JSON configuration files
- Runtime parameter adjustment
- Environment-specific settings

### 3. API Design
- Clean separation between interface and implementation
- Standardized data models
- Comprehensive documentation

## Testing Strategy

### 1. Unit Testing
- Individual component testing
- Mock dependencies for isolation
- Parametrized tests for different instruments

### 2. Integration Testing
- End-to-end pipeline testing
- Cross-component interaction validation
- Performance regression testing

### 3. Cultural Validation
- Expert musician validation of traditional technique detection
- Cross-cultural analysis verification
- Continuous accuracy monitoring

This architecture ensures the system is maintainable, extensible, and culturally accurate while providing high-performance analysis capabilities.