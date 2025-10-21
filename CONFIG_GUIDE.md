# InstrumentTimbre Configuration Guide

## Quick Start

```python
from config import get_config, production_config, studio_config

# Use default settings
config = get_config()

# Or choose a preset
config = production_config()  # For deployment
config = studio_config()      # For best quality
```

## Core Configuration Categories

### 🎵 Audio Settings
**What to configure**: Basic audio parameters for all processing
```python
sample_rate: int = 22050        # 22050/44100/48000 Hz
channels: int = 2               # 1=mono, 2=stereo  
bit_depth: int = 16             # 16/24 bit
max_duration: float = 30.0      # Max audio length (seconds)
```
**Used by**: All audio processing modules, feature extraction, professional audio

### 🧠 Model & Training
**What to configure**: AI model architecture and training process
```python
batch_size: int = 32            # Training batch size
learning_rate: float = 0.001    # Learning rate
epochs: int = 100               # Training epochs
hidden_dim: int = 512           # Model hidden dimensions
num_layers: int = 6             # Number of model layers
```
**Used by**: Training modules, model loading, inference

### 🎼 Music Generation  
**What to configure**: AI music generation behavior
```python
temperature: float = 0.8        # Creativity (0.1-2.0)
top_k: int = 50                 # Diversity control
max_length: int = 1024          # Max generated length
key_signature: str = "C_major"  # Musical key
tempo_bpm: float = 120.0        # Tempo
```
**Used by**: Music generation pipeline, style transfer, melody generation

### 🎚️ Professional Audio
**What to configure**: Audio quality and effects
```python
target_loudness_lufs: float = -16.0  # Target loudness
reverb_amount: float = 0.3           # Reverb strength (0-1)
compression_ratio: float = 3.0       # Compression ratio
eq_boost_amount: float = 0.4         # EQ enhancement (0-1)
stereo_width: float = 1.0            # Stereo width
```
**Used by**: Professional audio engine, mixing, effects processing

### ⚙️ System Performance
**What to configure**: Hardware and performance settings
```python
use_gpu: bool = True            # Use GPU acceleration
num_workers: int = 4            # CPU cores for processing
quality: Quality = NORMAL       # FAST/NORMAL/HIGH/ULTRA
processing_mode: ProcessingMode = BATCH  # REAL_TIME/BATCH/STUDIO
```
**Used by**: All modules for performance optimization

## Module Mapping

### Training Modules Need:
- `batch_size`, `learning_rate`, `epochs`
- `use_gpu`, `num_workers`
- `early_stopping_patience`, `weight_decay`

### Audio Processing Modules Need:
- `sample_rate`, `channels`, `bit_depth`
- `n_mels`, `n_fft`, `hop_length`
- `buffer_size`, `max_duration`

### Generation Modules Need:
- `temperature`, `top_k`, `max_length`
- `key_signature`, `tempo_bpm`
- `style_strength`, `generation_strategy`

### Professional Audio Modules Need:
- `target_loudness_lufs`, `max_peak_db`
- `reverb_amount`, `compression_ratio`
- `enable_reverb`, `enable_compression`

### Feature Extraction Modules Need:
- `sample_rate`, `n_mels`, `n_fft`
- `hop_length`, `feature_dim`

## Common Configuration Scenarios

### Real-time Music Performance
```python
from config import real_time_config, set_config

config = real_time_config()
config.buffer_size = 512      # Low latency
config.quality = Quality.FAST
set_config(config)
```

### High-Quality Studio Production
```python
from config import studio_config, set_config

config = studio_config()
config.sample_rate = 48000
config.bit_depth = 24
config.target_loudness_lufs = -18.0
set_config(config)
```

### Development and Testing
```python
from config import development_config, set_config

config = development_config()
config.debug = True
config.max_duration = 10.0    # Shorter audio for faster testing
config.epochs = 20            # Fewer training epochs
set_config(config)
```

### Mobile/Edge Deployment
```python
from config import minimal_config, set_config

config = minimal_config()
config.use_gpu = False        # CPU only
config.quality = Quality.FAST
config.enable_reverb = False  # Reduce processing
set_config(config)
```

## Environment Variables

Set these environment variables to override config:

```bash
export TIMBRE_USE_GPU=false
export TIMBRE_BATCH_SIZE=16
export TIMBRE_SAMPLE_RATE=44100
export TIMBRE_DEBUG=true
export TIMBRE_QUALITY=high
```

## Configuration Validation

```python
from config import validate_config, get_config

config = get_config()
if not validate_config(config):
    print("Invalid configuration!")
    # Fix issues or use defaults
```

## Custom Configuration

```python
from config import UltimateConfig, set_config

# Create custom config
custom_config = UltimateConfig(
    sample_rate=44100,
    quality=Quality.HIGH,
    reverb_amount=0.5,
    temperature=1.2,
    custom_params={
        'my_special_setting': 'value'
    }
)

set_config(custom_config)
```

## Troubleshooting

**Problem**: Out of memory during training
**Solution**: Reduce `batch_size`, set `quality=Quality.FAST`

**Problem**: Audio processing too slow
**Solution**: Set `processing_mode=ProcessingMode.REAL_TIME`, reduce `quality`

**Problem**: Generated music not creative enough
**Solution**: Increase `temperature` (0.8 → 1.2), increase `top_k`

**Problem**: Audio quality not good enough
**Solution**: Increase `sample_rate` (22050 → 44100), set `quality=Quality.HIGH`

## Important Notes

1. **Single Config File**: All settings are in `config.py` - no scattered configs
2. **Global State**: Use `get_config()` and `set_config()` for project-wide settings
3. **Presets Available**: `minimal_config()`, `development_config()`, `production_config()`, `studio_config()`, `real_time_config()`
4. **Validation**: Always validate configs with `validate_config()`
5. **Environment Override**: Environment variables automatically override settings