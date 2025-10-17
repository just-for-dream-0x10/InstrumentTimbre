# Project Structure Refactoring Plan
# 项目结构重构计划

**Priority**: 🔥 Critical  
**Duration**: 1 week  
**Phase**: 0 (Pre-development cleanup)

---

## Current Issues | 当前问题

### Script Organization Problems | 脚本组织问题

```
Current Mess | 当前混乱状态:
├── train.py                    # Training script
├── train.sh                    # Training shell script  
├── evaluate.py                 # Evaluation script
├── predict.py                  # Prediction script
├── convert_model.py            # Model conversion
├── demo.py                     # Demo script
├── run_tests.sh                # Test runner
├── enhanced_chinese_visualization.py  # Scattered in example/
├── extract_timbre_features.py         # Scattered in example/
└── app.py                      # Web app (incomplete)
```

### Problems Identified | 发现的问题

1. **🗂️ Scattered Scripts** - Scripts spread across root and example directories
2. **🔄 Duplicate Code** - Similar functionality in multiple files
3. **❌ Inconsistent APIs** - Different argument parsing and interfaces
4. **🏗️ No Architecture** - No clear separation of concerns
5. **🧪 Hard to Test** - Monolithic scripts difficult to unit test
6. **📚 Poor Documentation** - Each script has different usage patterns
7. **🔧 Maintenance Nightmare** - Changes require updating multiple files

---

## Target Architecture | 目标架构

### New Modular Structure | 新的模块化结构

```
InstrumentTimbre/
├── instrumenttimbre/                 # Main package
│   ├── __init__.py
│   ├── cli/                         # Command Line Interface
│   │   ├── __init__.py
│   │   ├── main.py                  # Unified entry point
│   │   ├── commands/                # Command modules
│   │   │   ├── __init__.py
│   │   │   ├── train.py            # Training commands
│   │   │   ├── evaluate.py         # Evaluation commands
│   │   │   ├── predict.py          # Prediction commands
│   │   │   ├── convert.py          # Model conversion
│   │   │   ├── visualize.py        # Visualization commands
│   │   │   └── test.py             # Testing commands
│   │   └── utils/                   # CLI utilities
│   │       ├── config.py           # Configuration management
│   │       ├── logging.py          # Logging setup
│   │       └── validation.py       # Input validation
│   ├── core/                        # Core business logic
│   │   ├── __init__.py
│   │   ├── data/                   # Data handling
│   │   │   ├── __init__.py
│   │   │   ├── datasets.py         # Dataset classes
│   │   │   ├── loaders.py          # Data loaders
│   │   │   └── preprocessors.py    # Data preprocessing
│   │   ├── models/                 # Model definitions
│   │   │   ├── __init__.py
│   │   │   ├── base.py             # Base model class
│   │   │   ├── classifiers.py      # Classification models
│   │   │   └── architectures.py    # Model architectures
│   │   ├── features/               # Feature extraction
│   │   │   ├── __init__.py
│   │   │   ├── base.py             # Base feature extractor
│   │   │   ├── chinese.py          # Chinese instrument features
│   │   │   ├── traditional.py      # Traditional audio features
│   │   │   └── deep.py             # Deep learning features
│   │   ├── training/               # Training logic
│   │   │   ├── __init__.py
│   │   │   ├── trainer.py          # Training orchestration
│   │   │   ├── optimizers.py       # Optimization strategies
│   │   │   └── schedulers.py       # Learning rate scheduling
│   │   ├── inference/              # Inference logic
│   │   │   ├── __init__.py
│   │   │   ├── predictor.py        # Prediction logic
│   │   │   ├── converter.py        # Model conversion
│   │   │   └── evaluator.py        # Model evaluation
│   │   └── visualization/          # Visualization logic
│   │       ├── __init__.py
│   │       ├── plotter.py          # Plotting utilities
│   │       ├── chinese_viz.py      # Chinese instrument viz
│   │       └── reports.py          # Report generation
│   ├── utils/                      # Shared utilities
│   │   ├── __init__.py
│   │   ├── audio.py                # Audio processing
│   │   ├── io.py                   # File I/O utilities
│   │   ├── metrics.py              # Evaluation metrics
│   │   └── helpers.py              # Common helpers
│   └── config/                     # Configuration schemas
│       ├── __init__.py
│       ├── base.py                 # Base config classes
│       ├── training.py             # Training config
│       └── inference.py            # Inference config
├── config/                         # Configuration files
│   ├── default.yaml                # Default settings
│   ├── training.yaml               # Training configurations
│   ├── models.yaml                 # Model specifications
│   └── instruments.yaml            # Instrument parameters
├── scripts/                        # Utility scripts
│   ├── setup_environment.sh        # Environment setup
│   ├── download_data.py            # Data downloading
│   ├── migrate_legacy.py           # Migration utilities
│   └── benchmarks.py               # Performance benchmarks
├── tests/                          # Test suite (existing)
├── docs/                           # Documentation (existing)
├── examples/                       # Clean examples
│   ├── basic_usage.py              # Simple usage examples
│   ├── advanced_features.py        # Advanced features
│   └── custom_models.py            # Custom model examples
├── legacy/                         # Legacy scripts (for compatibility)
│   ├── train.py -> ../scripts/legacy/train.py
│   ├── predict.py -> ../scripts/legacy/predict.py
│   └── README.md                   # Migration instructions
├── setup.py                        # Package installation
├── requirements.txt                # Dependencies
├── pyproject.toml                  # Modern Python packaging
└── instrumenttimbre                # Main CLI executable
```

---

## Implementation Plan | 实施计划

### Phase 0.1: Planning and Setup | 规划和设置 (Day 1)

**Tasks**:
- [ ] Create new package structure directories
- [ ] Design configuration schema
- [ ] Plan CLI interface design
- [ ] Identify reusable components

### Phase 0.2: Core Infrastructure | 核心基础设施 (Days 2-3)

**Tasks**:
- [ ] Implement base classes and interfaces
- [ ] Create configuration management system
- [ ] Set up logging infrastructure
- [ ] Implement CLI framework with Click/Typer

**CLI Framework**:
```python
# instrumenttimbre/cli/main.py
import click
from .commands import train, predict, evaluate, convert, visualize, test

@click.group()
@click.version_option()
def cli():
    """InstrumentTimbre: Chinese Traditional Instrument Analysis"""
    pass

cli.add_command(train.train)
cli.add_command(predict.predict)
cli.add_command(evaluate.evaluate)
cli.add_command(convert.convert)
cli.add_command(visualize.visualize)
cli.add_command(test.test)

if __name__ == "__main__":
    cli()
```

### Phase 0.3: Module Migration | 模块迁移 (Days 4-5)

**Migration Priority**:
1. **Core Features** → `instrumenttimbre/core/features/`
2. **Data Handling** → `instrumenttimbre/core/data/`
3. **Models** → `instrumenttimbre/core/models/`
4. **Training Logic** → `instrumenttimbre/core/training/`
5. **Inference Logic** → `instrumenttimbre/core/inference/`

**Migration Strategy**:
```python
# Example: Migrating chinese_instrument_features.py
# From: InstrumentTimbre/modules/utils/chinese_instrument_features.py
# To: instrumenttimbre/core/features/chinese.py

class ChineseInstrumentAnalyzer:
    """Migrated and enhanced Chinese instrument analyzer"""
    
    def __init__(self, config=None):
        self.config = config or get_default_config()
        self.logger = get_logger(__name__)
        
    def extract_features(self, audio_data, sample_rate, instrument_type=None):
        """Enhanced feature extraction with better error handling"""
        # Existing logic + improvements
        pass
```

### Phase 0.4: CLI Commands Implementation | CLI命令实现 (Day 6)

**Command Structure**:
```python
# instrumenttimbre/cli/commands/train.py
@click.command()
@click.option('--config', type=click.Path(exists=True), help='Training configuration file')
@click.option('--data-dir', type=click.Path(exists=True), help='Training data directory')
@click.option('--model-path', type=click.Path(), help='Output model path')
@click.option('--epochs', type=int, default=30, help='Number of training epochs')
@click.option('--batch-size', type=int, default=8, help='Batch size')
@click.option('--lr', type=float, default=0.001, help='Learning rate')
@click.option('--device', type=click.Choice(['auto', 'cpu', 'cuda']), default='auto')
@click.option('--chinese-instruments', is_flag=True, help='Use Chinese instrument optimization')
@click.option('--enhanced-features', is_flag=True, help='Use enhanced features')
def train(config, data_dir, model_path, epochs, batch_size, lr, device, chinese_instruments, enhanced_features):
    """Train InstrumentTimbre models with enhanced Chinese features"""
    
    from ...core.training import Trainer
    from ...core.data import get_dataloader
    from ...utils.config import load_config, merge_configs
    
    # Load and merge configurations
    base_config = load_config(config) if config else {}
    cli_config = {
        'data': {'data_dir': data_dir},
        'training': {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': lr,
            'device': device
        },
        'features': {
            'chinese_instruments': chinese_instruments,
            'enhanced_features': enhanced_features
        },
        'model': {'output_path': model_path}
    }
    
    final_config = merge_configs(base_config, cli_config)
    
    # Initialize trainer
    trainer = Trainer(final_config)
    
    # Run training
    trainer.train()
```

### Phase 0.5: Testing and Validation | 测试与验证 (Day 7)

**Tasks**:
- [ ] Update all existing tests to work with new structure
- [ ] Add integration tests for CLI commands
- [ ] Verify backward compatibility with legacy scripts
- [ ] Performance regression testing
- [ ] Documentation updates

---

## Configuration Management | 配置管理

### Unified Configuration Schema | 统一配置模式

```yaml
# config/default.yaml
project:
  name: "InstrumentTimbre"
  version: "2.0"
  author: "InstrumentTimbre Team"

data:
  sample_rate: 22050
  feature_size: 50
  supported_formats: [".wav", ".mp3", ".flac", ".m4a"]
  instruments:
    - name: "erhu"
      frequency_range: [196, 1568]
      techniques: ["hua_yin", "chan_yin"]
    - name: "pipa" 
      frequency_range: [220, 2093]
      techniques: ["lunzhi", "saofu"]
    - name: "guzheng"
      frequency_range: [196, 2093] 
      techniques: ["glissando", "tremolo"]

models:
  default_architecture: "enhanced_classifier"
  input_size: 50
  hidden_sizes: [256, 128, 64, 32]
  dropout: 0.3
  device: "auto"

training:
  epochs: 30
  batch_size: 8
  learning_rate: 0.001
  patience: 10
  validation_split: 0.2
  augmentation:
    enabled: true
    techniques: ["pitch_shift", "time_stretch", "noise_injection"]

inference:
  batch_size: 32
  confidence_threshold: 0.5
  output_format: "json"
  save_features: false

visualization:
  dpi: 300
  figure_size: [15, 10]
  color_scheme: "viridis"
  save_format: "png"
  
logging:
  level: "INFO"
  file: "instrumenttimbre.log"
  max_size: "10MB"
  backup_count: 5
```

### Configuration Loading | 配置加载

```python
# instrumenttimbre/utils/config.py
import yaml
from pathlib import Path
from typing import Dict, Any

class Config:
    """Configuration management class"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or self.get_default_config_path()
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with dot notation"""
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def update(self, updates: Dict[str, Any]):
        """Update configuration with new values"""
        self.config = merge_dicts(self.config, updates)
    
    @staticmethod
    def get_default_config_path() -> str:
        """Get default configuration file path"""
        return str(Path(__file__).parent.parent.parent / "config" / "default.yaml")
```

---

## CLI Interface Design | CLI接口设计

### New Unified Interface | 新的统一接口

```bash
# Training
instrumenttimbre train --config config/training.yaml --data-dir data/ --enhanced-features

# Prediction
instrumenttimbre predict --model models/enhanced.pt --input audio.wav --output-format json

# Batch prediction
instrumenttimbre predict --model models/enhanced.pt --input-dir audio_folder/ --output predictions.json

# Evaluation
instrumenttimbre evaluate --model models/enhanced.pt --test-dir test_data/ --metrics accuracy f1

# Model conversion
instrumenttimbre convert --input models/enhanced.pt --output models/enhanced.onnx --format onnx

# Visualization
instrumenttimbre visualize --input audio.wav --output analysis.png --type comprehensive

# Testing
instrumenttimbre test --suite chinese_features --verbose

# Configuration
instrumenttimbre config show  # Show current configuration
instrumenttimbre config set training.epochs 50  # Set configuration value
instrumenttimbre config reset  # Reset to defaults

# Help and information
instrumenttimbre --help
instrumenttimbre train --help
instrumenttimbre info models  # Show available models
instrumenttimbre info instruments  # Show supported instruments
```

### Backward Compatibility | 向后兼容性

```bash
# Legacy scripts still work through wrapper
./train.sh quick     # Calls: instrumenttimbre train --config config/quick.yaml
python predict.py     # Wrapper script that calls: instrumenttimbre predict
python evaluate.py    # Wrapper script that calls: instrumenttimbre evaluate
```

---

## Migration Strategy | 迁移策略

### Step-by-Step Migration | 分步迁移

#### Day 1: Infrastructure Setup
- [ ] Create package structure
- [ ] Set up configuration system
- [ ] Implement base classes

#### Day 2-3: Core Logic Migration
- [ ] Migrate feature extraction modules
- [ ] Migrate model definitions
- [ ] Migrate training logic

#### Day 4-5: CLI Implementation
- [ ] Implement CLI commands
- [ ] Add argument parsing and validation
- [ ] Create configuration integration

#### Day 6: Integration & Testing
- [ ] Integration testing
- [ ] Performance testing
- [ ] Backward compatibility verification

#### Day 7: Documentation & Cleanup
- [ ] Update documentation
- [ ] Clean up legacy files
- [ ] Final testing and validation

### Compatibility Guarantee | 兼容性保证

```python
# legacy/train.py - Backward compatibility wrapper
#!/usr/bin/env python3
"""
Legacy train.py wrapper for backward compatibility
"""
import sys
import subprocess
import argparse

def main():
    """Convert legacy arguments to new CLI format"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--chinese-instruments', action='store_true')
    parser.add_argument('--enhanced-features', action='store_true')
    parser.add_argument('--epochs', type=int, default=30)
    # ... other legacy arguments
    
    args = parser.parse_args()
    
    # Convert to new CLI command
    cmd = ['instrumenttimbre', 'train']
    
    if args.chinese_instruments:
        cmd.append('--chinese-instruments')
    if args.enhanced_features:
        cmd.append('--enhanced-features')
    cmd.extend(['--epochs', str(args.epochs)])
    
    # Execute new CLI
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
```

---

## Benefits of Refactoring | 重构的好处

### Immediate Benefits | 即时好处

1. **🎯 Unified Interface** - Single entry point for all functionality
2. **📋 Better Organization** - Clear separation of concerns
3. **🧪 Easier Testing** - Modular components are easier to test
4. **📚 Better Documentation** - Centralized help and documentation
5. **🔧 Easier Maintenance** - Changes in one place affect related functionality

### Long-term Benefits | 长期好处

1. **🚀 Faster Development** - New features easier to add
2. **🔌 Plugin Architecture** - Easy to extend with new models/features
3. **📦 Package Distribution** - Can be installed as proper Python package
4. **🏗️ Better Architecture** - Foundation for advanced features
5. **👥 Team Collaboration** - Clear ownership and responsibility

---

## Success Criteria | 成功标准

### Functional Requirements | 功能要求

- [ ] All existing functionality works through new CLI
- [ ] Backward compatibility maintained for legacy scripts
- [ ] Configuration system works correctly
- [ ] All tests pass with new structure
- [ ] Performance is same or better than before

### Non-Functional Requirements | 非功能要求

- [ ] Code coverage ≥ 90%
- [ ] Documentation updated and complete
- [ ] Package can be installed with `pip install -e .`
- [ ] CLI help system is comprehensive
- [ ] Migration completed within 1 week

---

This refactoring will create a solid foundation for all future development and make the codebase much more maintainable and extensible! 

**Should we start implementing this refactoring plan?** 🚀