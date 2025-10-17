"""
Configuration management utilities
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from copy import deepcopy

@dataclass
class Config:
    """Configuration container with dot notation access"""
    
    # Data configuration
    data: Dict[str, Any] = field(default_factory=dict)
    
    # Model configuration  
    models: Dict[str, Any] = field(default_factory=dict)
    
    # Training configuration
    training: Dict[str, Any] = field(default_factory=dict)
    
    # Inference configuration
    inference: Dict[str, Any] = field(default_factory=dict)
    
    # Visualization configuration
    visualization: Dict[str, Any] = field(default_factory=dict)
    
    # Logging configuration
    logging: Dict[str, Any] = field(default_factory=dict)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with dot notation"""
        keys = key.split('.')
        value = self.__dict__
        
        try:
            for k in keys:
                if isinstance(value, dict):
                    value = value[k]
                else:
                    value = getattr(value, k)
            return value
        except (KeyError, AttributeError):
            return default
    
    def set(self, key: str, value: Any):
        """Set configuration value with dot notation"""
        keys = key.split('.')
        target = self.__dict__
        
        # Navigate to parent
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]
        
        # Set final value
        target[keys[-1]] = value
    
    def update(self, other: Dict[str, Any]):
        """Update configuration with another dict"""
        for key, value in other.items():
            if hasattr(self, key) and isinstance(getattr(self, key), dict):
                getattr(self, key).update(value)
            else:
                setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'data': self.data,
            'models': self.models, 
            'training': self.training,
            'inference': self.inference,
            'visualization': self.visualization,
            'logging': self.logging
        }

def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from file"""
    if config_path is None:
        config_path = get_default_config_path()
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        # Return default config if file doesn't exist
        return get_default_config()
    
    with open(config_path, 'r', encoding='utf-8') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            data = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            data = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")
    
    config = Config()
    if data:
        config.update(data)
    
    return config

def save_config(config: Config, config_path: str):
    """Save configuration to file"""
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            yaml.dump(config.to_dict(), f, default_flow_style=False, indent=2)
        elif config_path.suffix.lower() == '.json':
            json.dump(config.to_dict(), f, indent=2)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")

def get_default_config_path() -> str:
    """Get default configuration file path"""
    return str(Path(__file__).parent.parent.parent.parent / "config" / "default.yaml")

def get_default_config() -> Config:
    """Get default configuration"""
    config = Config()
    
    # Data defaults
    config.data = {
        'sample_rate': 22050,
        'feature_size': 50,
        'supported_formats': ['.wav', '.mp3', '.flac', '.m4a'],
        'instruments': ['erhu', 'pipa', 'guzheng', 'dizi', 'guqin']
    }
    
    # Model defaults
    config.models = {
        'default_architecture': 'enhanced_classifier',
        'input_size': 50,
        'hidden_sizes': [256, 128, 64, 32],
        'dropout': 0.3,
        'device': 'auto'
    }
    
    # Training defaults
    config.training = {
        'epochs': 30,
        'batch_size': 8,
        'learning_rate': 0.001,
        'patience': 10,
        'validation_split': 0.2
    }
    
    # Inference defaults
    config.inference = {
        'batch_size': 32,
        'confidence_threshold': 0.5,
        'output_format': 'json'
    }
    
    # Visualization defaults
    config.visualization = {
        'dpi': 300,
        'figure_size': [15, 10],
        'color_scheme': 'viridis',
        'save_format': 'png'
    }
    
    # Logging defaults
    config.logging = {
        'level': 'INFO',
        'file': 'instrumenttimbre.log',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    }
    
    return config

def merge_configs(base_config: Config, override_config: Dict[str, Any]) -> Config:
    """Merge two configurations, with override taking precedence"""
    merged = deepcopy(base_config)
    merged.update(override_config)
    return merged