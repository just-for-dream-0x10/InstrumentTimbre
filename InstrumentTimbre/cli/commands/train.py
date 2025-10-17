"""
Training command for InstrumentTimbre CLI
"""

import click
from pathlib import Path
import sys

# Add project root for imports during transition
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from instrumenttimbre.cli.utils.config import load_config, merge_configs, get_default_config
from instrumenttimbre.utils import get_logger

@click.command()
@click.option('--config', type=click.Path(exists=True), help='Training configuration file')
@click.option('--data-dir', type=click.Path(exists=True), help='Training data directory')
@click.option('--model-path', type=click.Path(), help='Output model path')
@click.option('--epochs', type=int, help='Number of training epochs')
@click.option('--batch-size', type=int, help='Batch size')
@click.option('--lr', type=float, help='Learning rate')
@click.option('--device', type=click.Choice(['auto', 'cpu', 'cuda', 'mps']), help='Device for training')
@click.option('--chinese-instruments', is_flag=True, help='Use Chinese instrument optimization')
@click.option('--enhanced-features', is_flag=True, help='Use enhanced features')
@click.option('--debug', is_flag=True, help='Enable debug mode (fewer files)')
@click.pass_context
def train(ctx, config, data_dir, model_path, epochs, batch_size, lr, device, 
          chinese_instruments, enhanced_features, debug):
    """Train InstrumentTimbre models with enhanced Chinese features"""
    
    logger = get_logger(__name__)
    logger.info("🎵 Starting InstrumentTimbre training")
    
    try:
        # Load base configuration
        if config:
            base_config = load_config(config)
            logger.info(f"Loaded configuration from: {config}")
        else:
            base_config = get_default_config()
            logger.info("Using default configuration")
        
        # Build CLI overrides
        cli_overrides = {}
        
        if data_dir:
            cli_overrides['data'] = {'data_dir': str(data_dir)}
        if model_path:
            cli_overrides['model'] = {'output_path': str(model_path)}
        
        training_overrides = {}
        if epochs is not None:
            training_overrides['epochs'] = epochs
        if batch_size is not None:
            training_overrides['batch_size'] = batch_size
        if lr is not None:
            training_overrides['learning_rate'] = lr
        if device is not None:
            training_overrides['device'] = device
        
        if training_overrides:
            cli_overrides['training'] = training_overrides
        
        features_overrides = {}
        if chinese_instruments:
            features_overrides['chinese_instruments'] = True
        if enhanced_features:
            features_overrides['enhanced_features'] = True
        if debug:
            features_overrides['debug'] = True
            
        if features_overrides:
            cli_overrides['features'] = features_overrides
        
        # Merge configurations
        final_config = merge_configs(base_config, cli_overrides)
        
        # Use new modular training implementation
        logger.info("Using new modular training implementation...")
        
        try:
            # Import new training modules
            from ...core.training.trainer import Trainer
            from ...core.data.loaders import create_train_val_loaders
            from ...core.models.cnn import CNNClassifier, EnhancedCNNClassifier
            from ...modules.core.logger import setup_logging
            
            # Setup enhanced logging
            setup_logging(level=logging.DEBUG if debug else logging.INFO)
            
            # Validate required parameters
            data_directory = final_config.get('data', {}).get('data_dir')
            if not data_directory:
                logger.error("Data directory is required. Use --data-dir or set in config.")
                sys.exit(1)
            
            # Create data loaders
            logger.info("Creating data loaders...")
            train_loader, val_loader = create_train_val_loaders(
                data_dir=data_directory,
                train_split=final_config.get('data', {}).get('train_split', 0.8),
                batch_size=final_config.get('training', {}).get('batch_size', 32),
                config=final_config.get('features', {}),
                use_chinese_features=final_config.get('features', {}).get('chinese_instruments', True)
            )
            
            # Determine number of classes
            num_classes = len(train_loader.dataset.dataset.class_names)
            final_config.setdefault('model', {})['num_classes'] = num_classes
            
            logger.info(f"Dataset: {num_classes} classes, {len(train_loader.dataset)} train, {len(val_loader.dataset)} val samples")
            
            # Create model
            model_type = final_config.get('model', {}).get('type', 'enhanced_cnn')
            if model_type == 'enhanced_cnn':
                model = EnhancedCNNClassifier(final_config.get('model', {}))
            else:
                model = CNNClassifier(final_config.get('model', {}))
            
            logger.info(f"Created {model_type} model with {model.get_model_info()['total_parameters']} parameters")
            
            # Create trainer
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=final_config.get('training', {}),
                device=final_config.get('training', {}).get('device', 'auto')
            )
            
            # Start training
            logger.info("🚀 Starting training...")
            results = trainer.train()
            
            # Log results
            logger.info("🎉 Training completed!")
            logger.info(f"Best validation accuracy: {results['best_val_acc']:.4f}")
            logger.info(f"Training time: {results['training_time']:.2f}s")
            
            # Save final model info
            if model_path:
                output_dir = Path(model_path).parent
                output_dir.mkdir(exist_ok=True)
                
                # Save config used
                import yaml
                config_path = output_dir / 'training_config.yaml'
                with open(config_path, 'w') as f:
                    yaml.dump(final_config, f, default_flow_style=False)
                logger.info(f"Training config saved to: {config_path}")
            
        except ImportError as e:
            logger.error(f"Failed to import new training modules: {e}")
            logger.info("Falling back to legacy implementation...")
            # Fall back to legacy if new modules not ready
            import subprocess
            legacy_train_path = project_root / "train.py"
            if legacy_train_path.exists():
                cmd = ['python', str(legacy_train_path)]
                if data_dir:
                    cmd.extend(['--data-dir', str(data_dir)])
                result = subprocess.run(cmd, cwd=str(project_root))
                sys.exit(result.returncode)
            else:
                logger.error("Neither new nor legacy training available!")
                sys.exit(1)
        except Exception as e:
            logger.error(f"Training failed: {e}")
            if debug:
                import traceback
                traceback.print_exc()
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)