#!/usr/bin/env python3
"""
InstrumentTimbre Training Script
Simplified training interface for Chinese instrument classification

Usage:
    python train.py --data ./data/instruments
    python train.py --data ./data --config config.yaml --model enhanced_cnn
    python train.py --data ./data --epochs 50 --batch-size 64
"""

import sys
import argparse
import logging
from pathlib import Path
import json

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from InstrumentTimbre.modules.core.logger import setup_logging
    from InstrumentTimbre.modules.core.config import get_default_config, load_config
except ImportError:
    # Fallback implementations
    def setup_logging(level=logging.INFO):
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def get_default_config():
        return {
            'data': {'train_split': 0.8, 'use_chinese_features': True},
            'model': {'type': 'enhanced_cnn', 'input_dim': 128, 'num_classes': 5, 'dropout_rate': 0.1},
            'training': {'epochs': 100, 'batch_size': 32, 'optimizer': {'name': 'adamw', 'lr': 0.001}},
            'features': {'sample_rate': 22050, 'hop_length': 512, 'n_fft': 2048, 'n_mfcc': 50}
        }
    
    def load_config(path):
        import json
        import yaml
        if path.endswith('.json'):
            with open(path) as f:
                return json.load(f)
        else:
            with open(path) as f:
                return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='Train Chinese Instrument Classifier')
    
    # Required arguments
    parser.add_argument('--data', '-d', type=str, required=True,
                       help='Training data directory')
    
    # Optional arguments
    parser.add_argument('--config', '-c', type=str,
                       help='Configuration file (YAML/JSON)')
    parser.add_argument('--output', '-o', type=str, default='outputs',
                       help='Output directory')
    parser.add_argument('--model', '-m', choices=['cnn', 'enhanced_cnn', 'transformer', 'hybrid'],
                       default='enhanced_cnn', help='Model architecture')
    parser.add_argument('--epochs', '-e', type=int, help='Training epochs')
    parser.add_argument('--batch-size', '-b', type=int, help='Batch size')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto',
                       help='Training device')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=logging.DEBUG if args.debug else logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Import modules
        from InstrumentTimbre.core.training.trainer import Trainer
        from InstrumentTimbre.core.data.loaders import create_train_val_loaders
        from InstrumentTimbre.core.models.cnn import CNNClassifier, EnhancedCNNClassifier
        from InstrumentTimbre.core.models.transformer import TransformerClassifier
        from InstrumentTimbre.core.models.hybrid import HybridModel
        
        logger.info("🎵 InstrumentTimbre Training Started")
        logger.info(f"Data: {args.data}")
        logger.info(f"Model: {args.model}")
        
        # Load configuration
        if args.config:
            config = load_config(args.config)
            logger.info(f"Loaded config: {args.config}")
        else:
            config = get_default_config()
            logger.info("Using default configuration")
        
        # Override with command line arguments
        if args.epochs:
            config['training']['epochs'] = args.epochs
        if args.batch_size:
            config['training']['batch_size'] = args.batch_size
        if args.lr:
            config['training']['optimizer']['lr'] = args.lr
        
        # Setup output directory
        output_dir = Path(args.output)
        output_dir.mkdir(exist_ok=True)
        config['training']['save_dir'] = str(output_dir)
        config['training']['log_dir'] = str(output_dir / 'logs')
        
        # Create data loaders
        logger.info("Creating data loaders...")
        train_loader, val_loader = create_train_val_loaders(
            data_dir=args.data,
            train_split=config['data'].get('train_split', 0.8),
            batch_size=config['training']['batch_size'],
            config=config.get('features', {}),
            use_chinese_features=True
        )
        
        # Get dataset info
        num_classes = len(train_loader.dataset.dataset.class_names)
        class_names = train_loader.dataset.dataset.class_names
        config['model']['num_classes'] = num_classes
        
        logger.info(f"Dataset: {num_classes} classes")
        logger.info(f"Classes: {class_names}")
        logger.info(f"Train samples: {len(train_loader.dataset)}")
        logger.info(f"Val samples: {len(val_loader.dataset)}")
        
        # Create model
        logger.info(f"Creating {args.model} model...")
        if args.model == 'cnn':
            model = CNNClassifier(config['model'])
        elif args.model == 'enhanced_cnn':
            model = EnhancedCNNClassifier(config['model'])
        elif args.model == 'transformer':
            model = TransformerClassifier(config['model'])
        elif args.model == 'hybrid':
            model = HybridModel(config['model'])
        
        model_info = model.get_model_info()
        logger.info(f"Model parameters: {model_info['total_parameters']:,}")
        
        # Handle resume
        if args.resume:
            logger.info(f"Resuming from: {args.resume}")
            model, checkpoint = model.load_checkpoint(args.resume, args.device)
        
        # Create trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config['training'],
            device=args.device
        )
        
        # Start training
        logger.info("🚀 Starting training...")
        results = trainer.train()
        
        # Save training results
        final_results = {
            'best_val_acc': results['best_val_acc'],
            'best_val_loss': results['best_val_loss'],
            'training_time': results['training_time'],
            'final_epoch': results['final_epoch'],
            'class_names': class_names,
            'model_type': args.model,
            'model_params': model_info['total_parameters'],
            'config': config
        }
        
        results_file = output_dir / 'training_results.json'
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        # Summary
        logger.info("🎉 Training Completed!")
        logger.info(f"Best Validation Accuracy: {results['best_val_acc']:.4f}")
        logger.info(f"Best Validation Loss: {results['best_val_loss']:.4f}")
        logger.info(f"Training Time: {results['training_time']:.2f}s")
        logger.info(f"Results saved to: {output_dir}")
        logger.info(f"Best model: {output_dir}/best_acc_model.pth")
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())