#!/usr/bin/env python3
"""
InstrumentTimbre - Chinese Traditional Instrument Classification System
Main entry point for all functionalities

Usage:
    python main.py train --data ./data --config config.yaml
    python main.py predict --model model.pth --input audio.wav
    python main.py visualize --input audio.wav --output ./plots
    python main.py evaluate --model model.pth --data ./test_data
"""

import sys
import click
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from InstrumentTimbre.modules.core.logger import setup_logging
except ImportError:
    # Fallback logging setup
    def setup_logging(level=logging.INFO):
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

@click.group()
@click.option('--debug', is_flag=True, help='Enable debug logging')
@click.option('--quiet', is_flag=True, help='Suppress non-error output')
def cli(debug, quiet):
    """
    🎵 InstrumentTimbre - Chinese Traditional Instrument AI Classification
    
    A comprehensive system for analyzing and classifying Chinese traditional 
    instruments using advanced machine learning and audio processing techniques.
    
    Supported instruments: 二胡(Erhu), 琵琶(Pipa), 古筝(Guzheng), 笛子(Dizi), 古琴(Guqin)
    """
    # Setup logging
    if quiet:
        level = logging.ERROR
    elif debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    
    setup_logging(level=level)

@cli.command()
@click.option('--data', '-d', type=click.Path(exists=True), required=True,
              help='Training data directory')
@click.option('--config', '-c', type=click.Path(exists=True),
              help='Configuration file (YAML/JSON)')
@click.option('--output', '-o', type=click.Path(), default='outputs',
              help='Output directory for models and logs')
@click.option('--epochs', '-e', type=int, help='Number of training epochs')
@click.option('--batch-size', '-b', type=int, help='Training batch size')
@click.option('--model', '-m', type=click.Choice(['cnn', 'enhanced_cnn', 'transformer', 'hybrid']),
              default='enhanced_cnn', help='Model architecture')
@click.option('--lr', type=float, help='Learning rate')
@click.option('--device', type=click.Choice(['auto', 'cpu', 'cuda']), default='auto',
              help='Training device')
@click.option('--resume', type=click.Path(exists=True), help='Resume from checkpoint')
def train(data, config, output, epochs, batch_size, model, lr, device, resume):
    """
    🚀 Train Chinese instrument classification model
    
    Examples:
        python main.py train -d ./data/instruments
        python main.py train -d ./data -c config.yaml --model enhanced_cnn
        python main.py train -d ./data --epochs 50 --batch-size 64
    """
    try:
        from InstrumentTimbre.core.training.trainer import Trainer
        from InstrumentTimbre.core.data.loaders import create_train_val_loaders
        from InstrumentTimbre.core.models.cnn import CNNClassifier, EnhancedCNNClassifier
        from InstrumentTimbre.core.models.transformer import TransformerClassifier
        from InstrumentTimbre.core.models.hybrid import HybridModel
        from InstrumentTimbre.modules.core.config import load_config, get_default_config
        
        logger = logging.getLogger(__name__)
        logger.info("🎵 Starting Chinese Instrument Classification Training")
        
        # Load configuration
        if config:
            train_config = load_config(config)
            logger.info(f"Loaded config: {config}")
        else:
            train_config = get_default_config()
            logger.info("Using default configuration")
        
        # Override with command line args
        if epochs: train_config['training']['epochs'] = epochs
        if batch_size: train_config['training']['batch_size'] = batch_size
        if lr: train_config['training']['optimizer']['lr'] = lr
        
        # Setup output
        output_dir = Path(output)
        output_dir.mkdir(exist_ok=True)
        train_config['training']['save_dir'] = str(output_dir)
        train_config['training']['log_dir'] = str(output_dir / 'logs')
        
        # Create data loaders
        logger.info("Creating data loaders...")
        train_loader, val_loader = create_train_val_loaders(
            data_dir=str(data),
            train_split=train_config['data'].get('train_split', 0.8),
            batch_size=train_config['training']['batch_size'],
            config=train_config.get('features', {}),
            use_chinese_features=True
        )
        
        # Set number of classes
        num_classes = len(train_loader.dataset.dataset.class_names)
        train_config['model']['num_classes'] = num_classes
        class_names = train_loader.dataset.dataset.class_names
        
        logger.info(f"Dataset: {num_classes} classes, {len(train_loader.dataset)} train, {len(val_loader.dataset)} val")
        logger.info(f"Classes: {class_names}")
        
        # Create model
        logger.info(f"Creating {model} model...")
        if model == 'cnn':
            model_instance = CNNClassifier(train_config['model'])
        elif model == 'enhanced_cnn':
            model_instance = EnhancedCNNClassifier(train_config['model'])
        elif model == 'transformer':
            model_instance = TransformerClassifier(train_config['model'])
        elif model == 'hybrid':
            model_instance = HybridModel(train_config['model'])
        
        # Handle resume
        if resume:
            logger.info(f"Resuming from: {resume}")
            model_instance, checkpoint = model_instance.load_checkpoint(resume, device)
        
        # Create trainer
        trainer = Trainer(
            model=model_instance,
            train_loader=train_loader,
            val_loader=val_loader,
            config=train_config['training'],
            device=device
        )
        
        # Train
        logger.info("🚀 Starting training...")
        results = trainer.train()
        
        # Save results
        import json
        results_path = output_dir / 'training_results.json'
        with open(results_path, 'w') as f:
            json.dump({
                'best_val_acc': results['best_val_acc'],
                'best_val_loss': results['best_val_loss'],
                'training_time': results['training_time'],
                'class_names': class_names,
                'model_type': model
            }, f, indent=2)
        
        logger.info("🎉 Training completed!")
        logger.info(f"Best accuracy: {results['best_val_acc']:.4f}")
        logger.info(f"Results saved to: {output_dir}")
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Training failed: {e}")
        if logging.getLogger().level <= logging.DEBUG:
            import traceback
            traceback.print_exc()
        sys.exit(1)

@cli.command()
@click.option('--model', '-m', type=click.Path(exists=True), required=True,
              help='Trained model checkpoint')
@click.option('--input', '-i', type=click.Path(exists=True), required=True,
              help='Input audio file or directory')
@click.option('--output', '-o', type=click.Path(), help='Output file for results')
@click.option('--format', type=click.Choice(['json', 'csv', 'text']), default='text',
              help='Output format')
@click.option('--top-k', type=int, default=3, help='Show top-k predictions')
@click.option('--device', type=click.Choice(['auto', 'cpu', 'cuda']), default='auto')
def predict(model, input, output, format, top_k, device):
    """
    🎯 Predict Chinese instrument from audio
    
    Examples:
        python main.py predict -m model.pth -i audio.wav
        python main.py predict -m model.pth -i ./audio_files/ -o results.json --format json
    """
    try:
        from InstrumentTimbre.core.inference.predictor import InstrumentPredictor
        import json
        import csv
        
        logger = logging.getLogger(__name__)
        logger.info("🎯 Starting prediction...")
        
        # Initialize predictor
        predictor = InstrumentPredictor(model, device=device)
        logger.info(f"Model loaded: {predictor.get_model_info()['model_class']}")
        
        # Process input
        input_path = Path(input)
        if input_path.is_file():
            results = [predictor.predict_file(str(input_path), top_k)]
            results[0]['file'] = str(input_path)
        elif input_path.is_dir():
            audio_files = []
            for ext in ['.wav', '.mp3', '.flac', '.m4a']:
                audio_files.extend(input_path.glob(f'*{ext}'))
                audio_files.extend(input_path.glob(f'**/*{ext}'))
            results = predictor.predict_batch([str(f) for f in audio_files])
        
        # Output results
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(exist_ok=True)
            
            if format == 'json':
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2)
            elif format == 'csv':
                with open(output_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['file', 'predicted_class', 'confidence'])
                    for r in results:
                        if r.get('top_prediction'):
                            writer.writerow([r['file'], r['top_prediction']['class'], r['top_prediction']['confidence']])
            else:
                with open(output_path, 'w') as f:
                    for r in results:
                        f.write(f"File: {r['file']}\n")
                        for pred in r.get('predictions', []):
                            f.write(f"  {pred['class']}: {pred['confidence']:.4f}\n")
                        f.write("\n")
            logger.info(f"Results saved: {output_path}")
        else:
            # Console output
            for r in results:
                print(f"\n📁 {Path(r['file']).name}")
                for i, pred in enumerate(r.get('predictions', [])):
                    icon = ["🥇", "🥈", "🥉"][i] if i < 3 else "📌"
                    print(f"  {icon} {pred['class']}: {pred['confidence']:.4f}")
        
        logger.info("🎉 Prediction completed!")
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Prediction failed: {e}")
        sys.exit(1)

@cli.command()
@click.option('--input', '-i', type=click.Path(exists=True), required=True,
              help='Audio file or directory')
@click.option('--output', '-o', type=click.Path(), default='visualizations',
              help='Output directory')
@click.option('--style', type=click.Choice(['english', 'enhanced', 'both']), default='both',
              help='Visualization style')
@click.option('--instrument', type=str, help='Instrument type for enhanced analysis')
@click.option('--dpi', type=int, default=300, help='Output resolution')
def visualize(input, output, style, instrument, dpi):
    """
    🎨 Create comprehensive audio visualizations
    
    Examples:
        python main.py visualize -i audio.wav
        python main.py visualize -i erhu.wav --instrument erhu --style enhanced
        python main.py visualize -i ./audio_files/ --style both
    """
    try:
        from InstrumentTimbre.core.visualization.audio_viz import AudioVisualizer
        import librosa
        
        logger = logging.getLogger(__name__)
        logger.info("🎨 Creating visualizations...")
        
        # Setup visualizer
        visualizer = AudioVisualizer({
            'figure_size': (16, 12),
            'dpi': dpi,
            'save_format': 'png'
        })
        
        # Setup output
        output_dir = Path(output)
        if style in ['english', 'both']:
            (output_dir / 'english_visualizations').mkdir(parents=True, exist_ok=True)
        if style in ['enhanced', 'both']:
            (output_dir / 'enhanced_visualizations').mkdir(parents=True, exist_ok=True)
        
        # Process files
        input_path = Path(input)
        if input_path.is_file():
            audio_files = [input_path]
        else:
            audio_files = []
            for ext in ['.wav', '.mp3', '.flac', '.m4a']:
                audio_files.extend(input_path.glob(f'*{ext}'))
        
        for audio_file in audio_files:
            logger.info(f"Processing: {audio_file.name}")
            
            audio_data, sr = librosa.load(str(audio_file), sr=22050)
            filename = audio_file.stem
            
            # Auto-detect instrument
            if not instrument:
                name_lower = filename.lower()
                if 'erhu' in name_lower or '二胡' in name_lower:
                    file_instrument = 'erhu'
                elif 'pipa' in name_lower or '琵琶' in name_lower:
                    file_instrument = 'pipa'
                elif 'guzheng' in name_lower or '古筝' in name_lower:
                    file_instrument = 'guzheng'
                elif 'dizi' in name_lower or '笛子' in name_lower:
                    file_instrument = 'dizi'
                elif 'guqin' in name_lower or '古琴' in name_lower:
                    file_instrument = 'guqin'
                else:
                    file_instrument = 'Chinese Instrument'
            else:
                file_instrument = instrument
            
            # Create visualizations
            if style in ['english', 'both']:
                visualizer.create_comprehensive_analysis(
                    audio_data, sr, filename, str(output_dir / 'english_visualizations')
                )
            
            if style in ['enhanced', 'both']:
                enhanced_dir = output_dir / 'enhanced_visualizations'
                visualizer.plot_spectral_features(
                    audio_data, sr, f"{filename} - Spectral Analysis",
                    str(enhanced_dir / f"{filename}_spectral.png")
                )
                visualizer.plot_chinese_instrument_analysis(
                    audio_data, sr, file_instrument,
                    f"{filename} ({file_instrument}) - Enhanced Analysis",
                    str(enhanced_dir / f"{filename}_enhanced.png")
                )
        
        logger.info(f"🎉 Visualizations saved to: {output_dir}")
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Visualization failed: {e}")
        sys.exit(1)

@cli.command()
@click.option('--model', '-m', type=click.Path(exists=True), required=True,
              help='Model checkpoint to evaluate')
@click.option('--data', '-d', type=click.Path(exists=True), required=True,
              help='Test data directory')
@click.option('--output', '-o', type=click.Path(), help='Output file for results')
@click.option('--device', type=click.Choice(['auto', 'cpu', 'cuda']), default='auto')
def evaluate(model, data, output, device):
    """
    📊 Evaluate model performance on test data
    
    Examples:
        python main.py evaluate -m model.pth -d ./test_data
        python main.py evaluate -m model.pth -d ./test_data -o evaluation.json
    """
    try:
        from InstrumentTimbre.core.training.trainer import Trainer
        from InstrumentTimbre.core.data.loaders import create_train_val_loaders
        from InstrumentTimbre.core.models.base import BaseModel
        
        logger = logging.getLogger(__name__)
        logger.info("📊 Starting evaluation...")
        
        # Load model
        checkpoint = torch.load(model, map_location='cpu')
        model_class = checkpoint.get('model_class', 'EnhancedCNNClassifier')
        
        if model_class == 'CNNClassifier':
            from InstrumentTimbre.core.models.cnn import CNNClassifier
            model_instance = CNNClassifier(checkpoint['model_config'])
        elif model_class == 'TransformerClassifier':
            from InstrumentTimbre.core.models.transformer import TransformerClassifier
            model_instance = TransformerClassifier(checkpoint['model_config'])
        else:
            from InstrumentTimbre.core.models.cnn import EnhancedCNNClassifier
            model_instance = EnhancedCNNClassifier(checkpoint['model_config'])
        
        model_instance.load_state_dict(checkpoint['model_state_dict'])
        
        # Create test loader
        _, test_loader = create_train_val_loaders(
            data_dir=str(data),
            train_split=0.0,  # Use all data for testing
            batch_size=32,
            use_chinese_features=True
        )
        
        # Create dummy trainer for evaluation
        trainer = Trainer(model_instance, test_loader, test_loader, {}, device)
        
        # Evaluate
        results = trainer.evaluate(test_loader)
        
        logger.info("📊 Evaluation Results:")
        logger.info(f"  Accuracy: {results['accuracy']:.4f}")
        logger.info(f"  Loss: {results['loss']:.4f}")
        
        if output:
            import json
            with open(output, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved: {output}")
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)

@cli.command()
def info():
    """
    ℹ️ Show system information and available models
    """
    print("🎵 InstrumentTimbre - Chinese Traditional Instrument Classification")
    print("=" * 70)
    print("📱 System Status:")
    
    try:
        import torch
        print(f"  ✅ PyTorch: {torch.__version__}")
        print(f"  ✅ CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  📱 GPU: {torch.cuda.get_device_name()}")
    except:
        print("  ❌ PyTorch not available")
    
    try:
        import librosa
        print(f"  ✅ Librosa: {librosa.__version__}")
    except:
        print("  ❌ Librosa not available")
    
    print("\n🎭 Supported Instruments:")
    instruments = [
        ("二胡 (Erhu)", "Traditional two-stringed bowed instrument"),
        ("琵琶 (Pipa)", "Four-stringed plucked instrument"), 
        ("古筝 (Guzheng)", "Plucked zither with movable bridges"),
        ("笛子 (Dizi)", "Transverse bamboo flute"),
        ("古琴 (Guqin)", "Seven-stringed plucked instrument")
    ]
    
    for name, desc in instruments:
        print(f"  🎼 {name}: {desc}")
    
    print("\n🤖 Available Models:")
    models = [
        ("cnn", "Basic CNN classifier"),
        ("enhanced_cnn", "CNN with attention and residual connections"),
        ("transformer", "Transformer-based sequence model"),
        ("hybrid", "CNN + Transformer hybrid architecture")
    ]
    
    for name, desc in models:
        print(f"  🏗️  {name}: {desc}")
    
    print("\n🎨 Visualization Types:")
    viz_types = [
        "Waveform & Amplitude Envelope",
        "Spectrogram & Mel Spectrogram", 
        "MFCC & Chroma Features",
        "F0 Tracking & Voicing Detection",
        "Harmonic-Percussive Separation",
        "Chinese Pentatonic Analysis",
        "Spectral Feature Evolution"
    ]
    
    for viz in viz_types:
        print(f"  🎨 {viz}")

if __name__ == '__main__':
    cli()