"""
Prediction command for InstrumentTimbre CLI
"""

import click
from pathlib import Path
import sys
import subprocess

# Add project root for imports during transition
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from instrumenttimbre.utils import get_logger

@click.command()
@click.option('--model', required=True, type=click.Path(exists=True), help='Path to trained model')
@click.option('--input', 'input_path', required=True, type=click.Path(exists=True), help='Input audio file or directory')
@click.option('--output', type=click.Path(), help='Output file for results')
@click.option('--device', type=click.Choice(['auto', 'cpu', 'cuda', 'mps']), default='auto', help='Device for inference')
@click.option('--batch-size', type=int, help='Batch size for batch prediction')
@click.option('--features', is_flag=True, help='Include extracted features in output')
@click.option('--format', 'output_format', type=click.Choice(['json', 'csv', 'txt']), default='json', help='Output format')
@click.pass_context
def predict(ctx, model, input_path, output, device, batch_size, features, output_format):
    """Make predictions on audio files using trained models"""
    
    logger = get_logger(__name__)
    logger.info("🔮 Starting InstrumentTimbre prediction")
    
    try:
        # Try using new modular prediction implementation
        try:
            from ...core.inference.predictor import InstrumentPredictor
            from ...modules.core.logger import setup_logging
            import logging
            
            # Setup logging
            setup_logging(level=logging.INFO)
            
            # Initialize predictor
            logger.info(f"Loading model: {model}")
            predictor = InstrumentPredictor(model_path=str(model), device=device)
            
            # Make predictions
            input_path_obj = Path(input_path)
            if input_path_obj.is_file():
                # Single file prediction
                logger.info(f"Predicting: {input_path}")
                result = predictor.predict_file(str(input_path), return_features=features)
                results = [result]
            elif input_path_obj.is_dir():
                # Batch prediction
                audio_files = []
                for ext in ['.wav', '.mp3', '.flac', '.m4a']:
                    audio_files.extend(input_path_obj.glob(f'*{ext}'))
                    audio_files.extend(input_path_obj.glob(f'**/*{ext}'))
                
                logger.info(f"Found {len(audio_files)} audio files")
                results = predictor.predict_batch([str(f) for f in audio_files], 
                                                batch_size=batch_size or 8)
            else:
                logger.error(f"Input path not found: {input_path}")
                sys.exit(1)
            
            # Output results
            if output:
                output_path = Path(output)
                output_path.parent.mkdir(exist_ok=True)
                
                if output_format == 'json':
                    import json
                    with open(output_path, 'w') as f:
                        json.dump(results, f, indent=2)
                elif output_format == 'csv':
                    import csv
                    with open(output_path, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['file', 'predicted_class', 'confidence'])
                        for result in results:
                            if 'top_prediction' in result and result['top_prediction']:
                                writer.writerow([
                                    result.get('file', 'unknown'),
                                    result['top_prediction']['class'],
                                    result['top_prediction']['confidence']
                                ])
                else:  # txt
                    with open(output_path, 'w') as f:
                        for result in results:
                            f.write(f"File: {result.get('file', 'unknown')}\n")
                            if 'predictions' in result and result['predictions']:
                                for pred in result['predictions']:
                                    f.write(f"  {pred['class']}: {pred['confidence']:.4f}\n")
                            f.write("\n")
                
                logger.info(f"Results saved to: {output}")
            else:
                # Display results
                for result in results:
                    print(f"\n📁 File: {Path(result.get('file', 'unknown')).name}")
                    if 'predictions' in result and result['predictions']:
                        print("🎯 Predictions:")
                        for i, pred in enumerate(result['predictions']):
                            icon = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
                            print(f"  {icon} {pred['class']}: {pred['confidence']:.4f}")
                    else:
                        print("❌ No predictions available")
            
            logger.info("✅ Prediction completed successfully!")
            
        except ImportError as e:
            logger.warning(f"New prediction modules not available: {e}")
            logger.info("Falling back to legacy implementation...")
            
            # Fallback to legacy
            legacy_predict_path = project_root / "predict.py"
            
            if legacy_predict_path.exists():
                cmd = ['python', str(legacy_predict_path)]
                cmd.extend(['--model', str(model)])
                cmd.extend(['--input', str(input_path)])
                
                if output:
                    cmd.extend(['--output', str(output)])
                
                result = subprocess.run(cmd, cwd=str(project_root))
                sys.exit(result.returncode)
            else:
                logger.error("Neither new nor legacy prediction available!")
                sys.exit(1)
            
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        sys.exit(1)