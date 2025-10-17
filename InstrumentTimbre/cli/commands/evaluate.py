"""
Evaluation command for InstrumentTimbre CLI
"""

import click
from pathlib import Path
import sys
import subprocess

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from instrumenttimbre.utils import get_logger

@click.command()
@click.option('--model', required=True, type=click.Path(exists=True), help='Path to trained model')
@click.option('--test-dir', type=click.Path(exists=True), help='Directory with test audio files')
@click.option('--single-file', type=click.Path(exists=True), help='Evaluate single audio file')
@click.option('--output-dir', default='evaluation_results', help='Output directory for results')
@click.option('--device', type=click.Choice(['auto', 'cpu', 'cuda', 'mps']), default='auto', help='Device for evaluation')
@click.pass_context
def evaluate(ctx, model, test_dir, single_file, output_dir, device):
    """Evaluate model performance on test datasets"""
    
    logger = get_logger(__name__)
    logger.info("📊 Starting InstrumentTimbre evaluation")
    
    try:
        legacy_eval_path = project_root / "evaluate.py"
        
        if legacy_eval_path.exists():
            cmd = ['python', str(legacy_eval_path)]
            cmd.extend(['--model', str(model)])
            
            if test_dir:
                cmd.extend(['--test-dir', str(test_dir)])
            if single_file:
                cmd.extend(['--single-file', str(single_file)])
            if output_dir:
                cmd.extend(['--output-dir', str(output_dir)])
            if device != 'auto':
                cmd.extend(['--device', device])
            
            logger.info(f"Executing: {' '.join(cmd)}")
            result = subprocess.run(cmd, cwd=str(project_root))
            
            if result.returncode == 0:
                logger.info("✅ Evaluation completed successfully!")
            else:
                logger.error("❌ Evaluation failed!")
                sys.exit(result.returncode)
        else:
            logger.error("Legacy evaluation script not found.")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)