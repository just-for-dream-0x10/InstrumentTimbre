"""
Model conversion command for InstrumentTimbre CLI
"""

import click
from pathlib import Path
import sys
import subprocess

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from instrumenttimbre.utils import get_logger

@click.command()
@click.option('--input', 'input_model', required=True, type=click.Path(exists=True), help='Input PyTorch model')
@click.option('--output', help='Output file path')
@click.option('--format', 'output_format', required=True, 
              type=click.Choice(['onnx', 'torchscript', 'tensorrt', 'coreml', 'tflite', 'all']),
              help='Output format')
@click.option('--output-dir', default='converted_models', help='Output directory')
@click.option('--device', type=click.Choice(['cpu', 'cuda']), default='cpu', help='Device for conversion')
@click.option('--benchmark', is_flag=True, help='Benchmark converted model')
@click.pass_context
def convert(ctx, input_model, output, output_format, output_dir, device, benchmark):
    """Convert models to different deployment formats"""
    
    logger = get_logger(__name__)
    logger.info("🔄 Starting model conversion")
    
    try:
        legacy_convert_path = project_root / "convert_model.py"
        
        if legacy_convert_path.exists():
            cmd = ['python', str(legacy_convert_path)]
            cmd.extend(['--input', str(input_model)])
            cmd.extend(['--format', output_format])
            cmd.extend(['--output-dir', str(output_dir)])
            cmd.extend(['--device', device])
            
            if output:
                cmd.extend(['--output', str(output)])
            if benchmark:
                cmd.append('--benchmark')
            
            logger.info(f"Executing: {' '.join(cmd)}")
            result = subprocess.run(cmd, cwd=str(project_root))
            
            if result.returncode == 0:
                logger.info("✅ Model conversion completed successfully!")
            else:
                logger.error("❌ Model conversion failed!")
                sys.exit(result.returncode)
        else:
            logger.error("Legacy conversion script not found.")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Model conversion failed: {e}")
        sys.exit(1)