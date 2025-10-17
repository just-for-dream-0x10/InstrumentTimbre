"""
Visualization command for InstrumentTimbre CLI
"""

import click
from pathlib import Path
import sys
import subprocess

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from instrumenttimbre.utils import get_logger

@click.command()
@click.option('--input', 'input_file', required=True, type=click.Path(exists=True), help='Input audio file')
@click.option('--output', help='Output visualization file')
@click.option('--output-dir', default='visualizations', help='Output directory')
@click.option('--type', 'viz_type', type=click.Choice(['basic', 'comprehensive', 'chinese']), 
              default='comprehensive', help='Visualization type')
@click.pass_context
def visualize(ctx, input_file, output, output_dir, viz_type):
    """Generate visualizations for audio analysis"""
    
    logger = get_logger(__name__)
    logger.info("🎨 Starting visualization generation")
    
    try:
        if viz_type in ['comprehensive', 'chinese']:
            # Use enhanced Chinese visualization
            legacy_viz_path = project_root / "example" / "enhanced_chinese_visualization.py"
            
            if legacy_viz_path.exists():
                cmd = ['python', str(legacy_viz_path)]
                cmd.extend(['--input', str(input_file)])
                cmd.extend(['--output', str(output_dir)])
                
                logger.info(f"Executing: {' '.join(cmd)}")
                result = subprocess.run(cmd, cwd=str(project_root))
                
                if result.returncode == 0:
                    logger.info("✅ Visualization completed successfully!")
                else:
                    logger.error("❌ Visualization failed!")
                    sys.exit(result.returncode)
            else:
                logger.error("Enhanced visualization script not found.")
                sys.exit(1)
        else:
            # Use basic visualization
            legacy_viz_path = project_root / "example" / "timbre_extraction_visualization.py"
            
            if legacy_viz_path.exists():
                logger.info("Using basic visualization (legacy script)")
                result = subprocess.run(['python', str(legacy_viz_path)], cwd=str(project_root))
                
                if result.returncode == 0:
                    logger.info("✅ Basic visualization completed!")
                else:
                    logger.error("❌ Basic visualization failed!")
                    sys.exit(result.returncode)
            else:
                logger.error("Basic visualization script not found.")
                sys.exit(1)
            
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        sys.exit(1)