"""
Testing command for InstrumentTimbre CLI
"""

import click
from pathlib import Path
import sys
import subprocess

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from instrumenttimbre.utils import get_logger

@click.command()
@click.option('--suite', type=click.Choice(['all', 'chinese_features', 'training', 'utils']), 
              default='all', help='Test suite to run')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.option('--coverage', is_flag=True, help='Run with coverage analysis')
@click.pass_context
def test(ctx, suite, verbose, coverage):
    """Run test suites for InstrumentTimbre"""
    
    logger = get_logger(__name__)
    logger.info("🧪 Starting InstrumentTimbre tests")
    
    try:
        # Check if pytest is available
        test_runner_path = project_root / "run_tests.sh"
        
        if test_runner_path.exists():
            cmd = ['bash', str(test_runner_path)]
            
            if suite == 'all':
                cmd.append('test')
            else:
                # Run specific test file
                if suite == 'chinese_features':
                    cmd = ['python', '-m', 'pytest', 'tests/test_chinese_features.py']
                elif suite == 'training':
                    cmd = ['python', '-m', 'pytest', 'tests/test_training.py']
                elif suite == 'utils':
                    cmd = ['python', '-m', 'pytest', 'tests/test_utils.py']
                
                if verbose:
                    cmd.append('-v')
                if coverage:
                    cmd.extend(['--cov=instrumenttimbre', '--cov-report=html'])
            
            logger.info(f"Executing: {' '.join(cmd)}")
            result = subprocess.run(cmd, cwd=str(project_root))
            
            if result.returncode == 0:
                logger.info("✅ Tests completed successfully!")
            else:
                logger.error("❌ Some tests failed!")
                sys.exit(result.returncode)
        else:
            logger.error("Test runner script not found.")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Testing failed: {e}")
        sys.exit(1)