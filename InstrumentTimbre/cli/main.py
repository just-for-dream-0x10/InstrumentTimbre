#!/usr/bin/env python3
"""
Main CLI entry point for InstrumentTimbre
"""

import click
from pathlib import Path
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from instrumenttimbre.cli.commands import train, predict, evaluate, convert, visualize, test
from instrumenttimbre.utils.logging import setup_logging
from instrumenttimbre import __version__

@click.group()
@click.version_option(version=__version__)
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--config', type=click.Path(exists=True), help='Configuration file path')
@click.pass_context
def cli(ctx, verbose, config):
    """
    InstrumentTimbre: Chinese Traditional Instrument Analysis
    
    A comprehensive machine learning platform for analyzing and recognizing
    Chinese traditional instruments with enhanced cultural features.
    """
    # Ensure context object exists
    ctx.ensure_object(dict)
    
    # Setup logging
    log_level = 'DEBUG' if verbose else 'INFO'
    setup_logging(level=log_level)
    
    # Store global config
    ctx.obj['verbose'] = verbose
    ctx.obj['config'] = config

# Add command groups
cli.add_command(train.train)
cli.add_command(predict.predict) 
cli.add_command(evaluate.evaluate)
cli.add_command(convert.convert)
cli.add_command(visualize.visualize)
cli.add_command(test.test)

def main():
    """Main entry point"""
    cli()

if __name__ == '__main__':
    main()