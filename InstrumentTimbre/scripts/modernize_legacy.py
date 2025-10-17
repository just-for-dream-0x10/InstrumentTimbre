#!/usr/bin/env python3
"""
Script to help migrate legacy functionality to new modular architecture
This script provides utilities to update import paths and refactor code
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple

def find_legacy_imports(directory: str) -> List[Tuple[str, List[str]]]:
    """
    Find files with legacy import patterns
    
    Args:
        directory: Directory to search
        
    Returns:
        List of (file_path, legacy_imports) tuples
    """
    legacy_patterns = [
        r'from\s+train\s+import',
        r'import\s+train\b',
        r'from\s+predict\s+import',
        r'import\s+predict\b',
        r'from\s+evaluate\s+import',
        r'import\s+evaluate\b',
        r'from\s+convert_model\s+import',
        r'import\s+convert_model\b',
        r'from\s+demo\s+import',
        r'import\s+demo\b'
    ]
    
    files_with_legacy = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    legacy_imports = []
                    for pattern in legacy_patterns:
                        matches = re.findall(pattern, content)
                        if matches:
                            legacy_imports.extend(matches)
                    
                    if legacy_imports:
                        files_with_legacy.append((file_path, legacy_imports))
                        
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    
    return files_with_legacy

def generate_import_mapping() -> Dict[str, str]:
    """
    Generate mapping from legacy imports to new modular imports
    
    Returns:
        Dictionary mapping old imports to new imports
    """
    return {
        # Training
        'from train import': 'from InstrumentTimbre.core.training.trainer import',
        'import train': 'from InstrumentTimbre.core.training import trainer',
        
        # Prediction
        'from predict import': 'from InstrumentTimbre.core.inference.predictor import',
        'import predict': 'from InstrumentTimbre.core.inference import predictor',
        
        # Evaluation
        'from evaluate import': 'from InstrumentTimbre.core.training.metrics import',
        'import evaluate': 'from InstrumentTimbre.core.training import metrics',
        
        # Model conversion
        'from convert_model import': 'from InstrumentTimbre.utils.export import',
        'import convert_model': 'from InstrumentTimbre.utils import export',
        
        # Demo
        'from demo import': 'from InstrumentTimbre.cli.commands import',
        'import demo': 'from InstrumentTimbre.cli import commands',
        
        # Feature extraction
        'from utils.chinese_instrument_features import': 'from InstrumentTimbre.core.features.chinese import',
        'import utils.chinese_instrument_features': 'from InstrumentTimbre.core.features import chinese',
        
        # Data utilities
        'from utils.data import': 'from InstrumentTimbre.core.data.loaders import',
        'import utils.data': 'from InstrumentTimbre.core.data import loaders',
        
        # Models
        'from models import': 'from InstrumentTimbre.core.models import',
        'import models': 'from InstrumentTimbre.core import models'
    }

def update_file_imports(file_path: str, dry_run: bool = True) -> bool:
    """
    Update imports in a single file
    
    Args:
        file_path: Path to file to update
        dry_run: If True, only show what would be changed
        
    Returns:
        True if file was/would be updated
    """
    import_mapping = generate_import_mapping()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        updated_content = original_content
        changes_made = []
        
        for old_import, new_import in import_mapping.items():
            if old_import in updated_content:
                updated_content = updated_content.replace(old_import, new_import)
                changes_made.append(f"  {old_import} -> {new_import}")
        
        if changes_made:
            print(f"\n📄 {file_path}")
            for change in changes_made:
                print(change)
            
            if not dry_run:
                # Create backup
                backup_path = file_path + '.backup'
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(original_content)
                
                # Write updated content
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(updated_content)
                
                print(f"  ✅ Updated (backup: {backup_path})")
            else:
                print(f"  🔍 Would update (dry run)")
            
            return True
        
    except Exception as e:
        print(f"❌ Error updating {file_path}: {e}")
    
    return False

def modernize_directory(directory: str, dry_run: bool = True):
    """
    Modernize all Python files in a directory
    
    Args:
        directory: Directory to modernize
        dry_run: If True, only show what would be changed
    """
    print(f"🔍 Scanning {directory} for legacy imports...")
    
    files_with_legacy = find_legacy_imports(directory)
    
    if not files_with_legacy:
        print("✅ No legacy imports found!")
        return
    
    print(f"📋 Found {len(files_with_legacy)} files with legacy imports")
    
    if dry_run:
        print("\n🔍 DRY RUN - No files will be modified")
    else:
        print("\n🚀 UPDATING FILES")
    
    updated_count = 0
    for file_path, legacy_imports in files_with_legacy:
        if update_file_imports(file_path, dry_run):
            updated_count += 1
    
    print(f"\n📊 Summary:")
    print(f"  Files with legacy imports: {len(files_with_legacy)}")
    print(f"  Files updated: {updated_count}")
    
    if dry_run:
        print(f"\n💡 To apply changes, run with --apply")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Modernize legacy InstrumentTimbre imports"
    )
    parser.add_argument(
        'directory', 
        nargs='?', 
        default='.',
        help='Directory to modernize (default: current directory)'
    )
    parser.add_argument(
        '--apply', 
        action='store_true',
        help='Actually apply changes (default: dry run)'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.directory):
        print(f"❌ Directory not found: {args.directory}")
        sys.exit(1)
    
    modernize_directory(args.directory, dry_run=not args.apply)

if __name__ == '__main__':
    main()