#!/bin/bash
# Enhanced Chinese Instrument Training Script
# 增强版中国乐器训练脚本

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if conda environment exists
check_environment() {
    print_info "Checking conda environment..."
    
    if ! command -v conda &> /dev/null; then
        print_error "Conda not found! Please install Anaconda/Miniconda"
        exit 1
    fi
    
    if conda env list | grep -q "myenv"; then
        print_success "Found myenv conda environment"
    else
        print_warning "myenv environment not found, using base environment"
    fi
}

# Activate conda environment
activate_env() {
    print_info "Activating conda environment..."
    
    # Try different activation methods
    if [ -f "$CONDA_PREFIX/etc/profile.d/conda.sh" ]; then
        source "$CONDA_PREFIX/etc/profile.d/conda.sh"
        conda activate myenv 2>/dev/null || conda activate base
    elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
        conda activate myenv 2>/dev/null || conda activate base
    elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
        conda activate myenv 2>/dev/null || conda activate base
    else
        # Fallback methods
        source activate myenv 2>/dev/null || \
        conda activate myenv 2>/dev/null || \
        print_warning "Could not activate conda environment, using current environment"
    fi
    
    print_success "Environment activated"
}

# Check dependencies
check_dependencies() {
    print_info "Checking Python dependencies..."
    
    python -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null || {
        print_error "PyTorch not found! Please install: pip install torch"
        exit 1
    }
    
    python -c "import librosa; print(f'Librosa: {librosa.__version__}')" 2>/dev/null || {
        print_error "Librosa not found! Please install: pip install librosa"
        exit 1
    }
    
    print_success "All dependencies found"
}

# Quick training (debug mode)
train_quick() {
    print_info "🚀 Starting Quick Training (Debug Mode)"
    print_warning "This is for testing only - not suitable for production"
    
    python train.py \
        --chinese-instruments \
        --enhanced-features \
        --epochs 5 \
        --batch-size 4 \
        --lr 0.001 \
        --patience 3 \
        --debug \
        --device auto \
        --model-path "./saved_models/quick_model.pt"
}

# Standard training
train_standard() {
    print_info "🎵 Starting Standard Training"
    
    python train.py \
        --chinese-instruments \
        --enhanced-features \
        --data-dir ../wav \
        --epochs 30 \
        --batch-size 8 \
        --lr 0.001 \
        --patience 8 \
        --device auto \
        --model-path "./saved_models/standard_model.pt" \
        --cache-features
}

# Full training with all optimizations
train_full() {
    print_info "🔥 Starting Full Training (High Performance)"
    
    # Create larger model directory
    mkdir -p saved_models/full_training
    
    python train.py \
        --chinese-instruments \
        --enhanced-features \
        --data-dir ../wav \
        --epochs 100 \
        --batch-size 16 \
        --lr 0.0005 \
        --patience 15 \
        --device auto \
        --model-path "./saved_models/full_training/enhanced_model.pt" \
        --cache-features \
        --export-onnx \
        --augment
}

# Custom training with user parameters
train_custom() {
    print_info "⚙️  Starting Custom Training"
    print_info "Using parameters: $@"
    
    python train.py \
        --chinese-instruments \
        --enhanced-features \
        --device auto \
        "$@"
}

# Post-training analysis
post_training_analysis() {
    print_info "📊 Running Post-Training Analysis..."
    
    # Find the latest model
    latest_model=$(ls -t saved_models/*.pt 2>/dev/null | head -1)
    
    if [ -n "$latest_model" ]; then
        print_success "Latest model: $latest_model"
        
        # Model size
        model_size=$(du -h "$latest_model" | cut -f1)
        print_info "Model size: $model_size"
        
        # Model info
        python -c "
import torch
try:
    checkpoint = torch.load('$latest_model', map_location='cpu', weights_only=False)
    print(f'Classes: {checkpoint.get(\"class_names\", \"Unknown\")}')
    print(f'Feature size: {checkpoint.get(\"feature_size\", \"Unknown\")}')
    print(f'Best accuracy: {checkpoint.get(\"best_accuracy\", \"Unknown\")}')
    print(f'Enhanced features: {checkpoint.get(\"enhanced_features\", \"Unknown\")}')
except Exception as e:
    print(f'Error loading model info: {e}')
"
    else
        print_warning "No trained models found"
    fi
}

# Display help
show_help() {
    cat << EOF
🎵 Enhanced Chinese Instrument Training Script

Usage:
  ./train.sh [mode] [additional_options]

Training Modes:
  quick      Quick debug training (5 epochs, small batch)
  standard   Standard training (30 epochs, balanced settings)
  full       Full training (100 epochs, all optimizations)
  custom     Custom training (pass your own parameters)

System Commands:
  check      Check environment and dependencies
  info       Show model information
  help       Show this help message

Examples:
  ./train.sh quick                                    # Quick test run
  ./train.sh standard                                 # Standard training
  ./train.sh full                                     # Full training
  ./train.sh custom --epochs 50 --batch-size 12      # Custom parameters
  ./train.sh check                                    # Check system

Options (for custom mode):
  --epochs N              Number of training epochs
  --batch-size N          Batch size for training
  --lr RATE               Learning rate
  --data-dir PATH         Path to training data
  --model-path PATH       Where to save the model
  --device TYPE           Device: cuda, cpu, mps, auto
  --debug                 Enable debug mode (fewer files)
  --export-onnx           Export ONNX format
  --cache-features        Cache extracted features
  --augment               Apply data augmentation

Models are saved in: saved_models/
Training data expected in: ../wav/
EOF
}

# Main script logic
main() {
    echo "🎵 Enhanced Chinese Instrument Training Script"
    echo "=============================================="
    
    # Parse command line arguments
    mode="${1:-help}"
    shift 2>/dev/null || true  # Remove first argument, ignore errors
    
    case "$mode" in
        "quick")
            check_environment
            activate_env
            check_dependencies
            train_quick
            post_training_analysis
            ;;
        "standard")
            check_environment
            activate_env
            check_dependencies
            train_standard
            post_training_analysis
            ;;
        "full")
            check_environment
            activate_env
            check_dependencies
            train_full
            post_training_analysis
            ;;
        "custom")
            check_environment
            activate_env
            check_dependencies
            train_custom "$@"
            post_training_analysis
            ;;
        "check")
            check_environment
            activate_env
            check_dependencies
            print_success "System check completed!"
            ;;
        "info")
            post_training_analysis
            ;;
        "help"|"-h"|"--help"|"")
            show_help
            ;;
        *)
            print_error "Unknown mode: $mode"
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"