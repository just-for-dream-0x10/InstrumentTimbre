#!/bin/bash
# Test Runner for InstrumentTimbre
# InstrumentTimbre 测试运行脚本

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

# Run tests
run_tests() {
    print_info "🧪 Running InstrumentTimbre Test Suite"
    echo "=" * 50
    
    # Check if pytest is available
    if ! command -v pytest &> /dev/null; then
        print_warning "pytest not found, installing..."
        pip install pytest
    fi
    
    # Run different test categories
    print_info "Running unit tests..."
    pytest tests/test_utils.py -v
    
    print_info "Running feature tests..."
    pytest tests/test_chinese_features.py -v
    
    print_info "Running training tests..."
    pytest tests/test_training.py -v
    
    print_info "Running all tests with coverage..."
    pytest tests/ -v --tb=short
    
    print_success "All tests completed!"
}

# Demo prediction workflow
demo_prediction() {
    print_info "🔮 Demo: Prediction Workflow"
    echo "=" * 40
    
    # Check if model exists
    if [ ! -f "saved_models/chinese_instrument_enhanced.pt" ]; then
        print_warning "No trained model found, please run training first"
        return
    fi
    
    # Check if example files exist
    if [ ! -f "example/erhu1.wav" ]; then
        print_warning "Example audio files not found"
        return
    fi
    
    print_info "Making single file prediction..."
    python predict.py --model saved_models/chinese_instrument_enhanced.pt --input example/erhu1.wav
    
    print_info "Making batch predictions..."
    python predict.py --model saved_models/chinese_instrument_enhanced.pt --input example/ --output demo_predictions.json
    
    print_success "Prediction demo completed!"
}

# Demo evaluation workflow
demo_evaluation() {
    print_info "📊 Demo: Model Evaluation"
    echo "=" * 40
    
    if [ ! -f "saved_models/chinese_instrument_enhanced.pt" ]; then
        print_warning "No trained model found, please run training first"
        return
    fi
    
    if [ ! -d "example" ]; then
        print_warning "Example directory not found"
        return
    fi
    
    print_info "Running model evaluation..."
    python evaluate.py --model saved_models/chinese_instrument_enhanced.pt --test-dir example --output-dir evaluation_demo
    
    print_success "Evaluation demo completed!"
}

# Quick system check
system_check() {
    print_info "🔍 System Check"
    echo "=" * 30
    
    # Check Python version
    python_version=$(python --version 2>&1)
    print_info "Python: $python_version"
    
    # Check key dependencies
    python -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null && print_success "PyTorch OK" || print_error "PyTorch missing"
    python -c "import librosa; print(f'Librosa: {librosa.__version__}')" 2>/dev/null && print_success "Librosa OK" || print_error "Librosa missing"
    python -c "import numpy; print(f'NumPy: {numpy.__version__}')" 2>/dev/null && print_success "NumPy OK" || print_error "NumPy missing"
    python -c "import matplotlib; print(f'Matplotlib: {matplotlib.__version__}')" 2>/dev/null && print_success "Matplotlib OK" || print_warning "Matplotlib missing (optional)"
    
    # Check for trained models
    if [ -d "saved_models" ] && [ "$(ls -A saved_models 2>/dev/null)" ]; then
        model_count=$(ls saved_models/*.pt 2>/dev/null | wc -l)
        print_info "Found $model_count trained models"
    else
        print_warning "No trained models found"
    fi
    
    # Check for test data
    if [ -d "example" ]; then
        audio_count=$(find example -name "*.wav" 2>/dev/null | wc -l)
        print_info "Found $audio_count example audio files"
    else
        print_warning "Example directory not found"
    fi
    
    print_success "System check completed!"
}

# Show help
show_help() {
    cat << EOF
🧪 InstrumentTimbre Test & Demo Suite

Usage:
  ./run_tests.sh [command]

Commands:
  test        Run all tests
  predict     Demo prediction workflow
  evaluate    Demo evaluation workflow
  check       System check
  all         Run everything
  help        Show this help

Examples:
  ./run_tests.sh test       # Run test suite
  ./run_tests.sh predict   # Demo predictions
  ./run_tests.sh all       # Run everything

Requirements:
  - Trained model in saved_models/
  - Example audio files in example/
  - pytest installed (pip install pytest)

EOF
}

# Main script
main() {
    echo "🎵 InstrumentTimbre Test & Demo Suite"
    echo "====================================="
    
    command="${1:-help}"
    
    case "$command" in
        "test")
            run_tests
            ;;
        "predict")
            demo_prediction
            ;;
        "evaluate")
            demo_evaluation
            ;;
        "check")
            system_check
            ;;
        "all")
            system_check
            echo
            run_tests
            echo
            demo_prediction
            echo
            demo_evaluation
            ;;
        "help"|"-h"|"--help"|"")
            show_help
            ;;
        *)
            print_error "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac
}

main "$@"