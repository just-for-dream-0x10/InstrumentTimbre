#!/bin/bash

# Instrument Timbre Analysis and Conversion System - Training Script
# This script provides shortcuts for common training commands

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Display help information
show_help() {
    echo -e "${BLUE}Instrument Timbre Analysis and Conversion System - Training Script${NC}"
    echo "Usage: ./train.sh [command] [parameters]"
    echo ""
    echo "Available commands:"
    echo -e "  ${GREEN}standard${NC} \tStandard model training"
    echo -e "  ${GREEN}chinese${NC} \tChinese traditional instrument model training"
    echo -e "  ${GREEN}advanced${NC} \tAdvanced model training (with data augmentation and feature caching)"
    echo -e "  ${GREEN}quick${NC} \tQuick training (for testing)"
    echo -e "  ${GREEN}help${NC} \t\tDisplay this help information"
    echo ""
    echo "Examples:"
    echo "  ./train.sh standard --data-dir ../wav --model-path ./saved_models/standard_model.pt"
    echo "  ./train.sh chinese --data-dir ../wav --model-path ./saved_models/chinese_model.pt"
    echo "  ./train.sh advanced --data-dir ../wav --model-path ./saved_models/advanced_model.pt"
    echo "  ./train.sh quick --data-dir ../wav --model-path ./saved_models/test_model.pt"
}

# Check if Python and required dependencies are installed
check_dependencies() {
    # Check Python
    if ! command -v python &> /dev/null; then
        echo -e "${RED}Error: Python not found. Please install Python 3.7 or higher${NC}"
        exit 1
    fi
    
    # Check pip
    if ! command -v pip &> /dev/null; then
        echo -e "${RED}Error: pip not found. Please install pip${NC}"
        exit 1
    fi
    
    # Check required dependencies
    python -c "import torch" &> /dev/null || {
        echo -e "${YELLOW}PyTorch not found. Installing required dependencies...${NC}"
        pip install -r requirements.txt
    }
}

# If no parameters, show help information
if [ $# -eq 0 ]; then
    show_help
    exit 0
fi

# Check dependencies
check_dependencies

# Parse command
CMD="$1"
shift

case "$CMD" in
    standard)
        echo -e "${BLUE}Starting standard model training...${NC}"
        python train.py --model-path ./saved_models/standard_model.pt "$@"
        ;;
    chinese)
        echo -e "${BLUE}Starting Chinese traditional instrument model training...${NC}"
        python train.py --chinese-instruments --feature-type multi --model-path ./saved_models/chinese_model.pt "$@"
        ;;
    advanced)
        echo -e "${BLUE}Starting advanced model training (with data augmentation and feature caching)...${NC}"
        python train.py --chinese-instruments --feature-type multi --augment --cache-features --model-path ./saved_models/advanced_model.pt "$@"
        ;;
    quick)
        echo -e "${BLUE}Starting quick training (for testing)...${NC}"
        python train.py --epochs 3 --batch-size 16 --debug --model-path ./saved_models/test_model.pt "$@"
        ;;
    help)
        show_help
        ;;
    *)
        echo -e "${RED}Unknown command: $CMD${NC}"
        show_help
        exit 1
        ;;
esac

exit 0