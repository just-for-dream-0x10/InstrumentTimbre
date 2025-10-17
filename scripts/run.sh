#!/bin/bash

# InstrumentTimbre Run Script
# This script provides shortcuts for common commands

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Display help information
show_help() {
    echo -e "${BLUE}Instrument Timbre Analysis and Conversion System${NC}"
    echo "Usage: ./run.sh [command] [parameters]"
    echo ""
    echo "Available commands:"
    echo -e "  ${GREEN}train${NC} \t\tTrain model"
    echo -e "  ${GREEN}extract${NC} \tExtract timbre features"
    echo -e "  ${GREEN}apply${NC} \t\tApply timbre features"
    echo -e "  ${GREEN}separate${NC} \tSeparate audio sources"
    echo -e "  ${GREEN}cache-clear${NC} \tClear feature cache"
    echo -e "  ${GREEN}train-chinese${NC} \tTrain model with Chinese instrument mode"
    echo ""
    echo "Examples:"
    echo "  ./run.sh train --data-dir /path/to/data --model-path ./models/model.pt"
    echo "  ./run.sh extract --model-path ./models/model.pt --input-file ./audio/sample.wav"
    echo "  ./run.sh apply --model-path ./models/model.pt --target-file ./audio/piano.wav --timbre-file ./features/erhu_timbre.npz"
    echo "  ./run.sh separate --input-file ./audio/mixed.wav"
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

# Parse command
CMD="$1"
shift

case "$CMD" in
    train)
        echo -e "${BLUE}Starting model training...${NC}"
        python app.py train "$@"
        ;;
    extract)
        echo -e "${BLUE}Starting timbre feature extraction...${NC}"
        python app.py extract "$@"
        ;;
    apply)
        echo -e "${BLUE}Starting timbre feature application...${NC}"
        python app.py apply "$@"
        ;;
    separate)
        echo -e "${BLUE}Starting audio source separation...${NC}"
        python app.py separate "$@"
        ;;
    cache-clear)
        echo -e "${BLUE}Clearing feature cache...${NC}"
        python app.py cache --action clear
        ;;
    train-chinese)
        echo -e "${BLUE}Training model with Chinese instrument mode...${NC}"
        python app.py train --chinese-instruments --feature-type multi --augment --cache-features "$@"
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