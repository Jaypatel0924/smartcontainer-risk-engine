#!/bin/bash
# SmartContainer Risk Engine - Quick Start Script (Linux/Mac)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "======================================"
echo "SmartContainer Risk Engine"
echo "HackaMINEd-2026 Hackathon"
echo "======================================"

# Check Python
echo -e "\n${YELLOW}[CHECK]${NC} Python installation..."
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}[ERROR]${NC} Python 3 not found"
    exit 1
fi

echo -e "${GREEN}[OK]${NC} Python found: $(python3 --version)"

# Create venv
echo -e "\n${YELLOW}[SETUP]${NC} Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}[OK]${NC} Virtual environment created"
else
    echo -e "${GREEN}[OK]${NC} Virtual environment already exists"
fi

# Activate venv
echo -e "\n${YELLOW}[SETUP]${NC} Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo -e "\n${YELLOW}[SETUP]${NC} Installing dependencies..."
pip install -q -r requirements.txt
echo -e "${GREEN}[OK]${NC} Dependencies installed"

# Train model
echo -e "\n${YELLOW}[STEP 1]${NC} Training ML models..."
python3 model_training.py
if [ $? -ne 0 ]; then
    echo -e "${RED}[ERROR]${NC} Model training failed"
    exit 1
fi

# Generate predictions
echo -e "\n${YELLOW}[STEP 2]${NC} Generating predictions..."
python3 predict.py
if [ $? -ne 0 ]; then
    echo -e "${RED}[ERROR]${NC} Prediction generation failed"
    exit 1
fi

echo -e "\n${GREEN}[SUCCESS]${NC} Pipeline completed successfully!"
echo -e "\n${YELLOW}[INFO]${NC} Generated files:"
echo "  - models/random_forest_model.pkl"
echo "  - models/isolation_forest_model.pkl"
echo "  - output/risk_predictions.csv"

echo -e "\n${YELLOW}[NEXT]${NC} Launch dashboard:"
echo "  streamlit run dashboard.py"
