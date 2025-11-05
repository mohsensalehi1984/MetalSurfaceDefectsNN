#!/bin/bash
# ==============================================================
# Metal Surface Defect Classification - Run Script
# Author: Mohsen Salehi
# ==============================================================
# Usage:
#   chmod +x run.sh
#   ./run.sh train      # to train the model
#   ./run.sh evaluate   # to evaluate the best model
#   ./run.sh inspect    # to inspect the architecture
# ==============================================================

set -e

# 1️⃣ Activate virtual environment
if [ -d "venv" ]; then
    echo "✅ Using existing virtual environment..."
    source venv/bin/activate
else
    echo "🚀 Creating new virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
fi

# 2️⃣ Select action
case "$1" in
    train)
        echo "🧠 Training model..."
        python src/train.py
        ;;
    evaluate)
        echo "🔍 Evaluating best model..."
        python src/inference.py --checkpoint checkpoints/best_model.pt --evaluate
        ;;
    inspect)
        echo "🔧 Inspecting model..."
        python src/modelInspect.py
        ;;
    *)
        echo "❌ Invalid option. Use one of: train | evaluate | inspect"
        exit 1
        ;;
esac
