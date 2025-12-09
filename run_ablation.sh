#!/bin/bash

# Ablation Experiment Launch Script
# This script runs the ablation experiments comparing:
#   1. Full note diagnosis (no RAG)
#   2. Last section only diagnosis (no RAG)

set -e  # Exit on error

echo "======================================================================"
echo "ABLATION EXPERIMENT: Direct Diagnosis (No RAG)"
echo "======================================================================"

# Check if vLLM server is running on port 60363
echo ""
echo "Checking vLLM server status..."
if ! curl -s http://127.0.0.1:60363/health > /dev/null 2>&1; then
    echo "ERROR: vLLM server is not running on port 60363!"
    echo ""
    echo "Please start the vLLM server first:"
    echo "  bash vllm_launch.sh"
    echo "  OR"
    echo "  python -m vllm.entrypoints.openai.api_server \\"
    echo "    --model Qwen/Qwen2.5-7B-Instruct \\"
    echo "    --port 60363 \\"
    echo "    --tensor-parallel-size 1 \\"
    echo "    --gpu-memory-utilization 0.9"
    exit 1
fi
echo "✓ vLLM server is running on port 60363"

# Check if data file exists
DATA_FILE="$HOME/SRL/data/concatenated_notes_by_episode.pkl"
if [ ! -f "$DATA_FILE" ]; then
    echo "ERROR: Data file not found at $DATA_FILE"
    exit 1
fi
echo "✓ Data file found: $DATA_FILE"

# Check Python dependencies
echo ""
echo "Checking Python dependencies..."
python3 -c "import pandas; import sklearn; import numpy; import openai" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "ERROR: Missing required Python packages!"
    echo "Please install: pip install pandas scikit-learn numpy openai"
    exit 1
fi
echo "✓ All dependencies available"

echo ""
echo "======================================================================"
echo "Starting ablation experiment..."
echo "======================================================================"
echo ""
echo "Configuration:"
echo "  - vLLM Server: http://127.0.0.1:60363/v1"
echo "  - Model: Qwen/Qwen2.5-7B-Instruct"
echo "  - Data: $DATA_FILE"
echo "  - Total episodes: 775"
echo "  - Ablations: 2 (full_note, last_section)"
echo "  - Total runs: 1550"
echo ""
echo "Output files:"
echo "  - ablation_predictions.csv (predictions for all episodes)"
echo "  - ablation_metrics.csv (evaluation metrics)"
echo "  - ablation_checkpoint.pkl (progress checkpoint)"
echo ""
echo "Note: This script supports resuming from checkpoint if interrupted."
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."

# Run the ablation experiment
echo ""
echo "======================================================================"
python3 ablation_diagnosis.py

echo ""
echo "======================================================================"
echo "Ablation experiment completed!"
echo "======================================================================"
echo ""
echo "Output files created:"
ls -lh ablation_predictions.csv ablation_metrics.csv 2>/dev/null || echo "Warning: Output files not found"
echo ""
echo "To view results:"
echo "  - Predictions: cat ablation_predictions.csv | head -20"
echo "  - Metrics: cat ablation_metrics.csv"
echo ""
