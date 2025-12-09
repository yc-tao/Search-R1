#!/bin/bash

# Medical Diagnosis Inference Runner
# This script runs the diagnosis inference with proper environment setup

echo "========================================="
echo "Medical Diagnosis Inference"
echo "========================================="

# Check if retrieval server is running
echo "Checking retrieval server..."
if ! curl -s http://127.0.0.1:56321/retrieve > /dev/null 2>&1; then
    echo "ERROR: Retrieval server is not running!"
    echo "Please start it with: bash retrieval_launch.sh"
    exit 1
fi
echo "✓ Retrieval server is running"

# Check if data file exists
DATA_FILE="$HOME/SRL/data/concatenated_notes_by_episode.pkl"
if [ ! -f "$DATA_FILE" ]; then
    echo "ERROR: Data file not found at $DATA_FILE"
    exit 1
fi
echo "✓ Data file found"

# Activate conda environment (if not already activated)
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "searchr1" ]; then
    echo "Activating conda environment: searchr1"
    eval "$(conda shell.bash hook)"
    conda activate searchr1
else
    echo "✓ Environment searchr1 already activated"
fi

# Run the diagnosis script
echo ""
echo "Starting diagnosis inference..."
echo "========================================="
python infer_diagnosis.py

echo ""
echo "========================================="
echo "Diagnosis complete!"
echo "========================================="
