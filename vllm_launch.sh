#!/bin/bash

# Launch vLLM OpenAI-compatible server for Qwen2.5-7B-Instruct
# This server provides faster inference than loading the model locally

MODEL="Qwen/Qwen2.5-7B-Instruct"
HOST="127.0.0.1"
PORT=60363

echo "Launching vLLM server for $MODEL"
echo "Server will be available at http://$HOST:$PORT"
echo "OpenAI-compatible API endpoint: http://$HOST:$PORT/v1"

CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --host $HOST \
    --port $PORT \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --trust-remote-code \
    --gpu_memory_utilization 0.9