#!/bin/bash
# Train with search count reward on medical diagnosis data
#
# Reward: min(num_searches, 10) / 10
# Algorithm: GRPO (critic-free)
# Data: Medical diagnosis from concatenated_notes_by_episode.pkl
#
# Prerequisites:
#   1. Prepare data: python prepare_medical_data.py
#   2. Start retrieval server: bash retrieval_launch.sh
#   3. Run this script: bash train_medical_search_count.sh
#
# Note: Uses real retrieval server at http://127.0.0.1:56321/retrieve

set -e

# =========================
# RAY DEADLOCK PREVENTION
# =========================
export RAY_ADDRESS=10.55.164.88:6390
export RAY_OVERRIDE_NODE_IP=10.55.164.88

# GPU configuration
export CUDA_VISIBLE_DEVICES=4,5,6,7

# Data directory (created by prepare_medical_data.py)
export DATA_DIR='data/medical_diagnosis'
RETRIEVAL_URL="http://127.0.0.1:56321/retrieve"

# Weights & Biases project name
WANDB_PROJECT='Search-R1-Medical'

# Model configuration
export BASE_MODEL='Qwen/Qwen2.5-7B-Instruct'
export EXPERIMENT_NAME=medical-search-count-grpo-qwen2.5-7b-it

# Alternative models (uncomment to use):
# export BASE_MODEL='Qwen/Qwen2.5-3B-Instruct'
# export EXPERIMENT_NAME=medical-search-count-grpo-qwen2.5-3b-it
# export BASE_MODEL='meta-llama/Llama-3.1-8B-Instruct'
# export EXPERIMENT_NAME=medical-search-count-grpo-llama3.1-8b-it

# vLLM backend configuration
export VLLM_ATTENTION_BACKEND=XFORMERS

# Search count reward configuration
MAX_SEARCHES=10  # Cap for search count reward (reward = min(searches, 10) / 10)

# Retrieval mode: Using real retrieval server (must be running)
USE_MOCK_RETRIEVAL=false

# Training configuration
# max_prompt_length = max_start_length + max_response_length * (max_turns - 1) + max_obs_length * max_turns
MAX_TURNS=10  # Allow up to 10 search turns to match the reward cap

# Create logs directory
mkdir -p logs

echo "================================================"
echo "Training with Search Count Reward"
echo "================================================"
echo "Model: $BASE_MODEL"
echo "Data: $DATA_DIR"
echo "Max searches (reward cap): $MAX_SEARCHES"
echo "Max turns: $MAX_TURNS"
echo "Retrieval server: $RETRIEVAL_URL"
echo "================================================"

# Check if data exists
if [ ! -f "$DATA_DIR/train.parquet" ]; then
    echo "Error: Training data not found at $DATA_DIR/train.parquet"
    echo "Please run: python prepare_medical_data.py"
    exit 1
fi

# Check if retrieval server is running
echo "Checking if retrieval server is running..."
health_payload='{"queries":["health check"],"documents":[{"id":"ping","contents":"ping"}],"topk":1,"return_scores":false}'
health_check=$(curl -s -w "\n%{http_code}" -X POST "$RETRIEVAL_URL" -H "Content-Type: application/json" -d "$health_payload" || printf "000")
status_code=${health_check##*$'\n'}
response_body=${health_check%$'\n'*}

if [ "$status_code" = "000" ] || [ -z "$status_code" ]; then
    echo "Error: Retrieval server is not reachable at $RETRIEVAL_URL"
    echo "Please start the retrieval server: bash retrieval_launch.sh"
    exit 1
fi
if [[ "$status_code" =~ ^2 ]]; then
    echo "Retrieval server is running ✓"
else
    echo "Warning: Retrieval server responded with HTTP $status_code during the health check (continuing anyway)."
    if [ -n "$response_body" ] && [ "$response_body" != "$status_code" ]; then
        echo "Response: $response_body"
    fi
fi

echo "Starting isolated Ray cluster..."


PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_search_count \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=256 \
    data.val_batch_size=128 \
    data.max_prompt_length=8192 \
    data.max_response_length=512 \
    data.max_start_length=4096 \
    data.max_obs_length=512 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size=32 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=64 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=64 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    algorithm.no_think_rl=false \
    actor_rollout_ref.rollout.n_agent=5 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.actor.state_masking=true \
    trainer.logger=['wandb'] \
    +trainer.val_only=false \
    +trainer.val_before_train=true \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=25 \
    trainer.project_name=$WANDB_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=10 \
    trainer.total_training_steps=500 \
    trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
    max_turns=$MAX_TURNS \
    max_searches=$MAX_SEARCHES \
    use_mock_retrieval=$USE_MOCK_RETRIEVAL \
    retriever.url="$RETRIEVAL_URL" \
    retriever.topk=3 \
    2>&1 | tee logs/$EXPERIMENT_NAME.log
