#!/bin/bash
# Train with composite medical reward (search count + MDACE evidence hit) using GRPO.

set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export DATA_DIR="data/medical_evidence"
RETRIEVAL_URL="http://127.0.0.1:56321/retrieve"

WANDB_PROJECT="Search-R1-Medical"

export BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
export EXPERIMENT_NAME="medical-evidence-grpo-qwen2.5-7b-it"

export VLLM_ATTENTION_BACKEND=XFORMERS

REWARD_ALPHA=0.5
REWARD_BETA=0.5
MAX_SEARCHES=10
MAX_TURNS=10

mkdir -p logs

if [ ! -f "$DATA_DIR/train.parquet" ]; then
  echo "Training data not found at $DATA_DIR/train.parquet. Run prepare_medical_evidence_data.py first."
  exit 1
fi

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_medical_evidence \
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
    actor_rollout_ref.actor.state_masking=true \
    algorithm.no_think_rl=false \
    actor_rollout_ref.rollout.n_agent=5 \
    actor_rollout_ref.rollout.temperature=1 \
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
    trainer.total_epochs=8 \
    trainer.total_training_steps=480 \
    trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
    max_turns=$MAX_TURNS \
    max_searches=$MAX_SEARCHES \
    +reward_alpha=$REWARD_ALPHA \
    +reward_beta=$REWARD_BETA \
    use_mock_retrieval=false \
    retriever.url="$RETRIEVAL_URL" \
    retriever.topk=3 \
    2>&1 | tee logs/$EXPERIMENT_NAME.log
