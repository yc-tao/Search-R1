# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Main training script for search count reward.

The reward is based on the number of search queries issued:
    reward = min(num_searches, max_searches) / max_searches

This encourages the model to search more often (up to the cap).

Supports two modes:
1. Real retrieval: Uses the retrieval server for actual document search
2. Mock retrieval: Uses placeholder responses (no server needed, simpler training)
"""

from verl import DataProto
import torch
from verl.utils.reward_score import search_count
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
import numpy as np


class SearchCountRewardManager:
    """Reward manager that rewards based on search count.

    Reward = min(num_searches, max_searches) / max_searches
    """

    def __init__(self, tokenizer, num_examine: int = 0, max_searches: int = 10) -> None:
        """Initialize the reward manager.

        Args:
            tokenizer: The tokenizer for decoding sequences
            num_examine: Number of batches to print for debugging
            max_searches: Maximum number of searches to reward (cap)
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.max_searches = max_searches

    def __call__(self, data: DataProto) -> torch.Tensor:
        """Compute rewards for the batch.

        Args:
            data: DataProto containing batch data with prompts and responses

        Returns:
            Tensor of shape [batch_size, response_length] with rewards at the
            last valid token position for each sequence
        """
        # If there is rm score, we directly return rm score
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_printed = 0

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # Decode the full sequence
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            # Compute search count reward
            score = search_count.compute_score_search_count(
                solution_str=sequences_str,
                ground_truth=None,
                max_searches=self.max_searches
            )

            # Assign reward to the last valid response token
            reward_tensor[i, valid_response_length - 1] = score

            # Debug printing
            if already_printed < self.num_examine:
                already_printed += 1
                num_searches = search_count.count_searches(sequences_str)
                print(f"\n[SearchCountReward] Sample {i}:")
                print(f"  num_searches: {num_searches}")
                print(f"  capped: {min(num_searches, self.max_searches)}")
                print(f"  reward: {score:.4f}")
                print(f"  sequence (first 500 chars): {sequences_str[:500]}")

        return reward_tensor


import ray
import hydra


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # Reward model (optional)
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    # Get max_searches from config (default 10)
    max_searches = getattr(config, 'max_searches', 10)

    # Check mock retrieval mode
    use_mock_retrieval = getattr(config, 'use_mock_retrieval', True)
    if use_mock_retrieval:
        print("[SearchCountTraining] Using mock retrieval mode (no retrieval server needed)")
    else:
        print("[SearchCountTraining] Using real retrieval mode (requires retrieval server)")

    # Create search count reward manager
    reward_fn = SearchCountRewardManager(
        tokenizer=tokenizer,
        num_examine=0,
        max_searches=max_searches
    )

    # Note that we always use function-based RM for validation
    val_reward_fn = SearchCountRewardManager(
        tokenizer=tokenizer,
        num_examine=1,
        max_searches=max_searches
    )

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    trainer = RayPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        reward_fn=reward_fn,
        val_reward_fn=val_reward_fn,
    )

    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
