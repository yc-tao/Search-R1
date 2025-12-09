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
Training entrypoint for medical composite reward:
reward = alpha * search_count + beta * evidence_hit.
"""

from pprint import pprint

import hydra
import ray
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
from verl.utils.reward_score import medical_evidence
from verl.utils.reward_score.search_count import count_searches


class MedicalEvidenceRewardManager:
    """Compute composite reward combining search count and evidence hit."""

    def __init__(
        self,
        tokenizer,
        num_examine: int = 0,
        max_searches: int = 10,
        reward_alpha: float = 0.5,
        reward_beta: float = 0.5,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.max_searches = max_searches
        self.reward_alpha = reward_alpha
        self.reward_beta = reward_beta

    def __call__(self, data: DataProto) -> torch.Tensor:
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        printed = 0

        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            extra_info = data_item.non_tensor_batch.get('extra_info', {}) if isinstance(
                data_item.non_tensor_batch, dict) else {}
            documents = extra_info.get('documents', []) if isinstance(extra_info, dict) else []
            evidence_spans = extra_info.get('evidence_spans', []) if isinstance(extra_info, dict) else []

            score = medical_evidence.compute_composite_reward(
                solution_str=sequences_str,
                documents=documents,
                evidence_spans=evidence_spans,
                max_searches=self.max_searches,
                alpha=self.reward_alpha,
                beta=self.reward_beta,
            )

            reward_tensor[i, valid_response_length - 1] = score

            if printed < self.num_examine:
                printed += 1
                searches = count_searches(sequences_str)
                hit = medical_evidence.has_evidence_hit(
                    medical_evidence.extract_retrieved_documents(sequences_str, documents),
                    evidence_spans,
                )
                print("\n[MedicalReward] Example:")
                print(f"  searches: {searches} (capped at {self.max_searches})")
                print(f"  evidence_hit: {hit}")
                print(f"  reward: {score:.4f}")
                print(f"  sequence head: {sequences_str[:400]}")

        return reward_tensor


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from verl.utils import hf_tokenizer

    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)
    tokenizer = hf_tokenizer(local_path)

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

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes}
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    max_searches = getattr(config, 'max_searches', 10)
    reward_alpha = getattr(config, 'reward_alpha', 0.5)
    reward_beta = getattr(config, 'reward_beta', 0.5)

    reward_fn = MedicalEvidenceRewardManager(
        tokenizer=tokenizer,
        num_examine=0,
        max_searches=max_searches,
        reward_alpha=reward_alpha,
        reward_beta=reward_beta,
    )
    val_reward_fn = MedicalEvidenceRewardManager(
        tokenizer=tokenizer,
        num_examine=1,
        max_searches=max_searches,
        reward_alpha=reward_alpha,
        reward_beta=reward_beta,
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
