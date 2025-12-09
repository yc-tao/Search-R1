# Search-R1 Codebase Exploration Report

## Executive Summary
Search-R1 is a reinforcement learning framework for training LLMs to interleave reasoning and tool-use (search engine calling). It's built on veRL and extends DeepSeek-R1 with integrated search engine access. The codebase implements PPO and GRPO algorithms with multi-turn reasoning capabilities.

---

## 1. TRAINING INFRASTRUCTURE

### 1.1 Core Training Entry Points
- **`/verl/trainer/main_ppo.py`** (203 lines)
  - Main entry point for PPO training using Ray for distributed execution
  - Uses `RewardManager` class to compute rewards based on rule-based scoring (exact match for QA)
  - Instantiates `RayPPOTrainer` to manage distributed training across multiple nodes/GPUs

- **`/verl/trainer/ppo/ray_trainer.py`** (867 lines)
  - Core distributed trainer using Ray for resource management
  - Implements PPO training loop with data collection, advantage computation, and policy updates
  - Contains key functions:
    - `apply_kl_penalty()` - Applies KL divergence penalty to rewards
    - `compute_advantage()` - Computes advantages (GAE or GRPO)
    - `compute_data_metrics()` - Collects training metrics

### 1.2 Training Configuration
- **`/verl/trainer/config/ppo_trainer.yaml`** (180 lines)
  - Comprehensive config with sections for:
    - `data`: Batch sizes, sequence lengths (max_prompt=4096, max_response=500, max_obs=500)
    - `actor_rollout_ref`: Actor, rollout (vLLM-based), and reference policy configs
    - `critic`: Value function model configuration
    - `algorithm`: Advantage estimator (GAE or GRPO), KL penalty settings
    - `retriever`: URL and top-k for search engine access
    - `max_turns`: Maximum multi-turn interactions (default 10)

### 1.3 Training Scripts
- **`train_ppo.sh`** - Launches PPO training with 8 GPUs
  - Uses Llama-3.2-3B or Qwen2.5 models
  - Configurable turns (max_turns=2), retriever URL
  - Trains on NQ dataset with EM-based reward

- **`train_grpo.sh`** - Launches GRPO training variant
  - Similar setup but with GRPO advantage estimator
  - Multiple samples per prompt (n_agent=5)

---

## 2. SEARCH QUERY USAGE

### 2.1 Query Generation Pipeline
The model generates search queries through structured XML tags:
```
<search>query text</search>  -> Triggers search
<answer>answer text</answer> -> Provides final answer
```

### 2.2 Action Parsing and Execution (`/search_r1/llm_agent/generation.py`)

**`postprocess_predictions()` (lines 407-436)**
- Extracts search/answer actions from LLM outputs using regex pattern: `<(search|answer)>(.*?)</\1>`
- Returns tuple of (actions, contents)

**`execute_predictions()` (lines 353-405)**
- Core environment step function
- Routes search queries to search engine, returns formatted results
- Marks episodes as done only on `</answer>` tag
- Handles invalid actions with feedback: "My previous action is invalid..."

**`batch_search()` (lines 438-448)**
- Batches multiple search queries together
- Calls remote search server via HTTP
- Formats results as: `<information>passage1\n\npassage2\n...</information>`

### 2.3 Multi-Turn Interaction Flow (`run_llm_loop()`, lines 220-319)
1. Generate response token by token until `</search>` or `</answer>`
2. Parse action (search or answer)
3. Execute in environment (call search API or terminate)
4. Get observation (search results or empty for answer)
5. Continue loop for up to `max_turns` iterations
6. Force final answer generation at end

### 2.4 Retrieval Server (`/search_r1/search/retrieval_server.py`)
- FastAPI-based BM25 retriever
- Endpoint: `POST /retrieve`
- Input: `{"queries": [...], "documents": [...], "topk": 3}`
- Output: Top-k documents with scores

---

## 3. REWARD/EVALUATION MECHANISMS

### 3.1 Rule-Based Reward Function (`/verl/utils/reward_score/qa_em.py`)

**Exact Match (EM) Scoring for QA:**
- **`extract_solution()`** (lines 62-82)
  - Extracts answer from model output using regex: `<answer>(.*?)</answer>`
  - Returns last occurrence if multiple `</answer>` tags exist
  - Returns None if fewer than 2 matches (invalid format)

- **`compute_score_em()`** (lines 85-110)
  - Compares extracted answer with ground truth using normalized EM
  - Normalization: lowercase, remove articles/punctuation, fix whitespace
  - Returns 1.0 for correct answer, 0.0 for incorrect
  - Optional format_score for partial credit

- **`em_check()`** (lines 36-46)
  - Performs normalized string comparison
  - Supports multiple ground truth answers

### 3.2 Reward Manager in Training (`/verl/trainer/main_ppo.py`, lines 32-97)

**`RewardManager.__call__()`**
- Processes DataProto batch through PPO training
- For each sample:
  1. Extracts prompt IDs and response IDs using attention masks
  2. Decodes full sequence
  3. Computes reward using data-source-specific scoring function
  4. Places token-level reward at EOS position
- Supports multiple datasets (nq, triviaqa, hotpotqa, musique, etc.)

### 3.3 Reward Aggregation for Multi-Turn
- **Token-level rewards**: Reward placed only at final token of response
- **Response-level rewards**: Computed from final answer extraction
- **Multi-turn**: Rewards accumulated across turns until episode termination

---

## 4. MODEL ARCHITECTURE

### 4.1 Base Models Supported
- Llama-3.2-3B (base and instruct versions)
- Llama-3.1-8B (base and instruct versions)  
- Qwen2.5-3B/7B (base and instruct versions)
- Any HuggingFace causal LM

### 4.2 Training Model Architecture (`/verl/workers/fsdp_workers.py`)

**ActorRolloutRefWorker** (lines 47-150+)
- Can function as:
  - `actor`: Trained policy network
  - `rollout`: Generation engine (vLLM)
  - `ref`: Reference policy for KL penalty
  - Hybrid combinations

- FSDP distributed training with optional offloading:
  - Parameter offload to CPU
  - Gradient offload to CPU
  - Optimizer state offload to CPU

- Gradient checkpointing for memory efficiency
- Position ID and attention mask computation

### 4.3 Value Function (Critic)
- Separate model (same base model)
- Predicts value estimates for advantage computation
- FSDP training with configurable micro-batch sizes

### 4.4 Inference Engine (`/verl/workers/rollout/vllm_rollout/vllm_rollout.py`)

**vLLMRollout** (lines 57+)
- Uses vLLM for efficient generation
- Supports tensor parallelism
- Configurable:
  - Temperature: 1.0 (default)
  - Top-p: 0.95
  - GPU memory utilization: 0.6 (typical)
  - Max tokens: 500

- **`generate_sequences()`** (lines 142+)
  - Takes left-padded prompts
  - Returns responses with token-level log probabilities
  - Supports batch processing with dynamic batching

---

## 5. DATA PROCESSING PIPELINE

### 5.1 Dataset Format Requirements
**Expected data structure (from `/scripts/data_process/nq_search.py`):**
```python
{
    "data_source": "nq",  # Identifies dataset type
    "prompt": [{
        "role": "user",
        "content": "Answer the given question. You must conduct reasoning inside <think> and </think>..."
    }],
    "ability": "fact-reasoning",
    "reward_model": {
        "style": "rule",
        "ground_truth": {"target": ["answer1", "answer2"]}
    },
    "extra_info": {
        "split": "train",
        "index": 0
    }
}
```

### 5.2 Data Processing Scripts

**`/scripts/data_process/nq_search.py`** (100 lines)
- Loads NQ dataset from HuggingFace
- Adds prefix template for search+reason instruction
- Creates ground truth answer list
- Outputs train.parquet and test.parquet

**Prefix Template:**
```
"Answer the given question. You must conduct reasoning inside 
<think> and </think> first every time you get new information. 
After reasoning, if you find you lack some knowledge, you can call 
a search engine by <search> query </search> and it will return the 
top searched results between <information> and </information>. 
You can search as many times as your want. If you find no further 
external knowledge needed, you can directly provide the answer 
inside <answer> and </answer>..."
```

### 5.3 Corpus Format
**Wikipedia corpus (example from `example/corpus.jsonl`):**
```json
{"id": "0", "contents": "\"Document Title\"\nDocument text content..."}
{"id": "1", "contents": "\"Another Title\"\nMore content..."}
```

### 5.4 Data Loading in Training
- Uses PyArrow parquet files for efficiency
- Configurable train/val splits
- Batch size typically 512 (train) / 256 (val)
- Tokenized on-the-fly with padding to max lengths

---

## 6. PPO ALGORITHM IMPLEMENTATION

### 6.1 Advantage Estimation (`/verl/trainer/ppo/core_algos.py`)

**GAE (Generalized Advantage Estimation)** (lines 70-107)
```
advantages = cumsum of (delta + gamma * lam * lastgaelam)
returns = advantages + values
```
- Computes token-level advantages
- Normalizes advantages (masked whitening)
- Parameters: gamma=1.0 (discount), lambda=1.0 (trace decay)

**GRPO Outcome Advantage** (lines 111-155)
- Treats entire response as single outcome
- Groups samples by prompt (uid)
- Normalizes scores within each prompt group
- Returns normalized score replicated across response tokens

### 6.2 KL Penalty (`/verl/trainer/ppo/core_algos.py`, lines 242-274)

**KL Controller** (lines 28-66)
- `AdaptiveKLController`: Adjusts beta based on divergence from target
- `FixedKLController`: Maintains constant KL coefficient

**KL Divergence Computation:**
- kl_penalty='kl': Direct KL (log_prob - ref_log_prob)
- kl_penalty='low_var_kl': Low variance approximation
- Applied: reward_final = token_reward - beta * KL_divergence

### 6.3 Policy Loss (`/verl/trainer/ppo/core_algos.py`, lines 163-194)
```
ratio = exp(new_log_prob - old_log_prob)
pg_loss = max(-adv * ratio, -adv * clamp(ratio, 1-eps, 1+eps))
```
- Clip range: 0.2 (typical)
- Computes clipping fraction for monitoring

### 6.4 Value Loss (`/verl/trainer/ppo/core_algos.py`, lines 216-239)
```
vpred_clipped = clamp(vpred, value - cliprange, value + cliprange)
vf_loss = 0.5 * max((vpred - return)^2, (vpred_clipped - return)^2)
```

### 6.5 Training Loop Integration (`/verl/trainer/ppo/ray_trainer.py`)

**Key components:**
1. **Data Collection**
   - Generates trajectories using current policy (vLLM)
   - Collects log probabilities from actor
   - Calls reference policy for KL computation

2. **Advantage Computation** (lines 123-154)
   - KL penalty applied in `apply_kl_penalty()`
   - Advantage estimated via GAE or GRPO
   - Masked computation to exclude padding

3. **Policy Update**
   - Actor parameters updated with PPO loss
   - Critic parameters updated with value loss
   - Multiple mini-batch passes through data

4. **Metrics Collection**
   - response_score: Sum of token-level rewards
   - response_reward: After KL adjustment
   - advantages: Distribution statistics
   - KL divergence and coefficient

---

## 7. DIRECTORY STRUCTURE

```
/home/yichentao/Search-R1/
├── verl/                          # Core RL training framework
│   ├── trainer/
│   │   ├── main_ppo.py           # PPO entry point
│   │   ├── ppo/
│   │   │   ├── ray_trainer.py    # Distributed trainer
│   │   │   └── core_algos.py     # PPO/GRPO algorithms
│   │   └── config/
│   │       └── ppo_trainer.yaml  # Training configuration
│   ├── workers/
│   │   ├── fsdp_workers.py       # FSDP-based workers
│   │   ├── rollout/              # Generation/rollout implementations
│   │   │   └── vllm_rollout/     # vLLM-based efficient inference
│   │   └── reward_model/         # Reward model workers
│   ├── models/                    # Model architectures
│   │   ├── transformers/         # HuggingFace model wrappers
│   │   └── llama/                # Llama-specific models
│   ├── single_controller/         # Distributed orchestration
│   │   └── ray/                  # Ray-based controller
│   └── utils/
│       ├── reward_score/         # Reward computation functions
│       ├── torch_functional.py   # Masked operations
│       └── fsdp_utils.py         # FSDP utilities
│
├── search_r1/                     # Search+reasoning LLM logic
│   ├── llm_agent/
│   │   ├── generation.py         # Multi-turn generation manager
│   │   ├── tensor_helper.py      # Tensor operations
│   │   └── __init__.py
│   └── search/
│       ├── retrieval.py          # Retrieval interface
│       ├── retrieval_server.py   # BM25 API server
│       ├── google_search_server.py
│       └── index_builder.py      # FAISS indexing
│
├── scripts/
│   ├── data_process/
│   │   ├── nq_search.py         # NQ dataset processing
│   │   ├── nq_rag.py
│   │   └── qa_search_*.py
│   └── download.py
│
├── train_ppo.sh                   # Training launcher
├── train_grpo.sh
├── infer.py                       # Inference script
└── requirements.txt
```

---

## 8. KEY DATA STRUCTURES

### 8.1 DataProto (Core data structure)
- `batch`: Dict of tensors (input_ids, attention_mask, responses, etc.)
- `non_tensor_batch`: Dict of non-tensor data (metadata)
- `meta_info`: Additional metadata
- Supports masking and padding operations

### 8.2 Training Data Components
```
input_ids: (bs, prompt_length)
responses: (bs, response_length)
attention_mask: (bs, prompt_length + response_length)
position_ids: (bs, prompt_length + response_length)
old_log_probs: (bs, response_length)
ref_log_probs: (bs, response_length)
values: (bs, response_length)
token_level_scores: (bs, response_length)
token_level_rewards: (bs, response_length)
advantages: (bs, response_length)
returns: (bs, response_length)
```

### 8.3 Generation State
```
prompts: Initial question with instruction prefix
responses: Model-generated search/answer actions
responses_with_info_mask: Masked version for state masking
info_mask: Mask to identify search result information
```

---

## 9. CONFIGURATION PARAMETERS FOR PPO

### Critical Hyperparameters
- **Learning Rate (actor)**: 1e-6, warmup ratio: 0.285
- **Learning Rate (critic)**: 1e-5, warmup ratio: 0.015
- **Batch Size**: 512 (train), 256 (val)
- **Mini-batch Size**: 256, Micro-batch: 64
- **PPO Epochs**: 1 (full pass through data once)
- **Clip Range**: 0.2
- **Entropy Coefficient**: 0.001
- **KL Coefficient**: 0.001 (fixed)
- **Gamma (discount)**: 1.0
- **Lambda (trace decay)**: 1.0
- **Total Training Steps**: ~1000 (typical)

### Generation Parameters
- **Max Prompt Length**: 4096 tokens
- **Max Response Length**: 500 tokens
- **Max Observation Length**: 500 tokens (search results)
- **Max Turns**: 2 (number of search/answer cycles)
- **Temperature**: 1.0
- **Top-p**: 0.95
- **Retriever Top-k**: 3 documents

---

## 10. EXISTING RL MECHANISMS

### 10.1 What's Already in Place
1. ✓ Full PPO and GRPO implementations
2. ✓ Rule-based reward (exact match for QA)
3. ✓ Multi-turn environment with search/answer actions
4. ✓ Distributed training with FSDP + Ray
5. ✓ vLLM-based efficient inference
6. ✓ KL divergence penalty and control
7. ✓ GAE and GRPO advantage estimation
8. ✓ Value function training
9. ✓ Checkpointing and resumption
10. ✓ Wandb integration for logging

### 10.2 Design Patterns
- **State Masking**: Can mask information blocks during advantage computation
- **Token-Level Rewards**: Rewards only at sequence end
- **Multi-GPU**: FSDP + vLLM for scalable training
- **Modular Design**: Pluggable components (models, retrievers, reward functions)

---

## 11. PPO IMPLEMENTATION SUMMARY

### Current PPO Loop
```
1. Initialize: Actor, Critic, Reference Policy with same base model
2. For each epoch:
   a. Generate trajectories using current actor + vLLM
   b. Compute rewards using rule-based scoring
   c. Apply KL penalty: token_reward = score - beta * KL
   d. Estimate advantages: GAE or GRPO
   e. Compute critic values for returns
   f. Update actor with PPO loss (clipped objective)
   g. Update critic with value loss
   h. Update KL controller (adaptive)
   i. Log metrics to wandb
3. Save checkpoints, test on validation set
```

### Distributed Execution
- Ray orchestrates worker processes
- Each worker: FSDP across GPUs
- vLLM handles generation without gradient computation
- Separate reference policy for KL control

---

## 12. RECOMMENDATIONS FOR PPO ENHANCEMENTS

### Areas Ready for Enhancement
1. **Curriculum Learning**: Start with easier questions
2. **Reward Shaping**: Multi-component rewards (search quality, answer quality)
3. **Constrained Decoding**: Force search/answer tags
4. **Better KL Control**: Adaptive target, or different formulations
5. **Value Function Baseline**: Separate value network architecture
6. **Action Space Design**: Fine-grained control over search behavior
7. **Exploration**: Temperature scheduling or entropy bonuses
8. **Evaluation Metrics**: Beyond exact match (F1, BLEU, etc.)

---

