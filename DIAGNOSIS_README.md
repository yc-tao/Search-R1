# Medical Diagnosis Inference Script

This script performs retrieval-augmented generation to diagnose medical conditions from hospitalization notes.

## Overview

The script:
1. Loads concatenated hospital notes from a pickle file
2. Splits notes by newline into searchable pieces
3. Uses the Qwen2.5-7B-Instruct LLM with iterative retrieval
4. Generates a detailed diagnosis report with evidence

## Prerequisites

### 1. Environment Setup
```bash
conda activate searchr1
```

### 2. Start Retrieval Server
The retrieval server must be running before executing the script:
```bash
bash retrieval_launch.sh
```

The server should be accessible at `http://127.0.0.1:56321/retrieve`

### 3. Choose LLM Execution Mode

**Option A: vLLM Server (Recommended for Performance)**

Launch a vLLM server for faster inference:
```bash
bash vllm_launch.sh
```

The vLLM server will be available at `http://127.0.0.1:8000/v1`

Then set `USE_VLLM = True` in `infer_diagnosis.py` (default setting).

**Option B: Local Model Loading**

Set `USE_VLLM = False` in `infer_diagnosis.py` to load the model directly.
This requires more GPU memory and is slower but doesn't need a separate server.

### 4. Data Requirements
- Data file: `~/SRL/data/concatenated_notes_by_episode.pkl`
- Expected format: Dict with episode IDs as keys and concatenated notes as values, OR list of concatenated notes

## Usage

### Basic Execution
```bash
python infer_diagnosis.py
```

### What the Script Does

1. **Loads Data**: Reads the first episode from the pickle file
2. **Splits Documents**: Splits the concatenated note by `\n` into individual pieces
3. **Initializes LLM**: Loads Qwen/Qwen2.5-7B-Instruct model
4. **Runs Iterative Retrieval**:
   - LLM generates search queries in `<search>` tags
   - Retrieves top 3 most relevant pieces per query
   - LLM reasons with retrieved information in `<think>` tags
   - Process repeats until LLM provides final answer
5. **Generates Report**: Creates detailed output with:
   - All search queries made
   - Retrieved passages with scores
   - LLM reasoning steps
   - Final diagnosis list

### Output

The script produces:
1. **Console output**: Real-time inference progress
2. **Text file**: `diagnosis_report_episode_{id}.txt` with complete report

## Configuration

You can modify these settings in `infer_diagnosis.py`:

```python
DATA_PATH = os.path.expanduser("~/SRL/data/concatenated_notes_by_episode.pkl")
RETRIEVAL_SERVER_URL = "http://127.0.0.1:56321/retrieve"
VLLM_SERVER_URL = "http://127.0.0.1:8000/v1"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
TOPK_RETRIEVAL = 3  # Number of passages per search
USE_VLLM = True  # Set to False to use local model loading
```

### vLLM vs Local Model Loading

**vLLM Server Benefits:**
- ⚡ **Faster inference**: Optimized CUDA kernels and continuous batching
- 💾 **Better memory efficiency**: PagedAttention reduces memory usage
- 🔄 **Reusable**: Keep server running for multiple script executions
- 📊 **Scalable**: Can handle multiple concurrent requests

**When to use Local Model Loading:**
- Quick one-off tests without server setup
- Debugging model behavior
- Systems where vLLM installation is problematic

## Example Output Structure

```
================================================================================
MEDICAL DIAGNOSIS REPORT - EPISODE 123456
================================================================================

Question: Diagnose all conditions that happened during this patient's hospitalization stay.

Total document pieces available: 150
Number of searches performed: 4

--------------------------------------------------------------------------------
SEARCH QUERIES:
--------------------------------------------------------------------------------
1. symptoms and presenting complaints
2. diagnostic test results and findings
3. confirmed diagnoses and conditions
4. treatment plans and medications prescribed

--------------------------------------------------------------------------------
RETRIEVED PASSAGES (Top 3 per query):
--------------------------------------------------------------------------------

Passage 1 (ID: episode_123456_piece_42, Score: 5.23):
"Note Piece 42"
Patient presented with chest pain and shortness of breath...

[... more passages ...]

--------------------------------------------------------------------------------
LLM REASONING:
--------------------------------------------------------------------------------

Reasoning block 1:
Based on the presenting symptoms, I need to search for the initial complaints...

[... more reasoning ...]

================================================================================
FINAL DIAGNOSIS:
================================================================================
1. Acute Myocardial Infarction (AMI)
2. Congestive Heart Failure (CHF)
3. Type 2 Diabetes Mellitus
4. Hypertension

================================================================================
```

## Troubleshooting

### Error: Connection refused to retrieval server
```
Make sure the retrieval server is running:
bash retrieval_launch.sh
```

### Error: Connection refused to vLLM server
```
Make sure the vLLM server is running:
bash vllm_launch.sh

# Check if server is accessible:
curl http://127.0.0.1:8000/v1/models
```

Alternatively, set `USE_VLLM = False` to use local model loading.

### Error: File not found (pickle file)
```
Verify the data path:
ls ~/SRL/data/concatenated_notes_by_episode.pkl
```

### CUDA Out of Memory
The script uses bfloat16 precision. If you still encounter memory issues:
- Use vLLM server mode (more memory efficient)
- Close other GPU processes
- Consider reducing `max_new_tokens` in the generation config
- Reduce `--max-model-len` in vllm_launch.sh

## Extending the Script

### Process Multiple Episodes
Modify the `main()` function to loop through all episodes:

```python
for episode_id, concatenated_note in data.items():
    documents = split_and_format_documents(concatenated_note, episode_id)
    results = run_inference(question, documents, tokenizer, model, stopping_criteria)
    # ... generate report
```

### Custom Questions
Modify the question in `main()`:

```python
question = "What medications were prescribed during hospitalization?"
# or
question = "Did the patient experience any adverse events?"
```

### Adjust Retrieval Settings
Change `TOPK_RETRIEVAL` to retrieve more/fewer passages per search:

```python
TOPK_RETRIEVAL = 5  # Retrieve top 5 passages
```

## Technical Details

- **Model**: Qwen/Qwen2.5-7B-Instruct (7B parameter instruct-tuned model)
- **Retrieval**: BM25Okapi via dedicated retrieval server
- **Search Pattern**: Iterative retrieval with `<search>`, `<think>`, `<answer>` tags
- **Stopping Criteria**:
  - vLLM mode: Stop sequences `["</search>"]` + EOS token IDs
  - Local mode: Custom StoppingCriteria class
- **Device**: CUDA (GPU) with bfloat16 precision
- **LLM Execution**:
  - vLLM mode: OpenAI-compatible API with optimized inference
  - Local mode: HuggingFace Transformers with model.generate()

## Related Files

- `infer_diagnosis.py`: Medical diagnosis inference script (this script)
- `vllm_launch.sh`: Script to launch vLLM server
- `retrieval_launch.sh`: Script to launch the retrieval server
- `search_r1/search/retrieval_server.py`: Retrieval server implementation
- `infer.py`: Original inference script for Q&A tasks
- `DIAGNOSIS_README.md`: This documentation file

## Installation

Install the openai package if not already installed:
```bash
pip install openai
```

Or install all requirements:
```bash
pip install -r requirements.txt
```
