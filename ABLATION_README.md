# Ablation Experiment: Direct Diagnosis (No RAG)

This document describes the ablation experiment comparing direct LLM diagnosis without retrieval-augmented generation (RAG).

## Overview

The ablation experiment evaluates two simplified approaches to medical diagnosis:

1. **Ablation 1: Full Note** - Pass the entire concatenated medical note to the LLM for direct diagnosis
2. **Ablation 2: Last Section Only** - Pass only the last section of the note (typically discharge summary) for diagnosis

Both ablations skip the document splitting, retrieval, and iterative search process used in the main RAG-based approach.

## Key Differences from Main Approach

| Feature | Main Approach (infer_diagnosis.py) | Ablation Experiments |
|---------|-----------------------------------|---------------------|
| **Document Splitting** | ✓ Splits into 50+ word chunks | ✗ No splitting |
| **Retrieval Server** | ✓ Required | ✗ Not used |
| **Iterative Search** | ✓ Multiple search iterations | ✗ Single LLM call |
| **Input Processing** | RAG with top-k retrieval | Direct note/section input |
| **Server Port** | 60362 | 60363 |
| **Episodes Processed** | 1 (demo) | 775 (full dataset) |

## Files

- **`ablation_diagnosis.py`** - Main ablation experiment script
- **`run_ablation.sh`** - Launch script with environment checks
- **`ABLATION_README.md`** - This documentation file

### Output Files

- **`ablation_predictions.csv`** - Binary predictions for all episodes and both ablations
- **`ablation_metrics.csv`** - Evaluation metrics (AUROC, AUPRC, F1, accuracy, recall)
- **`ablation_checkpoint.pkl`** - Progress checkpoint (for resuming interrupted runs)

## Requirements

### 1. vLLM Server

Start the vLLM server on port **60363** (different from main approach):

```bash
# Option 1: Using vllm_launch.sh (modify to use port 60363)
bash vllm_launch.sh

# Option 2: Manual launch
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --port 60363 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9
```

### 2. Data File

Ensure the dataset is available at:
```
~/SRL/data/concatenated_notes_by_episode.pkl
```

This pickle file should contain a pandas DataFrame with:
- **TEXT column**: Concatenated medical notes
- **EPISODE_ID column**: Unique episode identifier
- **25 phenotype columns**: Binary ground truth labels (0/1)

### 3. Python Dependencies

```bash
pip install pandas scikit-learn numpy openai
```

## Usage

### Quick Start

```bash
# 1. Start vLLM server on port 60363
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --port 60363 \
  --tensor-parallel-size 1

# 2. Run ablation experiment
bash run_ablation.sh
```

### Manual Execution

```bash
python3 ablation_diagnosis.py
```

### Resuming from Checkpoint

The script automatically saves progress after each episode. If interrupted, simply re-run:

```bash
python3 ablation_diagnosis.py
```

It will detect `ablation_checkpoint.pkl` and resume from where it stopped.

To start fresh (ignoring checkpoint):

```bash
rm ablation_checkpoint.pkl
python3 ablation_diagnosis.py
```

## Experiment Details

### Ablation 1: Full Note

**Input**: Entire concatenated medical note (~72,000 characters on average)

**Prompt Template**:
```
Based on the following medical note, diagnose all conditions that
occurred during this patient's hospitalization. List all medical
conditions, diseases, complications, and health issues mentioned
or diagnosed.

Medical Note:
[FULL NOTE TEXT]

Diagnosis:
```

**Process**:
1. Pass full note to LLM → Get diagnosis text
2. Pass diagnosis to phenotype classifier → Get 25 binary labels (0/1)

### Ablation 2: Last Section Only

**Input**: Only the last section of the note (typically discharge summary)

**Extraction**: Splits note by `"=" * 80` separator, takes the last section

**Prompt Template**:
```
Based on the following medical note section, diagnose all conditions
that occurred during this patient's hospitalization. List all medical
conditions, diseases, complications, and health issues mentioned
or diagnosed.

Medical Note Section:
[LAST SECTION TEXT]

Diagnosis:
```

**Process**:
1. Extract last section using `extract_last_section()`
2. Pass last section to LLM → Get diagnosis text
3. Pass diagnosis to phenotype classifier → Get 25 binary labels (0/1)

### Phenotype Classification

Both ablations use the same 2-stage approach:

**Stage 1: Open-ended diagnosis**
- LLM generates natural language diagnosis

**Stage 2: Structured classification**
- LLM maps diagnosis to 25 predefined phenotypes
- Returns binary labels (0 or 1) for each phenotype

**Phenotype List** (25 clinical phenotypes):
1. Acute and unspecified renal failure
2. Acute cerebrovascular disease
3. Acute myocardial infarction
4. Cardiac dysrhythmias
5. Chronic kidney disease
6. Chronic obstructive pulmonary disease and bronchiectasis
7. Complications of surgical procedures or medical care
8. Conduction disorders
9. Congestive heart failure; nonhypertensive
10. Coronary atherosclerosis and other heart disease
11. Diabetes mellitus with complications
12. Diabetes mellitus without complication
13. Disorders of lipid metabolism
14. Essential hypertension
15. Fluid and electrolyte disorders
16. Gastrointestinal hemorrhage
17. Hypertension with complications and secondary hypertension
18. Other liver diseases
19. Other lower respiratory disease
20. Other upper respiratory disease
21. Pleurisy; pneumothorax; pulmonary collapse
22. Pneumonia (except that caused by tuberculosis or STD)
23. Respiratory failure; insufficiency; arrest (adult)
24. Septicemia (except in labor)
25. Shock

## Evaluation Metrics

For each ablation, the script computes:

### Per-Phenotype Metrics
- **AUROC** (Area Under ROC Curve)
- **AUPRC** (Area Under Precision-Recall Curve)
- **F1@0.5** (F1 score at threshold 0.5)
- **Optimal F1** (F1 at optimal threshold)
- **Accuracy** (Correct predictions / Total predictions)
- **Recall** (True Positives / (True Positives + False Negatives))
- **Support** (Number of positive samples in ground truth)

### Aggregated Metrics
- **Macro Average**: Mean across all 25 phenotypes
- **Micro Average**: Pooled across all predictions (flattened)

## Output Format

### ablation_predictions.csv

| Column | Type | Description |
|--------|------|-------------|
| EPISODE_ID | int | Episode identifier |
| ablation_type | str | "full_note" or "last_section" |
| timestamp | str | ISO timestamp of prediction |
| [phenotype]_prob | int | Binary prediction (0 or 1) for each of 25 phenotypes |

**Example**:
```csv
EPISODE_ID,ablation_type,timestamp,acute_and_unspecified_renal_failure_prob,...
200262,full_note,2024-01-15T10:30:45,1,...
200262,last_section,2024-01-15T10:32:18,1,...
200313,full_note,2024-01-15T10:35:22,0,...
```

**Total Rows**: 1550 (775 episodes × 2 ablations)

### ablation_metrics.csv

| Column | Type | Description |
|--------|------|-------------|
| ablation_type | str | "full_note" or "last_section" |
| phenotype | str | Phenotype name or "MACRO_AVERAGE"/"MICRO_AVERAGE" |
| auroc | float | Area Under ROC Curve |
| auprc | float | Area Under Precision-Recall Curve |
| f1_at_0.5 | float | F1 score at threshold 0.5 |
| optimal_f1 | float | Best F1 score (at optimal threshold) |
| optimal_threshold | float | Threshold that maximizes F1 |
| accuracy | float | Accuracy score |
| recall | float | Recall score |
| support | int | Number of positive ground truth samples |

**Example**:
```csv
ablation_type,phenotype,auroc,auprc,f1_at_0.5,optimal_f1,accuracy,recall,support
full_note,Acute and unspecified renal failure,0.7234,0.6891,0.5432,0.5432,0.8456,0.6123,142
full_note,Cardiac dysrhythmias,0.6891,0.6234,0.4987,0.4987,0.8234,0.5678,187
...
full_note,MACRO_AVERAGE,0.7012,0.6543,0.5123,0.5123,0.8345,0.5789,3685
full_note,MICRO_AVERAGE,0.7123,0.6678,0.5234,0.5234,0.8456,0.5891,3685
last_section,...
```

## Comparison with Main RAG Approach

To compare ablation results with the main RAG approach:

1. **Run ablation experiments** (this script):
   ```bash
   bash run_ablation.sh
   ```

2. **Run main RAG approach** (batch_infer_diagnosis.py):
   ```bash
   bash run_diagnosis.sh  # or similar
   ```

3. **Compare metrics**:
   - Ablation metrics: `ablation_metrics.csv`
   - RAG metrics: Output from batch_infer_diagnosis.py

Expected findings:
- **Full Note** may struggle with long context (avg. 72K characters)
- **Last Section** may miss earlier information but has focused input
- **RAG approach** should benefit from targeted retrieval and reasoning

## Runtime Estimates

- **Per episode**: ~30-60 seconds (2 ablations × 2 LLM calls each)
- **Total runtime**: 6-12 hours for 775 episodes
- **Checkpoint frequency**: After every episode (1550 checkpoints total)

## Troubleshooting

### vLLM Server Not Running
```
ERROR: vLLM server is not running on port 60363!
```
**Solution**: Start vLLM server:
```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --port 60363
```

### Data File Not Found
```
ERROR: Data file not found at ~/SRL/data/concatenated_notes_by_episode.pkl
```
**Solution**: Update `DATA_PATH` in `ablation_diagnosis.py` to correct location.

### Missing Dependencies
```
ERROR: Missing required Python packages!
```
**Solution**: Install dependencies:
```bash
pip install pandas scikit-learn numpy openai
```

### Checkpoint Issues

To clear checkpoint and restart:
```bash
rm ablation_checkpoint.pkl
```

To inspect checkpoint:
```python
import pickle
with open('ablation_checkpoint.pkl', 'rb') as f:
    results = pickle.load(f)
print(f"Processed: {len(results)} episode-ablation pairs")
```

## Configuration

Edit `ablation_diagnosis.py` to modify:

```python
# Server configuration
VLLM_SERVER_URL = "http://127.0.0.1:60363/v1"  # Change port if needed
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"          # Change model if needed

# Data path
DATA_PATH = os.path.expanduser("~/SRL/data/concatenated_notes_by_episode.pkl")

# Output files
OUTPUT_PREDICTIONS_CSV = "ablation_predictions.csv"
OUTPUT_METRICS_CSV = "ablation_metrics.csv"
CHECKPOINT_FILE = "ablation_checkpoint.pkl"
```

## Notes

- Binary predictions (0/1) are used instead of probabilities, which may limit AUROC/AUPRC evaluation quality
- The script uses temperature=0.0 for deterministic outputs
- Retry logic (3 attempts) handles transient API failures
- Ground truth labels come directly from the DataFrame phenotype columns
- The 25 phenotypes match Clinical Classification Software (CCS) categories

## Related Files

- **infer_diagnosis.py** - Single episode RAG-based diagnosis (demo)
- **batch_infer_diagnosis.py** - Batch RAG-based diagnosis with full evaluation
- **DIAGNOSIS_README.md** - Documentation for main RAG approach
- **retrieval_launch.sh** - Launch script for retrieval server (not used in ablation)
- **vllm_launch.sh** - Launch script for vLLM server

## Contact

For questions or issues with the ablation experiment, refer to the main project documentation or contact the development team.
