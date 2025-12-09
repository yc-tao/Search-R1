#!/usr/bin/env python3
"""
Batch inference script for medical diagnosis using vLLM.
Processes all episodes in the dataset, saves predictions to CSV, and computes evaluation metrics.
"""

import pickle
import requests
import re
import os
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from openai import OpenAI
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    recall_score,
    f1_score,
    precision_recall_curve
)

# Configuration
DATA_PATH = os.path.expanduser("~/SRL/data/concatenated_notes_by_episode.pkl")
RETRIEVAL_SERVER_URL = "http://127.0.0.1:56321/retrieve"
VLLM_SERVER_URL = "http://127.0.0.1:60362/v1"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
TOPK_RETRIEVAL = 3
CHECKPOINT_FREQUENCY = 10  # Save checkpoint every N episodes
MAX_SEARCHES = 50  # Maximum search iterations per episode

# Phenotype list for classification
PHENOTYPES = [
    'Acute and unspecified renal failure',
    'Acute cerebrovascular disease',
    'Acute myocardial infarction',
    'Cardiac dysrhythmias',
    'Chronic kidney disease',
    'Chronic obstructive pulmonary disease and bronchiectasis',
    'Complications of surgical procedures or medical care',
    'Conduction disorders',
    'Congestive heart failure; nonhypertensive',
    'Coronary atherosclerosis and other heart disease',
    'Diabetes mellitus with complications',
    'Diabetes mellitus without complication',
    'Disorders of lipid metabolism',
    'Essential hypertension',
    'Fluid and electrolyte disorders',
    'Gastrointestinal hemorrhage',
    'Hypertension with complications and secondary hypertension',
    'Other liver diseases',
    'Other lower respiratory disease',
    'Other upper respiratory disease',
    'Pleurisy; pneumothorax; pulmonary collapse',
    'Pneumonia (except that caused by tuberculosis or sexually transmitted disease)',
    'Respiratory failure; insufficiency; arrest (adult)',
    'Septicemia (except in labor)',
    'Shock'
]

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_inference.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# vLLM client setup
vllm_client = OpenAI(
    api_key="EMPTY",
    base_url=VLLM_SERVER_URL,
)


def phenotype_to_fieldname(phenotype: str) -> str:
    """Convert phenotype name to CSV field name."""
    # Lowercase, replace spaces with underscores, remove punctuation
    fieldname = phenotype.lower()
    fieldname = fieldname.replace(' ', '_')
    fieldname = re.sub(r'[^\w_]', '', fieldname)
    return f"{fieldname}_prob"


def load_data(data_path: str) -> pd.DataFrame:
    """Load the concatenated notes with labels from pickle file."""
    logger.info(f"Loading data from {data_path}...")

    # Compatibility shim for NumPy 2.x pickles with NumPy 1.x
    import sys
    import numpy as np
    sys.modules['numpy._core'] = np.core
    sys.modules['numpy._core.numeric'] = np.core.numeric

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    if not isinstance(data, pd.DataFrame):
        raise ValueError(f"Expected DataFrame, got {type(data)}")

    logger.info(f"Loaded {len(data)} episodes")
    logger.info(f"Columns: {list(data.columns)}")

    # Verify required columns exist
    required_cols = ['EPISODE_ID', 'SUBJECT_ID', 'HADM_ID']
    missing = [col for col in required_cols if col not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Verify phenotype columns exist
    missing_phenotypes = [p for p in PHENOTYPES if p not in data.columns]
    if missing_phenotypes:
        logger.warning(f"Missing phenotype columns: {missing_phenotypes}")

    return data


def extract_last_section(concatenated_note: str) -> str:
    """Extract the last section of the note split by '=' separator lines."""
    separator = "=" * 80
    sections = concatenated_note.split(separator)
    sections = [s.strip() for s in sections if s.strip()]
    return sections[-1] if sections else ""


def split_and_format_documents(concatenated_note: str, episode_id: str) -> List[Dict]:
    """
    Split concatenated note into chunks with the following rules:
    1. Each chunk must be at least 50 words in length
    2. Splits can only occur at punctuation marks
    3. Each chunk is extended to include one sentence before and after for context
    """
    # Split text into sentences at punctuation marks
    sentence_pattern = re.compile(r'([.!?]+[\s\n]+)')
    parts = sentence_pattern.split(concatenated_note)

    # Reconstruct sentences (combining text with their punctuation)
    sentences = []
    for i in range(0, len(parts) - 1, 2):
        if i + 1 < len(parts):
            sentence = parts[i] + parts[i + 1]
            sentences.append(sentence.strip())
    # Add last part if it doesn't end with punctuation
    if len(parts) % 2 == 1 and parts[-1].strip():
        sentences.append(parts[-1].strip())

    # Filter out empty sentences
    sentences = [s for s in sentences if s]

    if not sentences:
        logger.warning(f"No sentences found in episode {episode_id}")
        return []

    # Group sentences into chunks of at least 50 words
    MIN_WORDS = 50
    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        word_count = len(sentence.split())
        current_chunk.append(sentence)
        current_word_count += word_count

        if current_word_count >= MIN_WORDS:
            chunks.append(current_chunk)
            current_chunk = []
            current_word_count = 0

    # Add remaining sentences as a chunk if any
    if current_chunk:
        chunks.append(current_chunk)

    # Format as documents with context extension
    documents = []
    for idx, chunk in enumerate(chunks):
        chunk_start_idx = None
        for i in range(len(sentences)):
            if sentences[i] == chunk[0]:
                chunk_start_idx = i
                break

        if chunk_start_idx is None:
            continue

        chunk_end_idx = chunk_start_idx + len(chunk) - 1
        extended_start = max(0, chunk_start_idx - 1)
        extended_end = min(len(sentences) - 1, chunk_end_idx + 1)

        extended_sentences = sentences[extended_start:extended_end + 1]
        extended_text = ' '.join(extended_sentences)

        doc = {
            "id": f"episode_{episode_id}_piece_{idx}",
            "contents": f'"Note Piece {idx}"\n{extended_text}'
        }
        documents.append(doc)

    return documents


def get_query(text: str) -> str:
    """Extract query from <search> tags."""
    pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
    matches = pattern.findall(text)
    return matches[-1].strip() if matches else None


def extract_reasoning(text: str) -> List[str]:
    """Extract all reasoning blocks from <think> tags."""
    pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    return pattern.findall(text)


def extract_answer(text: str) -> str:
    """Extract answer from <answer> tags."""
    pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
    matches = pattern.findall(text)
    return matches[-1].strip() if matches else None


def summarize_last_section_vllm(last_section: str) -> str:
    """Summarize the last section of the note, focusing on diagnoses."""
    if not last_section.strip():
        return "No last section found to summarize."

    prompt = f"""Please provide a brief summary of the following medical note. Be concise but comprehensive.

Medical Note Section:
{last_section}

Summary of diagnoses and conditions:"""

    try:
        response = vllm_client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error summarizing last section: {e}")
        return "Error generating summary."


def search(query: str, documents: List[Dict]) -> Tuple[str, List[Dict]]:
    """Search for relevant documents given a query."""
    payload = {
        "queries": [query],
        "documents": documents,
        "topk": TOPK_RETRIEVAL,
        "return_scores": True
    }

    try:
        response = requests.post(RETRIEVAL_SERVER_URL, json=payload)
        response_json = response.json()
        results = response_json['result']
    except Exception as e:
        logger.error(f"Error calling retrieval server: {e}")
        raise

    def _passages2string(retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            score = doc_item.get('score', 'N/A')
            format_reference += f"Doc {idx+1}(Title: {title}, Score: {score:.2f}) {text}\n"
        return format_reference

    return _passages2string(results[0]), results[0]


def run_inference_vllm(question: str, documents: List[Dict], initial_summary: str = "",
                      max_searches: int = MAX_SEARCHES) -> Dict:
    """Run iterative inference with retrieval using vLLM server."""
    # Prepare the initial user message
    initial_prompt = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can provide the final answer inside <answer> and </answer>. Question: {question} \
"""

    # Add initial summary if available
    if initial_summary:
        initial_prompt += f"""

Initial Summary from Last Section of Note:
{initial_summary}
"""

    initial_prompt += "\n"

    # Initialize messages list
    messages = [{"role": "user", "content": initial_prompt}]

    # Track all searches and results
    search_history = []
    retrieved_passages = []
    full_output = ""
    iteration = 0

    # Stop sequences for vLLM
    stop_sequences = ["</search>"]

    while iteration < max_searches:
        try:
            # Call vLLM server with chat completion API
            response = vllm_client.chat.completions.create(
                model=MODEL_ID,
                messages=messages,
                max_tokens=1024,
                temperature=0.0,
                stop=stop_sequences,
                extra_body={
                    "stop_token_ids": [151645, 151643]  # Qwen EOS tokens
                }
            )

            output_text = response.choices[0].message.content

            # Add </search> back if it was removed by stop sequence
            if "<search>" in output_text and "</search>" not in output_text:
                output_text += "</search>"

            # Check if we hit EOS (no </search> found)
            if response.choices[0].finish_reason == "stop" and "</search>" not in output_text:
                full_output += output_text
                messages.append({"role": "assistant", "content": output_text})
                break

            full_output += output_text
            messages.append({"role": "assistant", "content": output_text})

            # Extract query from the assistant's output
            tmp_query = get_query(output_text)

            if tmp_query:
                search_results, raw_results = search(tmp_query, documents=documents)
                search_history.append(tmp_query)
                retrieved_passages.extend(raw_results)
            else:
                search_results = ''

            # Format search results as user message
            search_info = f"<information>{search_results}</information>\n"
            messages.append({"role": "user", "content": search_info})
            messages.append({"role": "user", "content": "If no information is found useful, approach the question in a different way, and search with a different query."})

            full_output += search_info
            iteration += 1

        except Exception as e:
            logger.error(f"Error calling vLLM server: {e}")
            raise

    # Extract reasoning and answer
    reasoning_blocks = extract_reasoning(full_output)
    final_answer = extract_answer(full_output)

    if final_answer is None:
        logger.warning("No final answer found in <answer> tags, retrying once...")
        # Retry once
        return run_inference_vllm(question, documents, initial_summary=initial_summary, max_searches=max_searches)

    return {
        'full_output': full_output,
        'search_history': search_history,
        'retrieved_passages': retrieved_passages,
        'reasoning_blocks': reasoning_blocks,
        'final_answer': final_answer,
        'num_searches': len(search_history),
        'initial_summary': initial_summary
    }


def classify_phenotypes_vllm(diagnosis_answer: str) -> List[str]:
    """Classify which phenotypes the patient has based on the diagnosis answer."""
    # Create numbered list of phenotypes
    phenotype_list = "\n".join([f"{i+1}. {p}" for i, p in enumerate(PHENOTYPES)])

    prompt = f"""Based on the following diagnosis, identify which phenotypes from the list apply to this patient.
The patient may have multiple phenotypes. Return ONLY the numbers of the applicable phenotypes, separated by commas.

Diagnosis:
{diagnosis_answer}

Phenotype List:
{phenotype_list}

Instructions:
- Review the diagnosis carefully
- Identify ALL phenotypes that match conditions mentioned in the diagnosis
- Return only the numbers (e.g., "1, 5, 12" or "3" if only one applies)
- If no phenotypes clearly match, return "None"

Answer (numbers only):"""

    try:
        response = vllm_client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.0,
        )

        answer = response.choices[0].message.content.strip()

        # Parse the response to get phenotype indices
        if answer.lower() == "none":
            return []

        # Extract numbers from the response
        numbers = re.findall(r'\d+', answer)
        phenotypes = []
        for num_str in numbers:
            num = int(num_str)
            if 1 <= num <= len(PHENOTYPES):
                phenotypes.append(PHENOTYPES[num - 1])

        return phenotypes

    except Exception as e:
        logger.error(f"Error in phenotype classification: {e}")
        return []


def process_episode(row: pd.Series, question: str) -> Dict:
    """Process a single episode and return prediction results."""
    episode_id = row['EPISODE_ID']
    subject_id = row['SUBJECT_ID']
    hadm_id = row['HADM_ID']

    # Find the text column
    text_col = None
    for col in ['TEXT', 'text', 'CONCATENATED_NOTE', 'concatenated_note']:
        if col in row.index:
            text_col = col
            break

    if text_col is None:
        # Try to find any column with 'text' or 'note' in the name
        text_cols = [col for col in row.index if 'text' in col.lower() or 'note' in col.lower()]
        if text_cols:
            text_col = text_cols[0]
        else:
            raise ValueError(f"Could not find text column for episode {episode_id}")

    concatenated_note = row[text_col]

    try:
        # Split and format documents
        documents = split_and_format_documents(concatenated_note, episode_id)

        if not documents:
            logger.warning(f"No documents created for episode {episode_id}")
            return None

        # Extract and summarize last section
        last_section = extract_last_section(concatenated_note)
        initial_summary = ""
        if last_section:
            initial_summary = summarize_last_section_vllm(last_section)

        # Run inference
        results = run_inference_vllm(question, documents, initial_summary=initial_summary)

        # Classify phenotypes
        if results['final_answer']:
            classified_phenotypes = classify_phenotypes_vllm(results['final_answer'])
        else:
            classified_phenotypes = []

        # Convert to binary probabilities
        probabilities = {}
        for phenotype in PHENOTYPES:
            fieldname = phenotype_to_fieldname(phenotype)
            probabilities[fieldname] = 1.0 if phenotype in classified_phenotypes else 0.0

        # Add identifiers
        prediction = {
            'SUBJECT_ID': subject_id,
            'HADM_ID': hadm_id,
            'EPISODE_ID': episode_id,
            **probabilities
        }

        return prediction

    except Exception as e:
        logger.error(f"Error processing episode {episode_id}: {e}")
        return None


def save_checkpoint(predictions: List[Dict], checkpoint_path: str):
    """Save intermediate predictions to checkpoint file."""
    df = pd.DataFrame(predictions)
    df.to_csv(checkpoint_path, index=False)
    logger.info(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint(checkpoint_path: str) -> List[Dict]:
    """Load predictions from checkpoint file."""
    if os.path.exists(checkpoint_path):
        df = pd.read_csv(checkpoint_path)
        logger.info(f"Loaded {len(df)} predictions from checkpoint: {checkpoint_path}")
        return df.to_dict('records')
    return []


def compute_metrics(predictions_df: pd.DataFrame, ground_truth_df: pd.DataFrame) -> pd.DataFrame:
    """Compute evaluation metrics comparing predictions to ground truth."""
    metrics_list = []

    # Merge predictions with ground truth on identifiers
    merged = predictions_df.merge(
        ground_truth_df,
        on=['SUBJECT_ID', 'HADM_ID', 'EPISODE_ID'],
        how='inner',
        suffixes=('_pred', '_gt')
    )

    if len(merged) == 0:
        logger.error("No matching episodes found between predictions and ground truth!")
        return pd.DataFrame()

    logger.info(f"Computing metrics for {len(merged)} episodes")

    # Compute metrics for each phenotype
    for phenotype in PHENOTYPES:
        fieldname = phenotype_to_fieldname(phenotype)

        # Check if columns exist
        if phenotype not in merged.columns:
            logger.warning(f"Ground truth column missing for phenotype: {phenotype}")
            continue
        if fieldname not in merged.columns:
            logger.warning(f"Prediction column missing for phenotype: {phenotype}")
            continue

        y_true = merged[phenotype].values
        y_pred_prob = merged[fieldname].values
        y_pred_binary = (y_pred_prob >= 0.5).astype(int)

        # Skip if no positive samples
        if y_true.sum() == 0:
            logger.warning(f"No positive samples for phenotype: {phenotype}")
            continue

        # Compute metrics
        try:
            auroc = roc_auc_score(y_true, y_pred_prob)
        except:
            auroc = np.nan

        try:
            auprc = average_precision_score(y_true, y_pred_prob)
        except:
            auprc = np.nan

        accuracy = accuracy_score(y_true, y_pred_binary)
        recall = recall_score(y_true, y_pred_binary, zero_division=0)
        f1_at_05 = f1_score(y_true, y_pred_binary, zero_division=0)

        # Compute optimal F1
        try:
            precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_prob)
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
            optimal_f1 = np.max(f1_scores)
            optimal_threshold = thresholds[np.argmax(f1_scores)] if len(thresholds) > 0 else 0.5
        except:
            optimal_f1 = np.nan
            optimal_threshold = np.nan

        metrics_list.append({
            'phenotype': phenotype,
            'AUROC': auroc,
            'AUPRC': auprc,
            'Accuracy': accuracy,
            'Recall': recall,
            'F1@0.5': f1_at_05,
            'Optimal_F1': optimal_f1,
            'Optimal_Threshold': optimal_threshold,
            'N_Positive': int(y_true.sum()),
            'N_Total': len(y_true)
        })

    metrics_df = pd.DataFrame(metrics_list)

    # Compute macro averages
    macro_avg = {
        'phenotype': 'MACRO_AVERAGE',
        'AUROC': metrics_df['AUROC'].mean(),
        'AUPRC': metrics_df['AUPRC'].mean(),
        'Accuracy': metrics_df['Accuracy'].mean(),
        'Recall': metrics_df['Recall'].mean(),
        'F1@0.5': metrics_df['F1@0.5'].mean(),
        'Optimal_F1': metrics_df['Optimal_F1'].mean(),
        'Optimal_Threshold': np.nan,
        'N_Positive': np.nan,
        'N_Total': np.nan
    }

    # Compute micro averages (pooled across all phenotypes)
    all_y_true = []
    all_y_pred_prob = []
    all_y_pred_binary = []

    for phenotype in PHENOTYPES:
        fieldname = phenotype_to_fieldname(phenotype)
        if phenotype in merged.columns and fieldname in merged.columns:
            all_y_true.extend(merged[phenotype].values)
            all_y_pred_prob.extend(merged[fieldname].values)
            all_y_pred_binary.extend((merged[fieldname].values >= 0.5).astype(int))

    if len(all_y_true) > 0:
        all_y_true = np.array(all_y_true)
        all_y_pred_prob = np.array(all_y_pred_prob)
        all_y_pred_binary = np.array(all_y_pred_binary)

        try:
            micro_auroc = roc_auc_score(all_y_true, all_y_pred_prob)
        except:
            micro_auroc = np.nan

        try:
            micro_auprc = average_precision_score(all_y_true, all_y_pred_prob)
        except:
            micro_auprc = np.nan

        micro_avg = {
            'phenotype': 'MICRO_AVERAGE',
            'AUROC': micro_auroc,
            'AUPRC': micro_auprc,
            'Accuracy': accuracy_score(all_y_true, all_y_pred_binary),
            'Recall': recall_score(all_y_true, all_y_pred_binary, zero_division=0),
            'F1@0.5': f1_score(all_y_true, all_y_pred_binary, zero_division=0),
            'Optimal_F1': np.nan,
            'Optimal_Threshold': np.nan,
            'N_Positive': np.nan,
            'N_Total': np.nan
        }
    else:
        micro_avg = macro_avg.copy()
        micro_avg['phenotype'] = 'MICRO_AVERAGE'

    # Append averages
    metrics_df = pd.concat([metrics_df, pd.DataFrame([macro_avg]), pd.DataFrame([micro_avg])], ignore_index=True)

    return metrics_df


def main():
    parser = argparse.ArgumentParser(description='Batch inference for medical diagnosis')
    parser.add_argument('--data-path', type=str, default=DATA_PATH, help='Path to data pickle file')
    parser.add_argument('--output-dir', type=str, default='.', help='Output directory for results')
    parser.add_argument('--checkpoint-freq', type=int, default=CHECKPOINT_FREQUENCY,
                       help='Save checkpoint every N episodes')
    parser.add_argument('--max-searches', type=int, default=MAX_SEARCHES,
                       help='Maximum search iterations per episode')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint if exists')
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    checkpoint_path = output_dir / 'checkpoint_predictions.csv'
    final_predictions_path = output_dir / 'predictions.csv'
    metrics_path = output_dir / 'metrics.csv'

    # Load data
    data = load_data(args.data_path)

    # Load checkpoint if resuming
    predictions = []
    start_idx = 0
    if args.resume:
        predictions = load_checkpoint(checkpoint_path)
        start_idx = len(predictions)
        if start_idx > 0:
            logger.info(f"Resuming from episode {start_idx}")

    # Define the diagnosis question
    question = "Diagnose all conditions that happened during this patient's hospitalization stay."

    # Process episodes
    logger.info(f"Processing {len(data)} episodes (starting from {start_idx})")

    for idx in tqdm(range(start_idx, len(data)), desc="Processing episodes"):
        row = data.iloc[idx]
        episode_id = row['EPISODE_ID']

        logger.info(f"\n{'='*80}")
        logger.info(f"Processing episode {idx+1}/{len(data)}: {episode_id}")
        logger.info(f"{'='*80}")

        prediction = process_episode(row, question)

        if prediction is not None:
            predictions.append(prediction)
            logger.info(f"Successfully processed episode {episode_id}")
        else:
            logger.error(f"Failed to process episode {episode_id}")

        # Save checkpoint
        if (idx + 1) % args.checkpoint_freq == 0:
            save_checkpoint(predictions, checkpoint_path)

    # Save final predictions
    logger.info(f"\nSaving final predictions to {final_predictions_path}")
    predictions_df = pd.DataFrame(predictions)
    predictions_df.to_csv(final_predictions_path, index=False)
    logger.info(f"Saved {len(predictions_df)} predictions")

    # Compute metrics
    logger.info("\nComputing evaluation metrics...")
    metrics_df = compute_metrics(predictions_df, data)
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Metrics saved to {metrics_path}")

    # Print summary
    print("\n" + "="*80)
    print("EVALUATION METRICS SUMMARY")
    print("="*80)
    print(metrics_df.to_string())
    print("="*80)

    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        logger.info("Checkpoint file removed")

    logger.info("\nBatch inference completed successfully!")


if __name__ == "__main__":
    main()
