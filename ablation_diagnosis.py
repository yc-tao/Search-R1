import pickle
import re
import os
import pandas as pd
import numpy as np
from pathlib import Path
from openai import OpenAI
from datetime import datetime
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    accuracy_score,
    recall_score,
    precision_recall_curve
)
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_PATH = os.path.expanduser("~/SRL/data/concatenated_notes_by_episode.pkl")
VLLM_SERVER_URL = "http://127.0.0.1:60362/v1"  # New port for ablation experiments
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_PREDICTIONS_CSV = "ablation_predictions.csv"
OUTPUT_METRICS_CSV = "ablation_metrics.csv"
CHECKPOINT_FILE = "ablation_checkpoint.pkl"

# Phenotype list for classification (same as original)
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

# vLLM client setup
vllm_client = OpenAI(
    api_key="EMPTY",
    base_url=VLLM_SERVER_URL,
)

def load_data(data_path):
    """Load the concatenated notes DataFrame with ground truth labels."""
    print(f"Loading data from {data_path}...")

    # Compatibility shim for NumPy 2.x pickles with NumPy 1.x
    import sys
    sys.modules['numpy._core'] = np.core
    sys.modules['numpy._core.numeric'] = np.core.numeric

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    print(f"Loaded {len(data)} episodes")
    print(f"Columns: {list(data.columns)}")
    return data

def extract_last_section(concatenated_note):
    """
    Extract the last section of the note split by '=' separator lines.

    Args:
        concatenated_note: The full concatenated note text

    Returns:
        The last section of the note, or empty string if no separator found
    """
    separator = "=" * 80
    sections = concatenated_note.split(separator)

    # Filter out empty sections
    sections = [s.strip() for s in sections if s.strip()]

    if sections:
        return sections[-1]
    else:
        return ""

def phenotype_to_fieldname(phenotype):
    """Convert phenotype name to CSV column name."""
    return phenotype.lower().replace(' ', '_').replace(';', '').replace(',', '').replace('(', '').replace(')', '').replace('/', '_') + '_prob'

def extract_diagnosis(text):
    """Extract diagnosis from <answer> tags, or return full text if no tags."""
    pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return matches[-1].strip()
    else:
        # No answer tags, return the full text
        return text.strip()

def diagnose_with_full_note(note, episode_id, max_retries=3):
    """
    Ablation 1: Diagnose using the full concatenated note.

    Args:
        note: Full concatenated note text
        episode_id: Episode identifier
        max_retries: Number of retries on API failure

    Returns:
        Diagnosis text from LLM
    """
    prompt = f"""Based on the following medical note, diagnose all conditions that occurred during this patient's hospitalization. List all medical conditions, diseases, complications, and health issues mentioned or diagnosed.

Medical Note:
{note}

Diagnosis:"""

    for attempt in range(max_retries):
        try:
            response = vllm_client.chat.completions.create(
                model=MODEL_ID,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0.0,
            )

            diagnosis = response.choices[0].message.content.strip()
            return diagnosis

        except Exception as e:
            print(f"Error diagnosing episode {episode_id} (attempt {attempt+1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                raise

    return None

def diagnose_with_last_section(note, episode_id, max_retries=3):
    """
    Ablation 2: Diagnose using only the last section of the note.

    Args:
        note: Full concatenated note text
        episode_id: Episode identifier
        max_retries: Number of retries on API failure

    Returns:
        Diagnosis text from LLM
    """
    # Extract last section
    last_section = extract_last_section(note)

    if not last_section:
        print(f"Warning: No last section found for episode {episode_id}, using full note")
        last_section = note

    prompt = f"""Based on the following medical note section, diagnose all conditions that occurred during this patient's hospitalization. List all medical conditions, diseases, complications, and health issues mentioned or diagnosed.

Medical Note Section:
{last_section}

Diagnosis:"""

    for attempt in range(max_retries):
        try:
            response = vllm_client.chat.completions.create(
                model=MODEL_ID,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0.0,
            )

            diagnosis = response.choices[0].message.content.strip()
            return diagnosis

        except Exception as e:
            print(f"Error diagnosing episode {episode_id} (attempt {attempt+1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                raise

    return None

def classify_phenotypes_vllm(diagnosis_answer, max_retries=3):
    """
    Classify which phenotypes the patient has based on the diagnosis answer.
    Returns binary labels (0 or 1) for each phenotype.

    Args:
        diagnosis_answer: The diagnosis text from LLM
        max_retries: Number of retries on API failure

    Returns:
        List of binary labels (0 or 1) for each phenotype, in same order as PHENOTYPES
    """
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

    for attempt in range(max_retries):
        try:
            response = vllm_client.chat.completions.create(
                model=MODEL_ID,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.0,
            )

            answer = response.choices[0].message.content.strip()

            # Parse the response to get phenotype indices
            binary_labels = [0] * len(PHENOTYPES)

            if answer.lower() == "none":
                return binary_labels

            # Extract numbers from the response
            numbers = re.findall(r'\d+', answer)
            for num_str in numbers:
                num = int(num_str)
                if 1 <= num <= len(PHENOTYPES):
                    binary_labels[num - 1] = 1

            return binary_labels

        except Exception as e:
            print(f"Error in phenotype classification (attempt {attempt+1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                # Return all zeros on failure
                return [0] * len(PHENOTYPES)

    return [0] * len(PHENOTYPES)

def process_episode(row, ablation_type):
    """
    Process a single episode with the specified ablation type.

    Args:
        row: DataFrame row containing episode data
        ablation_type: 'full_note' or 'last_section'

    Returns:
        Dictionary with episode_id, ablation_type, predictions, ground_truth
    """
    episode_id = row['EPISODE_ID']
    note = row['TEXT']

    # Get ground truth labels
    ground_truth = [int(row[phenotype]) for phenotype in PHENOTYPES]

    print(f"\n{'='*80}")
    print(f"Processing Episode {episode_id} - Ablation: {ablation_type}")
    print(f"{'='*80}")

    try:
        # Run diagnosis based on ablation type
        if ablation_type == 'full_note':
            print(f"Note length: {len(note)} characters")
            diagnosis = diagnose_with_full_note(note, episode_id)
        elif ablation_type == 'last_section':
            last_section = extract_last_section(note)
            print(f"Last section length: {len(last_section)} characters")
            diagnosis = diagnose_with_last_section(note, episode_id)
        else:
            raise ValueError(f"Unknown ablation type: {ablation_type}")

        if diagnosis is None:
            print(f"Failed to get diagnosis for episode {episode_id}")
            return None

        print(f"\nDiagnosis preview: {diagnosis[:200]}...")

        # Classify phenotypes
        predictions = classify_phenotypes_vllm(diagnosis)

        print(f"Predicted {sum(predictions)} phenotypes")
        print(f"Ground truth: {sum(ground_truth)} phenotypes")

        result = {
            'EPISODE_ID': episode_id,
            'ablation_type': ablation_type,
            'predictions': predictions,
            'ground_truth': ground_truth,
            'diagnosis': diagnosis,
            'timestamp': datetime.now().isoformat()
        }

        return result

    except Exception as e:
        print(f"Error processing episode {episode_id}: {e}")
        import traceback
        traceback.print_exc()
        return None

def compute_metrics(predictions_df, ground_truth_df, ablation_type):
    """
    Compute evaluation metrics for predictions.

    Args:
        predictions_df: DataFrame with prediction columns
        ground_truth_df: DataFrame with ground truth labels
        ablation_type: String identifier for this ablation

    Returns:
        DataFrame with metrics per phenotype
    """
    metrics_list = []

    for phenotype in PHENOTYPES:
        field_name = phenotype_to_fieldname(phenotype)

        # Get predictions and ground truth
        y_true = ground_truth_df[phenotype].values
        y_pred = predictions_df[field_name].values

        # Skip if no positive samples in ground truth
        if y_true.sum() == 0:
            print(f"Warning: No positive samples for {phenotype}, skipping metrics")
            continue

        # Compute metrics
        try:
            auroc = roc_auc_score(y_true, y_pred)
        except:
            auroc = np.nan

        try:
            auprc = average_precision_score(y_true, y_pred)
        except:
            auprc = np.nan

        try:
            # For binary predictions, F1 at 0.5 threshold
            f1 = f1_score(y_true, y_pred)
            f1_at_05 = f1

            # Optimal F1 (same as F1 for binary)
            optimal_f1 = f1
            optimal_threshold = 0.5
        except:
            f1_at_05 = np.nan
            optimal_f1 = np.nan
            optimal_threshold = np.nan

        try:
            accuracy = accuracy_score(y_true, y_pred)
        except:
            accuracy = np.nan

        try:
            recall = recall_score(y_true, y_pred)
        except:
            recall = np.nan

        metrics_list.append({
            'ablation_type': ablation_type,
            'phenotype': phenotype,
            'auroc': auroc,
            'auprc': auprc,
            'f1_at_0.5': f1_at_05,
            'optimal_f1': optimal_f1,
            'optimal_threshold': optimal_threshold,
            'accuracy': accuracy,
            'recall': recall,
            'support': int(y_true.sum())
        })

    metrics_df = pd.DataFrame(metrics_list)

    # Compute macro averages
    macro_avg = {
        'ablation_type': ablation_type,
        'phenotype': 'MACRO_AVERAGE',
        'auroc': metrics_df['auroc'].mean(),
        'auprc': metrics_df['auprc'].mean(),
        'f1_at_0.5': metrics_df['f1_at_0.5'].mean(),
        'optimal_f1': metrics_df['optimal_f1'].mean(),
        'optimal_threshold': metrics_df['optimal_threshold'].mean(),
        'accuracy': metrics_df['accuracy'].mean(),
        'recall': metrics_df['recall'].mean(),
        'support': int(metrics_df['support'].sum())
    }

    # Compute micro averages (pooled)
    all_y_true = []
    all_y_pred = []
    for phenotype in PHENOTYPES:
        field_name = phenotype_to_fieldname(phenotype)
        all_y_true.extend(ground_truth_df[phenotype].values)
        all_y_pred.extend(predictions_df[field_name].values)

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    micro_avg = {
        'ablation_type': ablation_type,
        'phenotype': 'MICRO_AVERAGE',
        'auroc': roc_auc_score(all_y_true, all_y_pred) if all_y_true.sum() > 0 else np.nan,
        'auprc': average_precision_score(all_y_true, all_y_pred) if all_y_true.sum() > 0 else np.nan,
        'f1_at_0.5': f1_score(all_y_true, all_y_pred),
        'optimal_f1': f1_score(all_y_true, all_y_pred),
        'optimal_threshold': 0.5,
        'accuracy': accuracy_score(all_y_true, all_y_pred),
        'recall': recall_score(all_y_true, all_y_pred),
        'support': int(all_y_true.sum())
    }

    # Append averages
    metrics_df = pd.concat([metrics_df, pd.DataFrame([macro_avg, micro_avg])], ignore_index=True)

    return metrics_df

def save_checkpoint(results, checkpoint_file=CHECKPOINT_FILE):
    """Save checkpoint with current results."""
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"Checkpoint saved to {checkpoint_file}")

def load_checkpoint(checkpoint_file=CHECKPOINT_FILE):
    """Load checkpoint if exists."""
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'rb') as f:
            results = pickle.load(f)
        print(f"Loaded checkpoint with {len(results)} results")
        return results
    return []

def main():
    print("="*80)
    print("ABLATION EXPERIMENT: Direct Diagnosis (No RAG)")
    print("="*80)
    print(f"vLLM Server: {VLLM_SERVER_URL}")
    print(f"Model: {MODEL_ID}")
    print(f"Output: {OUTPUT_PREDICTIONS_CSV}, {OUTPUT_METRICS_CSV}")
    print("="*80)

    # Load data
    data = load_data(DATA_PATH)

    print(f"\nDataset shape: {data.shape}")
    print(f"Total episodes: {len(data)}")

    # Load checkpoint if exists
    results = load_checkpoint()
    processed_keys = set()
    if results:
        processed_keys = {(r['EPISODE_ID'], r['ablation_type']) for r in results}
        print(f"Resuming from checkpoint: {len(processed_keys)} episode-ablation pairs already processed")

    # Process all episodes with both ablations
    ablation_types = ['last_section']  # Skip full_note ablation

    for idx, row in data.iterrows():
        episode_id = row['EPISODE_ID']

        for ablation_type in ablation_types:
            # Skip if already processed
            if (episode_id, ablation_type) in processed_keys:
                print(f"Skipping Episode {episode_id} - {ablation_type} (already processed)")
                continue

            # Process episode
            result = process_episode(row, ablation_type)

            if result is not None:
                results.append(result)
                processed_keys.add((episode_id, ablation_type))

                # Save checkpoint after each episode
                save_checkpoint(results)
            else:
                print(f"Skipped episode {episode_id} - {ablation_type} due to errors")

        print(f"\nProgress: {len(processed_keys)}/{len(data) * 2} episode-ablation pairs processed")

    # Convert results to DataFrame
    print("\n" + "="*80)
    print("PREPARING OUTPUT")
    print("="*80)

    predictions_rows = []
    for result in results:
        row_data = {
            'EPISODE_ID': result['EPISODE_ID'],
            'ablation_type': result['ablation_type'],
            'timestamp': result['timestamp']
        }

        # Add prediction columns
        for i, phenotype in enumerate(PHENOTYPES):
            field_name = phenotype_to_fieldname(phenotype)
            row_data[field_name] = result['predictions'][i]

        predictions_rows.append(row_data)

    predictions_df = pd.DataFrame(predictions_rows)

    # Create ground truth DataFrame
    ground_truth_data = data[['EPISODE_ID'] + PHENOTYPES].copy()

    # Compute metrics for each ablation
    all_metrics = []

    for ablation_type in ablation_types:
        print(f"\nComputing metrics for ablation: {ablation_type}")

        # Filter predictions for this ablation
        ablation_preds = predictions_df[predictions_df['ablation_type'] == ablation_type].copy()

        # Merge with ground truth
        ablation_preds = ablation_preds.merge(ground_truth_data, on='EPISODE_ID', how='inner')

        # Compute metrics
        metrics = compute_metrics(ablation_preds, ablation_preds, ablation_type)
        all_metrics.append(metrics)

        print(f"\n{ablation_type} - Macro Average Results:")
        macro_row = metrics[metrics['phenotype'] == 'MACRO_AVERAGE'].iloc[0]
        print(f"  AUROC: {macro_row['auroc']:.4f}")
        print(f"  AUPRC: {macro_row['auprc']:.4f}")
        print(f"  F1: {macro_row['f1_at_0.5']:.4f}")
        print(f"  Accuracy: {macro_row['accuracy']:.4f}")
        print(f"  Recall: {macro_row['recall']:.4f}")

    metrics_df = pd.concat(all_metrics, ignore_index=True)

    # Save outputs
    predictions_df.to_csv(OUTPUT_PREDICTIONS_CSV, index=False)
    print(f"\nPredictions saved to: {OUTPUT_PREDICTIONS_CSV}")

    metrics_df.to_csv(OUTPUT_METRICS_CSV, index=False)
    print(f"Metrics saved to: {OUTPUT_METRICS_CSV}")

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total episodes processed: {len(data)}")
    print(f"Total predictions: {len(predictions_df)}")
    print(f"Ablation types: {ablation_types}")

    print("\n" + "="*80)
    print("DONE!")
    print("="*80)

if __name__ == "__main__":
    main()
