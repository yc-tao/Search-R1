#!/usr/bin/env python3
"""
Prepare medical diagnosis data for RL training.

This script converts the concatenated_notes_by_episode.pkl data into the
parquet format expected by the Search-R1 RL training pipeline.

The output format matches what RLHFDataset expects:
- prompt: list of chat messages [{"role": "user", "content": "..."}]
- data_source: identifier for the data source
- reward_model: dict with ground_truth info (not used for search count reward)
- extra_info: additional metadata like episode_id, documents
"""

import pickle
import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from typing import List, Dict, Any


# Phenotype list for classification (same as infer_diagnosis.py)
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


def load_data(data_path: str) -> Any:
    """Load the concatenated notes from pickle file."""
    print(f"Loading data from {data_path}...")

    # Compatibility shim for NumPy 2.x pickles with NumPy 1.x
    import sys
    sys.modules['numpy._core'] = np.core
    sys.modules['numpy._core.numeric'] = np.core.numeric

    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded {len(data)} episodes")
    return data


def split_and_format_documents(concatenated_note: str, episode_id: Any) -> List[Dict]:
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

        # If we have at least 50 words, finalize this chunk
        if current_word_count >= MIN_WORDS:
            chunks.append(current_chunk)
            current_chunk = []
            current_word_count = 0

    # Add remaining sentences as a chunk if any
    if current_chunk:
        chunks.append(current_chunk)

    # Format as documents with context extension (one sentence before and after)
    documents = []
    for idx, chunk in enumerate(chunks):
        # Find the index range of this chunk in the original sentences list
        chunk_start_idx = None
        for i in range(len(sentences)):
            if sentences[i] == chunk[0]:
                chunk_start_idx = i
                break

        if chunk_start_idx is None:
            continue

        chunk_end_idx = chunk_start_idx + len(chunk) - 1

        # Extend with one sentence before and after
        extended_start = max(0, chunk_start_idx - 1)
        extended_end = min(len(sentences) - 1, chunk_end_idx + 1)

        # Build the extended chunk
        extended_sentences = sentences[extended_start:extended_end + 1]
        extended_text = ' '.join(extended_sentences)

        doc = {
            "id": f"episode_{episode_id}_piece_{idx}",
            "contents": f"\"Note Piece {idx}\"\n{extended_text}"
        }
        documents.append(doc)

    return documents


def create_prompt(question: str) -> List[Dict[str, str]]:
    """Create the prompt in chat format."""
    prompt_text = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can provide the final answer inside <answer> and </answer>. \
Question: {question}
"""
    return [{"role": "user", "content": prompt_text}]


def prepare_dataset(data: Any, output_dir: str, train_ratio: float = 0.9) -> None:
    """
    Prepare the dataset for RL training.

    Args:
        data: The loaded medical episode data
        output_dir: Directory to save the parquet files
        train_ratio: Ratio of data to use for training (rest for validation)
    """
    os.makedirs(output_dir, exist_ok=True)

    records = []

    # Handle different data formats
    if isinstance(data, pd.DataFrame):
        episodes = []
        for idx, row in data.iterrows():
            episode_id = row.get('EPISODE_ID', idx)
            # Find note column
            note_columns = [col for col in data.columns if 'note' in col.lower() or 'text' in col.lower()]
            if note_columns:
                concatenated_note = row[note_columns[0]]
            else:
                # Try first non-ID column
                for col in data.columns:
                    if col != 'EPISODE_ID' and isinstance(row[col], str) and len(row[col]) > 100:
                        concatenated_note = row[col]
                        break
                else:
                    continue
            episodes.append((episode_id, concatenated_note))
    elif isinstance(data, dict):
        episodes = list(data.items())
    elif isinstance(data, list):
        episodes = [(i, note) for i, note in enumerate(data)]
    else:
        raise ValueError(f"Unexpected data format: {type(data)}")

    print(f"Processing {len(episodes)} episodes...")

    for episode_id, concatenated_note in episodes:
        if not isinstance(concatenated_note, str) or len(concatenated_note) < 100:
            print(f"Skipping episode {episode_id}: invalid or too short note")
            continue

        # Split into documents for retrieval
        documents = split_and_format_documents(concatenated_note, episode_id)

        if not documents:
            print(f"Skipping episode {episode_id}: no valid documents")
            continue

        # Create the diagnosis question
        question = "Diagnose all conditions that happened during this patient's hospitalization stay."

        # Create record in the expected format
        record = {
            "data_source": "medical_diagnosis",
            "prompt": create_prompt(question),
            "ability": "medical-reasoning",
            "reward_model": {
                "style": "rule",
                "ground_truth": {"target": PHENOTYPES}  # Not used for search count reward
            },
            "extra_info": {
                "episode_id": str(episode_id),
                "documents": documents,
                "index": len(records)
            }
        }
        records.append(record)

    print(f"Created {len(records)} valid records")

    # Shuffle and split
    np.random.seed(42)
    indices = np.random.permutation(len(records))
    train_size = int(len(records) * train_ratio)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_records = [records[i] for i in train_indices]
    val_records = [records[i] for i in val_indices]

    print(f"Train set: {len(train_records)} records")
    print(f"Validation set: {len(val_records)} records")

    # Save as parquet files
    train_df = pd.DataFrame(train_records)
    val_df = pd.DataFrame(val_records)

    train_path = os.path.join(output_dir, "train.parquet")
    val_path = os.path.join(output_dir, "test.parquet")

    train_df.to_parquet(train_path)
    val_df.to_parquet(val_path)

    print(f"Saved train data to: {train_path}")
    print(f"Saved validation data to: {val_path}")

    # Also save documents index for retrieval server
    documents_index = {}
    for record in records:
        episode_id = record['extra_info']['episode_id']
        documents_index[episode_id] = record['extra_info']['documents']

    documents_path = os.path.join(output_dir, "documents_index.pkl")
    with open(documents_path, 'wb') as f:
        pickle.dump(documents_index, f)
    print(f"Saved documents index to: {documents_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare medical diagnosis data for RL training")
    parser.add_argument(
        "--data_path",
        type=str,
        default=os.path.expanduser("~/SRL/data/concatenated_notes_by_episode.pkl"),
        help="Path to the concatenated notes pickle file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/medical_diagnosis",
        help="Output directory for parquet files"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="Ratio of data to use for training"
    )
    args = parser.parse_args()

    # Load data
    data = load_data(args.data_path)

    # Prepare dataset
    prepare_dataset(data, args.output_dir, args.train_ratio)

    print("\nData preparation complete!")
    print(f"Next steps:")
    print(f"1. Start the retrieval server: bash retrieval_launch.sh")
    print(f"2. Run training: bash train_medical_search_count.sh")


if __name__ == "__main__":
    main()
