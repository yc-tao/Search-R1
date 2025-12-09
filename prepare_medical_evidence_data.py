#!/usr/bin/env python3
"""
Prepare medical diagnosis data with evidence-aware rewards.

This script builds an RL training dataset that pairs:
1) A search-count reward (number of <search> tags)
2) An evidence-intersection reward using MDACE annotations

Sources:
- Concatenated per-episode notes: ~/SRL/data/concatenated_notes_by_episode.pkl
- MDACE evidence JSON: /home/yichentao/MDACE/data/Inpatient/ICD-9/1.0
- Optional NOTEEVENTS.csv: used to inject text if MDACE JSON lack note text

Output:
- data/medical_evidence/train.parquet
- data/medical_evidence/test.parquet
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


def load_pickle_with_numpy(path: str) -> Any:
    """Load pickle files that depend on NumPy internals (shim for NumPy 2.x pickles)."""
    import pickle

    # Custom unpickler to handle numpy._core -> numpy.core redirects
    class CompatUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            # Redirect numpy._core imports to numpy.core for compatibility
            if module.startswith('numpy._core'):
                module = module.replace('numpy._core', 'numpy.core')
            return super().find_class(module, name)

    with open(path, 'rb') as f:
        return CompatUnpickler(f).load()


def load_concatenated_notes(path: str) -> Dict[str, str]:
    """Load concatenated notes keyed by episode/hadm id."""
    print(f"[prepare] Loading concatenated notes from {path} ...")
    data = load_pickle_with_numpy(path)

    if isinstance(data, pd.DataFrame):
        formatted: Dict[str, str] = {}

        # Use exact column names from the pickle file
        if 'HADM_ID' not in data.columns:
            raise ValueError(
                f"Expected 'HADM_ID' column in concatenated notes DataFrame. "
                f"Found columns: {list(data.columns)}"
            )

        if 'TEXT' not in data.columns:
            raise ValueError(
                f"Expected 'TEXT' column in concatenated notes DataFrame. "
                f"Found columns: {list(data.columns)}"
            )

        print(f"[prepare] Found {len(data)} rows with HADM_ID and TEXT columns")

        # Map hadm_id to concatenated text
        for _, row in data.iterrows():
            hadm_id = row['HADM_ID']
            note_text = row['TEXT']

            # Skip rows with missing hadm_id or text
            if pd.isna(hadm_id) or not isinstance(note_text, str) or not note_text.strip():
                continue

            # Convert hadm_id to string (it's float64 in the pickle)
            hadm_id_str = str(int(hadm_id))

            # If multiple rows have the same HADM_ID, concatenate their text
            if hadm_id_str in formatted:
                formatted[hadm_id_str] += "\n\n" + note_text
            else:
                formatted[hadm_id_str] = note_text

    elif isinstance(data, dict):
        formatted = {str(k): v for k, v in data.items() if isinstance(v, str)}
    elif isinstance(data, list):
        formatted = {str(idx): note for idx, note in enumerate(data) if isinstance(note, str)}
    else:
        raise ValueError(f"Unexpected format for concatenated notes: {type(data)}")

    print(f"[prepare] Loaded {len(formatted)} unique hadm_id episodes")
    return formatted


def load_noteevents_map(noteevents_path: str) -> Dict[int, str]:
    """Build a map from note_id to text using NOTEEVENTS.csv."""
    import csv

    if not noteevents_path:
        return {}

    path = Path(noteevents_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"NOTEEVENTS.csv not found at {path}")

    print(f"[prepare] Building NOTEEVENTS map from {path} ...")
    csv.field_size_limit(sys.maxsize)
    notes_map: Dict[int, str] = {}

    with path.open('r', encoding='utf8') as f:
        reader = csv.reader(f)
        header = next(reader)
        try:
            row_id_idx = header.index('ROW_ID')
            text_idx = header.index('TEXT')
        except ValueError as exc:
            raise ValueError("NOTEEVENTS.csv must contain ROW_ID and TEXT columns") from exc

        for row in tqdm(reader, desc="Reading NOTEEVENTS"):
            try:
                note_id = int(row[row_id_idx])
                notes_map[note_id] = row[text_idx]
            except Exception:
                continue

    print(f"[prepare] Loaded {len(notes_map)} notes from NOTEEVENTS")
    return notes_map


def ensure_note_text(admission: Dict, notes_map: Dict[int, str]) -> Dict:
    """Ensure each note in the admission has text, injecting from NOTEEVENTS if needed."""
    hadm_id = admission.get("hadm_id")
    updated_notes = []

    for note in admission.get("notes", []):
        text = note.get("text")
        if not text and notes_map:
            text = notes_map.get(note["note_id"])
            if text:
                note["text"] = text

        if not note.get("text"):
            print(f"[prepare] Skipping note {note.get('note_id')} in HADM {hadm_id}: missing text")
            continue

        for ann in note.get("annotations", []):
            begin, end = ann.get("begin"), ann.get("end")
            if begin is None or end is None:
                continue
            if "covered_text" not in ann or ann["covered_text"] is None:
                ann["covered_text"] = note["text"][begin:end]

        updated_notes.append(note)

    admission["notes"] = updated_notes
    return admission


def load_mdace_admissions(mdace_dir: str, noteevents_path: str | None) -> Dict[str, Dict]:
    """Load MDACE admissions, injecting text if available."""
    mdace_path = Path(mdace_dir)
    if not mdace_path.exists():
        raise FileNotFoundError(f"MDACE directory not found at {mdace_path}")

    notes_map = load_noteevents_map(noteevents_path) if noteevents_path else {}
    json_files = sorted(mdace_path.glob("*-ICD-9.json"))
    admissions: Dict[str, Dict] = {}
    missing_text = 0

    print(f"[prepare] Loading MDACE admissions from {mdace_path} ...")
    for json_file in tqdm(json_files, desc="MDACE files"):
        with json_file.open('r', encoding='utf8') as f:
            admission = json.load(f)

        admission = ensure_note_text(admission, notes_map)
        if not admission.get("notes"):
            missing_text += 1
            continue

        hadm_id = str(admission["hadm_id"])
        admissions[hadm_id] = admission

    print(f"[prepare] Loaded {len(admissions)} MDACE admissions with text ({missing_text} skipped)")
    if missing_text and not notes_map:
        print(
            "[prepare] Hint: rerun with --noteevents_path to inject missing note text "
            "if some admissions were skipped."
        )
    return admissions


def sentence_spans_with_offsets(text: str) -> List[Tuple[int, int]]:
    """Split text into sentence-like spans while retaining character offsets."""
    pattern = re.compile(r'[^.!?\n]+[.!?\n]+|[^.!?\n]+$')
    spans: List[Tuple[int, int]] = []
    for match in pattern.finditer(text):
        start, end = match.start(), match.end()
        if text[start:end].strip():
            spans.append((start, end))
    if not spans:
        spans.append((0, len(text)))
    return spans


def chunk_note_with_positions(
    note_text: str,
    hadm_id: str,
    note_id: str | int,
    min_words: int = 40,
    max_words: int = 180,
    context_sentences: int = 1,
) -> List[Dict]:
    """Split note text into chunks, keeping original character offsets."""
    spans = sentence_spans_with_offsets(note_text)
    documents: List[Dict] = []
    current: List[Tuple[int, int]] = []
    start_idx = 0
    word_count = 0

    # Ensure note_id is always a string for consistent parquet serialization
    note_id_str = str(note_id)

    def add_chunk(chunk_start_idx: int, chunk_end_idx: int):
        ext_start_idx = max(0, chunk_start_idx - context_sentences)
        ext_end_idx = min(len(spans) - 1, chunk_end_idx + context_sentences)
        start_char = spans[ext_start_idx][0]
        end_char = spans[ext_end_idx][1]
        chunk_text = note_text[start_char:end_char]
        chunk_idx = len(documents)

        title = f"HADM {hadm_id} NOTE {note_id_str} CHUNK {chunk_idx} [{start_char}:{end_char}]"
        documents.append({
            "id": f"hadm_{hadm_id}_note_{note_id_str}_chunk_{chunk_idx}",
            "contents": f"\"{title}\"\n{chunk_text}",
            "metadata": {
                "hadm_id": hadm_id,
                "note_id": note_id_str,
                "char_start": start_char,
                "char_end": end_char,
                "chunk_index": chunk_idx,
            }
        })

    for idx, (sent_start, sent_end) in enumerate(spans):
        segment = note_text[sent_start:sent_end]
        segment_words = len(segment.split())

        if not current:
            start_idx = idx
        current.append((sent_start, sent_end))
        word_count += segment_words

        if word_count >= min_words or word_count >= max_words:
            add_chunk(start_idx, idx)
            current = []
            word_count = 0

    if current:
        add_chunk(start_idx, start_idx + len(current) - 1)

    return documents


def collect_evidence_spans(admission: Dict) -> List[Dict]:
    """Extract evidence spans from an MDACE admission."""
    evidence_spans: List[Dict] = []
    for note in admission.get("notes", []):
        note_id = note.get("note_id")
        for ann in note.get("annotations", []):
            begin, end = ann.get("begin"), ann.get("end")
            if begin is None or end is None:
                continue
            evidence_spans.append({
                "note_id": note_id,
                "begin": int(begin),
                "end": int(end),
                "code": ann.get("code"),
                "description": ann.get("description"),
                "covered_text": ann.get("covered_text", "")
            })
    return evidence_spans


def create_prompt(question: str) -> List[Dict[str, str]]:
    """Create the prompt in chat format."""
    prompt_text = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as you want. \
If you find no further external knowledge needed, you can provide the final answer inside <answer> and </answer>. \
Question: {question}
"""
    return [{"role": "user", "content": prompt_text}]


def build_record(
    hadm_id: str,
    documents: List[Dict],
    evidence_spans: List[Dict],
    index: int,
    data_source: str = "medical_evidence",
) -> Dict:
    """Build a single training record."""
    question = "Diagnose all conditions that happened during this patient's hospitalization stay."
    return {
        "data_source": data_source,
        "prompt": create_prompt(question),
        "ability": "medical-reasoning",
        "reward_model": {
            "style": "rule",
            "ground_truth": {
                "evidence_present": bool(evidence_spans)
            }
        },
        "extra_info": {
            "hadm_id": hadm_id,
            "documents": documents,
            "evidence_spans": evidence_spans,
            "index": index
        }
    }


def build_records(
    concatenated_notes: Dict[str, str],
    mdace_admissions: Dict[str, Dict],
    min_words: int,
    max_words: int,
) -> List[Dict]:
    """Create dataset records combining concatenated notes and MDACE admissions."""
    records: List[Dict] = []
    seen_hadm: set[str] = set()

    # MDACE records with evidence
    for hadm_id, admission in mdace_admissions.items():
        documents: List[Dict] = []
        for note in admission.get("notes", []):
            text = note.get("text", "")
            if not text or len(text) < 50:
                continue
            note_id = note.get("note_id", "unknown")
            documents.extend(chunk_note_with_positions(
                note_text=text,
                hadm_id=hadm_id,
                note_id=note_id,
                min_words=min_words,
                max_words=max_words,
            ))

        if not documents:
            continue

        evidence_spans = collect_evidence_spans(admission)
        record = build_record(
            hadm_id=hadm_id,
            documents=documents,
            evidence_spans=evidence_spans,
            index=len(records),
            data_source="medical_evidence_mdace"
        )
        records.append(record)
        seen_hadm.add(hadm_id)

    # Additional episodes without evidence (search-count only)
    for hadm_id, note_text in concatenated_notes.items():
        if hadm_id in seen_hadm:
            continue
        if not isinstance(note_text, str) or len(note_text) < 50:
            continue

        documents = chunk_note_with_positions(
            note_text=note_text,
            hadm_id=hadm_id,
            note_id="episode",
            min_words=min_words,
            max_words=max_words,
        )
        if not documents:
            continue

        record = build_record(
            hadm_id=hadm_id,
            documents=documents,
            evidence_spans=[],
            index=len(records),
            data_source="medical_evidence"
        )
        records.append(record)

    print(f"[prepare] Built {len(records)} total records "
          f"({len(seen_hadm)} with evidence, {len(records) - len(seen_hadm)} without)")
    return records


def save_splits(records: List[Dict], output_dir: str, train_ratio: float) -> None:
    """Shuffle and save dataset splits."""
    if not records:
        raise ValueError("No records generated. Check input data paths.")

    os.makedirs(output_dir, exist_ok=True)

    np.random.seed(42)
    indices = np.random.permutation(len(records))
    train_size = int(len(records) * train_ratio)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_records = [records[i] for i in train_indices]
    val_records = [records[i] for i in val_indices]

    train_df = pd.DataFrame(train_records)
    val_df = pd.DataFrame(val_records)

    train_path = os.path.join(output_dir, "train.parquet")
    val_path = os.path.join(output_dir, "test.parquet")

    train_df.to_parquet(train_path)
    val_df.to_parquet(val_path)

    print(f"[prepare] Train records: {len(train_records)} -> {train_path}")
    print(f"[prepare] Test records:  {len(val_records)} -> {val_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare medical evidence-aware RL dataset")
    parser.add_argument(
        "--concatenated_notes_path",
        type=str,
        default=os.path.expanduser("~/SRL/data/concatenated_notes_by_episode.pkl"),
        help="Path to concatenated_notes_by_episode.pkl"
    )
    parser.add_argument(
        "--mdace_dir",
        type=str,
        default="/home/yichentao/MDACE/data/Inpatient/ICD-9/1.0",
        help="Directory containing MDACE JSON files"
    )
    parser.add_argument(
        "--noteevents_path",
        type=str,
        default=None,
        help="Path to NOTEEVENTS.csv (optional, used if MDACE JSON lack text)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/medical_evidence",
        help="Output directory for parquet files"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="Train split ratio"
    )
    parser.add_argument(
        "--min_chunk_words",
        type=int,
        default=40,
        help="Minimum words per chunk"
    )
    parser.add_argument(
        "--max_chunk_words",
        type=int,
        default=180,
        help="Maximum words per chunk before forcing a split"
    )
    args = parser.parse_args()

    concatenated_notes: Dict[str, str] = {}
    if args.concatenated_notes_path:
        try:
            concatenated_notes = load_concatenated_notes(args.concatenated_notes_path)
        except Exception as exc:
            print(f"[prepare] Warning: failed to load concatenated notes ({exc}); continuing with MDACE only.")

    mdace_admissions = load_mdace_admissions(args.mdace_dir, args.noteevents_path)

    records = build_records(
        concatenated_notes=concatenated_notes,
        mdace_admissions=mdace_admissions,
        min_words=args.min_chunk_words,
        max_words=args.max_chunk_words,
    )

    save_splits(records, args.output_dir, args.train_ratio)

    print("\n[prepare] Data preparation complete.")
    print(f"[prepare] Next: start retrieval server (bash retrieval_launch.sh) and launch training scripts.")


if __name__ == "__main__":
    main()
