#!/usr/bin/env python3
"""
Inject MIMIC-III note text into MDACE JSON files.

This script reads NOTEEVENTS.csv and injects the note text into MDACE
JSON files, adding:
- "text" field to each note
- "covered_text" field to each annotation

Usage:
    python prepare_mdace_with_text.py \
        --noteevents_path /path/to/NOTEEVENTS.csv \
        --mdace_dir /home/yichentao/MDACE/data/Inpatient/ICD-9/1.0 \
        --output_dir data/mdace_with_text
"""

import json
import csv
import sys
import argparse
from pathlib import Path
from typing import Dict
from tqdm import tqdm


def build_notes_map(noteevents_path: str) -> Dict[int, str]:
    """
    Build mapping from note_id to text from NOTEEVENTS.csv.

    Args:
        noteevents_path: Path to NOTEEVENTS.csv from MIMIC-III

    Returns:
        Dictionary mapping note_id (ROW_ID) to note text
    """
    print(f"Loading NOTEEVENTS.csv from {noteevents_path}...")

    id_text_map = {}
    csv.field_size_limit(sys.maxsize)

    with open(noteevents_path, 'r', encoding='utf8') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header

        # Find column indices
        row_id_idx = header.index('ROW_ID')
        text_idx = header.index('TEXT')

        for row in tqdm(reader, desc="Reading notes"):
            note_id = int(row[row_id_idx])
            text = row[text_idx]
            id_text_map[note_id] = text

    print(f"Loaded {len(id_text_map)} notes")
    return id_text_map


def inject_note_text(notes_map: Dict[int, str], admission: Dict) -> Dict:
    """
    Inject text into admission notes and add covered_text to annotations.

    Args:
        notes_map: Dictionary mapping note_id to text
        admission: MDACE admission dictionary

    Returns:
        Modified admission dictionary with text fields added
    """
    for note in admission["notes"]:
        note_id = note["note_id"]
        text = notes_map.get(note_id)

        if text is None:
            print(f"Warning: No text found for note_id {note_id} in HADM_ID {admission['hadm_id']}")
            continue

        # Add text to note
        note["text"] = text

        # Add covered_text to each annotation
        for annotation in note.get("annotations", []):
            begin = annotation["begin"]
            end = annotation["end"]
            annotation["covered_text"] = text[begin:end]

    return admission


def main():
    parser = argparse.ArgumentParser(description="Inject MIMIC-III note text into MDACE JSON files")
    parser.add_argument(
        "--noteevents_path",
        type=str,
        required=True,
        help="Path to NOTEEVENTS.csv from MIMIC-III"
    )
    parser.add_argument(
        "--mdace_dir",
        type=str,
        default="/home/yichentao/MDACE/data/Inpatient/ICD-9/1.0",
        help="Directory containing MDACE JSON files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/mdace_with_text",
        help="Output directory for MDACE files with text"
    )
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build notes map from NOTEEVENTS.csv
    notes_map = build_notes_map(args.noteevents_path)

    # Process each MDACE JSON file
    mdace_dir = Path(args.mdace_dir)
    json_files = sorted(mdace_dir.glob("*-ICD-9.json"))

    print(f"\nProcessing {len(json_files)} MDACE files...")

    successful = 0
    failed = 0

    for json_file in tqdm(json_files, desc="Injecting text"):
        try:
            with open(json_file, 'r', encoding='utf8') as f:
                admission = json.load(f)

            # Inject text
            admission = inject_note_text(notes_map, admission)

            # Save to output directory
            output_path = output_dir / json_file.name
            with open(output_path, 'w', encoding='utf8') as f:
                json.dump(admission, f, indent=2)

            successful += 1

        except Exception as e:
            print(f"\nError processing {json_file.name}: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"Text injection complete!")
    print(f"  Successful: {successful} files")
    print(f"  Failed: {failed} files")
    print(f"  Output directory: {output_dir}")
    print(f"{'='*60}")

    if failed > 0:
        print("\nWarning: Some files failed to process. Check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
