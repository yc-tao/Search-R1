"""
Composite medical reward: search-count + evidence hit.

Evidence reward is 1 if any retrieved document overlaps with a gold MDACE evidence
span from the same note_id. Overlap is computed on character offsets.
"""

import re
from typing import Dict, List, Optional, Tuple

from .search_count import count_searches


def extract_information_blocks(solution_str: str) -> List[str]:
    """Extract all <information>...</information> blocks from the solution."""
    return re.findall(r"<information>(.*?)</information>", solution_str, re.DOTALL)


def parse_retrieved_entries(information_block: str) -> List[Dict[str, str]]:
    """Parse retrieval results inside an <information> block."""
    doc_pattern = re.compile(
        r"Doc\s+\d+\(Title:\s*(?P<title>.*?)\)\s*(?P<body>(?:(?!Doc\s+\d+\(Title:).)*)",
        re.DOTALL,
    )
    entries: List[Dict[str, str]] = []
    for match in doc_pattern.finditer(information_block):
        title = match.group("title").strip()
        body = match.group("body").strip()
        entries.append({"title": title, "body": body})
    return entries


def normalize_title(title: str) -> str:
    """Normalize titles for matching."""
    return title.strip().strip('"').strip().lower()


def normalize_doc_title(doc: Dict) -> str:
    """Normalize the document title from its contents."""
    contents = doc.get("contents", "")
    if "\n" in contents:
        title_line = contents.split("\n", 1)[0]
    else:
        title_line = contents
    return normalize_title(title_line)


def build_document_lookup(documents: List[Dict]) -> Dict[str, List[Dict]]:
    """Build a lookup from normalized title to documents."""
    lookup: Dict[str, List[Dict]] = {}
    for doc in documents:
        title_key = normalize_doc_title(doc)
        lookup.setdefault(title_key, []).append(doc)
    return lookup


def _match_entry_to_document(
    entry: Dict[str, str],
    documents: List[Dict],
    lookup: Dict[str, List[Dict]],
) -> Optional[Dict]:
    """Match a retrieved entry to the original document list."""
    title_key = normalize_title(entry.get("title", ""))
    if title_key in lookup:
        return lookup[title_key][0]

    # Fallback: substring match on body text
    body = entry.get("body", "").lower()
    for doc in documents:
        doc_body = doc.get("contents", "").lower()
        if not doc_body:
            continue
        if body and (body in doc_body or doc_body in body):
            return doc
    return None


def extract_retrieved_documents(solution_str: str, documents: List[Dict]) -> List[Dict]:
    """Recover which documents were retrieved based on the formatted search output."""
    if not documents:
        return []

    lookup = build_document_lookup(documents)
    retrieved: List[Dict] = []
    seen_ids = set()

    for info_block in extract_information_blocks(solution_str):
        for entry in parse_retrieved_entries(info_block):
            doc = _match_entry_to_document(entry, documents, lookup)
            if not doc:
                continue
            doc_id = doc.get("id") or doc.get("metadata", {}).get("chunk_index")
            if doc_id in seen_ids:
                continue
            retrieved.append(doc)
            seen_ids.add(doc_id)
    return retrieved


def spans_intersect(span_a: Tuple[int, int], span_b: Tuple[int, int]) -> bool:
    """Return True if spans [a0, a1) and [b0, b1) overlap."""
    return span_a[0] < span_b[1] and span_b[0] < span_a[1]


def has_evidence_hit(retrieved_docs: List[Dict], evidence_spans: List[Dict]) -> bool:
    """Check if any retrieved doc overlaps with an evidence span on the same note_id."""
    if not retrieved_docs or not evidence_spans:
        return False

    for doc in retrieved_docs:
        metadata = doc.get("metadata", {})
        note_id = metadata.get("note_id")
        start = metadata.get("char_start")
        end = metadata.get("char_end")
        if note_id is None or start is None or end is None:
            continue

        for span in evidence_spans:
            if str(span.get("note_id")) != str(note_id):
                continue
            span_begin = span.get("begin")
            span_end = span.get("end")
            if span_begin is None or span_end is None:
                continue
            if spans_intersect((int(start), int(end)), (int(span_begin), int(span_end))):
                return True
    return False


def compute_composite_reward(
    solution_str: str,
    documents: List[Dict],
    evidence_spans: List[Dict],
    max_searches: int = 10,
    alpha: float = 0.5,
    beta: float = 0.5,
) -> float:
    """Compute composite reward = alpha * search_count + beta * evidence_hit."""
    num_searches = count_searches(solution_str)
    search_reward = min(num_searches, max_searches) / max_searches

    retrieved_docs = extract_retrieved_documents(solution_str, documents)
    evidence_reward = 1.0 if has_evidence_hit(retrieved_docs, evidence_spans) else 0.0

    return alpha * search_reward + beta * evidence_reward
