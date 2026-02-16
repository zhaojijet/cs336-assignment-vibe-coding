import os
import mmh3
from pathlib import Path
from tqdm import tqdm
import struct
import numpy as np
from typing import List, Set, Dict, Tuple
from collections import defaultdict


# Re-exporting run_exact_line_deduplication so it can be imported from here as expected
def run_exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
):
    """
    Deduplicates lines across multiple input files.
    The goal is to remove lines that appear in MORE THAN ONE document (boilerplate removal).
    Also deduplicates duplicate lines within the same document.
    """
    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pass 1: Count Document Frequency (DF) for each line
    # Map hash -> count of documents containing this line
    line_df = defaultdict(int)

    # We need to process all files first
    # Optimization: If files are huge, this might check memory.
    # But for this assignment, we assume it fits in memory (hashes).

    valid_input_files = [Path(p) for p in input_files if Path(p).exists()]

    for input_path in valid_input_files:
        with open(input_path, "r", encoding="utf-8", errors="replace") as f:
            # Get unique lines in this document
            unique_lines = set(f.readlines())

        for line in unique_lines:
            line_hash = mmh3.hash128(line)
            line_df[line_hash] += 1

    # Pass 2: Rewrite files
    for input_path in valid_input_files:
        output_path = output_dir / input_path.name

        with open(input_path, "r", encoding="utf-8", errors="replace") as infile, open(
            output_path, "w", encoding="utf-8"
        ) as outfile:

            # We enforce local deduplication too, logic:
            # If line DF == 1, it only appears in THIS document.
            # If it appears multiple times in THIS document, DF is still 1 (we used set above).
            # Should we dedup locally?
            # Standard "exact line deduplication" usually implies converting to set of lines.
            # So yes, keep first occurrence, skip subsequent.

            seen_in_doc = set()

            for line in infile:
                line_hash = mmh3.hash128(line)

                # Check global DF
                if line_df[line_hash] == 1:
                    # Check local seen
                    if line_hash not in seen_in_doc:
                        outfile.write(line)
                        seen_in_doc.add(line_hash)


# MinHash Implementation


def _get_ngrams(text: str, n: int) -> Set[str]:
    # White-space based tokenization for n-grams
    tokens = text.split()
    if len(tokens) < n:
        if tokens:
            return set([" ".join(tokens)])
        return set()

    shingles = set()
    for i in range(len(tokens) - n + 1):
        shingle = " ".join(tokens[i : i + n])
        shingles.add(shingle)
    return shingles


def _compute_minhash_signature(shingles: Set[str], num_hashes: int) -> List[int]:
    MAX_HASH = (1 << 32) - 1
    signature = [MAX_HASH] * num_hashes

    for shingle in shingles:
        for i in range(num_hashes):
            h = mmh3.hash(shingle, seed=i, signed=False)
            if h < signature[i]:
                signature[i] = h

    return signature


def _compute_jaccard_similarity(sig1: List[int], sig2: List[int]) -> float:
    if not sig1 or not sig2:
        return 0.0
    matches = sum(1 for i in range(len(sig1)) if sig1[i] == sig2[i])
    return matches / len(sig1)


class LSHIndex:
    def __init__(self, num_bands: int, num_hashes: int):
        self.num_bands = num_bands
        self.num_hashes = num_hashes
        self.rows_per_band = num_hashes // num_bands
        self.buckets: List[Dict[int, List[int]]] = [{} for _ in range(num_bands)]
        self.signatures: Dict[int, List[int]] = {}

    def query(self, signature: List[int], jaccard_threshold: float) -> int | None:
        candidates = set()

        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band_chunk = tuple(signature[start:end])
            band_hash = hash(band_chunk)

            if band_hash in self.buckets[band_idx]:
                candidates.update(self.buckets[band_idx][band_hash])

        for cand_id in candidates:
            cand_sig = self.signatures[cand_id]
            sim = _compute_jaccard_similarity(signature, cand_sig)
            if sim >= jaccard_threshold:
                return cand_id

        return None

    def add(self, doc_id: int, signature: List[int]):
        self.signatures[doc_id] = signature
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band_chunk = tuple(signature[start:end])
            band_hash = hash(band_chunk)

            if band_hash not in self.buckets[band_idx]:
                self.buckets[band_idx][band_hash] = []
            self.buckets[band_idx][band_hash].append(doc_id)


def run_minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    lsh_index = LSHIndex(num_bands=num_bands, num_hashes=num_hashes)
    doc_counter = 0

    # Sort input files to ensure deterministic processing order
    sorted_input_files = sorted([Path(p) for p in input_files], key=lambda x: x.name)

    for input_path in sorted_input_files:
        if not input_path.exists():
            continue

        with open(input_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()

        shingles = _get_ngrams(text, n=ngrams)

        if not shingles:
            signature = [0] * num_hashes
        else:
            signature = _compute_minhash_signature(shingles, num_hashes)

        duplicate_id = lsh_index.query(signature, jaccard_threshold)

        if duplicate_id is None:
            lsh_index.add(doc_counter, signature)

            output_path = output_dir / input_path.name
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)
        else:
            pass

        doc_counter += 1
