from __future__ import annotations

import os
from typing import Any


from cs336_data.extract import extract_text_from_html_bytes


def run_extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    return extract_text_from_html_bytes(html_bytes)


from cs336_data.langid import identify_language


def run_identify_language(text: str) -> tuple[Any, float]:
    return identify_language(text)


from cs336_data.pii import mask_emails, mask_ips, mask_phone_numbers
from cs336_data.toxicity import classify_nsfw, classify_toxic_speech


def run_mask_emails(text: str) -> tuple[str, int]:
    return mask_emails(text)


def run_mask_phone_numbers(text: str) -> tuple[str, int]:
    return mask_phone_numbers(text)


def run_mask_ips(text: str) -> tuple[str, int]:
    return mask_ips(text)


def run_classify_nsfw(text: str) -> tuple[Any, float]:
    return classify_nsfw(text)


def run_classify_toxic_speech(text: str) -> tuple[Any, float]:
    return classify_toxic_speech(text)


from cs336_data.quality import classify_quality, gopher_quality_filter


def run_classify_quality(text: str) -> tuple[Any, float]:
    return classify_quality(text)


def run_gopher_quality_filter(text: str) -> bool:
    return gopher_quality_filter(text)


from cs336_data.deduplication import (
    run_exact_line_deduplication as run_exact_line_deduplication_impl,
)
from cs336_data.deduplication import (
    run_minhash_deduplication as run_minhash_deduplication_impl,
)


def run_exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
):
    run_exact_line_deduplication_impl(input_files, output_directory)


def run_minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    run_minhash_deduplication_impl(
        input_files, num_hashes, num_bands, ngrams, jaccard_threshold, output_directory
    )
