from __future__ import annotations

import os
from typing import Any
from cs336_data.extraction import extract_text
from cs336_data.filter import *
from cs336_data.quality_classifier import QualityClassifier
from cs336_data.deduplication import exact_line_deduplication, minhash_deduplication



def run_extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    return extract_text(html_bytes)


def run_identify_language(text: str) -> tuple[Any, float]:
    return identify_language(text)


def run_mask_emails(text: str) -> tuple[str, int]:
    return mask_email(text)


def run_mask_phone_numbers(text: str) -> tuple[str, int]:
    return mask_phone_number(text)


def run_mask_ips(text: str) -> tuple[str, int]:
    return mask_ip_address(text)


def run_classify_nsfw(text: str) -> tuple[Any, float]:
    return classify_nsfw(text)


def run_classify_toxic_speech(text: str) -> tuple[Any, float]:
    return classify_toxic_speech(text)


def run_classify_quality(text: str) -> tuple[Any, float]:
    file_path = 'data/my_classifiers/quality_classifier_3gram.bin'
    mapping = {
                '__label__low_quality': 'cc',
                '__label__high_quality': 'wiki'
            }
    classifier = QualityClassifier(file_path, mapping)
    return classifier.predict(text)


def run_gopher_quality_filter(text: str) -> bool:
    return gopher_quality_filter(text)


def run_exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
):
    exact_line_deduplication(input_files, output_directory)


def run_minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    minhash_deduplication(input_files=input_files,
                          num_hashes=num_hashes,
                          num_bands=num_bands,
                          n=ngrams,
                          jaccard_threshold=jaccard_threshold,
                          output_dir=output_directory)
