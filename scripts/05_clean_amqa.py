#!/usr/bin/env python3
"""
05_clean_amqa.py

Create a reproducible cleaned copy of Henok/amharic-qa after manual
investigation of the 6 train/test question overlaps.

IMPORTANT:
- The original Hugging Face dataset is never modified.
- We remove only the examples documented in the manual overlap review.
- Validation is preserved unchanged.
"""

import json
from pathlib import Path

from datasets import load_dataset, DatasetDict


DATASET_NAME = "Henok/amharic-qa"
OUTPUT_DIR = Path("data/processed/amqa_cleaned")
REPORT_PATH = Path("data/processed/amqa_cleaning_report.json")

# Decisions from the manual overlap classification.
# Key = split, values = original row indices to remove.
REMOVE_INDICES = {
    "train": [657, 564, 1682, 147, 1266],
    "test": [174],
    "validation": [],
}

REMOVALS = [
    {
        "case": 1,
        "split": "test",
        "index": 174,
        "classification": "ANNOTATION_ERROR",
        "reason": (
            "The test answer conflicts with its context. The training example "
            "contains the context-supported answer."
        ),
    },
    {
        "case": 2,
        "split": "train",
        "index": 657,
        "classification": "VALID_DUPLICATE",
        "reason": (
            "Same question and underlying answer as the test example; "
            "keep the test instance and remove the training duplicate."
        ),
    },
    {
        "case": 3,
        "split": "train",
        "index": 564,
        "classification": "VALID_DUPLICATE",
        "reason": (
            "Same question and underlying answer as the test example; "
            "keep the test instance and remove the training duplicate."
        ),
    },
    {
        "case": 4,
        "split": "train",
        "index": 1682,
        "classification": "ANNOTATION_ERROR",
        "reason": (
            "The training answer does not answer the question asked by its "
            "context; the test example is the context-consistent instance."
        ),
    },
    {
        "case": 5,
        "split": "train",
        "index": 147,
        "classification": "ANNOTATION_ERROR",
        "reason": (
            "The training context is mismatched with the question, while "
            "the test context addresses Travis Kalanick."
        ),
    },
    {
        "case": 6,
        "split": "train",
        "index": 1266,
        "classification": "ANNOTATION_ERROR",
        "reason": (
            "The training context is mismatched with the Nietzsche question; "
            "the test context addresses Nietzsche."
        ),
    },
]


def main():
    print("=" * 72)
    print("AMQA CLEANING")
    print("=" * 72)
    print(f"Dataset: {DATASET_NAME}")
    print("Loading dataset...")

    dataset = load_dataset(DATASET_NAME)

    print("\nOriginal dataset:")
    for split in dataset:
        print(f"  {split:12s}: {len(dataset[split])}")

    # Safety checks: make sure the expected indices still exist.
    for split, indices in REMOVE_INDICES.items():
        if split not in dataset:
            raise KeyError(f"Expected split '{split}' was not found.")
        max_index = len(dataset[split]) - 1
        for idx in indices:
            if idx < 0 or idx > max_index:
                raise IndexError(
                    f"Removal index {idx} is invalid for {split}; "
                    f"valid range is 0..{max_index}."
                )

    cleaned = DatasetDict()

    for split, ds in dataset.items():
        remove_set = set(REMOVE_INDICES.get(split, []))

        # Dataset.select keeps all rows except the documented removals.
        keep_indices = [
            i for i in range(len(ds))
            if i not in remove_set
        ]

        cleaned[split] = ds.select(keep_indices)

    print("\nCleaned dataset:")
    for split in cleaned:
        print(f"  {split:12s}: {len(cleaned[split])}")

    print("\nChanges:")
    for split in dataset:
        before = len(dataset[split])
        after = len(cleaned[split])
        print(f"  {split:12s}: {before} -> {after}  (-{before - after})")

    # Save the cleaned DatasetDict in Hugging Face's native format.
    OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    cleaned.save_to_disk(str(OUTPUT_DIR))

    # Save an auditable machine-readable record.
    report = {
        "dataset": DATASET_NAME,
        "purpose": (
            "Create a cleaned research copy after manual investigation "
            "of train/test question overlaps."
        ),
        "original_sizes": {
            split: len(dataset[split]) for split in dataset
        },
        "cleaned_sizes": {
            split: len(cleaned[split]) for split in cleaned
        },
        "removed_indices": REMOVE_INDICES,
        "removals": REMOVALS,
        "validation_unchanged": (
            len(dataset["validation"]) == len(cleaned["validation"])
        ),
        "original_dataset_modified": False,
    }

    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nSaved:")
    print(f"  Dataset : {OUTPUT_DIR}")
    print(f"  Report  : {REPORT_PATH}")

    print("\nExpected final sizes:")
    print("  train      = 1826")
    print("  validation = 263")
    print("  test       = 522")

    print("\nCleaning completed successfully.")


if __name__ == "__main__":
    main()
