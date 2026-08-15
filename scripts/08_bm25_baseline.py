from datasets import load_from_disk
from rank_bm25 import BM25Okapi
import json
import re
from pathlib import Path


DATASET_PATH = "data/processed/amqa_structured"
OUTPUT_DIR = Path("models/bm25_baseline")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "results.json"


def normalize_text(text):
    """Normalize text for comparison."""
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text):
    """Simple whitespace tokenizer."""
    return str(text).lower().split()


def build_bm25(train_dataset):
    """Build BM25 index over training contexts."""

    contexts = [example["context"] for example in train_dataset]

    tokenized_contexts = [
        tokenize(context)
        for context in contexts
    ]

    print("Building BM25 index...")

    bm25 = BM25Okapi(tokenized_contexts)

    return bm25


def evaluate_split(bm25, train_dataset, split_dataset, split_name):
    """Evaluate BM25 retrieval."""

    print("=" * 70)
    print(f"EVALUATING: {split_name}")
    print("=" * 70)

    top1_correct = 0
    top5_correct = 0
    top10_correct = 0

    examples = []

    train_contexts = [
        normalize_text(example["context"])
        for example in train_dataset
    ]

    for i, example in enumerate(split_dataset):

        question = example["question"]
        true_context = normalize_text(example["context"])

        scores = bm25.get_scores(tokenize(question))

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda x: scores[x],
            reverse=True
        )

        top1 = ranked_indices[:1]
        top5 = ranked_indices[:5]
        top10 = ranked_indices[:10]

        def contains_correct(indices):
            return any(
                train_contexts[int(idx)] == true_context
                for idx in indices
            )

        correct1 = contains_correct(top1)
        correct5 = contains_correct(top5)
        correct10 = contains_correct(top10)

        if correct1:
            top1_correct += 1

        if correct5:
            top5_correct += 1

        if correct10:
            top10_correct += 1

        # Save a few examples for qualitative inspection
        if i < 5:

            retrieved = []

            for rank, idx in enumerate(top5, start=1):

                retrieved.append({
                    "rank": rank,
                    "training_index": int(idx),
                    "score": float(scores[idx]),
                    "context": train_dataset[int(idx)]["context"],
                    "is_correct": (
                        train_contexts[int(idx)] == true_context
                    )
                })

            examples.append({
                "question": question,
                "true_context": example["context"],
                "answer": example["answer"],
                "retrieved": retrieved
            })

    total = len(split_dataset)

    top1_accuracy = top1_correct / total
    top5_accuracy = top5_correct / total
    top10_accuracy = top10_correct / total

    print(
        f"Top-1  Retrieval Accuracy: "
        f"{top1_accuracy:.4f} "
        f"({top1_correct}/{total})"
    )

    print(
        f"Top-5  Retrieval Accuracy: "
        f"{top5_accuracy:.4f} "
        f"({top5_correct}/{total})"
    )

    print(
        f"Top-10 Retrieval Accuracy: "
        f"{top10_accuracy:.4f} "
        f"({top10_correct}/{total})"
    )

    print()

    return {
        "split": split_name,
        "total_examples": total,
        "top1_accuracy": top1_accuracy,
        "top5_accuracy": top5_accuracy,
        "top10_accuracy": top10_accuracy,
        "top1_correct": top1_correct,
        "top5_correct": top5_correct,
        "top10_correct": top10_correct,
        "examples": examples
    }


def main():

    print("=" * 70)
    print("BM25 RETRIEVAL BASELINE")
    print("=" * 70)
    print()

    print("Loading structured AMQA dataset...")

    dataset = load_from_disk(DATASET_PATH)

    train_dataset = dataset["train"]
    validation_dataset = dataset["validation"]
    test_dataset = dataset["test"]

    print(f"Train      : {len(train_dataset)}")
    print(f"Validation : {len(validation_dataset)}")
    print(f"Test       : {len(test_dataset)}")
    print()

    # Build BM25 index using training contexts
    bm25 = build_bm25(train_dataset)

    print(f"Indexed contexts : {len(train_dataset)}")
    print()

    # Validation evaluation
    validation_results = evaluate_split(
        bm25,
        train_dataset,
        validation_dataset,
        "validation"
    )

    # Test evaluation
    test_results = evaluate_split(
        bm25,
        train_dataset,
        test_dataset,
        "test"
    )

    # Save results
    results = {
        "method": "BM25",
        "dataset": "Henok/amharic-qa",
        "dataset_path": DATASET_PATH,
        "train_size": len(train_dataset),
        "validation_size": len(validation_dataset),
        "test_size": len(test_dataset),
        "validation": validation_results,
        "test": test_results
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("=" * 70)
    print("RESULTS SAVED")
    print("=" * 70)

    print(f"Results: {OUTPUT_FILE}")

    print()
    print("BM25 baseline completed successfully.")


if __name__ == "__main__":
    main()
