import json
from pathlib import Path


def load_results(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_metrics(results):
    """
    Extract retrieval metrics from the saved baseline results.
    Handles the expected structure produced by our baseline scripts.
    """

    metrics = {}

    for split in ["validation", "test"]:
        if split not in results:
            continue

        split_data = results[split]

        if isinstance(split_data, dict):
            for key, value in split_data.items():
                if "top" in key.lower():
                    metrics[f"{split}_{key}"] = value

    return metrics


def main():

    print("=" * 70)
    print("BASELINE MODEL COMPARISON")
    print("=" * 70)

    tfidf_path = Path("models/tfidf_baseline/results.json")
    bm25_path = Path("models/bm25_baseline/results.json")

    if not tfidf_path.exists():
        print(f"Missing: {tfidf_path}")
        return

    if not bm25_path.exists():
        print(f"Missing: {bm25_path}")
        return

    tfidf = load_results(tfidf_path)
    bm25 = load_results(bm25_path)

    print("\nTF-IDF")
    print("-" * 70)

    tfidf_metrics = extract_metrics(tfidf)

    for key, value in tfidf_metrics.items():
        print(f"{key:25}: {value}")

    print("\nBM25")
    print("-" * 70)

    bm25_metrics = extract_metrics(bm25)

    for key, value in bm25_metrics.items():
        print(f"{key:25}: {value}")

    # Save combined results
    comparison = {
        "tfidf": tfidf_metrics,
        "bm25": bm25_metrics,
    }

    output_path = Path("models/baseline_comparison.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("COMPARISON SAVED")
    print("=" * 70)
    print(f"Results: {output_path}")


if __name__ == "__main__":
    main()
