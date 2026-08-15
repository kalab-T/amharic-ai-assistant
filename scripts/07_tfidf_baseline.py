from datasets import load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import re
from pathlib import Path


DATASET_PATH = "data/processed/amqa_structured"
OUTPUT_DIR = Path("models/tfidf_baseline")


def normalize_text(text):
    """Normalize text for comparison."""
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def build_tfidf_index(train_dataset):
    """Build TF-IDF index using training contexts."""
    contexts = [example["context"] for example in train_dataset]

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        min_df=1,
        max_features=400000,
    )

    matrix = vectorizer.fit_transform(contexts)

    return vectorizer, matrix


def retrieve_top_k(question, vectorizer, matrix, k=5):
    """Retrieve top-k training contexts for a question."""
    question_vector = vectorizer.transform([question])

    scores = cosine_similarity(
        question_vector,
        matrix
    ).flatten()

    top_indices = scores.argsort()[::-1][:k]

    return [
        (int(index), float(scores[index]))
        for index in top_indices
    ]


def evaluate_split(
    split_dataset,
    train_dataset,
    vectorizer,
    matrix,
    split_name,
    k_values=(1, 5)
):
    """
    Evaluate retrieval against the training corpus.

    IMPORTANT:
    Retrieved indices always refer to train_dataset because
    the TF-IDF index was built from training contexts.
    """

    print()
    print("=" * 70)
    print(f"EVALUATING: {split_name}")
    print("=" * 70)

    total = len(split_dataset)

    results = {
        "split": split_name,
        "total_examples": total,
        "retrieval": {}
    }

    for k in k_values:
        correct = 0

        for example in split_dataset:
            question = example["question"]
            true_context = normalize_text(example["context"])

            retrieved = retrieve_top_k(
                question,
                vectorizer,
                matrix,
                k=k
            )

            retrieved_contexts = [
                normalize_text(train_dataset[index]["context"])
                for index, _ in retrieved
            ]

            if true_context in retrieved_contexts:
                correct += 1

        accuracy = correct / total if total > 0 else 0.0

        results["retrieval"][f"top_{k}"] = {
            "correct": correct,
            "total": total,
            "accuracy": accuracy
        }

        print(
            f"Top-{k:<2} Retrieval Accuracy: "
            f"{accuracy:.4f} "
            f"({correct}/{total})"
        )

    return results


def show_examples(
    split_dataset,
    train_dataset,
    vectorizer,
    matrix,
    num_examples=5
):
    """Display qualitative retrieval examples."""

    print()
    print("=" * 70)
    print("QUALITATIVE RETRIEVAL EXAMPLES")
    print("=" * 70)

    for i in range(min(num_examples, len(split_dataset))):

        example = split_dataset[i]

        question = example["question"]
        true_context = normalize_text(example["context"])
        answer = example["answer"]

        retrieved = retrieve_top_k(
            question,
            vectorizer,
            matrix,
            k=3
        )

        print()
        print("-" * 70)
        print(f"Example {i + 1}")
        print("-" * 70)

        print("QUESTION:")
        print(question)

        print()
        print("ANSWER:")
        print(answer)

        print()
        print("TRUE CONTEXT FOUND IN TOP-3:")

        found = False

        for rank, (index, score) in enumerate(retrieved, start=1):

            retrieved_context = normalize_text(
                train_dataset[index]["context"]
            )

            if retrieved_context == true_context:
                found = True

            print()
            print(f"Rank {rank}")
            print(f"Training index: {index}")
            print(f"Similarity: {score:.4f}")

            # Show shortened context
            original_context = train_dataset[index]["context"]

            if len(original_context) > 500:
                original_context = original_context[:500] + "..."

            print("Context:")
            print(original_context)

        print()
        print(f"Correct context in top-3: {found}")


def main():

    print("=" * 70)
    print("TF-IDF RETRIEVAL BASELINE")
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
    print("Building TF-IDF index...")

    vectorizer, matrix = build_tfidf_index(
        train_dataset
    )

    print(f"Vocabulary size : {len(vectorizer.vocabulary_)}")
    print(f"Matrix shape    : {matrix.shape}")

    # ---------------------------------------------------------
    # Validation evaluation
    # ---------------------------------------------------------

    validation_results = evaluate_split(
        validation_dataset,
        train_dataset,
        vectorizer,
        matrix,
        "validation",
        k_values=(1, 5)
    )

    # ---------------------------------------------------------
    # Test evaluation
    # ---------------------------------------------------------

    test_results = evaluate_split(
        test_dataset,
        train_dataset,
        vectorizer,
        matrix,
        "test",
        k_values=(1, 5)
    )

    # ---------------------------------------------------------
    # Qualitative examples
    # ---------------------------------------------------------

    show_examples(
        validation_dataset,
        train_dataset,
        vectorizer,
        matrix,
        num_examples=5
    )

    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results = {
        "method": "TF-IDF character n-gram retrieval",
        "dataset": DATASET_PATH,
        "train_size": len(train_dataset),
        "validation_size": len(validation_dataset),
        "test_size": len(test_dataset),
        "vocabulary_size": len(vectorizer.vocabulary_),
        "matrix_shape": list(matrix.shape),
        "validation": validation_results,
        "test": test_results,
        "parameters": {
            "analyzer": "char",
            "ngram_range": [2, 5],
            "max_features": 400000
        }
    }

    output_file = OUTPUT_DIR / "results.json"

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2
        )

    print()
    print("=" * 70)
    print("RESULTS SAVED")
    print("=" * 70)
    print(f"Results: {output_file}")

    print()
    print("TF-IDF baseline completed successfully.")


if __name__ == "__main__":
    main()
