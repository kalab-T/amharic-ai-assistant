from datasets import load_dataset
from collections import defaultdict


DATASET_NAME = "Henok/amharic-qa"


def normalize_text(text):
    """Basic normalization for comparison."""
    if text is None:
        return ""

    return " ".join(str(text).strip().split())


def extract_question(text):
    """
    Extract the question from the AmQA input field.

    The dataset contains context followed by:
    'ጥያቄ: ...'
    """
    text = str(text)

    markers = [
        "ጥያቄ:",
        "ጥያቄ :",
        "ጥያቄ፡",
        "ጥያቄ፡ "
    ]

    for marker in markers:
        if marker in text:
            question = text.split(marker, 1)[1]
            return normalize_text(question)

    return normalize_text(text)


def build_question_map(split):
    """
    Create:
        question -> list of dataset indices
    """
    question_map = defaultdict(list)

    for index, example in enumerate(split):
        question = extract_question(example["inputs"])
        question_map[question].append(index)

    return question_map


def analyze_internal_duplicates(split_name, split):
    """Find duplicate questions inside one split."""

    question_map = build_question_map(split)

    duplicates = {
        question: indices
        for question, indices in question_map.items()
        if len(indices) > 1
    }

    print(f"\n{split_name.upper()} INTERNAL DUPLICATES")
    print("-" * 60)
    print(f"Unique questions: {len(question_map)}")
    print(f"Questions appearing more than once: {len(duplicates)}")

    if duplicates:
        for question, indices in duplicates.items():
            print("\nQuestion:")
            print(question)
            print(f"Indices: {indices}")


def compare_splits(name_a, split_a, name_b, split_b):
    """Compare questions between two dataset splits."""

    map_a = build_question_map(split_a)
    map_b = build_question_map(split_b)

    questions_a = set(map_a.keys())
    questions_b = set(map_b.keys())

    overlap = questions_a.intersection(questions_b)

    print(f"\n{name_a.upper()} ∩ {name_b.upper()}")
    print("-" * 60)
    print(f"Overlapping questions: {len(overlap)}")

    if not overlap:
        print("No overlapping questions found.")
        return

    for question in sorted(overlap):
        print("\n" + "=" * 60)
        print("DUPLICATE QUESTION:")
        print(question)

        print(f"\n{name_a} indices: {map_a[question]}")
        print(f"{name_b} indices: {map_b[question]}")

        print("\nAnswers:")

        for index in map_a[question]:
            answer = split_a[index]["targets"]
            print(f"\n{name_a} answer:")
            print(answer)

        for index in map_b[question]:
            answer = split_b[index]["targets"]
            print(f"\n{name_b} answer:")
            print(answer)


def compare_contexts(name_a, split_a, name_b, split_b):
    """
    Check whether duplicated questions also have identical contexts.
    """

    map_a = build_question_map(split_a)
    map_b = build_question_map(split_b)

    overlap = set(map_a.keys()).intersection(map_b.keys())

    print(f"\n{'=' * 60}")
    print(f"CONTEXT COMPARISON: {name_a.upper()} vs {name_b.upper()}")
    print("=" * 60)

    if not overlap:
        print("No overlapping questions.")
        return

    identical_contexts = 0

    for question in sorted(overlap):

        for index_a in map_a[question]:
            context_a = normalize_text(split_a[index_a]["inputs"])

            for index_b in map_b[question]:
                context_b = normalize_text(split_b[index_b]["inputs"])

                if context_a == context_b:
                    identical_contexts += 1

                    print("\nIDENTICAL CONTEXT FOUND")
                    print("-" * 60)
                    print(f"Question: {question}")
                    print(f"{name_a} index: {index_a}")
                    print(f"{name_b} index: {index_b}")

    print(f"\nIdentical context pairs: {identical_contexts}")


def main():

    print("Loading AmQA dataset...")

    dataset = load_dataset(DATASET_NAME)

    print("\nDataset loaded successfully.")
    print(dataset)

    # ---------------------------------------------------------
    # 1. Internal duplicates
    # ---------------------------------------------------------

    for split_name in ["train", "validation", "test"]:
        analyze_internal_duplicates(
            split_name,
            dataset[split_name]
        )

    # ---------------------------------------------------------
    # 2. Cross-split duplicate questions
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("CROSS-SPLIT LEAKAGE ANALYSIS")
    print("=" * 60)

    compare_splits(
        "train",
        dataset["train"],
        "validation",
        dataset["validation"]
    )

    compare_splits(
        "train",
        dataset["train"],
        "test",
        dataset["test"]
    )

    compare_splits(
        "validation",
        dataset["validation"],
        "test",
        dataset["test"]
    )

    # ---------------------------------------------------------
    # 3. Compare contexts for train/test overlap
    # ---------------------------------------------------------

    compare_contexts(
        "train",
        dataset["train"],
        "test",
        dataset["test"]
    )

    print("\n" + "=" * 60)
    print("LEAKAGE ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
