from datasets import load_dataset
from collections import defaultdict
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

DATASET_NAME = "Henok/amharic-qa"
OUTPUT_FILE = "data/processed/overlap_classification_report.txt"


# ============================================================
# Helper functions
# ============================================================

def extract_question(text):
    """
    Extract the question from the AmQA input field.
    """
    if "ጥያቄ:" in text:
        return text.split("ጥያቄ:", 1)[1].strip()

    if "ጥያቄ :" in text:
        return text.split("ጥያቄ :", 1)[1].strip()

    return text.strip()


def extract_context(text):
    """
    Extract the context portion before the question.
    """
    if "ጥያቄ:" in text:
        return text.split("ጥያቄ:", 1)[0].strip()

    if "ጥያቄ :" in text:
        return text.split("ጥያቄ :", 1)[0].strip()

    return text.strip()


def find_question_indices(dataset_split):
    """
    Create a mapping:

        normalized question -> list of dataset indices
    """
    questions = defaultdict(list)

    for index, example in enumerate(dataset_split):
        question = extract_question(example["inputs"])
        questions[question].append(index)

    return questions


def separator(char="=", length=80):
    return char * length


# ============================================================
# Main analysis
# ============================================================

def main():

    print("Loading AmQA dataset...")

    dataset = load_dataset(DATASET_NAME)

    print("Dataset loaded successfully.")
    print(dataset)

    train = dataset["train"]
    test = dataset["test"]

    # --------------------------------------------------------
    # Find duplicate questions inside train
    # --------------------------------------------------------

    train_questions = find_question_indices(train)
    test_questions = find_question_indices(test)

    # --------------------------------------------------------
    # Find train/test overlap
    # --------------------------------------------------------

    overlapping_questions = sorted(
        set(train_questions.keys()) & set(test_questions.keys())
    )

    print()
    print(separator())
    print("TRAIN / TEST OVERLAP CLASSIFICATION")
    print(separator())

    print(f"\nOverlapping questions found: {len(overlapping_questions)}")

    if not overlapping_questions:
        print("No overlapping questions found.")
        return

    # --------------------------------------------------------
    # Prepare output directory
    # --------------------------------------------------------

    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = []

    report.append("AMQA TRAIN / TEST OVERLAP CLASSIFICATION REPORT")
    report.append(separator())
    report.append("")
    report.append(
        f"Dataset: {DATASET_NAME}"
    )
    report.append(
        f"Number of overlapping questions: {len(overlapping_questions)}"
    )
    report.append("")
    report.append(
        "IMPORTANT: This report does NOT modify the original dataset."
    )
    report.append(
        "Each case must be manually classified after examining the contexts."
    )
    report.append("")
    report.append(
        "Possible classifications:"
    )
    report.append(
        "  VALID_DUPLICATE     = same question and same underlying answer"
    )
    report.append(
        "  ANNOTATION_ERROR    = one answer appears incorrect for its context"
    )
    report.append(
        "  DIFFERENT_CONTEXT   = same question but contexts legitimately differ"
    )
    report.append(
        "  AMBIGUOUS           = insufficient evidence to decide"
    )
    report.append(
        "  UNRESOLVABLE        = requires external verification"
    )
    report.append("")

    # --------------------------------------------------------
    # Analyze each overlap
    # --------------------------------------------------------

    for number, question in enumerate(overlapping_questions, start=1):

        train_indices = train_questions[question]
        test_indices = test_questions[question]

        report.append(separator())
        report.append(f"CASE {number}")
        report.append(separator())

        report.append("")
        report.append("QUESTION:")
        report.append(question)

        # ----------------------------------------------------
        # TRAIN examples
        # ----------------------------------------------------

        report.append("")
        report.append(separator("-"))
        report.append("TRAIN EXAMPLE(S)")
        report.append(separator("-"))

        for index in train_indices:

            example = train[index]

            context = extract_context(example["inputs"])
            answer = example["targets"]

            report.append("")
            report.append(f"TRAIN INDEX: {index}")

            report.append("")
            report.append("TRAIN CONTEXT:")
            report.append(context)

            report.append("")
            report.append("TRAIN ANSWER:")
            report.append(answer)

        # ----------------------------------------------------
        # TEST examples
        # ----------------------------------------------------

        report.append("")
        report.append(separator("-"))
        report.append("TEST EXAMPLE(S)")
        report.append(separator("-"))

        for index in test_indices:

            example = test[index]

            context = extract_context(example["inputs"])
            answer = example["targets"]

            report.append("")
            report.append(f"TEST INDEX: {index}")

            report.append("")
            report.append("TEST CONTEXT:")
            report.append(context)

            report.append("")
            report.append("TEST ANSWER:")
            report.append(answer)

        # ----------------------------------------------------
        # Classification template
        # ----------------------------------------------------

        report.append("")
        report.append(separator("-"))
        report.append("MANUAL CLASSIFICATION")
        report.append(separator("-"))

        report.append("")
        report.append("Classification:")
        report.append(
            "[ ] VALID_DUPLICATE"
        )
        report.append(
            "[ ] ANNOTATION_ERROR"
        )
        report.append(
            "[ ] DIFFERENT_CONTEXT"
        )
        report.append(
            "[ ] AMBIGUOUS"
        )
        report.append(
            "[ ] UNRESOLVABLE"
        )

        report.append("")
        report.append("Reason:")
        report.append(
            "____________________________________________________________"
        )
        report.append(
            "____________________________________________________________"
        )

        report.append("")
        report.append("Recommended action:")
        report.append(
            "[ ] Keep train example"
        )
        report.append(
            "[ ] Keep test example"
        )
        report.append(
            "[ ] Remove train example"
        )
        report.append(
            "[ ] Remove test example"
        )
        report.append(
            "[ ] Keep both"
        )
        report.append(
            "[ ] Investigate further"
        )

        report.append("")
        report.append("")

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    report_text = "\n".join(report)

    output_path.write_text(
        report_text,
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print()
    print(separator())
    print("OVERLAP CASES")
    print(separator())

    for number, question in enumerate(overlapping_questions, start=1):

        train_indices = train_questions[question]
        test_indices = test_questions[question]

        print()
        print(f"CASE {number}")
        print("-" * 60)
        print(f"Train index: {train_indices}")
        print(f"Test index:  {test_indices}")
        print(f"Question: {question}")

    print()
    print(separator())
    print("REPORT CREATED")
    print(separator())

    print(f"\nSaved to:")
    print(f"  {OUTPUT_FILE}")

    print()
    print(
        "Next step: inspect the report and classify each case."
    )


if __name__ == "__main__":
    main()
