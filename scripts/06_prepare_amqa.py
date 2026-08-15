from datasets import load_from_disk, DatasetDict
import re
import json
from pathlib import Path


INPUT_PATH = "data/processed/amqa_cleaned"
OUTPUT_PATH = "data/processed/amqa_structured"
REPORT_PATH = "data/processed/amqa_structure_report.json"


def parse_input(text):
    """
    Extract context and question from the AMQA 'inputs' field.
    """

    if "ጥያቄ:" in text:
        context, question = text.rsplit("ጥያቄ:", 1)

        context = context.strip()
        question = question.strip()

    else:
        context = text.strip()
        question = ""

    return context, question


def clean_answer(answer):
    """
    Basic normalization of the target answer.
    We preserve the original Amharic content while
    removing unnecessary surrounding whitespace.
    """

    answer = answer.strip()
    answer = re.sub(r"\s+", " ", answer)

    return answer


def process_split(dataset):
    contexts = []
    questions = []
    answers = []
    original_inputs = []
    original_targets = []

    for example in dataset:
        original_input = example["inputs"]
        original_target = example["targets"]

        context, question = parse_input(original_input)
        answer = clean_answer(original_target)

        contexts.append(context)
        questions.append(question)
        answers.append(answer)

        original_inputs.append(original_input)
        original_targets.append(original_target)

    return {
        "context": contexts,
        "question": questions,
        "answer": answers,
        "original_inputs": original_inputs,
        "original_targets": original_targets,
    }


def main():

    print("=" * 70)
    print("AMQA STRUCTURED DATA PREPARATION")
    print("=" * 70)

    print(f"\nLoading dataset from: {INPUT_PATH}")

    dataset = load_from_disk(INPUT_PATH)

    print("\nOriginal dataset:")
    for split in dataset:
        print(f"  {split:12}: {len(dataset[split])}")

    structured = {}

    for split in dataset:
        print(f"\nProcessing {split}...")

        structured[split] = process_split(dataset[split])

        print(f"  Processed: {len(structured[split]['question'])}")

    structured_dataset = DatasetDict()

    for split, data in structured.items():
        from datasets import Dataset
        structured_dataset[split] = Dataset.from_dict(data)

    output_path = Path(OUTPUT_PATH)

    if output_path.exists():
        import shutil
        shutil.rmtree(output_path)

    structured_dataset.save_to_disk(OUTPUT_PATH)

    # Basic quality statistics
    report = {}

    for split in structured_dataset:

        data = structured_dataset[split]

        empty_questions = sum(
            1 for q in data["question"] if not q.strip()
        )

        empty_contexts = sum(
            1 for c in data["context"] if not c.strip()
        )

        empty_answers = sum(
            1 for a in data["answer"] if not a.strip()
        )

        report[split] = {
            "examples": len(data),
            "empty_questions": empty_questions,
            "empty_contexts": empty_contexts,
            "empty_answers": empty_answers,
        }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("STRUCTURED DATASET")
    print("=" * 70)

    for split in structured_dataset:
        print(f"  {split:12}: {len(structured_dataset[split])}")

    print("\nColumns:")
    print(structured_dataset["train"].column_names)

    print("\nFirst structured example:")
    print("-" * 70)

    example = structured_dataset["train"][0]

    print("QUESTION:")
    print(example["question"])

    print("\nCONTEXT:")
    print(example["context"])

    print("\nANSWER:")
    print(example["answer"])

    print("\n" + "=" * 70)
    print("QUALITY REPORT")
    print("=" * 70)

    for split, stats in report.items():
        print(f"\n{split}:")
        print(f"  Examples         : {stats['examples']}")
        print(f"  Empty questions  : {stats['empty_questions']}")
        print(f"  Empty contexts   : {stats['empty_contexts']}")
        print(f"  Empty answers    : {stats['empty_answers']}")

    print("\nSaved:")
    print(f"  Dataset : {OUTPUT_PATH}")
    print(f"  Report  : {REPORT_PATH}")

    print("\nPreparation completed successfully.")


if __name__ == "__main__":
    main()
