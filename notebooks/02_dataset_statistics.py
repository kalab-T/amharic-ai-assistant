from datasets import load_dataset
import re
from collections import Counter


DATASET_NAME = "Henok/amharic-qa"


def parse_example(text):
    """
    Extract context, question, and answer from an AmQA example.
    """

    # Find the question marker
    question_marker = "ጥያቄ:"

    if question_marker not in text:
        return text.strip(), "", ""

    context_part, question_part = text.rsplit(question_marker, 1)

    # Remove the standard instruction at the beginning
    instruction = (
        "ከዚህ በታች ያለውን ዝርዝር መረጃ "
        "በመጠቀም ለሚከተለው ጥያቄ መልስ ይስጡ፡"
    )

    context = context_part.replace(instruction, "", 1).strip()

    question = question_part.strip()

    return context, question, ""


def extract_answer(target):
    """
    Extract the answer from the target field.

    The dataset uses a template such as:
    '... 229 ቢሊየን ነው።'
    """

    if not target:
        return ""

    # Remove common answer prefixes
    prefixes = [
        "ከጥያቄው አንጻር ትክክለኛው መልስ",
        "ለተጠቀሰው ጥያቄ ትክክለኛው ምላሽ",
    ]

    answer = target.strip()

    for prefix in prefixes:
        if prefix in answer:
            answer = answer.split(prefix, 1)[1].strip()

    # Remove common connecting words
    answer = re.sub(r"^(ነው|ነበር|የሚለው)\s*", "", answer)

    return answer.strip()


def word_count(text):
    """
    Approximate word count using whitespace.
    """
    return len(text.split())


def char_count(text):
    """
    Number of Unicode characters.
    """
    return len(text)


def analyze_split(dataset, split_name):
    """
    Analyze one dataset split.
    """

    contexts = []
    questions = []
    answers = []

    malformed = 0

    for example in dataset[split_name]:

        input_text = example["inputs"]
        target_text = example["targets"]

        context, question, _ = parse_example(input_text)
        answer = extract_answer(target_text)

        if not question or not answer:
            malformed += 1

        contexts.append(context)
        questions.append(question)
        answers.append(answer)

    # Length statistics
    context_words = [word_count(x) for x in contexts]
    question_words = [word_count(x) for x in questions]
    answer_words = [word_count(x) for x in answers]

    context_chars = [char_count(x) for x in contexts]
    question_chars = [char_count(x) for x in questions]
    answer_chars = [char_count(x) for x in answers]

    # Duplicate questions
    question_counts = Counter(questions)
    duplicate_questions = sum(
        1 for count in question_counts.values() if count > 1
    )

    duplicated_examples = sum(
        count - 1 for count in question_counts.values() if count > 1
    )

    print(f"\n{'=' * 60}")
    print(f"{split_name.upper()} ANALYSIS")
    print(f"{'=' * 60}")

    print(f"Examples: {len(dataset[split_name])}")

    print("\n--- Word Statistics ---")

    print(
        f"Context words: "
        f"min={min(context_words)}, "
        f"max={max(context_words)}, "
        f"avg={sum(context_words) / len(context_words):.2f}"
    )

    print(
        f"Question words: "
        f"min={min(question_words)}, "
        f"max={max(question_words)}, "
        f"avg={sum(question_words) / len(question_words):.2f}"
    )

    print(
        f"Answer words: "
        f"min={min(answer_words)}, "
        f"max={max(answer_words)}, "
        f"avg={sum(answer_words) / len(answer_words):.2f}"
    )

    print("\n--- Character Statistics ---")

    print(
        f"Context characters: "
        f"min={min(context_chars)}, "
        f"max={max(context_chars)}, "
        f"avg={sum(context_chars) / len(context_chars):.2f}"
    )

    print(
        f"Question characters: "
        f"min={min(question_chars)}, "
        f"max={max(question_chars)}, "
        f"avg={sum(question_chars) / len(question_chars):.2f}"
    )

    print(
        f"Answer characters: "
        f"min={min(answer_chars)}, "
        f"max={max(answer_chars)}, "
        f"avg={sum(answer_chars) / len(answer_chars):.2f}"
    )

    print("\n--- Data Quality ---")

    print(f"Examples with missing question/answer: {malformed}")

    print(f"Unique questions: {len(question_counts)}")

    print(f"Questions appearing more than once: {duplicate_questions}")

    print(f"Additional duplicated examples: {duplicated_examples}")

    # Empty fields
    empty_contexts = sum(1 for x in contexts if not x)
    empty_questions = sum(1 for x in questions if not x)
    empty_answers = sum(1 for x in answers if not x)

    print(f"Empty contexts: {empty_contexts}")
    print(f"Empty questions: {empty_questions}")
    print(f"Empty answers: {empty_answers}")

    return {
        "examples": len(dataset[split_name]),
        "avg_context_words": sum(context_words) / len(context_words),
        "avg_question_words": sum(question_words) / len(question_words),
        "avg_answer_words": sum(answer_words) / len(answer_words),
        "duplicate_questions": duplicate_questions,
        "duplicated_examples": duplicated_examples,
        "malformed": malformed,
    }


def check_cross_split_leakage(dataset):
    """
    Check whether identical questions occur across splits.
    """

    print(f"\n{'=' * 60}")
    print("CROSS-SPLIT LEAKAGE CHECK")
    print(f"{'=' * 60}")

    split_questions = {}

    for split in ["train", "validation", "test"]:

        questions = set()

        for example in dataset[split]:
            _, question, _ = parse_example(example["inputs"])

            if question:
                questions.add(question)

        split_questions[split] = questions

        print(f"{split}: {len(questions)} unique questions")

    train_val = split_questions["train"] & split_questions["validation"]
    train_test = split_questions["train"] & split_questions["test"]
    val_test = split_questions["validation"] & split_questions["test"]

    print("\nOverlapping questions:")

    print(f"Train ∩ Validation: {len(train_val)}")
    print(f"Train ∩ Test: {len(train_test)}")
    print(f"Validation ∩ Test: {len(val_test)}")

    if not train_val and not train_test and not val_test:
        print("\nNo identical questions detected across splits.")
    else:
        print("\nWARNING: Potential data leakage detected.")


def main():

    print("Loading AmQA dataset...")

    dataset = load_dataset(DATASET_NAME)

    print("\nDataset loaded successfully.")

    print(dataset)

    results = {}

    for split in ["train", "validation", "test"]:
        results[split] = analyze_split(dataset, split)

    check_cross_split_leakage(dataset)

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")

    print(
        f"{'Split':<12}"
        f"{'Examples':>10}"
        f"{'Avg Context':>15}"
        f"{'Avg Question':>15}"
        f"{'Avg Answer':>15}"
    )

    print("-" * 67)

    for split, result in results.items():

        print(
            f"{split:<12}"
            f"{result['examples']:>10}"
            f"{result['avg_context_words']:>15.2f}"
            f"{result['avg_question_words']:>15.2f}"
            f"{result['avg_answer_words']:>15.2f}"
        )

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
