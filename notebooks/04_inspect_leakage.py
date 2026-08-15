from datasets import load_dataset
from collections import defaultdict


DATASET_NAME = "Henok/amharic-qa"


def normalize_text(text):
    """Normalize whitespace for comparison."""
    if text is None:
        return ""

    return " ".join(str(text).strip().split())


def extract_question(text):
    """Extract the question from the AmQA input field."""

    text = str(text)

    markers = [
        "ጥያቄ:",
        "ጥያቄ :",
        "ጥያቄ፡",
        "ጥያቄ፡ "
    ]

    for marker in markers:
        if marker in text:
            return normalize_text(text.split(marker, 1)[1])

    return normalize_text(text)


def build_question_map(split):
    """Map each question to its dataset indices."""

    question_map = defaultdict(list)

    for index, example in enumerate(split):
        question = extract_question(example["inputs"])
        question_map[question].append(index)

    return question_map


def main():

    print("Loading AmQA dataset...")

    dataset = load_dataset(DATASET_NAME)

    train = dataset["train"]
    test = dataset["test"]

    train_map = build_question_map(train)
    test_map = build_question_map(test)

    overlapping_questions = (
        set(train_map.keys())
        .intersection(set(test_map.keys()))
    )

    print("\n" + "=" * 70)
    print("TRAIN / TEST OVERLAPPING QUESTIONS")
    print("=" * 70)

    print(f"\nNumber of overlapping questions: {len(overlapping_questions)}")

    for number, question in enumerate(
        sorted(overlapping_questions), start=1
    ):

        print("\n" + "#" * 70)
        print(f"CASE {number}")
        print("#" * 70)

        print("\nQUESTION:")
        print(question)

        # --------------------------------------------------
        # TRAIN examples
        # --------------------------------------------------

        print("\n" + "-" * 70)
        print("TRAIN EXAMPLE")
        print("-" * 70)

        for index in train_map[question]:

            example = train[index]

            full_input = example["inputs"]
            answer = example["targets"]

            context = full_input

            for marker in [
                "ጥያቄ:",
                "ጥያቄ :",
                "ጥያቄ፡",
                "ጥያቄ፡ "
            ]:
                if marker in full_input:
                    context = full_input.split(marker, 1)[0]
                    break

            print(f"\nTrain index: {index}")

            print("\nCONTEXT:")
            print(context.strip())

            print("\nANSWER:")
            print(answer)

        # --------------------------------------------------
        # TEST examples
        # --------------------------------------------------

        print("\n" + "-" * 70)
        print("TEST EXAMPLE")
        print("-" * 70)

        for index in test_map[question]:

            example = test[index]

            full_input = example["inputs"]
            answer = example["targets"]

            context = full_input

            for marker in [
                "ጥያቄ:",
                "ጥያቄ :",
                "ጥያቄ፡",
                "ጥያቄ፡ "
            ]:
                if marker in full_input:
                    context = full_input.split(marker, 1)[0]
                    break

            print(f"\nTest index: {index}")

            print("\nCONTEXT:")
            print(context.strip())

            print("\nANSWER:")
            print(answer)

    print("\n" + "=" * 70)
    print("INSPECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
