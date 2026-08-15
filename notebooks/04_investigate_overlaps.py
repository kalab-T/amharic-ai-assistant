from datasets import load_dataset
from collections import defaultdict

DATASET_NAME = "Henok/amharic-qa"


def extract_question(text):
    """
    Extract the question from an AmQA input.
    """
    markers = ["ጥያቄ:", "ጥያቄ :"]

    for marker in markers:
        if marker in text:
            return text.split(marker, 1)[1].strip()

    return text.strip()


def get_overlapping_questions(train, test):
    """
    Find questions appearing in both train and test.
    """
    train_map = defaultdict(list)
    test_map = defaultdict(list)

    for i, example in enumerate(train):
        question = extract_question(example["inputs"])
        train_map[question].append(i)

    for i, example in enumerate(test):
        question = extract_question(example["inputs"])
        test_map[question].append(i)

    overlaps = sorted(set(train_map) & set(test_map))

    return overlaps, train_map, test_map


def print_example(label, example, index):
    print("\n" + "=" * 80)
    print(f"{label} INDEX: {index}")
    print("=" * 80)

    question = extract_question(example["inputs"])

    context = example["inputs"]

    # Remove the question from the displayed context
    for marker in ["ጥያቄ:", "ጥያቄ :"]:
        if marker in context:
            context = context.split(marker, 1)[0].strip()
            break

    print("\nCONTEXT:")
    print(context)

    print("\nQUESTION:")
    print(question)

    print("\nANSWER:")
    print(example["targets"])


def main():

    print("Loading AmQA dataset...")

    dataset = load_dataset(DATASET_NAME)

    print("\nDataset loaded successfully.")
    print(dataset)

    train = dataset["train"]
    test = dataset["test"]

    overlaps, train_map, test_map = get_overlapping_questions(
        train,
        test
    )

    print("\n" + "#" * 80)
    print("TRAIN / TEST OVERLAP INVESTIGATION")
    print("#" * 80)

    print(f"\nNumber of overlapping questions: {len(overlaps)}")

    for number, question in enumerate(overlaps, start=1):

        print("\n\n")
        print("#" * 80)
        print(f"OVERLAP {number} OF {len(overlaps)}")
        print("#" * 80)

        print("\nQUESTION:")
        print(question)

        train_indices = train_map[question]
        test_indices = test_map[question]

        print(f"\nTrain indices: {train_indices}")
        print(f"Test indices: {test_indices}")

        # Show every train occurrence
        for index in train_indices:
            print_example(
                "TRAIN EXAMPLE",
                train[index],
                index
            )

        # Show every test occurrence
        for index in test_indices:
            print_example(
                "TEST EXAMPLE",
                test[index],
                index
            )

        print("\n" + "-" * 80)
        print("INITIAL COMPARISON")
        print("-" * 80)

        train_answers = {
            train[index]["targets"]
            for index in train_indices
        }

        test_answers = {
            test[index]["targets"]
            for index in test_indices
        }

        print("\nTrain answer(s):")
        for answer in train_answers:
            print(f"- {answer}")

        print("\nTest answer(s):")
        for answer in test_answers:
            print(f"- {answer}")

        if train_answers == test_answers:
            print("\nSTATUS: Same answer across splits.")
        else:
            print("\nSTATUS: ⚠ CONFLICTING ANSWERS")

        print("\nIMPORTANT:")
        print(
            "Do NOT correct or delete this example automatically."
        )
        print(
            "The contexts must be manually investigated before "
            "making a dataset decision."
        )

    print("\n\n" + "#" * 80)
    print("INVESTIGATION COMPLETE")
    print("#" * 80)

    print("""
Next research step:

For every overlap, classify it as one of:

1. Valid duplicate
2. Annotation error
3. Ambiguous question
4. Different context / legitimately different answer
5. Unresolvable

Do not modify the original AmQA dataset yet.
""")


if __name__ == "__main__":
    main()
