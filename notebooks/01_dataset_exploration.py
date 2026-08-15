from datasets import load_dataset


DATASET_NAME = "Henok/amharic-qa"


def main():
    print("Loading AmQA dataset...")
    dataset = load_dataset(DATASET_NAME)

    print("\n=== Dataset Overview ===")
    print(dataset)

    for split_name, split in dataset.items():
        print(f"\n=== {split_name.upper()} ===")
        print(f"Examples: {len(split)}")
        print(f"Columns: {split.column_names}")

    # Inspect one training example
    example = dataset["train"][0]

    print("\n=== First Training Example ===")
    print("\nTARGET:")
    print(example["targets"])

    print("\nINPUT:")
    print(example["inputs"])


if __name__ == "__main__":
    main()
from datasets import load_dataset


DATASET_NAME = "Henok/amharic-qa"


def parse_example(example):
    """Separate AmQA input into context and question."""

    input_text = example["inputs"]
    target_text = example["targets"]

    marker = "ጥያቄ:"

    if marker not in input_text:
        return {
            "context": None,
            "question": None,
            "answer": target_text,
        }

    context, question = input_text.rsplit(marker, 1)

    return {
        "context": context.strip(),
        "question": question.strip(),
        "answer": target_text.strip(),
    }


def main():
    print("Loading AmQA dataset...")
    dataset = load_dataset(DATASET_NAME)

    print("\n=== Dataset Overview ===")
    print(dataset)

    for split_name, split in dataset.items():
        print(f"\n=== {split_name.upper()} ===")
        print(f"Examples: {len(split)}")

    # Parse the first example
    example = dataset["train"][0]
    parsed = parse_example(example)

    print("\n=== Parsed Example ===")

    print("\nCONTEXT:")
    print(parsed["context"])

    print("\nQUESTION:")
    print(parsed["question"])

    print("\nANSWER:")
    print(parsed["answer"])


if __name__ == "__main__":
    main()
