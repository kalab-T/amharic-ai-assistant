from datasets import load_dataset

print("Downloading AmQA dataset...")

dataset = load_dataset("Henok/amharic-qa")

print("\nDataset downloaded successfully!")
print(dataset)

for split in dataset:
    print(f"\n{split}:")
    print(f"Number of examples: {len(dataset[split])}")
    print(f"Columns: {dataset[split].column_names}")

    if len(dataset[split]) > 0:
        print("\nFirst example:")
        print(dataset[split][0])
