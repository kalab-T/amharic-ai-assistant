import json
import os
import time

from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    DefaultDataCollator,
)


# ======================================================================
# CONFIGURATION
# ======================================================================

DATASET_PATH = "data/processed/amqa_structured"
OUTPUT_DIR = "models/transformer_qa"

# Multilingual transformer model suitable for Amharic QA.
MODEL_NAME = "deepset/xlm-roberta-base-squad2"

# Reduced for CPU/RAM-constrained local training.
MAX_LENGTH = 256
DOC_STRIDE = 64

# Conservative first experiment for the local laptop.
NUM_EPOCHS = 1
BATCH_SIZE = 1
LEARNING_RATE = 3e-5


# ======================================================================
# LOAD DATASET
# ======================================================================

def load_dataset():
    print("=" * 70)
    print("TRANSFORMER QUESTION ANSWERING")
    print("=" * 70)
    print()

    print("Loading structured AMQA dataset...")

    dataset = load_from_disk(DATASET_PATH)

    print(f"Train      : {len(dataset['train'])}")
    print(f"Validation : {len(dataset['validation'])}")
    print(f"Test       : {len(dataset['test'])}")
    print()

    return dataset


# ======================================================================
# TOKENIZATION
# ======================================================================

def prepare_features(examples, tokenizer):

    questions = [
        q.strip()
        for q in examples["question"]
    ]

    contexts = [
        c.strip()
        for c in examples["context"]
    ]

    tokenized = tokenizer(
        questions,
        contexts,
        max_length=MAX_LENGTH,
        truncation="only_second",
        stride=DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,

        # Dynamic padding saves memory compared with
        # padding every example to MAX_LENGTH.
        padding=False,
    )

    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):

        input_ids = tokenized["input_ids"][i]

        # --------------------------------------------------------------
        # Find CLS token.
        # --------------------------------------------------------------

        cls_token_id = tokenizer.cls_token_id

        if cls_token_id is not None and cls_token_id in input_ids:
            cls_index = input_ids.index(cls_token_id)
        else:
            cls_index = 0

        sequence_ids = tokenized.sequence_ids(i)

        sample_index = sample_mapping[i]

        answer = examples["answer"][sample_index]
        context = contexts[sample_index]

        # --------------------------------------------------------------
        # Find the answer inside the context.
        # --------------------------------------------------------------

        answer_text = answer.strip()

        answer_start = context.find(answer_text)

        # Some answers may contain additional introductory wording.
        # If the complete answer is not found, try removing common
        # introductory phrases.
        if answer_start == -1:

            prefixes = [
                "ከጥያቄው አንጻር ትክክለኛው መልስ ",
                "ከጥያቄው አንጻር ትክክለኛው መልስ፡ ",
                "ትክክለኛው መልስ ",
                "መልስ፡ ",
            ]

            cleaned_answer = answer_text

            for prefix in prefixes:
                if cleaned_answer.startswith(prefix):
                    cleaned_answer = cleaned_answer[len(prefix):].strip()
                    break

            answer_start = context.find(cleaned_answer)

            if answer_start != -1:
                answer_text = cleaned_answer

        # --------------------------------------------------------------
        # If answer cannot be found, use CLS position.
        # --------------------------------------------------------------

        if answer_start == -1:

            start_positions.append(cls_index)
            end_positions.append(cls_index)

            continue

        answer_end = answer_start + len(answer_text)

        # --------------------------------------------------------------
        # Find context token boundaries.
        # --------------------------------------------------------------

        token_start_index = 0

        while (
            token_start_index < len(sequence_ids)
            and sequence_ids[token_start_index] != 1
        ):
            token_start_index += 1

        token_end_index = len(sequence_ids) - 1

        while (
            token_end_index >= 0
            and sequence_ids[token_end_index] != 1
        ):
            token_end_index -= 1

        # --------------------------------------------------------------
        # Answer is not inside this feature's context window.
        # --------------------------------------------------------------

        if (
            token_start_index > token_end_index
            or offsets[token_start_index][0] > answer_start
            or offsets[token_end_index][1] < answer_end
        ):

            start_positions.append(cls_index)
            end_positions.append(cls_index)

            continue

        # --------------------------------------------------------------
        # Find token containing answer start.
        # --------------------------------------------------------------

        while (
            token_start_index <= token_end_index
            and offsets[token_start_index][0] <= answer_start
        ):
            token_start_index += 1

        start_position = token_start_index - 1

        # --------------------------------------------------------------
        # Find token containing answer end.
        # --------------------------------------------------------------

        while (
            token_end_index >= token_start_index
            and offsets[token_end_index][1] >= answer_end
        ):
            token_end_index -= 1

        end_position = token_end_index + 1

        start_positions.append(start_position)
        end_positions.append(end_position)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions

    return tokenized


# ======================================================================
# MAIN
# ======================================================================

def main():

    # --------------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------------

    dataset = load_dataset()

    # --------------------------------------------------------------
    # Load tokenizer
    # --------------------------------------------------------------

    print("=" * 70)
    print("LOADING TOKENIZER")
    print("=" * 70)

    print(f"Model: {MODEL_NAME}")
    print()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    print("Tokenizer loaded successfully.")
    print()

    # --------------------------------------------------------------
    # Tokenize dataset
    # --------------------------------------------------------------

    print("=" * 70)
    print("TOKENIZING DATASET")
    print("=" * 70)

    tokenized_dataset = dataset.map(
        lambda examples: prepare_features(
            examples,
            tokenizer
        ),
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing",
    )

    print()
    print("Tokenization completed.")

    print()
    print("Tokenized sizes:")
    print(f"Train      : {len(tokenized_dataset['train'])}")
    print(f"Validation : {len(tokenized_dataset['validation'])}")
    print(f"Test       : {len(tokenized_dataset['test'])}")
    print()

    # --------------------------------------------------------------
    # Load model
    # --------------------------------------------------------------

    print("=" * 70)
    print("LOADING TRANSFORMER MODEL")
    print("=" * 70)

    model = AutoModelForQuestionAnswering.from_pretrained(
        MODEL_NAME
    )

    print("Transformer model loaded successfully.")
    print()

    # --------------------------------------------------------------
    # Training configuration
    # --------------------------------------------------------------

    print("=" * 70)
    print("TRAINING CONFIGURATION")
    print("=" * 70)

    print(f"Model           : {MODEL_NAME}")
    print(f"Epochs          : {NUM_EPOCHS}")
    print(f"Batch size      : {BATCH_SIZE}")
    print(f"Learning rate   : {LEARNING_RATE}")
    print(f"Max length      : {MAX_LENGTH}")
    print(f"Document stride : {DOC_STRIDE}")
    print("Padding         : Dynamic")
    print("Device          : CPU")
    print()

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,

        eval_strategy="epoch",

        save_strategy="epoch",

        learning_rate=LEARNING_RATE,

        per_device_train_batch_size=BATCH_SIZE,

        per_device_eval_batch_size=BATCH_SIZE,

        num_train_epochs=NUM_EPOCHS,

        weight_decay=0.01,

        logging_steps=50,

        save_total_limit=2,

        load_best_model_at_end=True,

        metric_for_best_model="eval_loss",

        greater_is_better=False,

        report_to="none",

        # CPU training.
        fp16=False,
    )

    data_collator = DefaultDataCollator()

    # --------------------------------------------------------------
    # Transformers 5.x compatibility:
    # processing_class replaces tokenizer=tokenizer.
    # --------------------------------------------------------------

    trainer = Trainer(
        model=model,

        args=training_args,

        train_dataset=tokenized_dataset["train"],

        eval_dataset=tokenized_dataset["validation"],

        processing_class=tokenizer,

        data_collator=data_collator,
    )

    # --------------------------------------------------------------
    # Train
    # --------------------------------------------------------------

    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    start_time = time.time()

    trainer.train()

    elapsed = time.time() - start_time

    print()
    print("=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)

    print(f"Training time: {elapsed / 60:.2f} minutes")
    print()

    # --------------------------------------------------------------
    # Save model
    # --------------------------------------------------------------

    final_model_path = os.path.join(
        OUTPUT_DIR,
        "final"
    )

    trainer.save_model(
        final_model_path
    )

    tokenizer.save_pretrained(
        final_model_path
    )

    print("Model saved to:")
    print(final_model_path)
    print()

    # --------------------------------------------------------------
    # Evaluate validation
    # --------------------------------------------------------------

    print("=" * 70)
    print("VALIDATION EVALUATION")
    print("=" * 70)

    validation_metrics = trainer.evaluate(
        tokenized_dataset["validation"]
    )

    for key, value in validation_metrics.items():
        print(f"{key}: {value}")

    # --------------------------------------------------------------
    # Save training information
    # --------------------------------------------------------------

    results = {
        "model_name": MODEL_NAME,
        "dataset": DATASET_PATH,

        "train_examples": len(
            dataset["train"]
        ),

        "validation_examples": len(
            dataset["validation"]
        ),

        "test_examples": len(
            dataset["test"]
        ),

        "max_length": MAX_LENGTH,
        "doc_stride": DOC_STRIDE,
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,

        "padding": "dynamic",

        "device": "cpu",

        "training_time_minutes": elapsed / 60,

        "validation_metrics": validation_metrics,

        "model_path": final_model_path,
    }

    results_path = os.path.join(
        OUTPUT_DIR,
        "training_results.json"
    )

    with open(
        results_path,
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

    print(f"Model   : {final_model_path}")
    print(f"Results : {results_path}")
    print()

    print("Transformer training completed successfully.")


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    main()
