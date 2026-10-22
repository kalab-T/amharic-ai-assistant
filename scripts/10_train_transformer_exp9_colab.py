
"""
Experiment 9: Transformer-Based Question Answering for Low-Resource Amharic
============================================================================

Status:
    Historical record of a Google Colab training run.

Model:
    google/mt5-small

Experiment configuration:
    Random seed: 42
    Maximum input length: 1536 tokens
    Maximum target length: 64 tokens
    Training epochs: 4
    Learning rate: 2e-5
    Per-device training batch size: 2
    Per-device evaluation batch size: 4
    Gradient accumulation steps: 4
    Weight decay: 0.01
    Generation beams: 4

Recorded validation results:
    Epoch 1: loss 2.30186, custom Token F1 15.02%
    Epoch 2: loss 1.45773, custom Token F1 21.72%
    Epoch 3: loss 1.23473, custom Token F1 22.00%
    Epoch 4: loss 1.17088, custom Token F1 22.38%

    Exact Match remained 0.00% at every recorded epoch.

IMPORTANT LIMITATIONS:
    1. This file is a historical record, not a standalone script.
    2. It expects a preloaded Hugging Face DatasetDict named `dataset`
       containing `train` and `validation` splits with `inputs` and
       `targets` columns. The original dataset-loading and
       input-construction steps have not yet been recovered.
    3. For inputs longer than 1536 tokens, the original code keeps the
       final 1536 tokens and discards the beginning. If the question
       appears at the beginning, it may be lost.
    4. The custom Token F1 uses set overlap, not standard
       count-aware token-level F1.
    5. The reported metrics are validation results, NOT test results.
       The held-out test split still needs to be evaluated.
    6. The original run left `final_best_model` empty in Google Drive.
       Model weights were found in the checkpoint directories.
       The usability of the saved model has not yet been verified.

Do not run this file as-is. It imports Google Colab modules and
depends on the original prepared dataset.
"""

# ============================================================================
# 1. Imports
# ============================================================================

from google.colab import drive
drive.mount("/content/drive")

import os
import torch
import numpy as np

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    set_seed,
)


# ============================================================================
# 2. Google Drive paths
# ============================================================================

DRIVE_ROOT = "/content/drive/MyDrive/amharic_qa_exp9_mt5"

CHECKPOINT_DIR = os.path.join(DRIVE_ROOT, "checkpoints")
FINAL_MODEL_DIR = os.path.join(DRIVE_ROOT, "final_best_model")

os.makedirs(DRIVE_ROOT, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(FINAL_MODEL_DIR, exist_ok=True)

print("Experiment directory:", DRIVE_ROOT)
print("Checkpoint directory:", CHECKPOINT_DIR)
print("Final model directory:", FINAL_MODEL_DIR)


# ============================================================================
# 3. Reproducibility and sequence configuration
# ============================================================================

SEED = 42
MAX_INPUT = 1536
MAX_TARGET = 64

set_seed(SEED)

print("Random seed:", SEED)
print("Maximum input length:", MAX_INPUT)
print("Maximum target length:", MAX_TARGET)


# ============================================================================
# 4. Load the model and tokenizer
# ============================================================================

MODEL_NAME = "google/mt5-small"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

print("Loaded model:", MODEL_NAME)


# ============================================================================
# 5. Tokenization
# ============================================================================
# Historical behavior:
# - Tokenize inputs without truncation first.
# - Keep the FINAL 1536 tokens if an input is too long.
# - Truncate target answers to 64 tokens.
#
# Note: Keeping the final tokens can discard the question or the beginning
# of the context. This behavior should be investigated in a future experiment.
# ============================================================================

def tokenize_function(examples):
    model_inputs = tokenizer(
        examples["inputs"],
        max_length=None,
        truncation=False,
        padding=False,
    )

    input_ids = []
    attention_mask = []

    for ids in model_inputs["input_ids"]:
        if len(ids) > MAX_INPUT:
            ids = ids[-MAX_INPUT:]

        input_ids.append(ids)
        attention_mask.append([1] * len(ids))

    labels = tokenizer(
        text_target=examples["targets"],
        max_length=MAX_TARGET,
        truncation=True,
        padding=False,
    )["input_ids"]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# ============================================================================
# 6. Prepare tokenized datasets
# ============================================================================
# `dataset` must already exist in the Colab runtime.
# Expected structure:
# dataset["train"]
# dataset["validation"]
#
# Each split must contain `inputs` and `targets` columns.
# The dataset-loading and input-construction code is not included here.
# ============================================================================

print("\nTokenizing dataset...")

tokenized = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset["train"].column_names,
    desc="Tokenizing",
)

print("Training examples:", len(tokenized["train"]))
print("Validation examples:", len(tokenized["validation"]))


# ============================================================================
# 7. Data collator
# ============================================================================

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    label_pad_token_id=-100,
)


# ============================================================================
# 8. Evaluation metrics
# ============================================================================
# Exact Match:
#   Percentage of normalized predictions exactly matching references.
#
# Custom Token F1:
#   Uses the intersection of UNIQUE whitespace-separated tokens, but divides
#   by the lengths of the original token lists. This is not standard,
#   count-aware token F1 and should be replaced in a future evaluation.
# ============================================================================

def compute_metrics(eval_preds):
    preds, labels = eval_preds

    if isinstance(preds, tuple):
        preds = preds[0]

    vocab_size = tokenizer.vocab_size
    preds = np.clip(preds, 0, vocab_size - 1)

    labels = np.where(
        labels != -100,
        labels,
        tokenizer.pad_token_id,
    )

    decoded_preds = tokenizer.batch_decode(
        preds,
        skip_special_tokens=True,
    )

    decoded_labels = tokenizer.batch_decode(
        labels,
        skip_special_tokens=True,
    )

    def normalize(text):
        return " ".join(text.strip().split())

    decoded_preds = [normalize(p) for p in decoded_preds]
    decoded_labels = [normalize(label) for label in decoded_labels]

    exact_match = np.mean([
        pred == label
        for pred, label in zip(decoded_preds, decoded_labels)
    ])

    f1_scores = []

    for pred, label in zip(decoded_preds, decoded_labels):
        pred_tokens = pred.split()
        label_tokens = label.split()

        common_tokens = set(pred_tokens) & set(label_tokens)

        if not pred_tokens or not label_tokens:
            f1_scores.append(0.0)
            continue

        precision = len(common_tokens) / len(pred_tokens)
        recall = len(common_tokens) / len(label_tokens)

        if precision + recall > 0:
            score = (
                2 * precision * recall
                / (precision + recall)
            )
        else:
            score = 0.0

        f1_scores.append(score)

    return {
        "exact_match": round(exact_match * 100, 2),
        "token_f1": round(float(np.mean(f1_scores)) * 100, 2),
    }


# ============================================================================
# 9. Training configuration
# ============================================================================

training_args = Seq2SeqTrainingArguments(
    output_dir=CHECKPOINT_DIR,

    eval_strategy="epoch",
    save_strategy="epoch",

    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,

    num_train_epochs=4,
    weight_decay=0.01,

    predict_with_generate=True,
    generation_max_length=MAX_TARGET,
    generation_num_beams=4,

    fp16=False,
    gradient_checkpointing=False,

    logging_steps=50,
    save_total_limit=3,

    load_best_model_at_end=True,
    metric_for_best_model="token_f1",
    greater_is_better=True,

    report_to="none",
    seed=SEED,
)


# ============================================================================
# 10. Initialize the trainer
# ============================================================================

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


# ============================================================================
# 11. Train the model
# ============================================================================
# This section records the original training procedure.
# Do not rerun it just to reproduce the report.
# ============================================================================

print("\nStarting historical Experiment 9 training...")
print("Training duration in the recorded run was approximately 27 minutes.")

train_result = trainer.train()


# ============================================================================
# 12. Evaluate on the validation split
# ============================================================================

print("\nFinal validation metrics:")

validation_metrics = trainer.evaluate()
print(validation_metrics)


# ============================================================================
# 13. Save model and tokenizer
# ============================================================================
# In the recorded run, final_best_model was later found to be empty.
# This save procedure therefore must not be assumed to have succeeded.
# ============================================================================

print("\nAttempting to save the selected model...")

trainer.save_model(FINAL_MODEL_DIR)
tokenizer.save_pretrained(FINAL_MODEL_DIR)

print("\nDirectory listing after save attempt:")
print("Checkpoints:", os.listdir(CHECKPOINT_DIR))
print("Final model directory:", os.listdir(FINAL_MODEL_DIR))

print("\nExperiment directory:", DRIVE_ROOT)


# ============================================================================
# 14. Historical outcome
# ============================================================================
# Best recorded checkpoint:
#   checkpoint-916
#
# Best recorded validation custom Token F1:
#   22.38%
#
# Best recorded validation loss:
#   1.17088
#
# Exact Match:
#   0.00%
#
# Test-set evaluation:
#   Not yet completed.
#
# Saved checkpoint weights have been observed in Google Drive, but the
# checkpoint must be loaded and verified before claiming it is usable.
# ============================================================================
