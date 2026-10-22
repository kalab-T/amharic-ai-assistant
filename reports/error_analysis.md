
# Experiment 9 Error Analysis

## 1. Purpose

This document records the observed evaluation problems in Experiment 9 and defines a plan for investigating them.

The aim is to understand why the mT5-small model achieved 0.00% Exact Match despite recording a custom Token F1 of 22.38% on the validation split.

The causes listed below are hypotheses to investigate, not confirmed explanations.

## 2. Observed Problems

### 2.1 Exact Match remained zero

The original evaluation recorded 0.00% Exact Match at every epoch.

This means none of the generated validation answers exactly matched the reference answers after whitespace normalization.

Possible explanations include:

- Generated answers may be incorrect or incomplete.
- The model may generate special tokens or empty answers.
- Reference answers may be difficult to reproduce exactly.
- The input format may not give the model enough information.
- Preprocessing may remove important parts of the question or context.

These possibilities require direct inspection of generated predictions and their corresponding reference answers.

### 2.2 Partial overlap does not guarantee a correct answer

The original custom Token F1 increased to 22.38%, even though Exact Match remained zero.

This is possible when generated answers share some words with reference answers but do not match them completely.

The original metric uses the intersection of unique whitespace-separated tokens while dividing by the lengths of the original token lists. Repeated tokens can therefore be handled inconsistently.

**Required action:** Implement and report a standard count-aware token-level precision, recall, and F1 metric, while preserving the original result as a historical metric.

### 2.3 Long-input truncation may remove useful information

The original preprocessing retains the last 1,536 tokens when an input exceeds the configured limit.

If the question or relevant context appears at the beginning, this strategy can discard it.

**Required action:**

1. Recover the exact input-construction format.
2. Inspect whether questions occur before or after contexts.
3. Measure how many examples exceed 1,536 tokens.
4. Check whether truncation removes the question or reference-answer evidence.
5. Compare safer input formats and truncation strategies in a separate experiment.

No conclusion about the actual number of affected examples should be made until the data has been inspected.

### 2.4 Special-token output needs investigation

Earlier experiments in this project encountered the output `<extra_id_0>`. The available Experiment 9 metrics do not establish whether this output occurred in Experiment 9.

**Required action:** Generate predictions from the restored checkpoint and inspect raw decoded outputs before removing special tokens. Record whether outputs are empty, contain special tokens, repeat phrases, or provide plausible answers.

### 2.5 Checkpoint recovery is incomplete

The checkpoint inspection found model weights and tokenizer/configuration files in Google Drive, including those in `checkpoint-916`.

However, the `final_best_model` directory was empty during inspection, and successful checkpoint loading has not yet been confirmed.

**Required action:** Load `checkpoint-916` in Google Colab, verify model and tokenizer initialization, and generate an answer for a small sample before running a full test evaluation.

## 3. Error Categories for Manual Inspection

After generating predictions, classify each incorrect answer using one or more categories.

| Category | Description |
|---|---|
| Empty output | The model generates no meaningful answer. |
| Special-token output | The answer contains unexpected model tokens. |
| Wrong answer | The output does not answer the question. |
| Partial answer | The output contains only part of the expected answer. |
| Unsupported answer | The output contains information not supported by the input evidence. |
| Truncation-related | The question or relevant evidence appears to be missing from the model input. |
| Reference mismatch | The generated answer may be reasonable but differs in wording from the reference. |
| Other | The error does not fit the categories above. |

Do not assign error counts until actual predictions have been reviewed.

## 4. Recommended Error-Analysis Procedure

1. Restore the best recorded checkpoint and verify that it can generate outputs.
2. Select validation examples for initial inspection.
3. Save each question, relevant input context, reference answer, raw prediction, and normalized prediction.
4. Calculate Exact Match and standard count-aware token F1.
5. Manually categorize incorrect predictions.
6. Summarize the frequency of each error category.
7. Review whether errors are associated with input length, answer length, or missing evidence.
8. Repeat the final evaluation on the held-out test split only after the evaluation procedure is finalized.

Validation examples can be used to develop and debug the method. Final reported test results should be produced separately and should not be used repeatedly to tune the model.

## 5. Research Implications

These issues are important for low-resource Amharic question answering because limited training data, tokenization behavior, input formatting, and reference-answer variation can complicate evaluation.

A defensible research report should distinguish between measured results and proposed explanations. Improvements should be supported by reproducible experiments rather than assumptions.

## 6. Current Status

- [x] Four training epochs recorded.
- [x] Validation metrics documented.
- [x] Checkpoint files located in Google Drive.
- [ ] Best checkpoint successfully reloaded.
- [ ] Generation smoke test completed.
- [ ] Standard evaluation metrics implemented.
- [ ] Held-out test split evaluated.
- [ ] Manual error categories populated with evidence.
- [ ] TF-IDF, BM25, and transformer comparisons completed consistently.

## 7. Conclusion

Experiment 9 provides a useful starting point, but its zero Exact Match score, nonstandard custom Token F1 implementation, potentially destructive truncation behavior, and unverified checkpoint recovery limit the conclusions that can currently be drawn.

The next stage is to restore and test the checkpoint, improve evaluation, and examine real predictions. This will make the project more scientifically credible and more useful as evidence of research potential.
