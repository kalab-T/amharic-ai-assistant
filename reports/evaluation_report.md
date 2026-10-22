
# Experiment 9 Evaluation Report

## 1. Experiment Overview

**Project:** Transformer-Based Question Answering for Low-Resource Amharic
**Experiment:** Experiment 9
**Model:** `google/mt5-small`
**Environment:** Google Colab
**Random seed:** 42
**Status:** Training completed; held-out test evaluation pending.

This experiment fine-tuned mT5-small for question answering in Amharic. The purpose was to investigate whether a multilingual sequence-to-sequence transformer could generate answers from the prepared Amharic question-answering dataset.

## 2. Dataset

The recorded dataset split sizes were:

| Split | Examples |
|---|---:|
| Training | 1,831 |
| Validation | 263 |
| Test | 523 |
| **Total** | **2,617** |

The training script expected a preloaded dataset with `inputs` and `targets` columns. The complete dataset-loading and input-construction procedure has not yet been recovered into the historical script.

The test split was tokenized during the original run, but it was not passed to the trainer for final evaluation. Therefore, no test-set performance is reported here.

## 3. Training Configuration

| Parameter | Value |
|---|---|
| Pretrained model | `google/mt5-small` |
| Maximum input length | 1,536 tokens |
| Maximum target length | 64 tokens |
| Epochs | 4 |
| Learning rate | 0.00002 |
| Training batch size per device | 2 |
| Evaluation batch size per device | 4 |
| Gradient accumulation steps | 4 |
| Weight decay | 0.01 |
| Generation beams | 4 |
| Random seed | 42 |
| Best-model selection metric | Custom Token F1 |

For inputs exceeding 1,536 tokens, the original preprocessing retained the final 1,536 tokens. This could remove the beginning of the input, including the question if it appears first. This behavior needs investigation.

## 4. Recorded Validation Results

| Epoch | Validation loss | Exact Match | Custom Token F1 |
|---:|---:|---:|---:|
| 1 | 2.30186 | 0.00% | 15.02% |
| 2 | 1.45773 | 0.00% | 21.72% |
| 3 | 1.23473 | 0.00% | 22.00% |
| 4 | 1.17088 | 0.00% | 22.38% |

The loss decreased at each recorded epoch, while the custom Token F1 increased from 15.02% to 22.38%. Exact Match remained at 0.00%.

These results show improvement according to the experiment's original evaluation procedure. However, the custom Token F1 implementation uses set overlap rather than standard count-aware token F1. Its value should therefore be treated as a preliminary metric, not as a directly comparable standard F1 score.

## 5. Interpretation

The recorded results provide evidence that the training procedure changed model behavior in a direction that improved the original custom overlap metric. However, the results do not yet establish that the model reliably answers Amharic questions.

The 0.00% Exact Match result indicates that no generated validation answer exactly matched its reference after the script's normalization procedure. Partial token overlap can occur even when the complete answer is incorrect.

The experiment therefore remains preliminary. Further evaluation is needed before drawing conclusions about model quality or research effectiveness.

## 6. Checkpoint Status

The Google Drive checkpoint inspection found model weights and tokenizer/configuration files in the recorded checkpoint directories.

The recorded checkpoints were:

| Checkpoint | Global step | Custom validation Token F1 |
|---|---:|---:|
| `checkpoint-229` | 229 | 15.02% |
| `checkpoint-458` | 458 | 21.72% |
| `checkpoint-687` | 687 | 22.00% |
| `checkpoint-916` | 916 | 22.38% |

`checkpoint-916` is the best recorded checkpoint according to the original custom validation metric.

The `final_best_model` directory was found empty during a later inspection. The checkpoint weights have not yet been successfully reloaded and verified, so model usability is not yet confirmed.

## 7. Limitations

1. Exact Match was 0.00% across all four recorded epochs.
2. The original Token F1 implementation is not standard count-aware token F1.
3. The held-out test split has not been evaluated.
4. The long-input truncation strategy may discard questions or relevant context.
5. The complete dataset-loading and input-construction steps are not included in the historical script.
6. The saved checkpoint has not yet passed a reload and generation test.
7. The experiment has not yet been compared with the TF-IDF and BM25 baselines using a unified evaluation procedure.

## 8. Next Steps

1. Restore `checkpoint-916` in Google Colab and verify that the model loads successfully.
2. Run a generation smoke test on a small number of validation examples.
3. Correct the evaluation implementation and calculate standard metrics.
4. Evaluate the verified model on the held-out test split.
5. Inspect incorrect predictions and categorize the failure types.
6. Compare the transformer against TF-IDF and BM25 baselines using consistent splits and metrics.
7. Investigate input formatting and truncation so that questions are retained.
8. Document the complete preprocessing and evaluation workflow.

## 9. Conclusion

Experiment 9 completed four training epochs and recorded a best validation loss of 1.17088 and a custom Token F1 of 22.38%, with Exact Match remaining at 0.00%.

These are preliminary validation results. A verified checkpoint, corrected metrics, held-out test evaluation, baseline comparisons, and error analysis are still required before making stronger claims about the system's performance.

---

**Evidence status:** Training metrics and checkpoint file listings were recorded from the original Google Colab run. Test-set performance and successful checkpoint reloading remain unverified.
