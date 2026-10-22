
# Amharic AI Assistant: Question Answering for a Low-Resource Language

A research-oriented NLP project investigating question answering for Amharic using information retrieval methods and multilingual transformer models.

**Research direction:** Transformer-Based Question Answering for Low-Resource Amharic
**Primary transformer:** `google/mt5-small`
**Project status:** In progress — baseline experiments and transformer training have been explored; checkpoint verification and held-out test evaluation remain pending.

## 1. Research Motivation

Amharic is a language with fewer publicly available NLP resources than many high-resource languages. Building reliable question-answering systems for Amharic presents challenges in dataset preparation, tokenization, model training, evaluation, and answer generation.

This project aims to investigate these challenges through reproducible experiments, comparisons between retrieval-based baselines and a generative transformer, and analysis of model errors.

The project is intended to develop into a documented research portfolio project, not to claim a solved question-answering task.

## 2. Research Questions

The project investigates the following questions:

1. How well do retrieval-based methods such as TF-IDF and BM25 perform on the prepared Amharic question-answering dataset?
2. What results does a fine-tuned multilingual transformer achieve under the same evaluation conditions?
3. How do input construction, tokenization, truncation, and answer length affect generated answers?
4. Which error patterns limit question-answering performance, and how can they be measured reproducibly?

These questions may be refined as the experiments progress.

## 3. Dataset

The recorded Experiment 9 run used the following split sizes:

| Split | Examples |
|---|---:|
| Training | 1,831 |
| Validation | 263 |
| Test | 523 |
| **Total** | **2,617** |

The historical training script expects a prepared dataset with `inputs` and `targets` columns. The full data-loading and input-construction workflow still needs to be documented.

Dataset access, licensing, provenance, and redistribution permissions should be confirmed before publishing the dataset or detailed examples. Do not commit private, restricted, or unnecessarily large data files.

## 4. Methods and Experiments

### Retrieval baselines

The repository contains scripts for exploring retrieval-based approaches:

- `scripts/07_tfidf_baseline.py`
- `scripts/08_bm25_baseline.py`
- `scripts/09_compare_baselines.py`

Their outputs should be verified and evaluated under consistent conditions before reporting a definitive comparison.

### Transformer experiment

Experiment 9 fine-tuned `google/mt5-small` using Google Colab.

Recorded configuration:

- Random seed: 42
- Maximum input length: 1,536 tokens
- Maximum target length: 64 tokens
- Training epochs: 4
- Learning rate: `2e-5`
- Training batch size per device: 2
- Evaluation batch size per device: 4
- Gradient accumulation steps: 4
- Generation beams: 4

The historical script is available at `scripts/10_train_transformer_exp9_colab.py`.

**Important:** This file documents the original training run. It is not standalone: it depends on a preloaded `dataset` variable and Google Colab. Do not run it as-is in a local environment.

## 5. Experiment 9: Recorded Validation Results

| Epoch | Validation loss | Exact Match | Custom Token F1 |
|---:|---:|---:|---:|
| 1 | 2.30186 | 0.00% | 15.02% |
| 2 | 1.45773 | 0.00% | 21.72% |
| 3 | 1.23473 | 0.00% | 22.00% |
| 4 | 1.17088 | 0.00% | 22.38% |

These are the results recorded during the original validation procedure.

The custom Token F1 uses set overlap rather than standard count-aware token-level F1. It should not be interpreted as a standard F1 score or compared directly with results calculated using a different metric.

Exact Match remained at 0.00%. The held-out test set has not yet been evaluated, so no test performance is claimed.

See [`reports/evaluation_report.md`](reports/evaluation_report.md) for details.

## 6. Checkpoint Status

The original Google Drive inspection found model weights and tokenizer/configuration files in the checkpoint directories, including `checkpoint-916`.

`checkpoint-916` had the highest recorded validation custom Token F1 at 22.38%. However, the `final_best_model` directory was empty during a later inspection, and successful checkpoint reloading and generation have not yet been verified.

The model should not be described as successfully restored or ready for deployment until those checks pass.

## 7. Error Analysis

The current investigation focuses on:

- Zero Exact Match despite partial token overlap
- Limitations of the original custom Token F1 implementation
- Potential loss of questions or context through long-input truncation
- Inspection of generated answers and unexpected special-token outputs
- Checkpoint recovery and reproducibility
- Consistent evaluation against retrieval baselines

See [`reports/error_analysis.md`](reports/error_analysis.md).

## 8. Repository Structure

```text
amharic-ai-assistant/
├── data/
│   ├── README.md
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
│   ├── 01_dataset_exploration.py
│   ├── 02_dataset_statistics.py
│   ├── 03_check_data_leakage.py
│   ├── 04_inspect_leakage.py
│   ├── 04_investigate_overlaps.py
│   └── 05_classify_overlaps.py
├── reports/
│   ├── evaluation_report.md
│   └── error_analysis.md
├── scripts/
│   ├── 05_clean_amqa.py
│   ├── 06_prepare_amqa.py
│   ├── 07_tfidf_baseline.py
│   ├── 08_bm25_baseline.py
│   ├── 09_compare_baselines.py
│   └── 10_train_transformer_exp9_colab.py
├── requirements.txt
├── .gitignore
└── README.md
```

The listing above highlights the main research files; other existing repository files may also be present.

## 9. Reproducibility and Environment

The transformer experiment was run in Google Colab because local hardware and memory constraints make training a large multilingual transformer difficult.

The historical training script does not contain all steps needed to reproduce the experiment from a fresh environment. To improve reproducibility, the project still needs:

1. Documented dataset acquisition and preparation.
2. A documented input format and truncation strategy.
3. A verified checkpoint-loading procedure.
4. A standard evaluation implementation.
5. Held-out test evaluation.
6. Saved prediction samples and documented error categories.
7. A verified dependency list and execution instructions.

Install dependencies in a suitable Python environment using:

```bash
python -m pip install -r requirements.txt
```

This command installs the packages listed in `requirements.txt`; it does not by itself reproduce the historical Colab run.

## 10. Current Limitations

- Exact Match was 0.00% during the recorded validation evaluations.
- The original custom Token F1 is not standard count-aware F1.
- The test split has not been evaluated.
- Long-input truncation may discard important input information.
- The complete dataset preparation workflow is not yet documented.
- Checkpoint loading and answer generation remain unverified.
- A consistent, completed comparison against TF-IDF and BM25 has not yet been established.

These limitations are reported transparently and form part of the ongoing research work.

## 11. Next Steps

1. Restore and verify `checkpoint-916` in Google Colab.
2. Run a small generation smoke test.
3. Inspect raw predictions and reference answers.
4. Implement standard evaluation metrics.
5. Evaluate the final model on the held-out test split.
6. Complete consistent TF-IDF, BM25, and transformer comparisons.
7. Investigate truncation and input construction.
8. Document reproducible procedures and evidence-based conclusions.

## 12. Research Integrity

Reported numbers in this README are historical validation results from the original run. They are not test results, deployment benchmarks, or evidence that the model reliably answers Amharic questions.

Future improvements should be supported by recorded experiments and reproducible evaluation.

---

**Author:** Kalab Tesfaneh
**GitHub:** [kalab-T](https://github.com/kalab-T)
