

# Face Side-View Keypoint Prediction

## Research-Grade Modular Codebase Documentation



## 1. Overview

This repository implements a **production-grade, research-oriented pipeline** for **face keypoint prediction**, with a strong focus on:

* **Correctness-first modeling**
* **Strict experimental isolation**
* **Modular preprocessing pipelines**
* **Robust cross-validation**
* **End-to-end inference latency measurement**
* **Reproducible benchmarking across multiple preprocessing strategies**

The system is designed to scale from local machines to **DGX A100–class servers**, while maintaining deterministic behavior and detailed logging.



## 2. Key Capabilities

The codebase supports:

* Multiple **preprocessing pipelines** (YOLO hybrid, DNN SSD, contrast-enhanced variants)
* **Heatmap-based keypoint regression**
* **Soft-Argmax coordinate extraction**
* **Hybrid loss (heatmap + coordinate)**
* **PCK@0.05 and PCK@0.02 metrics**
* **Per-keypoint PCK analysis**
* **N-fold cross-validation (configurable)**
* **Early stopping**
* **Mixed-precision training (AMP)**
* **Ensemble inference across folds**
* **True end-to-end inference latency measurement**
* **Comprehensive visualizations**
* **Benchmark suite for multi-pipeline comparison**



## 3. Repository Structure

```text
.
├── main_experiment.py        # Primary CLI entry point
├── src/                      # Active implementation
│   ├── config.py             # Global configuration
│   ├── utils.py              # Utilities (logging, seeding, IO)
│   ├── dataset.py            # Dataset + augmentation + heatmaps
│   ├── preprocess.py         # Preprocessing pipelines (v1–v4)
│   ├── model.py              # Model + SoftArgmax
│   ├── loss.py               # Losses + metrics
│   ├── train.py              # Training & validation logic
│   ├── infer.py              # End-to-end inference & ensemble
│   └── visualize.py          # All plotting & reporting
├── src_FREEZE_V1/             # Frozen reference implementation
└── experiments/              # Outputs (checkpoints, logs, viz)
```



## 4. Experimental Philosophy

This project follows **research-grade principles**:

### 4.1 Correctness Over Convenience

* No silent assumptions
* No hidden defaults
* No test leakage
* No auto-tuning during final training

### 4.2 Determinism

* Explicit seeding
* Controlled randomness
* Stable folds
* Reproducible results

### 4.3 Visibility as First-Class

* All metrics logged
* All plots saved
* Intermediate artifacts preserved
* Overlay visualizations for qualitative inspection

### 4.4 Isolation of Variables

When benchmarking preprocessing strategies:

* Same dataset splits
* Same model architecture
* Same hyperparameters
* Same training logic

Only the **preprocessing pipeline** changes.



## 5. Entry Point: `main_experiment.py`

`main_experiment.py` is the **only script the user runs directly**.

It exposes **three operating modes**:

| Mode              | Purpose                                       |
| ----------------- | --------------------------------------------- |
| `train`           | Train a single pipeline                       |
| `infer_raw_e2e`   | Run end-to-end inference on test data         |
| `benchmark_suite` | Train + infer multiple pipelines sequentially |



## 6. CLI Usage Overview

### 6.1 General CLI Pattern

```bash
python main_experiment.py \
  --mode <mode> \
  --pipeline <v1|v2|v3|v4> \
  [other arguments...]
```



## 7. CLI Arguments (Global)

These arguments are available across modes:

| Argument                | Description                                                  |
| ----------------------- | ------------------------------------------------------------ |
| `--mode`                | Execution mode (`train`, `infer_raw_e2e`, `benchmark_suite`) |
| `--pipeline`            | Preprocessing pipeline (`v1`, `v2`, `v3`, `v4`)              |
| `--folds`               | Number of CV folds                                           |
| `--epochs`              | Maximum epochs per fold                                      |
| `--early_stop_patience` | Early stopping patience                                      |
| `--lr`                  | Learning rate                                                |
| `--weight_decay`        | AdamW weight decay                                           |
| `--alpha`               | Heatmap vs coordinate loss weight                            |
| `--sigma`               | Heatmap Gaussian sigma                                       |
| `--softargmax_T`        | SoftArgmax temperature                                       |
| `--batch_size`          | Training batch size                                          |
| `--num_workers`         | DataLoader workers                                           |
| `--suite_root`          | Output directory for benchmark suite                         |
| `--pipelines`           | Subset of pipelines for suite                                |



## 8. Supported Preprocessing Pipelines

| Pipeline | Description                                         |
| -------- | --------------------------------------------------- |
| `v1`     | Hybrid YOLO det → ROI expand → YOLO seg → grayscale |
| `v2`     | v1 + CLAHE contrast enhancement                     |
| `v3`     | OpenCV DNN SSD ROI only (no segmentation)           |
| `v4`     | v3 + CLAHE contrast enhancement                     |

Each pipeline is **fully retrained**—no reuse of weights across pipelines.



## 9. Training Mode (`--mode train`)

### Purpose

Train **one preprocessing pipeline** using **N-fold cross-validation**.

### Example

```bash
python main_experiment.py --mode train --pipeline v1 --folds 5 \
  --epochs 150 --early_stop_patience 15 \
  --lr 0.0002376500626274199 \
  --weight_decay 0.000001986074040054842 \
  --sigma 2.58765228765362 \
  --alpha 0.6008870077358692 \
  --softargmax_T 9.58347264700672
```

### Outputs

Saved under:

```text
experiments/<pipeline>_experiment/
├── checkpoints/
├── logs/
├── viz/
└── cv_summary.json
```


## 10. What Happens During Training

For each fold:

1. Dataset split using `GroupKFold`
2. Dataset preprocessing + caching
3. Model initialization
4. Training loop with AMP
5. Validation per epoch
6. Early stopping
7. Best checkpoint saved
8. Per-keypoint PCK saved
9. Visualization plots generated



## 11. Metrics Computed During Training

| Metric           | Meaning                         |
| ---------------- | ------------------------------- |
| Train Loss       | Hybrid loss                     |
| Val Loss         | Hybrid loss                     |
| PCK@0.05         | Standard accuracy               |
| PCK@0.02         | High-precision accuracy         |
| Per-keypoint PCK | Individual landmark reliability |


## 12. Logging & Reproducibility

* Logs are written per fold
* Hyperparameters are logged verbatim
* Timing per epoch is recorded
* Fold histories saved as JSON
* All plots saved to disk

