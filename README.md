

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

Excellent.
This is **Chunk 2** of the documentation.



# Face Side-View Keypoint Prediction

## Detailed Codebase Documentation (Chunk 2)



## 13. Global Configuration (`src/config.py`)

### Purpose

`config.py` defines **all global, experiment-wide configuration** in one place.
This file ensures:

* No magic numbers scattered across the code
* Hardware-aware defaults
* Centralized reproducibility
* Safe runtime overrides via CLI



### 13.1 Design Philosophy

* **Stateless class**: configuration is accessed via class attributes
* **Pathlib everywhere**: avoids OS-dependent path bugs
* **Runtime override allowed**: values like folds, learning rate, sigma, etc. can be overridden by CLI
* **Hardware-aware defaults**: tuned for DGX-class machines but works everywhere



### 13.2 Configuration Fields (Explained)

#### Paths

```python
DATA_ROOT
TRAIN_IMG_DIR
TRAIN_COCO
TEST_IMG_DIR
TEST_COCO
```

* `DATA_ROOT`: Root folder containing datasets
* `TRAIN_IMG_DIR`: Folder containing training images
* `TRAIN_COCO`: COCO-format annotations for training
* `TEST_IMG_DIR`: Folder containing test images
* `TEST_COCO`: Optional COCO annotations for test (if available)

All paths are **resolved once** and reused everywhere.



#### Experiment Output

```python
WORK_DIR
```

* Root directory where **all outputs** are written:

  * checkpoints
  * logs
  * visualizations
  * inference results

Each pipeline and benchmark run creates its own isolated `WORK_DIR`.



#### Randomness & Reproducibility

```python
SEED
```

* Used for:

  * Python `random`
  * NumPy
  * PyTorch (CPU & CUDA)
* Guarantees deterministic splits and behavior (within CUDA limits)



#### Model Parameters

```python
IMG_SIZE = 512
HM_SIZE = 64
NUM_KPTS = 26
SIGMA
```

* `IMG_SIZE`: Input image resolution (square)
* `HM_SIZE`: Heatmap resolution
* `NUM_KPTS`: Number of facial keypoints
* `SIGMA`: Gaussian standard deviation for heatmap generation

`sigma` is **not fixed** — it is explicitly tunable and was optimized via Optuna.



#### Training Defaults

```python
BATCH_SIZE
NUM_WORKERS
FOLDS
EPOCHS
LR
WEIGHT_DECAY
ALPHA
SOFTARGMAX_T
EARLY_STOP_PATIENCE
```

* `FOLDS`: Number of cross-validation folds
* `ALPHA`: Weight between heatmap loss and coordinate loss
* `SOFTARGMAX_T`: Temperature controlling softargmax sharpness
* `EARLY_STOP_PATIENCE`: Stops training if no improvement is seen

All of these are **runtime-overridable via CLI**.



#### Hardware Settings

```python
DEVICE
AMP_ENABLED
AMP_DTYPE
PIN_MEMORY
PERSISTENT_WORKERS
GRAD_CLIP_NORM
```

* `DEVICE`: CUDA if available, else CPU
* `AMP_ENABLED`: Enables mixed precision
* `AMP_DTYPE`: Typically `float16` on A100
* `PIN_MEMORY`: Faster host→device transfers
* `PERSISTENT_WORKERS`: Keeps DataLoader workers alive
* `GRAD_CLIP_NORM`: Optional gradient clipping



### 13.3 `Config.setup()`

This method:

* Creates necessary directories
* Ensures `WORK_DIR` exists
* Is called **once at program start**

No configuration is assumed to exist beforehand.



## 14. Utilities (`src/utils.py`)

### Purpose

`utils.py` contains **infrastructure code** used across all modules:

* Logging
* Seeding
* Atomic file writes
* Worker initialization
* Work directory switching

None of these functions perform ML logic — they support reliability.



### 14.1 Logging (`get_logger`)

```python
get_logger(name, log_file)
```

Creates a **dual-output logger**:

* Writes to a log file
* Streams to console

Features:

* Timestamped entries
* One logger per logical unit (main, fold, dataset, inference)
* Prevents silent failures

Why this matters:

* Training on long runs (DGX) requires persistent logs
* Fold-level logs allow forensic debugging



### 14.2 Seeding (`seed_everything`)

```python
seed_everything(seed, deterministic=True)
```

Sets seeds for:

* Python `random`
* NumPy
* PyTorch CPU
* PyTorch CUDA

Also configures:

* `cudnn.deterministic`
* `cudnn.benchmark`

This ensures:

* Identical fold splits
* Comparable benchmark runs
* Stable Optuna evaluations


### 14.3 DataLoader Worker Init

```python
worker_init_fn(worker_id)
```

Ensures:

* Each DataLoader worker has a deterministic seed
* Augmentations are reproducible

This prevents subtle randomness differences between runs.



### 14.4 Atomic File Writes

```python
atomic_write_text(path, text)
```

Writes files safely by:

1. Writing to a temporary file
2. Renaming it atomically

This prevents:

* Partial JSON writes
* Corrupted summaries during crashes
* Incomplete visualization metadata

Used for:

* CV summaries
* Fold histories
* Benchmark results



### 14.5 Work Directory Switching

```python
set_work_dir(path)
```

Allows the benchmark suite to:

* Dynamically change `WORK_DIR`
* Isolate outputs per pipeline
* Run v1–v4 sequentially without conflicts

This is a **critical enabler** of the benchmark suite.



## 15. Why These Utilities Matter

Without these utilities:

* Logs would overwrite each other
* Results would not be reproducible
* Benchmarks would not be comparable
* Long runs could silently fail







# Face Side-View Keypoint Prediction





## 16. Preprocessing Pipelines (`src/preprocess.py`)

### Purpose

`preprocess.py` implements **all image preprocessing logic** used by the system.

This module is **the only place** where preprocessing differs across pipelines (v1–v4).
Everything downstream (dataset, model, training, inference) remains identical.

This strict separation ensures:

* Fair benchmarking
* No code duplication
* Clear attribution of performance differences



## 17. Preprocessing Interface (Critical Design)

All pipelines implement the same interface:

```python
gray_512, meta = preprocess_image(img_bgr, pipeline)
```

### Outputs

| Output     | Description                                                |
| ---------- | ---------------------------------------------------------- |
| `gray_512` | `float32` grayscale image of shape `(512, 512)` in `[0,1]` |
| `meta`     | Metadata used to map predictions back to original image    |

This design guarantees:

* Model always sees identical input shape
* Inference overlays are geometrically correct
* Ensemble predictions are aligned



## 18. Metadata (`meta`) Explained

The `meta` dictionary typically contains:

```python
meta = {
    "orig_shape": (H, W),
    "roi": (x1, y1, x2, y2),
    "scale_x": float,
    "scale_y": float,
}
```

Used during inference to:

* Convert predicted `(x, y)` from 512-space → original image space
* Draw overlays accurately
* Compute latency per image



## 19. Pipeline v1 — Hybrid YOLO (Baseline)

### Description

This is the **primary, most accurate pipeline**.

Steps:

1. **YOLO Detection**

   * Detect person / face bounding box
2. **ROI Expansion**

   * Expand bounding box to include hair, accessories
3. **Adaptive Resolution Selection**

   * Chooses segmentation resolution based on ROI size
4. **YOLO Segmentation**

   * Segment subject inside ROI
5. **Mask Cleanup**

   * Morphological closing
   * Largest connected component selection
6. **Background Removal**

   * Apply mask to image
7. **Grayscale Conversion**
8. **Resize to 512×512**

### Why this works well

* Removes clutter
* Preserves facial geometry
* Robust to side views and occlusions



## 20. Pipeline v2 — Hybrid YOLO + Contrast Enhancement

### Difference from v1

After grayscale conversion:

```text
Grayscale → CLAHE → Normalize
```

### CLAHE (Contrast Limited Adaptive Histogram Equalization)

* Improves local contrast
* Helps in:

  * Low-light images
  * Shadows
  * Overexposed regions

### Why benchmark this

* Improves edge visibility
* Can help heatmap learning
* May increase noise → needs empirical validation



## 21. Pipeline v3 — DNN SSD ROI Only

### Description

This pipeline **removes background removal entirely**.

Steps:

1. **OpenCV DNN SSD Face Detection**
2. **ROI Expansion**
3. **Crop ROI**
4. **Resize to 512×512**
5. **Grayscale Conversion**

No segmentation, no masking.

### Advantages

* Much faster preprocessing
* No YOLO segmentation overhead
* Lower latency

### Tradeoff

* Background clutter remains
* Model must learn robustness implicitly



## 22. Pipeline v4 — DNN SSD + Contrast Enhancement

Same as v3, plus:

```text
Grayscale → CLAHE → Normalize
```

This tests whether **contrast enhancement compensates for lack of segmentation**.



## 23. DNN SSD Details

Uses OpenCV’s pretrained SSD:

```python
deploy.prototxt
res10_300x300_ssd_iter_140000.caffemodel
```

Key characteristics:

* CPU-based
* Deterministic
* No CUDA dependency
* Lightweight

ROI padding ensures:

* No facial parts are clipped
* Comparable geometry to YOLO ROI



## 24. Fallback Logic (Robustness)

If detection fails:

* YOLO pipeline falls back to full-image segmentation
* SSD pipeline falls back to full-image crop

This ensures:

* No sample crashes the pipeline
* Dataset integrity is preserved



## 25. Why Preprocessing Is Central to This Project

The **entire benchmark suite exists to answer one question**:

> How much does preprocessing choice affect accuracy vs latency in face keypoint prediction?

By isolating preprocessing here:

* Training logic stays unchanged
* Metrics remain comparable
* Results are defensible in a paper or review






## 26. Dataset & Data Pipeline (`src/dataset.py`)

### Purpose

`dataset.py` is responsible for the entire **training-time data preparation** pipeline:

* COCO parsing
* Loading images and keypoints
* Applying preprocessing pipelines (v1–v4)
* Applying augmentations
* Generating heatmaps
* Returning tensors ready for training

This file is a major correctness-critical component because it directly controls:

* data integrity
* geometry correctness
* visibility logic
* heatmap alignment
* training stability



## 27. COCO Loader: `load_coco(...)`

### Signature

```python
coco_images, coco_anns, img_ids = load_coco(coco_json_path)
```

### Returns

| Variable      | Meaning                                             |
| ------------- | --------------------------------------------------- |
| `coco_images` | Dict mapping image_id → COCO image metadata         |
| `coco_anns`   | Dict mapping image_id → annotation (keypoints etc.) |
| `img_ids`     | Ordered list of image ids present in COCO           |

### Why it is written this way

* Fast random access by `img_id`
* Avoids scanning entire JSON per sample
* Keeps dataset deterministic



## 28. Augmentations

Two augmentation objects are used:

### `TRAIN_AUG`

Applied during training only.
Includes:

* Random affine transforms (rotation, scale, translation)
* Brightness/contrast perturbations
* Gaussian noise
* Coarse dropout

The keypoint augmentation mode is:

```python
keypoint_params=A.KeypointParams(format="xy", remove_invisible=False)
```

Meaning:

* Keypoints are treated as `(x,y)` pairs
* Keypoints are transformed consistently with the image
* Invisible points are not removed automatically (visibility is handled explicitly)



### `VAL_AUG`

Empty augmentation pipeline

* Preserves original data
* Ensures validation is stable and comparable



## 29. Heatmap Generation: `generate_heatmaps(...)`

### Purpose

Convert sparse keypoints into dense supervision targets.

### Inputs

```python
generate_heatmaps(kps, hm_size, sigma)
```

* `kps`: `(K, 3)` array, each keypoint is `(x, y, v)`
* `hm_size`: heatmap resolution (64)
* `sigma`: gaussian blur radius

### Output

* Heatmaps shape: `(K, hm_size, hm_size)`
* Each heatmap is a 2D gaussian centered at the keypoint

### Visibility handling

* If `v == 0` (not visible), that heatmap stays all zeros

### Scaling

Keypoints are in `IMG_SIZE` coordinate space (512).
Heatmaps are in `HM_SIZE` space (64).
So the gaussian center is scaled by:

```python
scale = HM_SIZE / IMG_SIZE
```



## 30. Dataset Class: `FaceKeypointDataset`

### Signature

```python
FaceKeypointDataset(
    img_ids,
    coco_images,
    coco_anns,
    img_dir,
    transform,
    cache=True,
    sigma=Config.SIGMA,
    pipeline="v1",
    logger_name="dataset",
    log_file=...
)
```

### Key Responsibilities

For each sample:

1. Load raw BGR image
2. Retrieve keypoints from COCO annotation
3. Apply preprocessing pipeline (v1–v4)
4. Convert to grayscale tensor
5. Apply Albumentations transforms (with keypoints)
6. Generate heatmaps (gt_hm)
7. Build visibility mask (vis)
8. Return tensors



## 31. Dataset Caching (DGX Optimization)

Caching is enabled by default:

```python
cache=True
```

### What caching means

* Samples are preprocessed once
* Stored in RAM as tensors/arrays
* Avoids running YOLO/SSD for every epoch

### Why this matters

Preprocessing is expensive:

* YOLO seg is GPU-heavy
* SSD is CPU-heavy but still costly

Caching ensures:

* Training loop is dominated by model forward/backward (desired)
* Validation becomes fast and stable
* Benchmark comparisons are fair



## 32. Caching Preload: `_preload()`

When caching is enabled:

* Dataset iterates over `img_ids`
* For each:

  * runs full preprocessing + augmentation (if training dataset)
  * stores preprocessed output in an internal dict

Progress is shown using `tqdm`, so you can see exactly how long caching takes.



## 33. Failure Handling

Dataset is designed to never crash training due to one bad sample.

Common failure cases:

* image missing on disk
* corrupted file
* ROI cropping returns invalid rectangle
* detector returns no box
* segmentation mask is empty

In such cases the dataset:

* logs the issue
* falls back to full image
* returns a valid tensor

This prevents:

* failed training jobs after hours of execution
* incomplete CV folds
* bias due to sample dropping



## 34. Returned Tensors (Training Contract)

Each `__getitem__()` returns:

```python
img, hm, vis, kps
```

### Shapes

| Tensor | Shape           | Notes                                       |
| ------ | --------------- | ------------------------------------------- |
| `img`  | `[1, 512, 512]` | grayscale in float32                        |
| `hm`   | `[K, 64, 64]`   | heatmaps                                    |
| `vis`  | `[K]`           | visibility mask {0,1}                       |
| `kps`  | `[K, 3]`        | keypoints in 512-space, includes visibility |

This is the contract expected by training.



## 35. Geometry Consistency

A core property of the dataset:

✅ Keypoints always match image content in `512×512` space
because preprocessing and keypoints are transformed together.

This guarantees:

* correct heatmap centers
* correct coordinate loss
* correct PCK evaluation



## 36. Why Dataset Correctness Dominates Results

If the dataset is wrong, you can still get:

* low loss
* apparently improving metrics

…but the model will fail on real images.

This dataset implementation is designed to avoid:

* silent keypoint drift
* misaligned heatmaps
* incorrect visibility
* train/val mismatch

Below is a **single, consolidated, professional documentation section** that covers **everything from Chunk 5 to the end of the codebase**, rewritten cleanly, **without redundancy**, but with **all functions, logic, CLI usage, and experimental details preserved**.

This is written in a style suitable for:

* a serious GitHub repository
* a technical report appendix
* or the “Methods / Experimental Setup” section of a paper



# Model, Training, Inference, and Benchmarking

## Detailed Codebase Documentation (Model → End)



## 1. Model Architecture (`src/model.py`)

### Purpose

`src/model.py` defines the **entire learnable component** of the system.
It implements a **heatmap-based facial keypoint regression model** with **Soft-Argmax coordinate extraction**, designed for:

* side-view and non-frontal faces
* partial visibility and occlusion
* stable training under cross-validation
* ensemble inference

The architecture is intentionally lightweight, interpretable, and reproducible.

---

### Architecture Overview

```
Input: 1 × 512 × 512 (grayscale)
  ↓
MobileNetV3-Large (feature extractor)
  ↓
Heatmap Head (upsampling CNN)
  ↓
K heatmaps (64 × 64)
  ↓
Soft-Argmax
  ↓
K (x, y) coordinates
```

The network **predicts heatmaps only**.
Coordinates are derived externally using a differentiable Soft-Argmax.



### Backbone: MobileNetV3-Large (Grayscale Adaptation)

The pretrained MobileNetV3 backbone is adapted from RGB to grayscale by:

* replacing the first convolution layer to accept 1 channel
* initializing its weights as the mean of pretrained RGB filters

This preserves:

* edge detectors
* spatial inductive bias
* pretrained benefits


### Heatmap Head

The heatmap head upsamples backbone features to produce spatially precise keypoint heatmaps.

Structure:

* Conv → BatchNorm → ReLU
* Upsample ×2
* Conv → BatchNorm → ReLU
* Upsample ×2
* Final Conv → `NUM_KPTS` channels

Output shape:

```
[B, NUM_KPTS, HM_SIZE, HM_SIZE]
```

Each channel corresponds to one facial keypoint.



### Soft-Argmax (`softargmax_2d`)

#### Function Signature

```python
softargmax_2d(hm: Tensor, temperature: float) -> Tensor
```

#### Inputs

* `hm`: `[B, K, H, W]` heatmaps
* `temperature`: sharpness control

#### Output

* `[B, K, 2]` normalized coordinates in `[0,1]`

Coordinates are later scaled by `IMG_SIZE` to pixel space.



### Soft-Argmax Temperature (`softargmax_T`)

Controls how peaked the heatmap distribution is.

* Too low → blurry localization
* Too high → unstable gradients

This parameter is:

* explicitly tuned (e.g. via Optuna)
* passed consistently to training, validation, and inference

CLI usage:

```bash
--softargmax_T 9.58347264700672
```



## 2. Losses and Metrics (`src/loss.py`)

### Hybrid Loss Function

Training uses a **hybrid loss** combining:

1. **Heatmap MSE loss** (masked by visibility)
2. **Coordinate L1 loss** (masked by visibility)

```text
Total Loss = α · Heatmap Loss + (1 − α) · Coordinate Loss
```

Where:

* `α` is configurable (`--alpha`)
* invisible keypoints do not contribute to the loss

This balances:

* spatial supervision (heatmaps)
* geometric precision (coordinates)



### PCK Metrics (Percentage of Correct Keypoints)

Two thresholds are computed:

* **PCK@0.05** → standard accuracy
* **PCK@0.02** → high-precision accuracy

Distance is measured in pixel space:

```text
‖pred − gt‖ < threshold × IMG_SIZE
```



### Per-Keypoint PCK

In addition to overall PCK, the code computes:

* per-keypoint PCK values
* per-keypoint sample counts

These are used to:

* identify weak landmarks
* generate bar plots
* support detailed analysis in reports


## 3. Training & Validation (`src/train.py`)

### Training Strategy

* N-fold cross-validation (`GroupKFold`)
* One model per fold
* Early stopping on validation PCK@0.05
* Mixed precision training (AMP) on GPU



### Training Loop (`train_one_epoch`)

For each batch:

1. Forward pass (AMP-enabled)
2. Heatmap prediction
3. Soft-Argmax coordinate extraction
4. Hybrid loss computation
5. Backpropagation
6. Optional gradient clipping
7. Optimizer + scheduler step

Training loss and timing are logged per epoch.



### Validation Loop (`validate`)

For each validation batch:

* forward pass (no gradients)
* loss computation
* prediction storage on CPU

After the epoch:

* overall PCK@0.05, PCK@0.02
* per-keypoint PCK arrays
* per-keypoint counts

All metrics are returned explicitly.



### Fold Execution (`run_fold`)

For each fold:

1. Initialize model, optimizer, scheduler
2. Train for up to `epochs`
3. Track best PCK@0.05
4. Save best checkpoint
5. Apply early stopping
6. Save:

   * fold history JSON
   * per-keypoint PCK arrays
   * visualizations

Saved under:

```
experiments/<pipeline>_experiment/
  checkpoints/
  viz/
  logs/
```



## 4. Inference & Ensemble (`src/infer.py`)

### Inference Mode: `infer_raw_e2e`

This is a **true end-to-end inference pipeline**, measuring:

```text
raw image
 → preprocessing
 → model forward (all folds)
 → ensemble aggregation
 → coordinate projection
 → overlay generation
```



### Ensemble Strategy

* Load best checkpoint from each fold
* Run all models on the same preprocessed input
* Average predicted coordinates across folds
* Produce:

  * per-fold overlays
  * ensemble overlays

This improves robustness and stability.



### End-to-End Timing

The following timings are measured per image:

* preprocessing time
* per-model forward time
* ensemble aggregation time
* full end-to-end latency

Statistics reported:

* mean
* median
* p90 / p95

Hardware metadata logged:

* GPU model
* AMP dtype
* CPU core count

Saved as:

```
infer_raw_e2e/timing_ensemble.json
```



## 5. Visualization (`src/visualize.py`)

### Training Visualizations (per fold)

* training vs validation loss
* PCK@0.05 / PCK@0.02 curves
* learning rate schedule
* per-keypoint PCK bar charts

Saved under:

```
viz/
```



### Inference Visualizations

* predicted keypoints overlaid on **original images**
* fold-level overlays
* ensemble overlays

Saved under:

```
infer_raw_e2e/
  foldX/overlays/
  ensemble/overlays/
```



### Benchmark Comparison Visualizations

When running the benchmark suite:

* accuracy comparison across pipelines
* latency comparison across pipelines
* accuracy vs latency tradeoff plot

Saved under:

```
benchmark_comparison/
```



## 6. Benchmark Suite (`--mode benchmark_suite`)

### Purpose

Run **multiple preprocessing pipelines sequentially**, with:

* identical hyperparameters
* identical CV splits
* identical model architecture

Only preprocessing changes.



### Supported Pipelines

| Pipeline | Description                      |
| -------- | -------------------------------- |
| v1       | Hybrid YOLO det → ROI → YOLO seg |
| v2       | v1 + contrast correction         |
| v3       | DNN SSD ROI only                 |
| v4       | v3 + contrast correction         |



### Example Command (5-fold CV)

```bash
python main_experiment.py --mode benchmark_suite --folds 5 \
  --epochs 150 --early_stop_patience 15 \
  --lr 0.0002376500626274199 \
  --weight_decay 0.000001986074040054842 \
  --sigma 2.58765228765362 \
  --alpha 0.6008870077358692 \
  --softargmax_T 9.58347264700672 \
  --batch_size 64 --num_workers 8 \
  --suite_root experiments/benchmark_suite
```



### Outputs

For each pipeline:

```
experiments/benchmark_suite/vX_experiment/
```

Global comparison:

```
experiments/benchmark_suite/benchmark_comparison/
```



## 7. Reproducibility & Reliability Guarantees

This codebase guarantees:

* deterministic splits
* no test leakage
* isolated experiment folders
* explicit hyperparameter control
* full logging of metrics and timing
* qualitative and quantitative validation



## 8. Intended Use

This codebase is suitable for:

* academic research
* ablation studies
* benchmarking preprocessing strategies
* deployment feasibility studies
* reproducible ML experimentation





