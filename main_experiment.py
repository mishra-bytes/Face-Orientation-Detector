from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader

from src.config import Config
from src.dataset import load_coco, FaceKeypointDataset, TRAIN_AUG, VAL_AUG
from src.model import KeypointNet
from src.train import run_fold
from src.utils import (
    get_logger,
    seed_everything,
    worker_init_fn,
    atomic_write_text,
    set_work_dir,
)
from src.infer import infer_raw_e2e
from src.visualize import plot_suite_comparison


def _collect_ckpts(work_dir: Path, folds: int) -> List[Path]:
    ckpts: List[Path] = []
    for f in range(1, folds + 1):
        p = work_dir / "checkpoints" / f"fold{f}_best.pt"
        if p.exists():
            ckpts.append(p)
    return ckpts


def run_train_cv(*, args, work_dir: Path, pipeline: str) -> Dict[str, object]:
    set_work_dir(work_dir)

    # IMPORTANT: runtime override
    Config.FOLDS = int(args.folds)

    seed_everything(Config.SEED, deterministic=True)
    logger = get_logger(f"main_{pipeline}", Config.WORK_DIR / "logs" / "main.log")

    t0 = time.perf_counter()

    logger.info("=== Face Keypoint Prediction: Robust CV Training ===")
    logger.info(f"Pipeline: {pipeline}")
    logger.info(f"FOLDS: {Config.FOLDS}")
    logger.info(f"Device: {Config.DEVICE}")
    logger.info(f"WORK_DIR: {Config.WORK_DIR.resolve()}")
    logger.info(f"TRAIN_COCO: {Config.TRAIN_COCO.resolve()}")
    logger.info(f"TRAIN_IMG_DIR: {Config.TRAIN_IMG_DIR.resolve()}")
    logger.info(
        f"Hyperparams: epochs={args.epochs}, lr={args.lr:.5g}, wd={args.weight_decay:.5g}, "
        f"alpha={args.alpha:.4g}, sigma={args.sigma:.4g}, softargmax_T={args.softargmax_T:.4g}, "
        f"batch_size={args.batch_size}, workers={args.num_workers}, early_stop_patience={args.early_stop_patience}"
    )

    coco_images, coco_anns, img_ids = load_coco(Config.TRAIN_COCO)

    # GroupKFold: group per image id (stable, no leakage across same image)
    groups = np.array(img_ids, dtype=np.int64)
    gkf = GroupKFold(n_splits=Config.FOLDS)

    fold_scores_05: List[float] = []
    fold_scores_02: List[float] = []
    fold_best_ckpts: List[str] = []

    for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(img_ids, y=img_ids, groups=groups), start=1):
        train_ids = [img_ids[i] for i in tr_idx]
        val_ids = [img_ids[i] for i in va_idx]

        logger.info(f"\n--- Fold {fold_idx}/{Config.FOLDS} ---")
        logger.info(f"Train size: {len(train_ids)} | Val size: {len(val_ids)}")

        train_ds = FaceKeypointDataset(
            img_ids=train_ids,
            coco_images=coco_images,
            coco_anns=coco_anns,
            img_dir=Config.TRAIN_IMG_DIR,
            transform=TRAIN_AUG,
            cache=True,
            sigma=args.sigma,
            pipeline=pipeline,
            logger_name=f"dataset_train_fold{fold_idx}",
            log_file=Config.WORK_DIR / "logs" / f"dataset_train_fold{fold_idx}.log",
        )
        val_ds = FaceKeypointDataset(
            img_ids=val_ids,
            coco_images=coco_images,
            coco_anns=coco_anns,
            img_dir=Config.TRAIN_IMG_DIR,
            transform=VAL_AUG,
            cache=True,
            sigma=args.sigma,
            pipeline=pipeline,
            logger_name=f"dataset_val_fold{fold_idx}",
            log_file=Config.WORK_DIR / "logs" / f"dataset_val_fold{fold_idx}.log",
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=getattr(Config, "PIN_MEMORY", True),
            persistent_workers=getattr(Config, "PERSISTENT_WORKERS", True) if args.num_workers > 0 else False,
            worker_init_fn=worker_init_fn,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=getattr(Config, "PIN_MEMORY", True),
            persistent_workers=getattr(Config, "PERSISTENT_WORKERS", True) if args.num_workers > 0 else False,
            worker_init_fn=worker_init_fn,
            drop_last=False,
        )

        model = KeypointNet()
        fold_logger = get_logger(f"{pipeline}_fold{fold_idx}", Config.WORK_DIR / "logs" / f"fold{fold_idx}.log")

        state = run_fold(
            fold_idx=fold_idx,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=Config.DEVICE,
            logger=fold_logger,
            lr=args.lr,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            softargmax_T=args.softargmax_T,
            alpha=args.alpha,
            work_dir=Config.WORK_DIR,
            early_stop_patience=args.early_stop_patience,
        )

        fold_scores_05.append(float(state.best_pck05))
        fold_scores_02.append(float(state.best_pck02))
        fold_best_ckpts.append(str(state.best_ckpt_path) if state.best_ckpt_path else "")

    t1 = time.perf_counter()

    summary = {
        "pipeline": pipeline,
        "folds": int(Config.FOLDS),
        "scores_pck05": fold_scores_05,
        "mean_pck05": float(np.mean(fold_scores_05)) if fold_scores_05 else 0.0,
        "std_pck05": float(np.std(fold_scores_05)) if fold_scores_05 else 0.0,
        "scores_pck02": fold_scores_02,
        "mean_pck02": float(np.mean(fold_scores_02)) if fold_scores_02 else 0.0,
        "std_pck02": float(np.std(fold_scores_02)) if fold_scores_02 else 0.0,
        "best_ckpts": fold_best_ckpts,
        "hyperparams": {
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "alpha": args.alpha,
            "sigma": args.sigma,
            "softargmax_T": args.softargmax_T,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "early_stop_patience": args.early_stop_patience,
        },
        "total_train_time_s": float(t1 - t0),
        "work_dir": str(Config.WORK_DIR),
    }

    logger.info("\n=== CV Summary ===")
    logger.info(json.dumps(summary, indent=2))
    atomic_write_text(Config.WORK_DIR / "cv_summary.json", json.dumps(summary, indent=2))
    return summary


def run_infer_raw_e2e(*, args, work_dir: Path, pipeline: str) -> Dict[str, object]:
    set_work_dir(work_dir)

    # IMPORTANT: runtime override for checkpoint collection
    Config.FOLDS = int(args.folds)

    seed_everything(Config.SEED, deterministic=True)
    logger = get_logger(f"infer_raw_e2e_{pipeline}", Config.WORK_DIR / "logs" / "infer_raw_e2e.log")

    test_dir = Config.TEST_IMG_DIR
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = sorted([p for p in test_dir.glob("*") if p.suffix.lower() in exts])
    if len(image_paths) == 0:
        raise RuntimeError(f"No test images found in {test_dir}")

    ckpts = _collect_ckpts(Config.WORK_DIR, Config.FOLDS)
    if len(ckpts) == 0:
        raise RuntimeError("No fold best checkpoints found. Train first.")

    out_root = Config.WORK_DIR / "infer_raw_e2e"
    out_root.mkdir(parents=True, exist_ok=True)

    timing = infer_raw_e2e(
        ckpt_paths=ckpts,
        image_paths=image_paths,
        device=Config.DEVICE,
        softargmax_T=args.softargmax_T,
        out_root=out_root,
        pipeline=pipeline,
        logger=logger,
    )
    return {"pipeline": pipeline, "infer_timing": timing, "work_dir": str(Config.WORK_DIR)}


def run_benchmark_suite(args) -> None:
    """
    Runs v1->v4 sequentially:
      - train (Config.FOLDS folds)
      - infer_raw_e2e
      - writes a global summary and global comparison plots
    """
    base_root = Path(args.suite_root).resolve()
    base_root.mkdir(parents=True, exist_ok=True)

    pipelines = ["v1", "v2", "v3", "v4"] if args.pipelines is None else args.pipelines
    suite_results: List[Dict[str, object]] = []

    for p in pipelines:
        work_dir = base_root / f"{p}_experiment"
        work_dir.mkdir(parents=True, exist_ok=True)

        train_summary = run_train_cv(args=args, work_dir=work_dir, pipeline=p)
        infer_summary = run_infer_raw_e2e(args=args, work_dir=work_dir, pipeline=p)

        suite_results.append({"train": train_summary, "infer": infer_summary})

    suite_dir = base_root / "benchmark_comparison"
    suite_dir.mkdir(parents=True, exist_ok=True)

    atomic_write_text(suite_dir / "suite_results.json", json.dumps(suite_results, indent=2))
    plot_suite_comparison(suite_dir=suite_dir, suite_results=suite_results)

    print(f"\nSuite complete. Results: {suite_dir}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", type=str, default="train", choices=["train", "infer_raw_e2e", "benchmark_suite"])
    p.add_argument("--pipeline", type=str, default="v1", choices=["v1", "v2", "v3", "v4"])

    # NEW: folds override
    p.add_argument("--folds", type=int, default=5, help="Number of CV folds (e.g., 3)")

    p.add_argument("--epochs", type=int, default=Config.EPOCHS)
    p.add_argument("--lr", type=float, default=Config.LR)
    p.add_argument("--weight_decay", type=float, default=Config.WEIGHT_DECAY)
    p.add_argument("--alpha", type=float, default=Config.ALPHA)
    p.add_argument("--sigma", type=float, default=Config.SIGMA)
    p.add_argument("--softargmax_T", type=float, default=Config.SOFTARGMAX_T)

    p.add_argument("--batch_size", type=int, default=Config.BATCH_SIZE)
    p.add_argument("--num_workers", type=int, default=Config.NUM_WORKERS)
    p.add_argument("--early_stop_patience", type=int, default=Config.EARLY_STOP_PATIENCE)

    # suite options
    p.add_argument("--suite_root", type=str, default="experiments/benchmark_suite")
    p.add_argument("--pipelines", nargs="*", default=None, help="Optional subset, e.g. --pipelines v1 v3")
    return p.parse_args()


def main():
    args = parse_args()
    Config.setup()

    if args.mode == "train":
        run_train_cv(args=args, work_dir=Config.WORK_DIR, pipeline=args.pipeline)
    elif args.mode == "infer_raw_e2e":
        run_infer_raw_e2e(args=args, work_dir=Config.WORK_DIR, pipeline=args.pipeline)
    else:
        run_benchmark_suite(args)


if __name__ == "__main__":
    main()
