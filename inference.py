"""Inference orchestration for the hybrid pipeline."""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

from models import HybridModel
from processor import apply_black_bg, load_rgb, save_image, save_mask
from telemetry import TelemetryLogger


def run_pipeline_on_image(model: HybridModel, image_path: str, output_dir: str) -> Tuple[str, str]:
    """Run the full hybrid pipeline on a single image.

    Returns the saved mask path and composited image path.
    """

    img_rgb = load_rgb(image_path)
    with TelemetryLogger(items_processed=1):
        mask = model.predict_mask(img_rgb)
    visual = apply_black_bg(img_rgb, mask)

    name = Path(image_path).stem
    os.makedirs(output_dir, exist_ok=True)
    mask_path = os.path.join(output_dir, f"{name}_mask.png")
    visual_path = os.path.join(output_dir, f"{name}_hybrid.png")
    save_mask(mask_path, mask)
    save_image(visual_path, visual)
    return mask_path, visual_path


def collect_images(input_path: str) -> List[str]:
    path = Path(input_path)
    if path.is_file():
        return [str(path)]
    if path.is_dir():
        return [
            str(p)
            for p in path.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def run_batch(model: HybridModel, input_path: str, output_dir: str) -> List[Tuple[str, str]]:
    """Process an image file or directory and return saved artifact paths."""

    images = collect_images(input_path)
    results: List[Tuple[str, str]] = []
    for image_path in images:
        results.append(run_pipeline_on_image(model, image_path, output_dir))
    return results
