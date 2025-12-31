"""Image processing utilities for the hybrid pipeline."""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

YOLO_DET_CONF: float = 0.25
YOLO_SEG_CONF: float = 0.25
ROI_EXPAND_RATIO: float = 0.55


def clamp(value: int, lo: int, hi: int) -> int:
    """Clamp *value* to the inclusive ``[lo, hi]`` range."""

    return max(lo, min(hi, value))


def largest_cc_mask(mask01: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component of a binary mask."""

    mask = (mask01.astype(np.uint8) * 255)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return mask01
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == best).astype(np.uint8)


def cleanup_mask(mask01: np.ndarray) -> np.ndarray:
    """Morphologically close small holes and keep the largest component."""

    mask = (mask01.astype(np.uint8) * 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask01 = (mask > 127).astype(np.uint8)
    return largest_cc_mask(mask01)


def apply_black_bg(img_rgb: np.ndarray, mask01: np.ndarray) -> np.ndarray:
    """Apply a binary mask and black out the background."""

    mask3 = np.repeat(mask01[:, :, None], 3, axis=2)
    return np.where(mask3 > 0, img_rgb, 0).astype(np.uint8)


def pick_adaptive_imgsz(roi_w: int, roi_h: int, full_w: int, full_h: int) -> int:
    """Select YOLO segmentation resolution based on ROI size."""

    frac = (roi_w * roi_h) / float(full_w * full_h + 1e-6)
    m = max(roi_w, roi_h)
    if frac <= 0.10 or m <= 220:
        return 320
    if frac <= 0.25 or m <= 420:
        return 416
    return 640


def yolo_det_box(model, img_rgb: np.ndarray, conf: float) -> Optional[Tuple[int, int, int, int]]:
    """Return the largest detection box as (x1, y1, x2, y2)."""

    result = model(img_rgb, conf=conf, verbose=False)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return None
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    idx = int(np.argmax(areas))
    x1, y1, x2, y2 = boxes[idx].astype(int)
    return int(x1), int(y1), int(x2), int(y2)


def expand_box(box: Tuple[int, int, int, int], w: int, h: int, ratio: float = ROI_EXPAND_RATIO) -> Tuple[int, int, int, int]:
    """Expand a bounding box by *ratio* while keeping it within image bounds."""

    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    pad_w, pad_h = int(bw * ratio), int(bh * ratio)
    ex1 = clamp(x1 - pad_w, 0, w - 1)
    ey1 = clamp(y1 - pad_h, 0, h - 1)
    ex2 = clamp(x2 + pad_w, 0, w - 1)
    ey2 = clamp(y2 + pad_h, 0, h - 1)
    if ex2 <= ex1 or ey2 <= ey1:
        return box
    return ex1, ey1, ex2, ey2


def yolo_seg_mask_for_image(model, img_rgb: np.ndarray, imgsz: int, conf: float) -> np.ndarray:
    """Run YOLO segmentation and return a cleaned binary mask."""

    height, width = img_rgb.shape[:2]
    result = model(img_rgb, imgsz=imgsz, conf=conf, verbose=False)[0]
    mask01 = np.zeros((height, width), dtype=np.uint8)
    if result.masks is not None and result.masks.data is not None and len(result.masks.data) > 0:
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        idx = int(np.argmax(areas))
        mask = result.masks.data[idx].detach().cpu().numpy()
        mask01 = (cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR) > 0.5).astype(np.uint8)
    return cleanup_mask(mask01)


def load_rgb(path: str) -> np.ndarray:
    """Load an image as RGB; raises ``ValueError`` if it cannot be read."""

    bgr = cv2.imread(path)
    if bgr is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_mask(path: str, mask: np.ndarray) -> None:
    """Save a binary mask to disk as PNG."""

    cv2.imwrite(path, (mask * 255).astype(np.uint8))


def save_image(path: str, img_rgb: np.ndarray) -> None:
    """Save an RGB image to disk."""

    cv2.imwrite(path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
