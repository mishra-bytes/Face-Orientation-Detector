"""Model definitions for the Face Orientation hybrid pipeline.

This module implements the final hybrid approach tested in the notebook:
1. YOLOv8 detection to get the largest face/person box.
2. ROI expansion to include context.
3. Adaptive segmentation crop with YOLOv8-seg at dynamic resolution.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from processor import (
    ROI_EXPAND_RATIO,
    YOLO_DET_CONF,
    YOLO_SEG_CONF,
    cleanup_mask,
    expand_box,
    pick_adaptive_imgsz,
    yolo_det_box,
    yolo_seg_mask_for_image,
)


@dataclass
class HybridModel:
    """Hybrid YOLO detection + segmentation pipeline.

    The architecture mirrors the final experimental configuration from the
    notebook: YOLOv8 detection feeds an expanded ROI into YOLOv8 segmentation
    with an adaptive input resolution. If detection fails, segmentation runs on
    the full frame.
    """

    yolo_det_weights: str = "yolov8n.pt"
    yolo_seg_weights: str = "yolov8n-seg.pt"

    def __post_init__(self) -> None:
        from ultralytics import YOLO  # imported here to keep optional dependency lazy

        self.detector = YOLO(self.yolo_det_weights)
        self.segmenter = YOLO(self.yolo_seg_weights)

    def predict_mask(self, img_rgb: np.ndarray) -> np.ndarray:
        """Return a cleaned binary mask for the primary subject.

        Args:
            img_rgb: Input image as HxWx3 RGB array.
        Returns:
            Binary mask (uint8) in the original image shape.
        """

        height, width = img_rgb.shape[:2]
        box = yolo_det_box(self.detector, img_rgb, conf=YOLO_DET_CONF)

        if box is None:
            return yolo_seg_mask_for_image(self.segmenter, img_rgb, imgsz=640, conf=YOLO_SEG_CONF)

        ex1, ey1, ex2, ey2 = expand_box(box, width, height, ROI_EXPAND_RATIO)
        roi = img_rgb[ey1:ey2, ex1:ex2]
        roi_height, roi_width = roi.shape[:2]
        imgsz = pick_adaptive_imgsz(roi_width, roi_height, width, height)

        roi_mask = yolo_seg_mask_for_image(self.segmenter, roi, imgsz=imgsz, conf=YOLO_SEG_CONF)
        full_mask = np.zeros((height, width), dtype=np.uint8)
        full_mask[ey1:ey2, ex1:ex2] = roi_mask
        return cleanup_mask(full_mask)

    def __call__(self, img_rgb: np.ndarray) -> np.ndarray:
        """Alias for :py:meth:`predict_mask`."""

        return self.predict_mask(img_rgb)
