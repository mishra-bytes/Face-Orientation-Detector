"""Command-line entry point for the hybrid face orientation pipeline."""
from __future__ import annotations

import argparse

from inference import run_batch
from models import HybridModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hybrid YOLO detection + segmentation pipeline")
    parser.add_argument("input", help="Input image path or directory")
    parser.add_argument("output", help="Directory to store masks and visualizations")
    parser.add_argument("--det-weights", default="yolov8n.pt", help="YOLO detection weights path")
    parser.add_argument("--seg-weights", default="yolov8n-seg.pt", help="YOLO segmentation weights path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = HybridModel(yolo_det_weights=args.det_weights, yolo_seg_weights=args.seg_weights)
    run_batch(model, args.input, args.output)


if __name__ == "__main__":
    main()
