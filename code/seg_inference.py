# seg_inference.py
from ultralytics import YOLO
import numpy as np
import cv2

class SegmentationWrapper:
    def __init__(self, weights="yolov8n-seg.pt", device=None, conf=0.25):
        # device like "cuda:0" or "cpu" or None (auto)
        self.model = YOLO(weights)
        if device:
            self.model.to(device)
        self.conf = conf

    def get_person_mask(self, img_bgr: np.ndarray):
        """
        Returns binary mask (uint8) same HxW where person pixels are 255.
        Handles cases where multiple instances exist by ORing masks.
        """
        # ensure RGB input as ultralytics expects
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = self.model.predict(source=img_rgb, imgsz=img_rgb.shape[:2], conf=self.conf, verbose=False)
        r = results[0]
        H, W = img_rgb.shape[:2]
        mask = np.zeros((H, W), dtype=np.uint8)
        if hasattr(r, 'masks') and r.masks is not None:
            try:
                masks = r.masks.data.cpu().numpy()  # shape (n,H,W)
            except Exception:
                masks = np.array(r.masks.data)
            for m in masks:
                mask = np.maximum(mask, (m*255).astype(np.uint8))
        # fallback: if no mask, return zero mask
        return mask
