"""YOLO face detector (Ultralytics), with automatic weight download."""

from __future__ import annotations

import os
import sys
from typing import Optional

from .vision import Rect

WEIGHTS_URL = "https://github.com/lindevs/yolov8-face/releases/latest/download/yolov8n-face-lindevs.pt"
DEFAULT_WEIGHTS = os.path.join("weights", "yolov8n-face-lindevs.pt")


def ensure_weights(path: str, url: str = WEIGHTS_URL) -> str:
    """Download the default face weights on first run. Returns ``path`` either way."""
    if os.path.exists(path):
        return path
    import requests

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    partial = path + ".part"
    print(f"Downloading face weights to {path} ...", file=sys.stderr)
    with requests.get(url, stream=True, timeout=30) as response:
        response.raise_for_status()
        with open(partial, "wb") as f:
            for chunk in response.iter_content(chunk_size=1 << 16):
                f.write(chunk)
    os.replace(partial, path)
    return path


class FaceDetector:
    """Returns the single most confident face box, as (x, y, w, h)."""

    def __init__(self, weights: str, imgsz: int = 640, device: str = "") -> None:
        from ultralytics import YOLO  # imported here so the rest of the package works without torch

        self.model = YOLO(weights)
        self.imgsz = imgsz
        self.device = device or None

    def detect(self, frame, min_conf: float) -> Optional[Rect]:
        results = self.model.predict(source=frame, verbose=False, imgsz=self.imgsz, device=self.device, conf=min_conf)
        best = None
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for conf, xyxy in zip(boxes.conf.tolist(), boxes.xyxy.tolist()):
                if conf >= min_conf and (best is None or conf > best[0]):
                    x1, y1, x2, y2 = (int(v) for v in xyxy)
                    best = (conf, (x1, y1, x2 - x1, y2 - y1))
        return best[1] if best else None
