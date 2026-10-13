"""Small OpenCV helpers: regions of interest and the pupil finder."""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

Rect = Tuple[int, int, int, int]  # x, y, w, h


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def ema(previous: float, new: float, smoothing: float) -> float:
    """Exponential moving average; ``smoothing`` is the weight kept from the previous value."""
    return smoothing * previous + (1.0 - smoothing) * new


def clip_rect(rect: Rect, width: int, height: int) -> Rect:
    """Clip a face box to the frame (always at least 1x1)."""
    x, y, w, h = rect
    x = int(clamp(x, 0, width - 1))
    y = int(clamp(y, 0, height - 1))
    w = int(max(1, min(w, width - x)))
    h = int(max(1, min(h, height - y)))
    return x, y, w, h


def eye_rects(face: Rect) -> Tuple[Rect, Rect]:
    """The top third of the face box, split down the middle: (left eye, right eye) as seen in the image."""
    x, y, w, h = face
    bottom = y + int(h / 3.0)
    mid = x + w // 2
    return (x, y, mid - x, bottom - y), (mid, y, x + w - mid, bottom - y)


def crop(frame: np.ndarray, rect: Rect) -> Optional[np.ndarray]:
    x, y, w, h = rect
    height, width = frame.shape[:2]
    x1, x2 = int(clamp(x, 0, width)), int(clamp(x + w, 0, width))
    y1, y2 = int(clamp(y, 0, height)), int(clamp(y + h, 0, height))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def pupil_position(
    gray: Optional[np.ndarray], min_area_px: int = 80, min_area_frac: float = 0.002
) -> Optional[Tuple[float, float]]:
    """Centroid of the darkest large blob in an eye crop, as (x, y) in 0..1, or None.

    Otsu thresholding separates dark (pupil, iris, lashes) from light (sclera,
    skin). Blobs smaller than ``min_area_px`` pixels or ``min_area_frac`` of the
    crop are treated as noise.
    """
    if gray is None or gray.size == 0:
        return None
    h, w = gray.shape[:2]
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    blob = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(blob))
    if area < min_area_px or area / float(max(1, h * w)) < min_area_frac:
        return None
    m = cv2.moments(blob)
    if m["m00"] == 0:
        return None
    return clamp(m["m10"] / m["m00"] / w, 0.0, 1.0), clamp(m["m01"] / m["m00"] / h, 0.0, 1.0)


def to_gray(image: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if image is None or image.size == 0:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
