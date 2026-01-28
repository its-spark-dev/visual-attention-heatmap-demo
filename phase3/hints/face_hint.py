from __future__ import annotations

from typing import Optional

import numpy as np


def build_face_hint_map(image: np.ndarray) -> np.ndarray:
    """Return a soft face-prior mask in [0, 1] with shape (H, W)."""
    if image.ndim not in (2, 3):
        raise ValueError("image must be a 2D or 3D array")

    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "OpenCV is required for face hints. Install opencv-python."
        ) from exc

    height, width = image.shape[:2]
    if height == 0 or width == 0:
        return np.zeros((height, width), dtype=np.float32)

    gray = _to_uint8_gray(image, cv2)
    detector = _load_face_detector(cv2)
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(24, 24),
    )

    if faces is None or len(faces) == 0:
        return np.zeros((height, width), dtype=np.float32)

    mask = np.zeros((height, width), dtype=np.float32)
    for x, y, w, h in faces:
        x0 = max(0, int(x))
        y0 = max(0, int(y))
        x1 = min(width, int(x + w))
        y1 = min(height, int(y + h))
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 1.0

    sigma = max(1.0, min(height, width) * 0.02)
    blurred = cv2.GaussianBlur(mask, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)
    return _normalize(blurred)


def _to_uint8_gray(image: np.ndarray, cv2_module) -> np.ndarray:
    if image.ndim == 2:
        gray = image
    else:
        if image.shape[2] >= 3:
            gray = cv2_module.cvtColor(image, cv2_module.COLOR_RGB2GRAY)
        else:
            gray = image[..., 0]

    gray = np.asarray(gray, dtype=np.float32)
    max_val = float(np.max(gray)) if gray.size else 0.0
    if max_val <= 1.0:
        gray = gray * 255.0
    gray = np.clip(gray, 0.0, 255.0)
    return gray.astype(np.uint8)


def _load_face_detector(cv2_module):
    cascade_path = cv2_module.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2_module.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError("Failed to load OpenCV Haar cascade for face detection.")
    return detector


def _normalize(mask: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    min_val = float(np.min(mask))
    max_val = float(np.max(mask))
    if max_val - min_val < eps:
        return np.zeros_like(mask, dtype=np.float32)
    normalized = (mask - min_val) / (max_val - min_val)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)
