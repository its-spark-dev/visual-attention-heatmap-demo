from __future__ import annotations

import numpy as np


def build_object_hint_map(image: np.ndarray) -> np.ndarray:
    """Return a placeholder object-prior mask in [0, 1] with shape (H, W)."""
    if image.ndim not in (2, 3):
        raise ValueError("image must be a 2D or 3D array")

    height, width = image.shape[:2]
    if height == 0 or width == 0:
        return np.zeros((height, width), dtype=np.float32)

    # TODO: integrate a pretrained object detector (e.g., lightweight SSD or YOLO)
    # and convert detected boxes into a soft mask, similar to face/text hints.
    return np.zeros((height, width), dtype=np.float32)
