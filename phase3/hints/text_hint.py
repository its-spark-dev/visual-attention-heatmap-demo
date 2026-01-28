from __future__ import annotations

from typing import List, Optional, Tuple

import os

import numpy as np


def build_text_hint_map(image: np.ndarray) -> np.ndarray:
    """Return a soft text-prior mask in [0, 1] with shape (H, W)."""
    if image.ndim not in (2, 3):
        raise ValueError("image must be a 2D or 3D array")

    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "OpenCV is required for text hints. Install opencv-python."
        ) from exc

    height, width = image.shape[:2]
    if height == 0 or width == 0:
        return np.zeros((height, width), dtype=np.float32)

    gray = _to_uint8_gray(image, cv2)
    boxes = _detect_text_regions(gray, cv2)
    if not boxes:
        return np.zeros((height, width), dtype=np.float32)

    mask = np.zeros((height, width), dtype=np.float32)
    for x0, y0, x1, y1 in boxes:
        x0 = max(0, int(x0))
        y0 = max(0, int(y0))
        x1 = min(width, int(x1))
        y1 = min(height, int(y1))
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 1.0

    sigma = max(1.0, min(height, width) * 0.02)
    blurred = cv2.GaussianBlur(mask, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)
    return _normalize(blurred)


def _detect_text_regions(gray: np.ndarray, cv2_module) -> List[Tuple[int, int, int, int]]:
    model_path = _east_model_path()
    if model_path and hasattr(cv2_module, "dnn"):
        try:
            return _detect_text_east(gray, cv2_module, model_path)
        except Exception:
            pass
    return _detect_text_heuristic(gray, cv2_module)


def _east_model_path() -> Optional[str]:
    env_path = os.getenv("EAST_TEXT_MODEL_PATH")
    if env_path and os.path.isfile(env_path):
        return env_path
    local_path = os.path.join(
        os.path.dirname(__file__), "frozen_east_text_detection.pb"
    )
    if os.path.isfile(local_path):
        return local_path
    return None


def _detect_text_east(
    gray: np.ndarray,
    cv2_module,
    model_path: str,
    score_threshold: float = 0.5,
    nms_threshold: float = 0.4,
) -> List[Tuple[int, int, int, int]]:
    height, width = gray.shape[:2]
    target_max = 320
    scale = min(target_max / max(height, width), 1.0)
    new_w = max(32, int(width * scale))
    new_h = max(32, int(height * scale))
    new_w = (new_w // 32) * 32
    new_h = (new_h // 32) * 32

    resized = cv2_module.resize(gray, (new_w, new_h))
    resized = cv2_module.cvtColor(resized, cv2_module.COLOR_GRAY2BGR)

    net = cv2_module.dnn.readNet(model_path)
    blob = cv2_module.dnn.blobFromImage(
        resized, 1.0, (new_w, new_h), (123.68, 116.78, 103.94), swapRB=False, crop=False
    )
    net.setInput(blob)
    layer_names = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
    scores, geometry = net.forward(layer_names)

    rectangles: List[Tuple[int, int, int, int]] = []
    confidences: List[float] = []

    rows, cols = scores.shape[2], scores.shape[3]
    for y in range(rows):
        scores_data = scores[0, 0, y]
        x_data0 = geometry[0, 0, y]
        x_data1 = geometry[0, 1, y]
        x_data2 = geometry[0, 2, y]
        x_data3 = geometry[0, 3, y]
        angles = geometry[0, 4, y]
        for x in range(cols):
            score = scores_data[x]
            if score < score_threshold:
                continue
            offset_x = x * 4.0
            offset_y = y * 4.0
            angle = angles[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            h = x_data0[x] + x_data2[x]
            w = x_data1[x] + x_data3[x]
            end_x = int(offset_x + (cos * x_data1[x]) + (sin * x_data2[x]))
            end_y = int(offset_y - (sin * x_data1[x]) + (cos * x_data2[x]))
            start_x = int(end_x - w)
            start_y = int(end_y - h)
            rectangles.append((start_x, start_y, int(w), int(h)))
            confidences.append(float(score))

    if not rectangles:
        return []

    indices = cv2_module.dnn.NMSBoxes(rectangles, confidences, score_threshold, nms_threshold)
    if indices is None or len(indices) == 0:
        return []

    scale_x = width / float(new_w)
    scale_y = height / float(new_h)

    boxes: List[Tuple[int, int, int, int]] = []
    for idx in indices.flatten():
        x, y, w, h = rectangles[idx]
        start_x = int(x * scale_x)
        start_y = int(y * scale_y)
        end_x = int((x + w) * scale_x)
        end_y = int((y + h) * scale_y)
        boxes.append((start_x, start_y, end_x, end_y))
    return boxes


def _detect_text_heuristic(
    gray: np.ndarray,
    cv2_module,
) -> List[Tuple[int, int, int, int]]:
    blurred = cv2_module.GaussianBlur(gray, (3, 3), 0)
    grad_x = cv2_module.Sobel(blurred, cv2_module.CV_32F, 1, 0, ksize=3)
    grad_y = cv2_module.Sobel(blurred, cv2_module.CV_32F, 0, 1, ksize=3)
    magnitude = cv2_module.magnitude(grad_x, grad_y)

    mag_uint8 = cv2_module.normalize(magnitude, None, 0, 255, cv2_module.NORM_MINMAX)
    mag_uint8 = mag_uint8.astype(np.uint8)
    _, thresh = cv2_module.threshold(
        mag_uint8, 0, 255, cv2_module.THRESH_BINARY + cv2_module.THRESH_OTSU
    )

    kernel_w = max(3, gray.shape[1] // 40)
    kernel_h = max(3, gray.shape[0] // 80)
    kernel = cv2_module.getStructuringElement(
        cv2_module.MORPH_RECT, (kernel_w, kernel_h)
    )
    closed = cv2_module.morphologyEx(thresh, cv2_module.MORPH_CLOSE, kernel, iterations=1)
    closed = cv2_module.dilate(closed, kernel, iterations=1)

    contours, _ = cv2_module.findContours(
        closed, cv2_module.RETR_EXTERNAL, cv2_module.CHAIN_APPROX_SIMPLE
    )

    height, width = gray.shape[:2]
    min_area = max(50, int(height * width * 0.001))
    boxes: List[Tuple[int, int, int, int]] = []
    for contour in contours:
        x, y, w, h = cv2_module.boundingRect(contour)
        area = w * h
        if area < min_area:
            continue
        aspect = w / float(h + 1e-6)
        if aspect < 0.5 or aspect > 15.0:
            continue
        boxes.append((x, y, x + w, y + h))
    return boxes


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


def _normalize(mask: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    min_val = float(np.min(mask))
    max_val = float(np.max(mask))
    if max_val - min_val < eps:
        return np.zeros_like(mask, dtype=np.float32)
    normalized = (mask - min_val) / (max_val - min_val)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)
