from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image


def build_heatmap_overlay(
    image: Image.Image,
    attention_map: np.ndarray,
    alpha: float = 0.45,
) -> Image.Image:
    """Return an RGB image with a heatmap overlay."""
    base = image.convert("RGB")
    heatmap = build_heatmap_image(attention_map, base.size)
    overlay = Image.blend(base, heatmap, alpha=alpha)
    return overlay


def build_heatmap_image(attention_map: np.ndarray, size: Tuple[int, int]) -> Image.Image:
    """Return an RGB heatmap image resized to the given size."""
    return _prepare_heatmap(attention_map, size)


def _prepare_heatmap(attention_map: np.ndarray, size: Tuple[int, int]) -> Image.Image:
    normalized = _normalize_attention(attention_map)
    heatmap_rgb = _apply_colormap(normalized)
    heatmap_image = Image.fromarray(heatmap_rgb, mode="RGB")
    return heatmap_image.resize(size, resample=Image.BILINEAR)


def _normalize_attention(attention_map: np.ndarray) -> np.ndarray:
    if attention_map.ndim == 3:
        attention_map = attention_map.mean(axis=2)
    attention_map = attention_map.astype("float32")
    min_val = float(np.min(attention_map))
    max_val = float(np.max(attention_map))
    if max_val - min_val < 1e-6:
        return np.zeros_like(attention_map, dtype="float32")
    return (attention_map - min_val) / (max_val - min_val)


def _apply_colormap(normalized: np.ndarray) -> np.ndarray:
    """Simple warm colormap from dark to yellow-white."""
    normalized = np.clip(normalized, 0.0, 1.0)
    red = (normalized * 255).astype(np.uint8)
    green = (np.clip((normalized - 0.3) / 0.7, 0.0, 1.0) * 255).astype(np.uint8)
    blue = (np.clip((normalized - 0.75) / 0.25, 0.0, 1.0) * 255).astype(np.uint8)
    return np.stack([red, green, blue], axis=2)


def build_heatmap_legend(width: int = 240, height: int = 16) -> Image.Image:
    """Create a horizontal gradient legend matching the heatmap colormap."""
    gradient = np.linspace(0.0, 1.0, width, dtype=np.float32)
    gradient = np.tile(gradient, (height, 1))
    legend_rgb = _apply_colormap(gradient)
    return Image.fromarray(legend_rgb, mode="RGB")
