from __future__ import annotations

from typing import Optional

import numpy as np


def modulate_attention(
    core_attention_map: np.ndarray,
    face_hint_map: Optional[np.ndarray] = None,
    alpha: float = 0.6,
    blend: float = 1.0,
) -> np.ndarray:
    """Apply a simple, explainable face prior to the core attention map.

    If no face_hint_map is provided, the output equals core_attention_map.
    """
    core = np.asarray(core_attention_map, dtype=np.float32)
    if core.ndim != 2:
        raise ValueError("core_attention_map must be a 2D (H, W) array")

    if face_hint_map is None:
        return core

    hint = np.asarray(face_hint_map, dtype=np.float32)
    if hint.shape != core.shape:
        raise ValueError("face_hint_map must match core_attention_map shape")

    alpha = max(0.0, float(alpha))
    blend = float(np.clip(blend, 0.0, 1.0))

    core = np.clip(core, 0.0, 1.0)
    hint = np.clip(hint, 0.0, 1.0)

    multiplier = 1.0 + alpha * hint
    modulated = core * multiplier
    modulated = np.clip(modulated, 0.0, 1.0 + alpha)

    normalized = _normalize(modulated)
    if blend >= 1.0:
        return normalized

    return (1.0 - blend) * core + blend * normalized


def _normalize(attention_map: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    min_val = float(np.min(attention_map))
    max_val = float(np.max(attention_map))
    if max_val - min_val < eps:
        return np.zeros_like(attention_map, dtype=np.float32)
    normalized = (attention_map - min_val) / (max_val - min_val)
    return np.clip(normalized, 0.0, 1.0)
