from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
from PIL import Image

from core.features import (
    CenterBiasFeature,
    CenterSurroundFeature,
    ContrastFeature,
    EdgeDensityFeature,
)
from core.fusion import fuse_features


@dataclass
class AttentionResult:
    attention_map: np.ndarray
    feature_scores: Dict[str, float]


def run_attention(image: Image.Image) -> AttentionResult:
    """Run the core visual attention pipeline on a PIL image."""
    if image.mode != "RGB":
        image = image.convert("RGB")

    image_array = np.asarray(image).astype(np.float32)
    return _run_feature_pipeline(image_array)


def _run_feature_pipeline(image: np.ndarray) -> AttentionResult:
    features, feature_names, weights = _load_core_features()
    feature_maps = [feature(image) for feature in features]
    fused = fuse_features(features, image, weights=weights)
    feature_scores = _score_features(feature_maps, feature_names, weights)
    return AttentionResult(attention_map=fused, feature_scores=feature_scores)


def _load_core_features():
    features = [
        CenterBiasFeature(),
        ContrastFeature(),
        EdgeDensityFeature(),
        CenterSurroundFeature(),
    ]
    feature_names = [
        "center_bias",
        "contrast",
        "edge_density",
        "center_surround",
    ]
    weights = np.full(len(features), 1.0 / len(features), dtype=np.float32)
    return features, feature_names, weights


def _score_features(
    feature_maps: Sequence[np.ndarray],
    feature_names: Sequence[str],
    weights: Sequence[float],
) -> Dict[str, float]:
    contributions: List[float] = []
    for feature_map, weight in zip(feature_maps, weights):
        contributions.append(float(np.mean(feature_map)) * float(weight))

    total = float(np.sum(contributions))
    if total <= 0.0:
        return {name: 0.0 for name in feature_names}

    return {
        name: value / total for name, value in zip(feature_names, contributions)
    }
