from __future__ import annotations

from typing import Dict, Iterable, List, Tuple


_FEATURE_EXPLANATIONS = {
    "center_bias": "Central regions often attract first glances.",
    "contrast": "High-contrast areas stand out against their surroundings.",
    "edge_density": "Dense edges hint at detail and structure.",
    "center_surround": "Local contrast changes can draw early attention.",
    "color_contrast": "Color differences pull attention.",
    "brightness": "Brighter regions tend to pop forward.",
    "saliency": "Overall visual saliency emphasizes standout regions.",
}


def format_feature_explanation(feature_scores: Dict[str, float], top_k: int = 4) -> List[str]:
    """Return human-readable bullet strings based on feature scores."""
    if not feature_scores:
        return [
            "The attention map highlights regions that visually stand out.",
            "Strong contrast and central placement often drive early attention.",
        ]

    ranked = sorted(feature_scores.items(), key=lambda item: item[1], reverse=True)
    bullets: List[str] = []
    for feature, score in ranked[:top_k]:
        description = _FEATURE_EXPLANATIONS.get(
            feature,
            "This feature emphasizes visually distinctive regions.",
        )
        bullets.append(f"{feature.replace('_', ' ').title()}: {description}")

    return bullets
