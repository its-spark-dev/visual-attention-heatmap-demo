from __future__ import annotations

from typing import Dict, List, Tuple


_FEATURE_EXPLANATIONS = {
    "center_bias": "Central regions often attract first glances.",
    "contrast": "High-contrast areas stand out against their surroundings.",
    "edge_density": "Dense edges hint at detail and structure.",
    "center_surround": "Local contrast changes can draw early attention.",
    "color_contrast": "Color differences pull attention.",
    "brightness": "Brighter regions tend to pop forward.",
    "saliency": "Overall visual saliency emphasizes standout regions.",
}

_FEATURE_LABELS = {
    "center_bias": "Center placement",
    "contrast": "Contrast",
    "edge_density": "Edges and detail",
    "center_surround": "Local contrast changes",
    "color_contrast": "Color contrast",
    "brightness": "Brightness",
    "saliency": "Overall saliency",
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


def summarize_feature_contributions(
    feature_scores: Dict[str, float],
    top_k: int = 6,
) -> List[Tuple[str, float, str]]:
    """Return (label, percentage, description) tuples for UI display."""
    if not feature_scores:
        return []

    ranked = sorted(feature_scores.items(), key=lambda item: item[1], reverse=True)
    summaries: List[Tuple[str, float, str]] = []
    for feature, score in ranked[:top_k]:
        label = _FEATURE_LABELS.get(feature, feature.replace("_", " ").title())
        description = _FEATURE_EXPLANATIONS.get(
            feature, "This feature emphasizes visually distinctive regions."
        )
        summaries.append((label, float(score) * 100.0, description))
    return summaries
