from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from phase3.hints.face_hint import build_face_hint_map
from phase3.hints.text_hint import build_text_hint_map
from phase3.modulator import modulate_attention


def run_phase3(
    image: np.ndarray,
    core_attention_map: np.ndarray,
    alpha: float,
    beta: float,
    blend: float,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Run Phase 3 hint generation and modulation."""
    face_hint = build_face_hint_map(image)
    text_hint = build_text_hint_map(image)

    combined_hint = np.clip(alpha * face_hint + beta * text_hint, 0.0, 1.0)
    final_attention = modulate_attention(
        core_attention_map,
        face_hint_map=combined_hint,
        alpha=1.0,
        blend=blend,
    )

    return final_attention, {"face": face_hint, "text": text_hint}
