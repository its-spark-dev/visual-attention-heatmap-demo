from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from explanation import format_feature_explanation, summarize_feature_contributions
from visualization import build_heatmap_image, build_heatmap_legend, build_heatmap_overlay
from core_adapter.attention_runner import run_attention
from phase3.runner import run_phase3


def main() -> None:
    st.set_page_config(page_title="Visual Attention Heatmap Demo", layout="wide")

    st.title("Visual Attention Heatmap Demo")
    st.write(
        "Upload an image to see a rule-based visual attention heatmap and a short explanation."
    )
    st.info(
        "How to read this: warmer colors indicate regions that are more visually "
        "attention-grabbing. This is a deterministic, rule-based saliency map — "
        "it does not classify or interpret image content."
    )

    uploaded_file = st.file_uploader(
        "Upload an image (PNG or JPG)", type=["png", "jpg", "jpeg"]
    )

    if not uploaded_file:
        st.info("Awaiting an image upload.")
        return

    image = Image.open(uploaded_file).convert("RGB")

    with st.spinner("Computing attention map..."):
        result = run_attention(image)

    legend = build_heatmap_legend()

    st.subheader("Phase 3 (Layer 2 hints)")
    enable_phase3 = st.toggle("Enable Phase 3 (Layer 2 hints)", value=False)
    controls = st.columns(3)
    with controls[0]:
        alpha = st.slider(
            "Alpha (face hint strength)",
            0.0,
            2.0,
            0.6,
            0.05,
            disabled=not enable_phase3,
        )
    with controls[1]:
        beta = st.slider(
            "Beta (text hint strength)",
            0.0,
            2.0,
            0.6,
            0.05,
            disabled=not enable_phase3,
        )
    with controls[2]:
        blend = st.slider(
            "Blend (core ↔ hints)",
            0.0,
            1.0,
            1.0,
            0.05,
            disabled=not enable_phase3,
        )

    image_array = np.asarray(image, dtype=np.float32)
    final_attention = result.attention_map
    hint_maps = {}
    if enable_phase3:
        try:
            final_attention, hint_maps = run_phase3(
                image_array, result.attention_map, alpha, beta, blend
            )
        except Exception as exc:
            st.error(f"Phase 3 failed to run: {exc}")
            final_attention = result.attention_map
            hint_maps = {}

    core_overlay = build_heatmap_overlay(image, result.attention_map)
    final_overlay = build_heatmap_overlay(image, final_attention)

    st.subheader("Original")
    st.image(image, use_column_width=True)

    st.subheader("Before / After")
    left, right = st.columns(2)
    with left:
        st.markdown("**Core Attention (Overlay)**")
        st.image(core_overlay, use_column_width=True)
    with right:
        st.markdown("**Final Attention (Overlay)**")
        st.image(final_overlay, use_column_width=True)

    st.caption("Low attention → High attention")
    st.image(legend, use_column_width=False)

    if enable_phase3 and hint_maps:
        st.subheader("Hint Maps (Heatmap-only)")
        show_face = st.checkbox("Show face hint map", value=False)
        show_text = st.checkbox("Show text hint map", value=False)
        hint_columns = []
        if show_face:
            hint_columns.append(("Face hint", hint_maps.get("face")))
        if show_text:
            hint_columns.append(("Text hint", hint_maps.get("text")))

        if hint_columns:
            cols = st.columns(len(hint_columns))
            for col, (label, hint_map) in zip(cols, hint_columns):
                with col:
                    st.markdown(f"**{label}**")
                    if hint_map is None:
                        st.caption("No hint map available.")
                    else:
                        hint_heatmap = build_heatmap_image(hint_map, image.size)
                        st.image(hint_heatmap, use_column_width=True)

    st.subheader("Why these regions stand out")
    bullets = format_feature_explanation(result.feature_scores)
    for bullet in bullets:
        st.markdown(f"- {bullet}")

    contributions = summarize_feature_contributions(result.feature_scores)
    if contributions:
        st.subheader("Feature contributions")
        for label, percent, description in contributions:
            st.markdown(f"**{label}** — {percent:.0f}%")
            st.caption(description)

    st.caption(
        "This demo is deterministic and does not perform classification or learning."
    )


if __name__ == "__main__":
    main()
