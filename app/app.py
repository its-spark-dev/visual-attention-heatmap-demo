from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from explanation import format_feature_explanation, summarize_feature_contributions
from visualization import build_heatmap_image, build_heatmap_legend, build_heatmap_overlay
from core_adapter.attention_runner import run_attention


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

    view_mode = st.radio(
        "View mode",
        ["Overlay (image + heatmap)", "Heatmap only"],
        horizontal=True,
    )

    overlay = build_heatmap_overlay(image, result.attention_map)
    heatmap_only = build_heatmap_image(result.attention_map, image.size)
    legend = build_heatmap_legend()

    left, right = st.columns(2)
    with left:
        st.subheader("Original")
        st.image(image, use_column_width=True)
    with right:
        st.subheader("Attention Heatmap")
        if view_mode == "Heatmap only":
            st.image(heatmap_only, use_column_width=True)
        else:
            st.image(overlay, use_column_width=True)
        st.caption("Low attention → High attention")
        st.image(legend, use_column_width=False)

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
