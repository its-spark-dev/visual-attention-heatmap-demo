from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app.explanation import format_feature_explanation
from app.visualization import build_heatmap_overlay
from core_adapter.attention_runner import run_attention


def main() -> None:
    st.set_page_config(page_title="Visual Attention Heatmap Demo", layout="wide")

    st.title("Visual Attention Heatmap Demo")
    st.write(
        "Upload an image to see a rule-based visual attention heatmap and a short explanation."
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

    overlay = build_heatmap_overlay(image, result.attention_map)

    left, right = st.columns(2)
    with left:
        st.subheader("Original")
        st.image(image, use_column_width=True)
    with right:
        st.subheader("Attention Heatmap")
        st.image(overlay, use_column_width=True)

    st.subheader("Why these regions stand out")
    bullets = format_feature_explanation(result.feature_scores)
    for bullet in bullets:
        st.markdown(f"- {bullet}")

    st.caption(
        "This demo is deterministic and does not perform classification or learning."
    )


if __name__ == "__main__":
    main()
