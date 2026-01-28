# visual-attention-heatmap-demo

Interactive demo for visualizing and explaining predicted human visual attention
on images using a rule-based attention core.

This repository is a **demo and visualization layer**, built on top of the
`visual-attention-heatmap` core library.  
It focuses on *interpretability, clarity, and qualitative insight* rather than
model training or quantitative benchmarks.

---

## 🎯 Purpose

The goal of this demo is to answer the question:

> “Given an image, where is a human likely to look first — and why?”

This project provides:
- A visual **attention heatmap overlay** on top of an input image
- A **feature-level explanation** of why certain regions are emphasized
- A lightweight way to explore human visual attention **without eye-tracking hardware**

This is **not** intended to replace eye-tracking systems, but to:
- Provide intuition
- Enable rapid qualitative comparisons
- Serve as a foundation for future applications (UX analysis, design review, education, etc.)

---

## 🧠 How It Works (High Level)

1. An image is uploaded by the user.
2. The image is passed to a **rule-based visual attention core**.
3. Multiple low-level visual features are computed (e.g. center bias, contrast, edge density).
4. Feature maps are normalized and fused into a single attention map.
5. The attention map is overlaid on the original image.
6. A textual explanation describes which features contributed most and why.

---

## 🧩 Project Structure

```text
visual-attention-heatmap-demo/
├─ app/
│  ├─ app.py              # Streamlit entry point
│  ├─ visualization.py    # Heatmap overlay & rendering logic
│  ├─ explanation.py      # Human-readable feature explanations
│
├─ core_adapter/
│  └─ attention_runner.py # Adapter for calling the core library
│
├─ README.md
├─ DESIGN.md
├─ requirements.txt
```

This repository does not contain the core attention algorithms themselves.
Those live in a separate repository and are imported as a dependency.

---

## 🚫 Explicit Non-Goals

To keep the demo focused and maintainable, this repository intentionally excludes:
- Machine learning or model training
- Eye-tracking hardware integration
- Quantitative accuracy benchmarks
- Dataset management
- Video or real-time processing
- Authentication, storage, or backend services

---

## 🛠️ Technology Stack

- Python
- Streamlit — interactive UI
- NumPy / PIL — image handling
- visual-attention-heatmap — rule-based attention core

---

## 🚀 Getting Started (Planned)

`Detailed setup instructions will be added once the first runnable demo is complete.`

At a high level:
1. Install dependencies 
2. Run the Streamlit app 
3. Upload an image and explore the results

---

## 🔮 Future Directions

Possible next steps include:

- Side-by-side comparison of multiple images 
- Feature weight sliders for exploration 
- Exportable reports (image + explanation)
- Web deployment 
- Integration with empirical eye-tracking data for comparison