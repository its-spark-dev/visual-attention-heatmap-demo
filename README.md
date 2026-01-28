# visual-attention-heatmap-demo

Interactive demo for visualizing and explaining **rule-based visual saliency**
on images using a deterministic attention core (v1.0).

This repository is a **demo and visualization layer**, built on top of the
`visual-attention-heatmap` core library.  
It focuses on *interpretability, clarity, and qualitative insight* rather than
model training or quantitative benchmarks.

---

## 🎯 Purpose

The goal of this demo is to answer the question:

> “Given an image, which regions are visually salient — and why?”

This project provides:
- A visual **attention heatmap overlay** on top of an input image
- A **feature-level explanation** of why certain regions are emphasized
- A lightweight way to explore human visual attention **without eye-tracking hardware**

This is **not** intended to replace eye-tracking systems. It is intended to:
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

## ✅ What the System Does Well

- Produces deterministic, repeatable saliency maps for the same image
- Highlights **low-level visual cues** (contrast, edges, center bias)
- Offers **transparent explanations** tied directly to feature definitions
- Supports fast, qualitative comparisons between images

---

## ❌ What the System Does NOT Do

- It does **not** measure or predict actual human eye-tracking data
- It does **not** interpret semantics (faces, text meaning, objects, intent)
- It does **not** learn, adapt, or improve with usage
- It does **not** provide calibrated probabilities of gaze

---

## ⚠️ Known Limitations & Biases

- **Center bias is strong**: central regions tend to dominate even when peripheral content is compelling.
- **Low-level bias**: the system favors contrast and edges over semantic importance.
- **Context-blind**: it does not understand task, viewer intent, or cultural salience.
- **Resolution sensitivity**: extremely small or noisy images can produce unstable maps.

---

## 🧪 Qualitative Test Plan (Phase 2)

**Image categories**
- Synthetic patterns (grids, single shapes, uniform fields)
- Natural photos (people, landscapes, street scenes)
- Documents (text-heavy pages, forms, posters)
- Layouts / UI (web pages, app screens, thumbnails)

**Expected feature behavior**
- Center bias peaks near image center across categories
- High-contrast edges produce hotspots (logos, headlines, sharp borders)
- Center–surround responds to local contrast changes (icons, separators)

**Failure cases worth documenting**
- Saliency highlights non-semantic textures (grass, noise) over key content
- Dominant center bias overwhelms meaningful off-center elements
- Complex documents where dense text drowns titles or callouts

---

## 🧭 Current Observed Issues

- **Center dominance** is often stronger than desired in wide layouts.
- **Saliency ≠ gaze**: maps can feel plausible but do not reflect true human fixation patterns.
- **Semantics missing**: faces or key objects are not privileged unless they have strong low-level cues.

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
