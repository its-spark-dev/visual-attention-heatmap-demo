# Design Principles – Visual Attention Heatmap Demo

This document describes the architectural principles and design constraints
of the **visual-attention-heatmap-demo** repository.

---

## 1. Role of This Repository

This repository is a **presentation and exploration layer**.

It is responsible for:
- Visualizing attention maps
- Explaining feature contributions
- Providing an intuitive interface for humans

It is **not** responsible for:
- Defining attention algorithms
- Feature engineering
- Numerical optimization
- Learning or adaptation

All core computation lives in the separate core repository.

---

## 2. Separation of Concerns

### Core vs Demo

| Layer | Responsibility |
|-----|----------------|
| Core | Deterministic attention computation |
| Demo | Visualization, explanation, interaction |

The demo must treat the core as a **black box**:
- No modification of core logic
- No assumptions about internal implementation
- Only consume documented interfaces

---

## 3. Determinism & Explainability

The demo assumes that:
- Given the same image, the core produces the same output
- Feature contributions are interpretable and stable
- Explanations can be mapped directly to feature definitions

No randomness or hidden state should be introduced at the demo layer.

---

## 4. Visualization Philosophy

Visualization is intended to:
- Aid human understanding
- Reveal relative importance, not exact probabilities
- Support qualitative comparison

Design choices should favor:
- Clarity over visual flair
- Simplicity over configurability
- Explanation over automation

---

## 5. Feature Explanations

Each feature explanation should answer:
- What visual property does this feature capture?
- Why does it matter for human attention?
- What kind of regions does it emphasize?

Explanations should be:
- Human-readable
- Non-technical where possible
- Grounded in perceptual intuition, not neuroscience jargon

---

## 6. Input Assumptions

The demo may accept:
- Natural images
- Documents
- Posters, thumbnails, UI screenshots

The demo **must not**:
- Attempt to classify image type
- Apply semantic understanding
- Make claims about intent or meaning

All outputs are *visual saliency predictions*, not semantic judgments.

---

## 7. Extensibility Rules

Future extensions must:
- Preserve core/demo separation
- Avoid leaking application logic into the core
- Keep the demo runnable with minimal setup

If a feature requires:
- Persistent state
- Learning
- External services

…it likely belongs in a **different repository**.

---

## 8. Success Criteria

This demo is considered successful if:
- A user can upload an image and immediately understand the output
- The heatmap “feels reasonable” even if imperfect
- The explanation increases trust rather than confusion

Accuracy is secondary to *insight* at this stage.

---
