# Phase 3: High-level Hint Modulation (Demo-only)

## Goal
Make the simulated attention map feel closer to where humans actually look,
by adding lightweight high-level priors (face/text/object hints) on top of the
core low-level saliency baseline.

Core remains unchanged and deterministic. All Phase 3 experiments live in demo repo.

## Why Phase 3
Observed issue in Phase 2:
- Strong center bias often dominates
- Low-level saliency reacts to contrast/edges but has no semantic understanding
- "saliency ≠ gaze" for many real images

Phase 3 adds a thin semantic prior layer to reduce "always center wins" behavior.

## Scope (In)
- Add Layer 2: hint extraction modules (face/text/[optional object])
- Add a modulator that combines core_attention + hint_maps into final_attention
- Streamlit UI: alpha/beta sliders + before/after comparison
- Qualitative evaluation protocol update (Phase 2 -> Phase 3)

## Non-goals (Out)
- No changes to core repo logic
- No end-to-end trained gaze model
- Not claiming eye-tracking equivalence
- No heavy dataset collection pipeline in MVP

## Architecture
Input image
  -> Core adapter (black box):
       core_attention_map (H x W, [0,1])
  -> Phase3 hint extractors:
       face_hint_map (H x W, [0,1])
       text_hint_map (H x W, [0,1])
       (optional) object_hint_map
  -> Modulator:
       final_attention_map (H x W, [0,1])
  -> Visualization + Explanation (demo UI)

## Modulation Strategy (Explainable)
We treat hints as multiplicative priors on the baseline saliency:

final = normalize( core * (1 + α*face + β*text + γ*object) )

Properties:
- If no hints: final == core (safe fallback)
- Hints only reweight existing saliency, not replace it
- α/β sliders are directly interpretable: "how much to trust the hint"

Optional stabilization:
- Clamp multiplier: (1 + ...) capped to avoid blow-ups
- Soft floor: final = normalize( (1-λ)*core + λ*core*multiplier )

## Hint Maps (MVP)
Face hint:
- Use a pretrained face detector to get bounding boxes
- Convert boxes to soft masks (Gaussian / distance transform)
- Normalize to [0,1], optionally dilate slightly

Text hint:
- Use a text detector (EAST/CRAFT) or OCR bounding boxes
- Convert boxes to soft masks, normalize to [0,1]

## Evaluation Plan (Qualitative MVP)
Goal is not perfect eye-tracking match, but "doesn't look weird to humans."

Test set (manual, small):
- 10–30 images covering:
  - portrait (face centered / face off-center)
  - product photo with text
  - poster/thumbnail with big title
  - document photo (dense text)
  - natural scene with no face/text

Procedure:
- For each image, record:
  - Core baseline result (before)
  - Modulated result with default α/β (after)
  - Notes: "where would a human look first?" + "does it feel plausible?"

Pass criteria (MVP):
- In face images, face region is consistently among top-attended areas
- In posters/docs, text blocks gain attention vs center-only blobs
- No catastrophic failures: full-map saturation, random artifacts, unstable outputs
- A small set of viewers (>=3) say "after looks more natural" more often than "before"

## Experiment Plan
Milestone 1: Face-only hint (α slider) + before/after UI
Milestone 2: Add text hint (β slider)
Milestone 3 (optional): Object hint and/or per-image auto-weights
Milestone 4: Better explanations (show hint masks + contribution deltas)

## Known Risks
- Face/text detector misses -> fallback to core
- Over-amplification -> clamp multiplier
- Domain mismatch (cartoons, low-res) -> degrade gracefully

## Version note (core metadata mismatch)
Core version metadata mismatch (tag v1.0.1 vs pyproject 1.0.0) is low priority.
Fix later; Phase 3 does not depend on it.
