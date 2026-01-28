from __future__ import annotations

import numpy as np

from phase3.modulator import modulate_attention


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_no_hints_passthrough() -> None:
    core = np.array([[0.1, 0.5], [0.2, 0.9]], dtype=np.float32)
    output = modulate_attention(core, face_hint_map=None, alpha=0.6, blend=1.0)
    _assert(np.array_equal(core, output), "Output should equal core when no hint map.")


def test_face_hint_increases_attention() -> None:
    core = np.array([[0.2, 0.2], [0.2, 0.2]], dtype=np.float32)
    hint = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    output = modulate_attention(core, face_hint_map=hint, alpha=1.0, blend=1.0)
    _assert(output[0, 0] > output[0, 1], "Hinted region should increase attention.")


def test_output_range_and_dtype() -> None:
    core = np.random.rand(4, 4).astype(np.float32)
    hint = (np.random.rand(4, 4) > 0.5).astype(np.float32)
    output = modulate_attention(core, face_hint_map=hint, alpha=1.5, blend=0.8)
    _assert(output.dtype == np.float32, "Output must be float32.")
    _assert(np.min(output) >= 0.0, "Output must be >= 0.")
    _assert(np.max(output) <= 1.0, "Output must be <= 1.")


def run_smoke_tests() -> None:
    test_no_hints_passthrough()
    test_face_hint_increases_attention()
    test_output_range_and_dtype()
    print("Phase 3 smoke tests passed.")


if __name__ == "__main__":
    run_smoke_tests()
