"""Utilities for dual-hand feature extraction.

Format data yang didukung:
- satu tangan: one side present (left_present atau right_present = 1)
- dua tangan: both sides present
- setiap row menyimpan 63 koordinat untuk setiap sisi dan mask presence.
"""

from __future__ import annotations

import numpy as np

from src.features.landmark_features import FEATURE_COLS, build_feature_vector

LEFT_PREFIX = "left_"
RIGHT_PREFIX = "right_"
PRESENCE_COLS = ("left_present", "right_present")

LEFT_FEATURE_COLS = [f"{LEFT_PREFIX}{col}" for col in FEATURE_COLS]
RIGHT_FEATURE_COLS = [f"{RIGHT_PREFIX}{col}" for col in FEATURE_COLS]
DUAL_FEATURE_COLS = LEFT_FEATURE_COLS + RIGHT_FEATURE_COLS + list(PRESENCE_COLS)


def _to_numpy_coords(raw_coords) -> np.ndarray:
    if raw_coords is None:
        return np.zeros(63, dtype=np.float32)
    arr = np.asarray(raw_coords, dtype=np.float32).reshape(-1)
    if arr.size != 63:
        raise ValueError(f"Expected 63 coords, got {arr.size}")
    return arr


def build_dual_feature_vector(
    left_coords=None,
    right_coords=None,
    left_label: str = "unknown",
    right_label: str = "unknown",
    left_present: bool = False,
    right_present: bool = False,
) -> np.ndarray:
    """Build a unified 85*2 + 2 feature vector for one or two-hand samples."""
    left_vec = np.zeros(len(FEATURE_COLS), dtype=np.float32)
    right_vec = np.zeros(len(FEATURE_COLS), dtype=np.float32)

    if left_present and left_coords is not None:
        left_vec = build_feature_vector(_to_numpy_coords(left_coords), left_label)

    if right_present and right_coords is not None:
        right_vec = build_feature_vector(_to_numpy_coords(right_coords), right_label)

    return np.concatenate(
        [
            left_vec,
            right_vec,
            np.array([float(left_present), float(right_present)], dtype=np.float32),
        ]
    )


def make_dual_feature_columns(base_prefix: str = "") -> list[str]:
    """Helper untuk membuat kolom-kanonikal untuk script lain."""
    prefix = f"{base_prefix}_" if base_prefix else ""
    return [
        f"{prefix}{LEFT_PREFIX}{col}" for col in FEATURE_COLS
    ] + [
        f"{prefix}{RIGHT_PREFIX}{col}" for col in FEATURE_COLS
    ] + [
        f"{prefix}left_present",
        f"{prefix}right_present",
    ]
