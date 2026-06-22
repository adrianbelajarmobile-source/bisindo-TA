from __future__ import annotations

from typing import Optional

import numpy as np

try:
    from src.features.landmark_features import build_frame_feature_vector
except Exception:
    build_frame_feature_vector = None

def extract_landmark_row(hand_landmarks):
    row = []

    if not hand_landmarks or len(hand_landmarks) < 21:
        return None

    for lm in hand_landmarks:
        row.extend([lm.x, lm.y, lm.z])

    return np.asarray(row, dtype=np.float32)


def normalize_hand_block(coords: Optional[np.ndarray]) -> np.ndarray:
    """Fallback 63 fitur per tangan: wrist sebagai origin dan skala dari ukuran tangan."""
    if coords is None:
        return np.zeros(63, dtype=np.float32)

    pts = np.asarray(coords, dtype=np.float32).reshape(21, 3)
    origin = pts[0].copy()
    centered = pts - origin

    # Skala memakai jarak wrist ke middle_mcp + fallback bbox.
    scale = float(np.linalg.norm(pts[9, :2] - pts[0, :2]))
    if scale < 1e-6:
        xy = pts[:, :2]
        bbox = np.max(xy, axis=0) - np.min(xy, axis=0)
        scale = float(max(bbox[0], bbox[1], 1e-6))

    normalized = centered / scale
    return normalized.reshape(-1).astype(np.float32)


def build_sequence_frame_feature(
    *,
    left_coords: Optional[np.ndarray],
    right_coords: Optional[np.ndarray],
    left_label: str,
    right_label: str,
    left_present: bool,
    right_present: bool,
) -> np.ndarray:
    """Membuat 1 frame sequence dengan kontrak 127 fitur.

    Kontrak yang dipakai: [num_hands, right_63, left_63].
    Jika fungsi training src.features.landmark_features.build_frame_feature_vector tersedia,
    fungsi itu dicoba dulu agar inference sama dengan dataset sequence.
    """
    if build_frame_feature_vector is not None:
        call_attempts = [
            lambda: build_frame_feature_vector(
                left_coords=left_coords,
                right_coords=right_coords,
                left_label=left_label,
                right_label=right_label,
                left_present=left_present,
                right_present=right_present,
            ),
            lambda: build_frame_feature_vector(
                right_coords=right_coords,
                left_coords=left_coords,
                right_present=right_present,
                left_present=left_present,
            ),
            lambda: build_frame_feature_vector(
                left_coords,
                right_coords,
                left_present,
                right_present,
            ),
        ]

        for call in call_attempts:
            try:
                feature = np.asarray(call(), dtype=np.float32).reshape(-1)
                if feature.shape[0] == 127:
                    return feature
            except TypeError:
                continue
            except Exception as exc:
                print("build_frame_feature_vector gagal, pakai fallback lokal:", exc)
                break

    feature = np.zeros(127, dtype=np.float32)
    feature[0] = float(int(left_present) + int(right_present))
    feature[1:64] = normalize_hand_block(right_coords)
    feature[64:127] = normalize_hand_block(left_coords)
    return feature



def build_motion_raw_feature(
    *,
    left_coords: Optional[np.ndarray],
    right_coords: Optional[np.ndarray],
    left_present: bool,
    right_present: bool,
) -> np.ndarray:
    """Fitur raw 126 dimensi khusus untuk mendeteksi gerakan.

    Ini tidak dipakai untuk input model, hanya untuk menentukan kapan sequence mode mulai.
    Raw x/y/z dipakai agar perpindahan posisi tangan ikut terbaca.
    """
    feature = np.zeros(126, dtype=np.float32)
    if right_present and right_coords is not None:
        feature[0:63] = np.asarray(right_coords, dtype=np.float32).reshape(-1)
    if left_present and left_coords is not None:
        feature[63:126] = np.asarray(left_coords, dtype=np.float32).reshape(-1)
    return feature


