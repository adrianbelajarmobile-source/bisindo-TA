"""Shared landmark feature engineering utilities."""

from itertools import combinations

import numpy as np


NUM_LANDMARKS = 21
LANDMARK_COLS = [f"{coord}{i}" for i in range(NUM_LANDMARKS) for coord in ("x", "y", "z")]
FINGERTIP_IDS = [4, 8, 12, 16, 20]
ANGLE_TRIPLETS = [
    (1, 2, 4),    # thumb
    (5, 6, 8),    # index
    (9, 10, 12),  # middle
    (13, 14, 16), # ring
    (17, 18, 20), # pinky
]

TIP_TO_WRIST_FEATURES = [f"dist_tip{i}_to_wrist" for i in FINGERTIP_IDS]
TIP_PAIR_FEATURES = [
    f"dist_tip{start}_tip{end}" for start, end in combinations(FINGERTIP_IDS, 2)
]
FINGER_ANGLE_FEATURES = [
    f"angle_{name}" for name in ("thumb", "index", "middle", "ring", "pinky")
]
HANDEDNESS_FEATURES = ["hand_left", "hand_right"]

ENGINEERED_FEATURE_COLS = (
    TIP_TO_WRIST_FEATURES + TIP_PAIR_FEATURES + FINGER_ANGLE_FEATURES + HANDEDNESS_FEATURES
)
FEATURE_COLS = LANDMARK_COLS + ENGINEERED_FEATURE_COLS


def normalize_landmarks(coords: np.ndarray) -> np.ndarray:
    coords = coords.reshape(NUM_LANDMARKS, 3).astype(np.float32)

    wrist = coords[0:1, :]
    coords = coords - wrist

    # Canonical rotation:
    # rotate hand so wrist -> middle_mcp points downward/upward consistently
    ref = coords[9, :2]  # middle_mcp x,y
    ref_norm = np.linalg.norm(ref)

    if ref_norm > 1e-6 and not np.isnan(ref_norm):
        angle = np.arctan2(ref[1], ref[0])
        target_angle = np.pi / 2.0
        rot = target_angle - angle

        cos_r = np.cos(rot)
        sin_r = np.sin(rot)

        x = coords[:, 0].copy()
        y = coords[:, 1].copy()

        coords[:, 0] = x * cos_r - y * sin_r
        coords[:, 1] = x * sin_r + y * cos_r

    pairwise = np.linalg.norm(
        coords[:, None, :] - coords[None, :, :],
        axis=-1,
    )

    max_dist = np.max(pairwise)

    if max_dist <= 0 or np.isnan(max_dist):
        return coords.flatten()

    return (coords / max_dist).flatten()


def encode_handedness(hand_label: str) -> np.ndarray:
    """Encode handedness as two binary features."""
    hand = str(hand_label or "").strip().lower()
    if hand == "left":
        return np.array([1.0, 0.0], dtype=np.float32)
    if hand == "right":
        return np.array([0.0, 1.0], dtype=np.float32)
    return np.array([0.0, 0.0], dtype=np.float32)


def _euclidean_distance(coords: np.ndarray, start: int, end: int) -> float:
    return float(np.linalg.norm(coords[start] - coords[end]))


def _joint_angle(coords: np.ndarray, start: int, mid: int, end: int) -> float:
    v1 = coords[start] - coords[mid]
    v2 = coords[end] - coords[mid]
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm_product <= 1e-8:
        return 0.0

    cosine = np.dot(v1, v2) / norm_product
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.arccos(cosine) / np.pi)


def extract_engineered_features(normalized_coords: np.ndarray) -> np.ndarray:
    """Create distance and angle features that help separate similar letters."""
    coords = normalized_coords.reshape(NUM_LANDMARKS, 3).astype(np.float32)
    features = []

    for fingertip_id in FINGERTIP_IDS:
        features.append(_euclidean_distance(coords, 0, fingertip_id))

    for start, end in combinations(FINGERTIP_IDS, 2):
        features.append(_euclidean_distance(coords, start, end))

    for start, mid, end in ANGLE_TRIPLETS:
        features.append(_joint_angle(coords, start, mid, end))

    return np.asarray(features, dtype=np.float32)


def build_feature_vector(raw_coords: np.ndarray, hand_label: str = "unknown") -> np.ndarray:
    """Build the full feature vector used by training and inference."""
    normalized = normalize_landmarks(raw_coords)
    engineered = extract_engineered_features(normalized)
    handedness = encode_handedness(hand_label)
    return np.concatenate([normalized, engineered, handedness]).astype(np.float32)


def resample_sequence(sequence: np.ndarray, target_len: int) -> np.ndarray:
    """Resample a sequence to a fixed temporal length using linear interpolation.

    This keeps gesture timing more consistent than simple truncation/padding.
    """
    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.ndim != 2:
        raise ValueError(
            f"Expected sequence with shape (frames, features), got {sequence.shape}"
        )

    if sequence.shape[0] == 0:
        return np.zeros((target_len, sequence.shape[1]), dtype=np.float32)
    if sequence.shape[0] == target_len:
        return sequence.copy()

    src_indices = np.arange(sequence.shape[0], dtype=np.float32)
    dst_indices = np.linspace(0, sequence.shape[0] - 1, target_len)

    resampled = np.empty((target_len, sequence.shape[1]), dtype=np.float32)
    for i in range(sequence.shape[1]):
        resampled[:, i] = np.interp(dst_indices, src_indices, sequence[:, i])
    return resampled

