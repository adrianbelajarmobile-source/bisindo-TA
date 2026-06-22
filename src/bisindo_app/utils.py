from __future__ import annotations

import json
from pathlib import Path

def normalize_key(label: str) -> str:
    return str(label).strip().lower().replace(" ", "_").replace("-", "_")



def is_alphabet_label(label: str) -> bool:
    label = str(label).strip().upper()
    return len(label) == 1 and "A" <= label <= "Z"


def load_labels(labels_path: Path):
    """Mendukung format {"A": 0}, {"0": "A"}, atau ["A", "B"]."""
    with open(labels_path, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        return {idx: str(label) for idx, label in enumerate(data)}

    if not isinstance(data, dict):
        raise ValueError(f"Format labels tidak dikenali: {labels_path}")

    # Format idx -> label, contoh {"0": "AKU"}
    if all(str(k).isdigit() for k in data.keys()):
        return {int(k): str(v) for k, v in data.items()}

    # Format label -> idx, contoh {"AKU": 0}
    return {int(v): str(k) for k, v in data.items()}


