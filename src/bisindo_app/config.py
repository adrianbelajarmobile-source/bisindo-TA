from __future__ import annotations

import threading
from pathlib import Path

# ============================================================
# PATH MODEL
# ============================================================
STATIC_MODEL_PATH = Path("models/best_model_dual.h5")
STATIC_LABELS_PATH = Path("models/labels_dual.json")

SEQUENCE_MODEL_PATH = Path("models/best_sequence_model.h5")
SEQUENCE_LABELS_PATH = Path("models/gesture_labels.json")

# Router model: memilih IDLE / STATIC / DYNAMIC sebelum alfabet/sequence.
ROUTER_MODEL_PATH = Path("models/router_model_final.h5")
ROUTER_LABELS_PATH = Path("models/router_labels.json")

MP_MODEL_PATH = Path("models/hand_landmarker.task")

# ============================================================
# PARAMETER STATIC / ALFABET
# ============================================================
BUFFER_SIZE = 6
VOTE_THRESHOLD = 3
CONFIDENCE_THRESHOLD = 0.25
MARGIN_THRESHOLD = 0.02
PROCESS_EVERY_N_FRAMES = 1

# ============================================================
# PARAMETER SEQUENCE / GERAKAN
# ============================================================
SEQUENCE_LENGTH = 30
SEQUENCE_CONFIDENCE_THRESHOLD = 0.40
SEQUENCE_MARGIN_THRESHOLD = 0.05
SEQUENCE_NO_HAND_CANCEL_FRAMES = 10
SEQUENCE_COOLDOWN_FRAMES = 15

AUTO_SEQUENCE_ON_MOTION = False
MOTION_START_THRESHOLD = 0.018
MOTION_TRIGGER_FRAMES = 3
MOTION_FORCE_SEQUENCE_THRESHOLD = 0.035

SEQUENCE_TRIGGER_USE_RAW_FALLBACK = False
SEQUENCE_TRIGGER_CONFIDENCE = 0.60
SEQUENCE_TRIGGER_MARGIN = 0.08

# ============================================================
# TRAINED ROUTER MODEL STATIC VS DYNAMIC
# ============================================================
USE_TRAINED_ROUTER = True
ROUTER_WINDOW_FRAMES = 12
ROUTER_CONFIDENCE_THRESHOLD = 0.60
ROUTER_STATIC_CONFIDENCE_THRESHOLD = 0.55
ROUTER_DYNAMIC_CONFIDENCE_THRESHOLD = 0.65
ROUTER_DYNAMIC_STABLE_FRAMES = 2
HAND_SETTLE_FRAMES = 3

STATIC_OVERRIDE_CONFIDENCE = 0.60
STATIC_OVERRIDE_MARGIN = 0.05
STATIC_OVERRIDE_MAX_MOTION = 0.010
LOCK_SEQUENCE_WHILE_SPELLING = True

# ============================================================
# TOKEN COMPOSER
# ============================================================
TOKEN_IDLE = "IDLE"
TOKEN_SEQUENCE = "SEQUENCE"
TOKEN_SPELLING = "SPELLING"
LOCK_SEQUENCE_DURING_SPELLING_TOKEN = True
SEQUENCE_RELEASE_FRAMES = 5

# ============================================================
# SMART COMPOSER / PENYUSUN KATA-KALIMAT
# ============================================================
LETTER_COMMIT_CONFIDENCE = 0.35
LETTER_COMMIT_MARGIN = 0.02
LETTER_STEADY_MOTION_MAX = 0.018
LETTER_COMMIT_STABLE_FRAMES = 2

CLEAR_UNCOMMITTED_LETTERS_ON_SEQUENCE = True
AUTO_COMMIT_SEQUENCE_WORD = True
SEQUENCE_DUPLICATE_COOLDOWN_FRAMES = 45
AUTO_SAVE_SPELLED_WORD_ON_GAP = True
WORD_GAP_NO_HAND_FRAMES = 12

SEQUENCE_TRIGGER_LABELS = {"sequel"}

# ============================================================
# RECOGNITION MODE
# ============================================================
DEFAULT_RECOGNITION_MODE = "auto"
MODE_AUTO = "auto"
MODE_ALPHABET = "alphabet"
MODE_KATA = "kata"
VALID_RECOGNITION_MODES = {MODE_AUTO, MODE_ALPHABET, MODE_KATA}

MIRROR_CAMERA_OUTPUT = True

STATIC_MODEL_LOCK = threading.Lock()
SEQUENCE_MODEL_LOCK = threading.Lock()
ROUTER_MODEL_LOCK = threading.Lock()
