# from __future__ import annotations

# from collections import deque
# from typing import Deque, Optional

# import av
# import cv2
# import mediapipe as mp
# import numpy as np
# import tensorflow as tf
# from streamlit_webrtc import VideoProcessorBase
# from mediapipe.tasks.python import vision
# from mediapipe.tasks.python.vision import HandLandmarkerOptions, RunningMode

# from src.features.landmark_features_dual import build_dual_feature_vector

# from .config import *
# from .drawing import center_crop_to_4_3, draw_landmarks, draw_scanner_overlay
# from .features import build_motion_raw_feature, build_sequence_frame_feature, extract_landmark_row
# from .models import (
#     load_cached_router_model,
#     load_cached_sequence_model,
#     load_cached_static_model,
#     load_router_labels_safe,
#     predict_with_model,
# )
# from .utils import is_alphabet_label, load_labels, normalize_key

# class BisindoVideoProcessor(VideoProcessorBase):
#     def __init__(self):
#         self.static_model = load_cached_static_model()
#         self.sequence_model = load_cached_sequence_model()
#         self.router_model = load_cached_router_model()

#         self.static_idx_to_label = load_labels(STATIC_LABELS_PATH)
#         self.sequence_idx_to_label = load_labels(SEQUENCE_LABELS_PATH)
#         self.router_idx_to_label = load_router_labels_safe(ROUTER_LABELS_PATH)

#         static_labels_norm = {normalize_key(v) for v in self.static_idx_to_label.values()}
#         self.available_sequence_triggers = static_labels_norm.intersection(SEQUENCE_TRIGGER_LABELS)
#         if not self.available_sequence_triggers:
#             print(
#                 "PERINGATAN: Tidak ada label trigger sequence di static labels. "
#                 "Tambahkan class seperti 'sequel' / 'sequence' ke model static, "
#                 "atau ubah SEQUENCE_TRIGGER_LABELS agar cocok dengan labels_dual.json."
#             )

#         self.prediction_buffer: Deque[Optional[int]] = deque(maxlen=BUFFER_SIZE)
#         self.sequence_buffer: Deque[np.ndarray] = deque(maxlen=SEQUENCE_LENGTH)

#         # Router buffer: 12 frame fitur 127 untuk model router IDLE / STATIC / DYNAMIC.
#         self.router_motion_buffer: Deque[np.ndarray] = deque(maxlen=ROUTER_WINDOW_FRAMES)
#         self.router_sequence_buffer: Deque[np.ndarray] = deque(maxlen=ROUTER_WINDOW_FRAMES)

#         self.predicted_letter = "..."
#         self.raw_prediction = "..."
#         self.confidence = 0.0
#         self.margin = 0.0
#         self.hand_count = 0

#         self.sequence_prediction = "..."
#         self.sequence_confidence = 0.0
#         self.sequence_margin = 0.0
#         self.sequence_progress = 0
#         self.mode = "STATIC"  # STATIC atau SEQUENCE internal
#         self.recognition_mode = DEFAULT_RECOGNITION_MODE
#         self.token_mode = TOKEN_IDLE
#         self.sequence_no_hand_frames = 0
#         self.sequence_cooldown = 0
#         self.sequence_locked_until_release = False
#         self.last_motion_feature = None
#         self.motion_active_frames = 0
#         self.motion_score = 0.0
#         self.router_prediction = "WAIT"
#         self.router_confidence = 0.0
#         self.router_margin = 0.0
#         self.router_dynamic_streak = 0
#         self.hand_present_frames = 0

#         self.frame_count = 0

#         # Buffer teks
#         self.hasil_kata = ""
#         self.sentence_words = []
#         self.last_added_letter = None

#         # Smart composer state.
#         self.pending_letter = None
#         self.pending_letter_frames = 0
#         self.no_hand_frames = 0
#         self.hasil_kata_committed_preview = False
#         self.last_sequence_word = None
#         self.last_sequence_commit_frame = -10_000
#         self.last_static_letter_preview = None
#         self.last_static_letter_confidence = 0.0
#         self.last_static_letter_margin = 0.0

#         self.lock = threading.Lock()

#         options = HandLandmarkerOptions(
#             base_options=mp.tasks.BaseOptions(
#                 model_asset_path=str(MP_MODEL_PATH)
#             ),
#             running_mode=RunningMode.IMAGE,
#             min_hand_detection_confidence=0.3,
#             min_tracking_confidence=0.3,
#             num_hands=2,
#         )

#         self.landmarker = vision.HandLandmarker.create_from_options(options)

#     def set_recognition_mode(self, mode: str):
#         mode = str(mode).strip().lower()
#         if mode not in VALID_RECOGNITION_MODES:
#             mode = DEFAULT_RECOGNITION_MODE

#         with self.lock:
#             if mode != self.recognition_mode:
#                 self.recognition_mode = mode
#                 self.prediction_buffer.clear()
#                 self.reset_pending_letter()
#                 self.last_added_letter = None
#                 self.last_motion_feature = None
#                 self.motion_active_frames = 0
#                 self.motion_score = 0.0
#                 self.router_prediction = "WAIT"
#                 self.router_confidence = 0.0
#                 self.router_margin = 0.0

#                 # Kalau pindah mode, batalkan rekaman sequence yang sedang setengah jalan.
#                 if self.mode == "SEQUENCE":
#                     self.cancel_sequence_mode()

#     def run_static_inference(self, input_data: np.ndarray):
#         return predict_with_model(self.static_model, input_data, STATIC_MODEL_LOCK)

#     def run_sequence_inference(self, sequence_data: np.ndarray):
#         return predict_with_model(self.sequence_model, sequence_data, SEQUENCE_MODEL_LOCK)

#     def run_router_inference(self):
#         if len(self.router_sequence_buffer) < ROUTER_WINDOW_FRAMES:
#             return "WAIT", 0.0, 0.0

#         input_data = np.asarray(self.router_sequence_buffer, dtype=np.float32).reshape(
#             1, ROUTER_WINDOW_FRAMES, 127
#         )

#         with ROUTER_MODEL_LOCK:
#             input_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)
#             probs = self.router_model(input_tensor, training=False).numpy()[0]

#         ranked = np.argsort(probs)[::-1]
#         pred_idx = int(ranked[0])
#         conf = float(probs[pred_idx])
#         second_conf = float(probs[ranked[1]]) if len(ranked) > 1 else 0.0
#         margin = conf - second_conf
#         label = self.router_idx_to_label.get(pred_idx, str(pred_idx))
#         return label, conf, margin

#     def clear_text(self):
#         with self.lock:
#             self.hasil_kata = ""
#             self.sentence_words = []
#             self.last_added_letter = None
#             self.pending_letter = None
#             self.pending_letter_frames = 0
#             self.no_hand_frames = 0
#             self.hasil_kata_committed_preview = False
#             self.last_sequence_word = None
#             self.last_sequence_commit_frame = -10_000
#             self.last_static_letter_preview = None
#             self.last_static_letter_confidence = 0.0
#             self.last_static_letter_margin = 0.0
#             self.prediction_buffer.clear()
#             self.sequence_buffer.clear()
#             self.router_motion_buffer.clear()
#             self.router_sequence_buffer.clear()
#             self.sequence_prediction = "..."
#             self.sequence_confidence = 0.0
#             self.sequence_margin = 0.0
#             self.sequence_progress = 0
#             self.mode = "STATIC"
#             self.token_mode = TOKEN_IDLE
#             self.sequence_no_hand_frames = 0
#             self.sequence_cooldown = 0
#             self.sequence_locked_until_release = False
#             self.last_motion_feature = None
#             self.motion_active_frames = 0
#             self.motion_score = 0.0

#     def force_commit_pending_letter(self):
#         """Dipanggil saat tombol Simpan ditekan agar huruf terakhir tidak hilang."""
#         label = self.pending_letter or self.last_static_letter_preview
#         if not label:
#             return

#         if not is_alphabet_label(label):
#             return

#         if self.hasil_kata_committed_preview:
#             self.hasil_kata = ""
#             self.hasil_kata_committed_preview = False

#         # Hindari dobel saat huruf yang sama sudah terakhir masuk.
#         if label != self.last_added_letter:
#             self.hasil_kata += label
#             self.last_added_letter = label

#         self.reset_pending_letter()

#     def save_current_word(self):
#         with self.lock:
#             # Kalau sedang spelling, tombol Simpan harus mengunci huruf pending dulu.
#             if not self.hasil_kata_committed_preview:
#                 self.force_commit_pending_letter()

#             # Kalau hasil_kata adalah preview dari sequence yang sudah auto masuk kalimat,
#             # tombol Simpan cukup membersihkan preview supaya tidak dobel.
#             if self.hasil_kata_committed_preview:
#                 self.hasil_kata = ""
#                 self.hasil_kata_committed_preview = False
#                 self.last_added_letter = None
#                 self.pending_letter = None
#                 self.pending_letter_frames = 0
#                 return

#             word = self.hasil_kata.strip()

#             if not word and self.predicted_letter not in ("", "..."):
#                 # Jangan simpan label trigger sequence sebagai kata.
#                 if normalize_key(self.predicted_letter) not in SEQUENCE_TRIGGER_LABELS:
#                     word = self.predicted_letter

#             if word:
#                 self.sentence_words.append(word.upper())
#                 self.hasil_kata = ""
#                 self.hasil_kata_committed_preview = False
#                 self.last_added_letter = None
#                 self.pending_letter = None
#                 self.pending_letter_frames = 0

#     def get_ui_state(self):
#         with self.lock:
#             extra_current_word = []
#             if self.hasil_kata and not self.hasil_kata_committed_preview:
#                 extra_current_word = [self.hasil_kata]

#             sentence_preview = " ".join(self.sentence_words + extra_current_word)

#             return {
#                 "predicted_letter": self.predicted_letter,
#                 "raw_prediction": self.raw_prediction,
#                 "confidence": self.confidence,
#                 "margin": self.margin,
#                 "hand_count": self.hand_count,
#                 "hasil_kata": self.hasil_kata,
#                 "hasil_kalimat": sentence_preview,
#                 "mode": self.mode,
#                 "sequence_prediction": self.sequence_prediction,
#                 "sequence_confidence": self.sequence_confidence,
#                 "sequence_margin": self.sequence_margin,
#                 "sequence_progress": self.sequence_progress,
#                 "sequence_trigger_ready": bool(self.available_sequence_triggers),
#                 "auto_sequence_on_motion": AUTO_SEQUENCE_ON_MOTION,
#                 "motion_score": self.motion_score,
#                 "pending_letter": self.pending_letter or "",
#                 "pending_letter_frames": self.pending_letter_frames,
#                 "hasil_kata_committed_preview": self.hasil_kata_committed_preview,
#                 "last_static_letter_preview": self.last_static_letter_preview or "",
#                 "recognition_mode": self.recognition_mode,
#                 "router_frames": len(self.router_motion_buffer),
#                 "router_window": ROUTER_WINDOW_FRAMES,
#                 "token_mode": self.token_mode,
#                 "sequence_locked_until_release": self.sequence_locked_until_release,
#                 "router_prediction": self.router_prediction,
#                 "router_confidence": self.router_confidence,
#                 "router_margin": self.router_margin,
#                 "router_dynamic_streak": self.router_dynamic_streak,
#                 "hand_present_frames": self.hand_present_frames,
#             }

#     def stabilize_prediction(
#         self,
#         predicted_idx: Optional[int],
#         is_confident: bool,
#     ) -> str:
#         self.prediction_buffer.append(predicted_idx if is_confident else None)

#         if len(self.prediction_buffer) < BUFFER_SIZE:
#             return "..."

#         valid_predictions = [
#             idx for idx in self.prediction_buffer if idx is not None
#         ]

#         if len(valid_predictions) < VOTE_THRESHOLD:
#             return "..."

#         counts = {}

#         for idx in valid_predictions:
#             counts[idx] = counts.get(idx, 0) + 1

#         max_count = max(counts.values())

#         if max_count >= VOTE_THRESHOLD:
#             stable_idx = max(counts, key=counts.get)
#             return self.static_idx_to_label[stable_idx]

#         return "..."

#     def reset_pending_letter(self):
#         self.pending_letter = None
#         self.pending_letter_frames = 0

#     def commit_spelled_word_if_needed(self):
#         if not AUTO_SAVE_SPELLED_WORD_ON_GAP:
#             return

#         if self.hasil_kata and not self.hasil_kata_committed_preview:
#             word = self.hasil_kata.strip().upper()
#             if word:
#                 self.sentence_words.append(word)
#             self.hasil_kata = ""
#             self.hasil_kata_committed_preview = False
#             self.last_added_letter = None
#             self.reset_pending_letter()
#             self.token_mode = TOKEN_IDLE

#     def handle_static_letter_candidate(
#         self,
#         label: str,
#         confidence: float,
#         margin: float,
#         hand_count: int,
#     ):
#         """Composer alfabet yang lebih pintar.

#         Huruf tidak langsung ditulis. Huruf hanya masuk jika:
#         - label stabil beberapa frame,
#         - confidence/margin cukup,
#         - motion rendah / tangan diam,
#         - bukan sedang sequence,
#         - bukan pengulangan huruf yang sama tanpa release tangan.
#         """
#         if hand_count <= 0:
#             self.reset_pending_letter()
#             return

#         if label in ("", "..."):
#             self.reset_pending_letter()
#             return

#         if normalize_key(label) in SEQUENCE_TRIGGER_LABELS:
#             self.reset_pending_letter()
#             return

#         # Kalau tangan sedang bergerak, jangan pernah tulis alfabet.
#         if self.motion_score > LETTER_STEADY_MOTION_MAX:
#             self.reset_pending_letter()
#             return

#         if confidence < LETTER_COMMIT_CONFIDENCE or margin < LETTER_COMMIT_MARGIN:
#             self.reset_pending_letter()
#             return

#         if label != self.pending_letter:
#             self.pending_letter = label
#             self.pending_letter_frames = 1
#             return

#         self.pending_letter_frames += 1

#         if self.pending_letter_frames < LETTER_COMMIT_STABLE_FRAMES:
#             return

#         # Jangan menulis huruf sama berkali-kali saat pose masih ditahan.
#         if label == self.last_added_letter:
#             return

#         # Kalau sebelumnya hasil_kata adalah preview sequence yang sudah masuk kalimat,
#         # mulai ejaan baru dengan buffer kosong.
#         if self.hasil_kata_committed_preview:
#             self.hasil_kata = ""
#             self.hasil_kata_committed_preview = False

#         self.enter_spelling_mode()
#         self.hasil_kata += label
#         self.last_added_letter = label
#         self.reset_pending_letter()

#     def commit_sequence_word(self, word: str):
#         word = str(word).strip().upper()
#         if not word or word in ("...", "-"):
#             return

#         # Anti dobel: kalau kata yang sama baru saja masuk, jangan append lagi.
#         is_duplicate = (
#             self.last_sequence_word == word
#             and (self.frame_count - self.last_sequence_commit_frame) < SEQUENCE_DUPLICATE_COOLDOWN_FRAMES
#         )

#         if AUTO_COMMIT_SEQUENCE_WORD and not is_duplicate:
#             self.sentence_words.append(word)
#             self.last_sequence_word = word
#             self.last_sequence_commit_frame = self.frame_count
#             self.hasil_kata_committed_preview = True
#         else:
#             self.hasil_kata_committed_preview = False

#         # Hasil kata tetap menampilkan kata terakhir, tapi tidak didobel di kalimat.
#         self.hasil_kata = word
#         self.last_added_letter = None
#         self.reset_pending_letter()
#         self.token_mode = TOKEN_IDLE

#     def reset_router_buffers(self):
#         self.router_motion_buffer.clear()
#         self.router_sequence_buffer.clear()
#         self.router_prediction = "WAIT"
#         self.router_confidence = 0.0
#         self.router_margin = 0.0
#         self.router_dynamic_streak = 0

#     def update_temporal_router(
#         self,
#         motion_feature: Optional[np.ndarray],
#         sequence_feature: Optional[np.ndarray],
#         hand_count: int,
#     ):
#         # Nama fungsi dipertahankan agar perubahan di recv minimal.
#         # Sekarang isinya bukan threshold motion, tapi buffer untuk trained router model.
#         if not USE_TRAINED_ROUTER or hand_count <= 0 or sequence_feature is None:
#             self.reset_router_buffers()
#             self.router_prediction = "IDLE" if hand_count <= 0 else "WAIT"
#             self.router_confidence = 0.0
#             self.router_margin = 0.0
#             self.motion_score = 0.0
#             return

#         self.router_sequence_buffer.append(sequence_feature.copy())
#         self.router_motion_buffer.append(sequence_feature.copy())

#         # Motion score hanya debug visual, bukan penentu router.
#         self.motion_score = self.compute_router_motion_score()

#         if self.router_ready():
#             label, conf, margin = self.run_router_inference()
#             self.router_prediction = label
#             self.router_confidence = conf
#             self.router_margin = margin
#         else:
#             self.router_prediction = "WAIT"
#             self.router_confidence = 0.0
#             self.router_margin = 0.0

#     def compute_router_motion_score(self) -> float:
#         if len(self.router_sequence_buffer) < 2:
#             return 0.0

#         frames = list(self.router_sequence_buffer)
#         diffs = [float(np.mean(np.abs(frames[i][1:] - frames[i - 1][1:]))) for i in range(1, len(frames))]
#         return float(max(diffs)) if diffs else 0.0

#     def router_ready(self) -> bool:
#         return len(self.router_sequence_buffer) >= ROUTER_WINDOW_FRAMES

#     def router_sees_dynamic(self) -> bool:
#         if not self.router_ready():
#             self.router_dynamic_streak = 0
#             return False

#         # Saat tangan baru masuk kamera, frame awal menuju pose A/B/C sering terlihat bergerak.
#         # Jangan langsung anggap dynamic sampai tangan stabil beberapa frame.
#         if self.hand_present_frames < HAND_SETTLE_FRAMES:
#             self.router_dynamic_streak = 0
#             return False

#         is_dynamic = (
#             normalize_key(self.router_prediction) == "dynamic"
#             and self.router_confidence >= ROUTER_DYNAMIC_CONFIDENCE_THRESHOLD
#         )

#         if is_dynamic:
#             self.router_dynamic_streak += 1
#         else:
#             self.router_dynamic_streak = 0

#         return self.router_dynamic_streak >= ROUTER_DYNAMIC_STABLE_FRAMES

#     def router_sees_static(self) -> bool:
#         if not self.router_ready():
#             return False
#         return (
#             normalize_key(self.router_prediction) == "static"
#             and self.router_confidence >= ROUTER_STATIC_CONFIDENCE_THRESHOLD
#         )

#     def enter_spelling_mode(self):
#         if self.token_mode != TOKEN_SPELLING:
#             self.token_mode = TOKEN_SPELLING
#             # Saat spelling dimulai, preview sequence lama tidak boleh dianggap kata aktif.
#             if self.hasil_kata_committed_preview:
#                 self.hasil_kata = ""
#                 self.hasil_kata_committed_preview = False
#             self.reset_router_buffers()

#     def currently_spelling_word(self) -> bool:
#         if self.token_mode == TOKEN_SPELLING:
#             return True
#         return bool(self.hasil_kata and not self.hasil_kata_committed_preview)

#     def preload_sequence_from_router(self):
#         # Masukkan frame awal gerakan yang sudah ada di router agar sequence tidak kehilangan start motion.
#         for feat in list(self.router_sequence_buffer):
#             if feat is not None and len(self.sequence_buffer) < SEQUENCE_LENGTH:
#                 self.sequence_buffer.append(feat)

#     def update_motion_state(self, motion_feature: Optional[np.ndarray], hand_count: int) -> bool:
#         if not AUTO_SEQUENCE_ON_MOTION or motion_feature is None or hand_count <= 0:
#             self.last_motion_feature = None
#             self.motion_active_frames = 0
#             self.motion_score = 0.0
#             return False

#         if self.last_motion_feature is None:
#             self.last_motion_feature = motion_feature.copy()
#             self.motion_active_frames = 0
#             self.motion_score = 0.0
#             return False

#         diff = np.abs(motion_feature - self.last_motion_feature)
#         self.motion_score = float(np.mean(diff))
#         self.last_motion_feature = motion_feature.copy()

#         if self.motion_score >= MOTION_START_THRESHOLD:
#             self.motion_active_frames += 1
#         else:
#             self.motion_active_frames = 0

#         return self.motion_active_frames >= MOTION_TRIGGER_FRAMES

#     def start_sequence_mode(self):
#         self.mode = "SEQUENCE"
#         self.token_mode = TOKEN_SEQUENCE
#         self.sequence_buffer.clear()
#         self.sequence_progress = 0
#         self.sequence_no_hand_frames = 0
#         self.prediction_buffer.clear()
#         self.last_added_letter = None
#         self.reset_pending_letter()

#         # Ini bagian penting: begitu sequence mulai, alfabet yang belum benar-benar
#         # dimaksudkan dibersihkan agar gerakan kata tidak diawali huruf sampah.
#         if CLEAR_UNCOMMITTED_LETTERS_ON_SEQUENCE and not self.hasil_kata_committed_preview:
#             # Untuk keamanan, hanya buang buffer pendek yang biasanya noise awal gerakan.
#             if len(self.hasil_kata.strip()) <= 3:
#                 self.hasil_kata = ""

#         self.last_motion_feature = None
#         self.motion_active_frames = 0
#         self.motion_score = 0.0

#     def cancel_sequence_mode(self):
#         self.mode = "STATIC"
#         self.token_mode = TOKEN_IDLE
#         self.sequence_buffer.clear()
#         self.sequence_progress = 0
#         self.sequence_no_hand_frames = 0
#         self.sequence_cooldown = SEQUENCE_COOLDOWN_FRAMES
#         self.last_motion_feature = None
#         self.motion_active_frames = 0
#         self.motion_score = 0.0

#     def finish_sequence_mode(self):
#         sequence_array = np.asarray(self.sequence_buffer, dtype=np.float32)
#         sequence_input = sequence_array.reshape(1, SEQUENCE_LENGTH, 127)

#         seq_idx, seq_conf, seq_margin = self.run_sequence_inference(sequence_input)
#         seq_label = self.sequence_idx_to_label[seq_idx]

#         is_sequence_confident = (
#             seq_conf >= SEQUENCE_CONFIDENCE_THRESHOLD
#             and seq_margin >= SEQUENCE_MARGIN_THRESHOLD
#         )

#         if is_sequence_confident:
#             self.sequence_prediction = seq_label
#             self.sequence_confidence = seq_conf
#             self.sequence_margin = seq_margin
#             self.predicted_letter = seq_label
#             self.raw_prediction = f"SEQ:{seq_label}"
#             self.confidence = seq_conf
#             self.margin = seq_margin
#             self.commit_sequence_word(seq_label)
#         else:
#             self.sequence_prediction = "..."
#             self.sequence_confidence = seq_conf
#             self.sequence_margin = seq_margin
#             self.predicted_letter = "..."
#             self.raw_prediction = f"SEQ_LOW:{seq_label}"
#             self.confidence = seq_conf
#             self.margin = seq_margin

#         self.mode = "STATIC"
#         self.token_mode = TOKEN_IDLE
#         self.sequence_buffer.clear()
#         self.sequence_progress = 0
#         self.sequence_no_hand_frames = 0
#         self.sequence_cooldown = SEQUENCE_COOLDOWN_FRAMES
#         self.sequence_locked_until_release = True
#         self.last_motion_feature = None
#         self.motion_active_frames = 0
#         self.motion_score = 0.0

#     def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
#         image = frame.to_ndarray(format="bgr24")

#         # Paksa frame yang diproses model selalu landscape 4:3 640x480.
#         image = center_crop_to_4_3(image)

#         # Mirror untuk kamera depan maupun kamera belakang.
#         # Urutan crop -> mirror dibuat agar preview dan input model konsisten.
#         if MIRROR_CAMERA_OUTPUT:
#             image = cv2.flip(image, 1)

#         self.frame_count += 1

#         # Skip sebagian frame agar latency turun.
#         # Frame yang diskip tetap ditampilkan, tapi tidak diproses MediaPipe/model.
#         if self.frame_count % PROCESS_EVERY_N_FRAMES != 0:
#             with self.lock:
#                 predicted_letter = self.predicted_letter
#                 raw_prediction = self.raw_prediction
#                 confidence = self.confidence
#                 margin = self.margin
#                 hand_count = self.hand_count

#             image = draw_scanner_overlay(
#                 image=image,
#                 predicted_letter=predicted_letter,
#                 raw_prediction=raw_prediction,
#                 confidence=confidence,
#                 margin=margin,
#                 hand_count=hand_count,
#             )

#             return av.VideoFrame.from_ndarray(image, format="bgr24")

#         rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#         mp_image = mp.Image(
#             image_format=mp.ImageFormat.SRGB,
#             data=rgb_image,
#         )

#         result = self.landmarker.detect(mp_image)

#         predicted_letter = "..."
#         raw_prediction = "..."
#         confidence = 0.0
#         margin = 0.0
#         hand_count = 0
#         sequence_feature = None
#         motion_feature = None

#         if self.sequence_cooldown > 0:
#             self.sequence_cooldown -= 1

#         if result and result.hand_landmarks:
#             left_coords = None
#             right_coords = None

#             left_label = "unknown"
#             right_label = "unknown"

#             left_present = False
#             right_present = False

#             for idx, hand_landmarks in enumerate(result.hand_landmarks):
#                 draw_landmarks(image, hand_landmarks)

#                 row = extract_landmark_row(hand_landmarks)

#                 if row is None:
#                     continue

#                 hand_count += 1

#                 side = "unknown"

#                 if (
#                     hasattr(result, "handedness")
#                     and result.handedness
#                     and len(result.handedness) > idx
#                 ):
#                     side = result.handedness[idx][0].category_name.lower()

#                 if side == "left":
#                     left_present = True
#                     left_coords = row
#                     left_label = side

#                 elif side == "right":
#                     right_present = True
#                     right_coords = row
#                     right_label = side

#             if left_present or right_present:
#                 # Fitur static untuk model alfabet/gestur.
#                 static_feature_vector = build_dual_feature_vector(
#                     left_coords=left_coords,
#                     right_coords=right_coords,
#                     left_label=left_label,
#                     right_label=right_label,
#                     left_present=left_present,
#                     right_present=right_present,
#                 )

#                 static_input = static_feature_vector.reshape(1, -1).astype(np.float32)

#                 predicted_idx, confidence, margin = self.run_static_inference(static_input)

#                 is_confident = (
#                     confidence >= CONFIDENCE_THRESHOLD
#                     and margin >= MARGIN_THRESHOLD
#                 )

#                 predicted_letter = self.stabilize_prediction(
#                     predicted_idx,
#                     is_confident,
#                 )

#                 raw_prediction = self.static_idx_to_label[predicted_idx]

#                 # Preview alfabet mentah untuk tombol Simpan dan debug.
#                 if is_alphabet_label(raw_prediction) and confidence >= LETTER_COMMIT_CONFIDENCE:
#                     self.last_static_letter_preview = raw_prediction
#                     self.last_static_letter_confidence = confidence
#                     self.last_static_letter_margin = margin

#                 # Fitur sequence 127 dimensi untuk model gerakan.
#                 sequence_feature = build_sequence_frame_feature(
#                     left_coords=left_coords,
#                     right_coords=right_coords,
#                     left_label=left_label,
#                     right_label=right_label,
#                     left_present=left_present,
#                     right_present=right_present,
#                 )

#                 motion_feature = build_motion_raw_feature(
#                     left_coords=left_coords,
#                     right_coords=right_coords,
#                     left_present=left_present,
#                     right_present=right_present,
#                 )

#         else:
#             predicted_letter = self.stabilize_prediction(None, False)

#         with self.lock:
#             self.hand_count = hand_count

#             if hand_count > 0:
#                 self.hand_present_frames += 1
#             else:
#                 self.hand_present_frames = 0

#             if hand_count == 0:
#                 self.no_hand_frames += 1
#                 self.last_added_letter = None
#                 self.reset_pending_letter()
#                 self.reset_router_buffers()
#                 if self.no_hand_frames >= SEQUENCE_RELEASE_FRAMES:
#                     self.sequence_locked_until_release = False
#                 if self.no_hand_frames >= WORD_GAP_NO_HAND_FRAMES:
#                     self.commit_spelled_word_if_needed()
#             else:
#                 self.no_hand_frames = 0

#             if self.mode == "SEQUENCE":
#                 if sequence_feature is not None and hand_count > 0:
#                     self.sequence_buffer.append(sequence_feature)
#                     self.sequence_no_hand_frames = 0
#                 else:
#                     self.sequence_no_hand_frames += 1

#                 self.sequence_progress = len(self.sequence_buffer)
#                 self.predicted_letter = f"SEQ {self.sequence_progress}/{SEQUENCE_LENGTH}"
#                 self.raw_prediction = "Merekam gerakan"
#                 self.confidence = confidence
#                 self.margin = margin

#                 if self.sequence_no_hand_frames >= SEQUENCE_NO_HAND_CANCEL_FRAMES:
#                     self.cancel_sequence_mode()
#                     self.predicted_letter = "..."
#                     self.raw_prediction = "SEQ_CANCEL"

#                 elif len(self.sequence_buffer) >= SEQUENCE_LENGTH:
#                     self.finish_sequence_mode()

#             else:
#                 stable_key = normalize_key(predicted_letter)
#                 raw_key = normalize_key(raw_prediction)

#                 # ============================================================
#                 # 3 MODE + TRAINED ROUTER MODEL
#                 # ============================================================
#                 # AUTO    : pakai router_model_final.h5 untuk memilih STATIC/DYNAMIC.
#                 # ALFABET : router tidak dipakai; sequence dimatikan total.
#                 # KATA    : router tidak dipakai; langsung rekam 30 frame untuk sequence.
#                 auto_mode = self.recognition_mode == MODE_AUTO
#                 alphabet_mode = self.recognition_mode == MODE_ALPHABET
#                 kata_mode = self.recognition_mode == MODE_KATA

#                 if auto_mode:
#                     self.update_temporal_router(
#                         motion_feature=motion_feature,
#                         sequence_feature=sequence_feature,
#                         hand_count=hand_count,
#                     )
#                 else:
#                     # Di mode ALFABET/KATA, router model tidak dijalankan supaya tidak mengganggu UX.
#                     self.reset_router_buffers()
#                     self.router_prediction = "OFF"
#                     self.router_confidence = 0.0
#                     self.router_margin = 0.0

#                 stable_sequence_trigger = stable_key in SEQUENCE_TRIGGER_LABELS
#                 raw_sequence_trigger = (
#                     SEQUENCE_TRIGGER_USE_RAW_FALLBACK
#                     and raw_key in SEQUENCE_TRIGGER_LABELS
#                     and confidence >= SEQUENCE_TRIGGER_CONFIDENCE
#                     and margin >= SEQUENCE_TRIGGER_MARGIN
#                 )

#                 locked_by_spelling = LOCK_SEQUENCE_WHILE_SPELLING and self.currently_spelling_word()

#                 strong_static_candidate = (
#                     (
#                         is_alphabet_label(predicted_letter)
#                         or is_alphabet_label(raw_prediction)
#                     )
#                     and confidence >= STATIC_OVERRIDE_CONFIDENCE
#                     and margin >= STATIC_OVERRIDE_MARGIN
#                     and self.motion_score <= STATIC_OVERRIDE_MAX_MOTION
#                 )

#                 router_dynamic_trigger = (
#                     auto_mode
#                     and USE_TRAINED_ROUTER
#                     and self.router_sees_dynamic()
#                     and not locked_by_spelling
#                     and not strong_static_candidate
#                 )

#                 if kata_mode:
#                     should_start_sequence = (
#                         self.sequence_cooldown <= 0
#                         and not self.sequence_locked_until_release
#                         and hand_count > 0
#                     )
#                 elif alphabet_mode:
#                     should_start_sequence = False
#                 else:
#                     should_start_sequence = (
#                         self.sequence_cooldown <= 0
#                         and not self.sequence_locked_until_release
#                         and hand_count > 0
#                         and (
#                             stable_sequence_trigger
#                             or raw_sequence_trigger
#                             or router_dynamic_trigger
#                         )
#                     )

#                 if should_start_sequence:
#                     self.start_sequence_mode()
#                     if auto_mode:
#                         self.preload_sequence_from_router()
#                     if sequence_feature is not None and len(self.sequence_buffer) < SEQUENCE_LENGTH:
#                         self.sequence_buffer.append(sequence_feature)
#                     self.reset_router_buffers()
#                     self.sequence_progress = len(self.sequence_buffer)
#                     self.predicted_letter = f"SEQ {self.sequence_progress}/{SEQUENCE_LENGTH}"
#                     self.raw_prediction = "Mode KATA" if kata_mode else "Router DYNAMIC"
#                     self.confidence = confidence
#                     self.margin = margin

#                 else:
#                     self.predicted_letter = predicted_letter
#                     self.raw_prediction = raw_prediction
#                     self.confidence = confidence
#                     self.margin = margin
#                     self.sequence_progress = 0

#                     if kata_mode:
#                         # Mode KATA: alfabet tidak boleh ditulis sama sekali.
#                         self.reset_pending_letter()

#                     elif alphabet_mode:
#                         # Mode ALFABET: langsung tulis alfabet tanpa menunggu router.
#                         if is_alphabet_label(predicted_letter) or is_alphabet_label(raw_prediction):
#                             self.enter_spelling_mode()

#                         self.handle_static_letter_candidate(
#                             label=predicted_letter,
#                             confidence=confidence,
#                             margin=margin,
#                             hand_count=hand_count,
#                         )

#                     else:
#                         # Mode AUTO: router belum cukup frame, jangan tulis alfabet dulu.
#                         # Ini mencegah awal gerakan kata terbaca sebagai huruf.
#                         if USE_TRAINED_ROUTER and hand_count > 0 and not self.router_ready():
#                             self.reset_pending_letter()
#                             self.predicted_letter = f"WAIT {len(self.router_sequence_buffer)}/{ROUTER_WINDOW_FRAMES}"
#                             self.raw_prediction = raw_prediction

#                         else:
#                             # Jika router melihat static/diam, token ini dianggap SPELLING.
#                             # Setelah masuk SPELLING, gerakan transisi antar huruf tidak akan memicu sequence.
#                             if is_alphabet_label(predicted_letter) or is_alphabet_label(raw_prediction):
#                                 self.enter_spelling_mode()

#                             self.handle_static_letter_candidate(
#                                 label=predicted_letter,
#                                 confidence=confidence,
#                                 margin=margin,
#                                 hand_count=hand_count,
#                             )

#             predicted_letter_for_overlay = self.predicted_letter
#             raw_prediction_for_overlay = self.raw_prediction
#             confidence_for_overlay = self.confidence
#             margin_for_overlay = self.margin
#             hand_count_for_overlay = self.hand_count

#         image = draw_scanner_overlay(
#             image=image,
#             predicted_letter=predicted_letter_for_overlay,
#             raw_prediction=raw_prediction_for_overlay,
#             confidence=confidence_for_overlay,
#             margin=margin_for_overlay,
#             hand_count=hand_count_for_overlay,
#         )

#         return av.VideoFrame.from_ndarray(image, format="bgr24")


from __future__ import annotations

from collections import deque
import threading
import traceback
from typing import Deque, Optional

import av
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from streamlit_webrtc import VideoProcessorBase
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarkerOptions, RunningMode

from src.features.landmark_features_dual import build_dual_feature_vector

from .config import *
from .drawing import center_crop_to_4_3, draw_landmarks, draw_scanner_overlay
from .features import build_motion_raw_feature, build_sequence_frame_feature, extract_landmark_row
from .models import (
    load_cached_router_model,
    load_cached_sequence_model,
    load_cached_static_model,
    load_router_labels_safe,
    predict_with_model,
)
from .utils import is_alphabet_label, load_labels, normalize_key

class BisindoVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.static_model = load_cached_static_model()
        self.sequence_model = load_cached_sequence_model()
        self.router_model = load_cached_router_model()

        self.static_idx_to_label = load_labels(STATIC_LABELS_PATH)
        self.sequence_idx_to_label = load_labels(SEQUENCE_LABELS_PATH)
        self.router_idx_to_label = load_router_labels_safe(ROUTER_LABELS_PATH)

        static_labels_norm = {normalize_key(v) for v in self.static_idx_to_label.values()}
        self.available_sequence_triggers = static_labels_norm.intersection(SEQUENCE_TRIGGER_LABELS)
        if not self.available_sequence_triggers:
            print(
                "PERINGATAN: Tidak ada label trigger sequence di static labels. "
                "Tambahkan class seperti 'sequel' / 'sequence' ke model static, "
                "atau ubah SEQUENCE_TRIGGER_LABELS agar cocok dengan labels_dual.json."
            )

        self.prediction_buffer: Deque[Optional[int]] = deque(maxlen=BUFFER_SIZE)
        self.sequence_buffer: Deque[np.ndarray] = deque(maxlen=SEQUENCE_LENGTH)

        # Router buffer: 12 frame fitur 127 untuk model router IDLE / STATIC / DYNAMIC.
        self.router_motion_buffer: Deque[np.ndarray] = deque(maxlen=ROUTER_WINDOW_FRAMES)
        self.router_sequence_buffer: Deque[np.ndarray] = deque(maxlen=ROUTER_WINDOW_FRAMES)

        self.predicted_letter = "..."
        self.raw_prediction = "..."
        self.confidence = 0.0
        self.margin = 0.0
        self.hand_count = 0

        self.sequence_prediction = "..."
        self.sequence_confidence = 0.0
        self.sequence_margin = 0.0
        self.sequence_progress = 0
        self.mode = "STATIC"  # STATIC atau SEQUENCE internal
        self.recognition_mode = DEFAULT_RECOGNITION_MODE
        self.token_mode = TOKEN_IDLE
        self.sequence_no_hand_frames = 0
        self.sequence_cooldown = 0
        self.sequence_locked_until_release = False
        self.last_motion_feature = None
        self.motion_active_frames = 0
        self.motion_score = 0.0
        self.router_prediction = "WAIT"
        self.router_confidence = 0.0
        self.router_margin = 0.0
        self.router_dynamic_streak = 0
        self.hand_present_frames = 0

        self.frame_count = 0
        self.last_error = None

        # Buffer teks
        self.hasil_kata = ""
        self.sentence_words = []
        self.last_added_letter = None

        # Smart composer state.
        self.pending_letter = None
        self.pending_letter_frames = 0
        self.no_hand_frames = 0
        self.hasil_kata_committed_preview = False
        self.last_sequence_word = None
        self.last_sequence_commit_frame = -10_000
        self.last_static_letter_preview = None
        self.last_static_letter_confidence = 0.0
        self.last_static_letter_margin = 0.0

        self.lock = threading.Lock()

        options = HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(MP_MODEL_PATH)
            ),
            running_mode=RunningMode.IMAGE,
            min_hand_detection_confidence=0.3,
            min_tracking_confidence=0.3,
            num_hands=2,
        )

        self.landmarker = vision.HandLandmarker.create_from_options(options)

    def set_recognition_mode(self, mode: str):
        mode = str(mode).strip().lower()
        if mode not in VALID_RECOGNITION_MODES:
            mode = DEFAULT_RECOGNITION_MODE

        with self.lock:
            if mode != self.recognition_mode:
                self.recognition_mode = mode
                self.prediction_buffer.clear()
                self.reset_pending_letter()
                self.last_added_letter = None
                self.last_motion_feature = None
                self.motion_active_frames = 0
                self.motion_score = 0.0
                self.router_prediction = "WAIT"
                self.router_confidence = 0.0
                self.router_margin = 0.0

                # Kalau pindah mode, batalkan rekaman sequence yang sedang setengah jalan.
                if self.mode == "SEQUENCE":
                    self.cancel_sequence_mode()

    def run_static_inference(self, input_data: np.ndarray):
        return predict_with_model(self.static_model, input_data, STATIC_MODEL_LOCK)

    def run_sequence_inference(self, sequence_data: np.ndarray):
        return predict_with_model(self.sequence_model, sequence_data, SEQUENCE_MODEL_LOCK)

    def run_router_inference(self):
        if len(self.router_sequence_buffer) < ROUTER_WINDOW_FRAMES:
            return "WAIT", 0.0, 0.0

        input_data = np.asarray(self.router_sequence_buffer, dtype=np.float32).reshape(
            1, ROUTER_WINDOW_FRAMES, 127
        )

        with ROUTER_MODEL_LOCK:
            input_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)
            probs = self.router_model(input_tensor, training=False).numpy()[0]

        ranked = np.argsort(probs)[::-1]
        pred_idx = int(ranked[0])
        conf = float(probs[pred_idx])
        second_conf = float(probs[ranked[1]]) if len(ranked) > 1 else 0.0
        margin = conf - second_conf
        label = self.router_idx_to_label.get(pred_idx, str(pred_idx))
        return label, conf, margin

    def clear_text(self):
        with self.lock:
            self.hasil_kata = ""
            self.sentence_words = []
            self.last_added_letter = None
            self.pending_letter = None
            self.pending_letter_frames = 0
            self.no_hand_frames = 0
            self.hasil_kata_committed_preview = False
            self.last_sequence_word = None
            self.last_sequence_commit_frame = -10_000
            self.last_static_letter_preview = None
            self.last_static_letter_confidence = 0.0
            self.last_static_letter_margin = 0.0
            self.prediction_buffer.clear()
            self.sequence_buffer.clear()
            self.router_motion_buffer.clear()
            self.router_sequence_buffer.clear()
            self.sequence_prediction = "..."
            self.sequence_confidence = 0.0
            self.sequence_margin = 0.0
            self.sequence_progress = 0
            self.mode = "STATIC"
            self.token_mode = TOKEN_IDLE
            self.sequence_no_hand_frames = 0
            self.sequence_cooldown = 0
            self.sequence_locked_until_release = False
            self.last_motion_feature = None
            self.motion_active_frames = 0
            self.motion_score = 0.0

    def force_commit_pending_letter(self):
        """Dipanggil saat tombol Simpan ditekan agar huruf terakhir tidak hilang."""
        label = self.pending_letter or self.last_static_letter_preview
        if not label:
            return

        if not is_alphabet_label(label):
            return

        if self.hasil_kata_committed_preview:
            self.hasil_kata = ""
            self.hasil_kata_committed_preview = False

        # Hindari dobel saat huruf yang sama sudah terakhir masuk.
        if label != self.last_added_letter:
            self.hasil_kata += label
            self.last_added_letter = label

        self.reset_pending_letter()

    def save_current_word(self):
        with self.lock:
            # Kalau sedang spelling, tombol Simpan harus mengunci huruf pending dulu.
            if not self.hasil_kata_committed_preview:
                self.force_commit_pending_letter()

            # Kalau hasil_kata adalah preview dari sequence yang sudah auto masuk kalimat,
            # tombol Simpan cukup membersihkan preview supaya tidak dobel.
            if self.hasil_kata_committed_preview:
                self.hasil_kata = ""
                self.hasil_kata_committed_preview = False
                self.last_added_letter = None
                self.pending_letter = None
                self.pending_letter_frames = 0
                return

            word = self.hasil_kata.strip()

            if not word and self.predicted_letter not in ("", "..."):
                # Jangan simpan label trigger sequence sebagai kata.
                if normalize_key(self.predicted_letter) not in SEQUENCE_TRIGGER_LABELS:
                    word = self.predicted_letter

            if word:
                self.sentence_words.append(word.upper())
                self.hasil_kata = ""
                self.hasil_kata_committed_preview = False
                self.last_added_letter = None
                self.pending_letter = None
                self.pending_letter_frames = 0

    def get_ui_state(self):
        with self.lock:
            extra_current_word = []
            if self.hasil_kata and not self.hasil_kata_committed_preview:
                extra_current_word = [self.hasil_kata]

            sentence_preview = " ".join(self.sentence_words + extra_current_word)

            return {
                "predicted_letter": self.predicted_letter,
                "raw_prediction": self.raw_prediction,
                "confidence": self.confidence,
                "margin": self.margin,
                "hand_count": self.hand_count,
                "hasil_kata": self.hasil_kata,
                "hasil_kalimat": sentence_preview,
                "mode": self.mode,
                "sequence_prediction": self.sequence_prediction,
                "sequence_confidence": self.sequence_confidence,
                "sequence_margin": self.sequence_margin,
                "sequence_progress": self.sequence_progress,
                "sequence_trigger_ready": bool(self.available_sequence_triggers),
                "auto_sequence_on_motion": AUTO_SEQUENCE_ON_MOTION,
                "motion_score": self.motion_score,
                "pending_letter": self.pending_letter or "",
                "pending_letter_frames": self.pending_letter_frames,
                "hasil_kata_committed_preview": self.hasil_kata_committed_preview,
                "last_static_letter_preview": self.last_static_letter_preview or "",
                "recognition_mode": self.recognition_mode,
                "router_frames": len(self.router_motion_buffer),
                "router_window": ROUTER_WINDOW_FRAMES,
                "token_mode": self.token_mode,
                "sequence_locked_until_release": self.sequence_locked_until_release,
                "router_prediction": self.router_prediction,
                "router_confidence": self.router_confidence,
                "router_margin": self.router_margin,
                "router_dynamic_streak": self.router_dynamic_streak,
                "hand_present_frames": self.hand_present_frames,
            }

    def stabilize_prediction(
        self,
        predicted_idx: Optional[int],
        is_confident: bool,
    ) -> str:
        self.prediction_buffer.append(predicted_idx if is_confident else None)

        if len(self.prediction_buffer) < BUFFER_SIZE:
            return "..."

        valid_predictions = [
            idx for idx in self.prediction_buffer if idx is not None
        ]

        if len(valid_predictions) < VOTE_THRESHOLD:
            return "..."

        counts = {}

        for idx in valid_predictions:
            counts[idx] = counts.get(idx, 0) + 1

        max_count = max(counts.values())

        if max_count >= VOTE_THRESHOLD:
            stable_idx = max(counts, key=counts.get)
            return self.static_idx_to_label[stable_idx]

        return "..."

    def reset_pending_letter(self):
        self.pending_letter = None
        self.pending_letter_frames = 0

    def commit_spelled_word_if_needed(self):
        if not AUTO_SAVE_SPELLED_WORD_ON_GAP:
            return

        if self.hasil_kata and not self.hasil_kata_committed_preview:
            word = self.hasil_kata.strip().upper()
            if word:
                self.sentence_words.append(word)
            self.hasil_kata = ""
            self.hasil_kata_committed_preview = False
            self.last_added_letter = None
            self.reset_pending_letter()
            self.token_mode = TOKEN_IDLE

    def handle_static_letter_candidate(
        self,
        label: str,
        confidence: float,
        margin: float,
        hand_count: int,
    ):
        """Composer alfabet yang lebih pintar.

        Huruf tidak langsung ditulis. Huruf hanya masuk jika:
        - label stabil beberapa frame,
        - confidence/margin cukup,
        - motion rendah / tangan diam,
        - bukan sedang sequence,
        - bukan pengulangan huruf yang sama tanpa release tangan.
        """
        if hand_count <= 0:
            self.reset_pending_letter()
            return

        if label in ("", "..."):
            self.reset_pending_letter()
            return

        if normalize_key(label) in SEQUENCE_TRIGGER_LABELS:
            self.reset_pending_letter()
            return

        # Kalau tangan sedang bergerak, jangan pernah tulis alfabet.
        if self.motion_score > LETTER_STEADY_MOTION_MAX:
            self.reset_pending_letter()
            return

        if confidence < LETTER_COMMIT_CONFIDENCE or margin < LETTER_COMMIT_MARGIN:
            self.reset_pending_letter()
            return

        if label != self.pending_letter:
            self.pending_letter = label
            self.pending_letter_frames = 1
            return

        self.pending_letter_frames += 1

        if self.pending_letter_frames < LETTER_COMMIT_STABLE_FRAMES:
            return

        # Jangan menulis huruf sama berkali-kali saat pose masih ditahan.
        if label == self.last_added_letter:
            return

        # Kalau sebelumnya hasil_kata adalah preview sequence yang sudah masuk kalimat,
        # mulai ejaan baru dengan buffer kosong.
        if self.hasil_kata_committed_preview:
            self.hasil_kata = ""
            self.hasil_kata_committed_preview = False

        self.enter_spelling_mode()
        self.hasil_kata += label
        self.last_added_letter = label
        self.reset_pending_letter()

    def commit_sequence_word(self, word: str):
        word = str(word).strip().upper()
        if not word or word in ("...", "-"):
            return

        # Anti dobel: kalau kata yang sama baru saja masuk, jangan append lagi.
        is_duplicate = (
            self.last_sequence_word == word
            and (self.frame_count - self.last_sequence_commit_frame) < SEQUENCE_DUPLICATE_COOLDOWN_FRAMES
        )

        if AUTO_COMMIT_SEQUENCE_WORD and not is_duplicate:
            self.sentence_words.append(word)
            self.last_sequence_word = word
            self.last_sequence_commit_frame = self.frame_count
            self.hasil_kata_committed_preview = True
        else:
            self.hasil_kata_committed_preview = False

        # Hasil kata tetap menampilkan kata terakhir, tapi tidak didobel di kalimat.
        self.hasil_kata = word
        self.last_added_letter = None
        self.reset_pending_letter()
        self.token_mode = TOKEN_IDLE

    def reset_router_buffers(self):
        self.router_motion_buffer.clear()
        self.router_sequence_buffer.clear()
        self.router_prediction = "WAIT"
        self.router_confidence = 0.0
        self.router_margin = 0.0
        self.router_dynamic_streak = 0

    def update_temporal_router(
        self,
        motion_feature: Optional[np.ndarray],
        sequence_feature: Optional[np.ndarray],
        hand_count: int,
    ):
        # Nama fungsi dipertahankan agar perubahan di recv minimal.
        # Sekarang isinya bukan threshold motion, tapi buffer untuk trained router model.
        if not USE_TRAINED_ROUTER or hand_count <= 0 or sequence_feature is None:
            self.reset_router_buffers()
            self.router_prediction = "IDLE" if hand_count <= 0 else "WAIT"
            self.router_confidence = 0.0
            self.router_margin = 0.0
            self.motion_score = 0.0
            return

        self.router_sequence_buffer.append(sequence_feature.copy())
        self.router_motion_buffer.append(sequence_feature.copy())

        # Motion score hanya debug visual, bukan penentu router.
        self.motion_score = self.compute_router_motion_score()

        if self.router_ready():
            label, conf, margin = self.run_router_inference()
            self.router_prediction = label
            self.router_confidence = conf
            self.router_margin = margin
        else:
            self.router_prediction = "WAIT"
            self.router_confidence = 0.0
            self.router_margin = 0.0

    def compute_router_motion_score(self) -> float:
        if len(self.router_sequence_buffer) < 2:
            return 0.0

        frames = list(self.router_sequence_buffer)
        diffs = [float(np.mean(np.abs(frames[i][1:] - frames[i - 1][1:]))) for i in range(1, len(frames))]
        return float(max(diffs)) if diffs else 0.0

    def router_ready(self) -> bool:
        return len(self.router_sequence_buffer) >= ROUTER_WINDOW_FRAMES

    def router_sees_dynamic(self) -> bool:
        if not self.router_ready():
            self.router_dynamic_streak = 0
            return False

        # Saat tangan baru masuk kamera, frame awal menuju pose A/B/C sering terlihat bergerak.
        # Jangan langsung anggap dynamic sampai tangan stabil beberapa frame.
        if self.hand_present_frames < HAND_SETTLE_FRAMES:
            self.router_dynamic_streak = 0
            return False

        is_dynamic = (
            normalize_key(self.router_prediction) == "dynamic"
            and self.router_confidence >= ROUTER_DYNAMIC_CONFIDENCE_THRESHOLD
        )

        if is_dynamic:
            self.router_dynamic_streak += 1
        else:
            self.router_dynamic_streak = 0

        return self.router_dynamic_streak >= ROUTER_DYNAMIC_STABLE_FRAMES

    def router_sees_static(self) -> bool:
        if not self.router_ready():
            return False
        return (
            normalize_key(self.router_prediction) == "static"
            and self.router_confidence >= ROUTER_STATIC_CONFIDENCE_THRESHOLD
        )

    def enter_spelling_mode(self):
        if self.token_mode != TOKEN_SPELLING:
            self.token_mode = TOKEN_SPELLING
            # Saat spelling dimulai, preview sequence lama tidak boleh dianggap kata aktif.
            if self.hasil_kata_committed_preview:
                self.hasil_kata = ""
                self.hasil_kata_committed_preview = False
            self.reset_router_buffers()

    def currently_spelling_word(self) -> bool:
        if self.token_mode == TOKEN_SPELLING:
            return True
        return bool(self.hasil_kata and not self.hasil_kata_committed_preview)

    def preload_sequence_from_router(self):
        # Masukkan frame awal gerakan yang sudah ada di router agar sequence tidak kehilangan start motion.
        for feat in list(self.router_sequence_buffer):
            if feat is not None and len(self.sequence_buffer) < SEQUENCE_LENGTH:
                self.sequence_buffer.append(feat)

    def update_motion_state(self, motion_feature: Optional[np.ndarray], hand_count: int) -> bool:
        if not AUTO_SEQUENCE_ON_MOTION or motion_feature is None or hand_count <= 0:
            self.last_motion_feature = None
            self.motion_active_frames = 0
            self.motion_score = 0.0
            return False

        if self.last_motion_feature is None:
            self.last_motion_feature = motion_feature.copy()
            self.motion_active_frames = 0
            self.motion_score = 0.0
            return False

        diff = np.abs(motion_feature - self.last_motion_feature)
        self.motion_score = float(np.mean(diff))
        self.last_motion_feature = motion_feature.copy()

        if self.motion_score >= MOTION_START_THRESHOLD:
            self.motion_active_frames += 1
        else:
            self.motion_active_frames = 0

        return self.motion_active_frames >= MOTION_TRIGGER_FRAMES

    def start_sequence_mode(self):
        self.mode = "SEQUENCE"
        self.token_mode = TOKEN_SEQUENCE
        self.sequence_buffer.clear()
        self.sequence_progress = 0
        self.sequence_no_hand_frames = 0
        self.prediction_buffer.clear()
        self.last_added_letter = None
        self.reset_pending_letter()

        # Ini bagian penting: begitu sequence mulai, alfabet yang belum benar-benar
        # dimaksudkan dibersihkan agar gerakan kata tidak diawali huruf sampah.
        if CLEAR_UNCOMMITTED_LETTERS_ON_SEQUENCE and not self.hasil_kata_committed_preview:
            # Untuk keamanan, hanya buang buffer pendek yang biasanya noise awal gerakan.
            if len(self.hasil_kata.strip()) <= 3:
                self.hasil_kata = ""

        self.last_motion_feature = None
        self.motion_active_frames = 0
        self.motion_score = 0.0

    def cancel_sequence_mode(self):
        self.mode = "STATIC"
        self.token_mode = TOKEN_IDLE
        self.sequence_buffer.clear()
        self.sequence_progress = 0
        self.sequence_no_hand_frames = 0
        self.sequence_cooldown = SEQUENCE_COOLDOWN_FRAMES
        self.last_motion_feature = None
        self.motion_active_frames = 0
        self.motion_score = 0.0

    def finish_sequence_mode(self):
        sequence_array = np.asarray(self.sequence_buffer, dtype=np.float32)
        sequence_input = sequence_array.reshape(1, SEQUENCE_LENGTH, 127)

        seq_idx, seq_conf, seq_margin = self.run_sequence_inference(sequence_input)
        seq_label = self.sequence_idx_to_label[seq_idx]

        is_sequence_confident = (
            seq_conf >= SEQUENCE_CONFIDENCE_THRESHOLD
            and seq_margin >= SEQUENCE_MARGIN_THRESHOLD
        )

        if is_sequence_confident:
            self.sequence_prediction = seq_label
            self.sequence_confidence = seq_conf
            self.sequence_margin = seq_margin
            self.predicted_letter = seq_label
            self.raw_prediction = f"SEQ:{seq_label}"
            self.confidence = seq_conf
            self.margin = seq_margin
            self.commit_sequence_word(seq_label)
        else:
            self.sequence_prediction = "..."
            self.sequence_confidence = seq_conf
            self.sequence_margin = seq_margin
            self.predicted_letter = "..."
            self.raw_prediction = f"SEQ_LOW:{seq_label}"
            self.confidence = seq_conf
            self.margin = seq_margin

        self.mode = "STATIC"
        self.token_mode = TOKEN_IDLE
        self.sequence_buffer.clear()
        self.sequence_progress = 0
        self.sequence_no_hand_frames = 0
        self.sequence_cooldown = SEQUENCE_COOLDOWN_FRAMES
        self.sequence_locked_until_release = True
        self.last_motion_feature = None
        self.motion_active_frames = 0
        self.motion_score = 0.0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        try:
            return self._recv_impl(frame)
        except Exception:
            self.last_error = traceback.format_exc()
            print(self.last_error)

            # Jangan return frame mentah saat error, karena itu membuat preview
            # terlihat "lompat/pindah-pindah" antara raw frame dan processed frame.
            try:
                image = frame.to_ndarray(format="bgr24")
                image = center_crop_to_4_3(image)

                if MIRROR_CAMERA_OUTPUT:
                    image = cv2.flip(image, 1)

                with self.lock:
                    predicted_letter = self.predicted_letter
                    raw_prediction = self.raw_prediction
                    confidence = self.confidence
                    margin = self.margin
                    hand_count = self.hand_count

                try:
                    image = draw_scanner_overlay(
                        image=image,
                        predicted_letter=predicted_letter,
                        raw_prediction="PROCESS ERROR",
                        confidence=confidence,
                        margin=margin,
                        hand_count=hand_count,
                    )
                except Exception:
                    cv2.putText(
                        image,
                        # "PROCESS ERROR",
                        (16, 32),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )

                return av.VideoFrame.from_ndarray(image, format="bgr24")

            except Exception:
                return frame

    def _recv_impl(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")

        # Paksa frame yang diproses model selalu landscape 4:3 640x480.
        image = center_crop_to_4_3(image)

        # Mirror untuk kamera depan maupun kamera belakang.
        # Urutan crop -> mirror dibuat agar preview dan input model konsisten.
        if MIRROR_CAMERA_OUTPUT:
            image = cv2.flip(image, 1)

        self.frame_count += 1

        # Skip sebagian frame agar latency turun.
        # Frame yang diskip tetap ditampilkan, tapi tidak diproses MediaPipe/model.
        if self.frame_count % PROCESS_EVERY_N_FRAMES != 0:
            with self.lock:
                predicted_letter = self.predicted_letter
                raw_prediction = self.raw_prediction
                confidence = self.confidence
                margin = self.margin
                hand_count = self.hand_count

            image = draw_scanner_overlay(
                image=image,
                predicted_letter=predicted_letter,
                raw_prediction=raw_prediction,
                confidence=confidence,
                margin=margin,
                hand_count=hand_count,
            )

            return av.VideoFrame.from_ndarray(image, format="bgr24")

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_image,
        )

        result = self.landmarker.detect(mp_image)

        predicted_letter = "..."
        raw_prediction = "..."
        confidence = 0.0
        margin = 0.0
        hand_count = 0
        sequence_feature = None
        motion_feature = None

        if self.sequence_cooldown > 0:
            self.sequence_cooldown -= 1

        if result and result.hand_landmarks:
            left_coords = None
            right_coords = None

            left_label = "unknown"
            right_label = "unknown"

            left_present = False
            right_present = False

            for idx, hand_landmarks in enumerate(result.hand_landmarks):
                draw_landmarks(image, hand_landmarks)

                row = extract_landmark_row(hand_landmarks)

                if row is None:
                    continue

                hand_count += 1

                side = "unknown"

                if (
                    hasattr(result, "handedness")
                    and result.handedness
                    and len(result.handedness) > idx
                ):
                    side = result.handedness[idx][0].category_name.lower()

                if side == "left":
                    left_present = True
                    left_coords = row
                    left_label = side

                elif side == "right":
                    right_present = True
                    right_coords = row
                    right_label = side

            if left_present or right_present:
                # Fitur static untuk model alfabet/gestur.
                static_feature_vector = build_dual_feature_vector(
                    left_coords=left_coords,
                    right_coords=right_coords,
                    left_label=left_label,
                    right_label=right_label,
                    left_present=left_present,
                    right_present=right_present,
                )

                static_input = static_feature_vector.reshape(1, -1).astype(np.float32)

                predicted_idx, confidence, margin = self.run_static_inference(static_input)

                is_confident = (
                    confidence >= CONFIDENCE_THRESHOLD
                    and margin >= MARGIN_THRESHOLD
                )

                predicted_letter = self.stabilize_prediction(
                    predicted_idx,
                    is_confident,
                )

                raw_prediction = self.static_idx_to_label[predicted_idx]

                # Preview alfabet mentah untuk tombol Simpan dan debug.
                if is_alphabet_label(raw_prediction) and confidence >= LETTER_COMMIT_CONFIDENCE:
                    self.last_static_letter_preview = raw_prediction
                    self.last_static_letter_confidence = confidence
                    self.last_static_letter_margin = margin

                # Fitur sequence 127 dimensi untuk model gerakan.
                sequence_feature = build_sequence_frame_feature(
                    left_coords=left_coords,
                    right_coords=right_coords,
                    left_label=left_label,
                    right_label=right_label,
                    left_present=left_present,
                    right_present=right_present,
                )

                motion_feature = build_motion_raw_feature(
                    left_coords=left_coords,
                    right_coords=right_coords,
                    left_present=left_present,
                    right_present=right_present,
                )

        else:
            predicted_letter = self.stabilize_prediction(None, False)

        with self.lock:
            self.hand_count = hand_count

            if hand_count > 0:
                self.hand_present_frames += 1
            else:
                self.hand_present_frames = 0

            if hand_count == 0:
                self.no_hand_frames += 1
                self.last_added_letter = None
                self.reset_pending_letter()
                self.reset_router_buffers()
                if self.no_hand_frames >= SEQUENCE_RELEASE_FRAMES:
                    self.sequence_locked_until_release = False
                if self.no_hand_frames >= WORD_GAP_NO_HAND_FRAMES:
                    self.commit_spelled_word_if_needed()
            else:
                self.no_hand_frames = 0

            if self.mode == "SEQUENCE":
                if sequence_feature is not None and hand_count > 0:
                    self.sequence_buffer.append(sequence_feature)
                    self.sequence_no_hand_frames = 0
                else:
                    self.sequence_no_hand_frames += 1

                self.sequence_progress = len(self.sequence_buffer)
                self.predicted_letter = f"SEQ {self.sequence_progress}/{SEQUENCE_LENGTH}"
                self.raw_prediction = "Merekam gerakan"
                self.confidence = confidence
                self.margin = margin

                if self.sequence_no_hand_frames >= SEQUENCE_NO_HAND_CANCEL_FRAMES:
                    self.cancel_sequence_mode()
                    self.predicted_letter = "..."
                    self.raw_prediction = "SEQ_CANCEL"

                elif len(self.sequence_buffer) >= SEQUENCE_LENGTH:
                    self.finish_sequence_mode()

            else:
                stable_key = normalize_key(predicted_letter)
                raw_key = normalize_key(raw_prediction)

                # ============================================================
                # 3 MODE + TRAINED ROUTER MODEL
                # ============================================================
                # AUTO    : pakai router_model_final.h5 untuk memilih STATIC/DYNAMIC.
                # ALFABET : router tidak dipakai; sequence dimatikan total.
                # KATA    : router tidak dipakai; langsung rekam 30 frame untuk sequence.
                auto_mode = self.recognition_mode == MODE_AUTO
                alphabet_mode = self.recognition_mode == MODE_ALPHABET
                kata_mode = self.recognition_mode == MODE_KATA

                if auto_mode:
                    self.update_temporal_router(
                        motion_feature=motion_feature,
                        sequence_feature=sequence_feature,
                        hand_count=hand_count,
                    )
                else:
                    # Di mode ALFABET/KATA, router model tidak dijalankan supaya tidak mengganggu UX.
                    self.reset_router_buffers()
                    self.router_prediction = "OFF"
                    self.router_confidence = 0.0
                    self.router_margin = 0.0

                stable_sequence_trigger = stable_key in SEQUENCE_TRIGGER_LABELS
                raw_sequence_trigger = (
                    SEQUENCE_TRIGGER_USE_RAW_FALLBACK
                    and raw_key in SEQUENCE_TRIGGER_LABELS
                    and confidence >= SEQUENCE_TRIGGER_CONFIDENCE
                    and margin >= SEQUENCE_TRIGGER_MARGIN
                )

                locked_by_spelling = LOCK_SEQUENCE_WHILE_SPELLING and self.currently_spelling_word()

                strong_static_candidate = (
                    (
                        is_alphabet_label(predicted_letter)
                        or is_alphabet_label(raw_prediction)
                    )
                    and confidence >= STATIC_OVERRIDE_CONFIDENCE
                    and margin >= STATIC_OVERRIDE_MARGIN
                    and self.motion_score <= STATIC_OVERRIDE_MAX_MOTION
                )

                router_dynamic_trigger = (
                    auto_mode
                    and USE_TRAINED_ROUTER
                    and self.router_sees_dynamic()
                    and not locked_by_spelling
                    and not strong_static_candidate
                )

                if kata_mode:
                    should_start_sequence = (
                        self.sequence_cooldown <= 0
                        and not self.sequence_locked_until_release
                        and hand_count > 0
                    )
                elif alphabet_mode:
                    should_start_sequence = False
                else:
                    should_start_sequence = (
                        self.sequence_cooldown <= 0
                        and not self.sequence_locked_until_release
                        and hand_count > 0
                        and (
                            stable_sequence_trigger
                            or raw_sequence_trigger
                            or router_dynamic_trigger
                        )
                    )

                if should_start_sequence:
                    self.start_sequence_mode()
                    if auto_mode:
                        self.preload_sequence_from_router()
                    if sequence_feature is not None and len(self.sequence_buffer) < SEQUENCE_LENGTH:
                        self.sequence_buffer.append(sequence_feature)
                    self.reset_router_buffers()
                    self.sequence_progress = len(self.sequence_buffer)
                    self.predicted_letter = f"SEQ {self.sequence_progress}/{SEQUENCE_LENGTH}"
                    self.raw_prediction = "Mode KATA" if kata_mode else "Router DYNAMIC"
                    self.confidence = confidence
                    self.margin = margin

                else:
                    self.predicted_letter = predicted_letter
                    self.raw_prediction = raw_prediction
                    self.confidence = confidence
                    self.margin = margin
                    self.sequence_progress = 0

                    if kata_mode:
                        # Mode KATA: alfabet tidak boleh ditulis sama sekali.
                        self.reset_pending_letter()

                    elif alphabet_mode:
                        # Mode ALFABET: langsung tulis alfabet tanpa menunggu router.
                        if is_alphabet_label(predicted_letter) or is_alphabet_label(raw_prediction):
                            self.enter_spelling_mode()

                        self.handle_static_letter_candidate(
                            label=predicted_letter,
                            confidence=confidence,
                            margin=margin,
                            hand_count=hand_count,
                        )

                    else:
                        # Mode AUTO: router belum cukup frame, jangan tulis alfabet dulu.
                        # Ini mencegah awal gerakan kata terbaca sebagai huruf.
                        if USE_TRAINED_ROUTER and hand_count > 0 and not self.router_ready():
                            self.reset_pending_letter()
                            self.predicted_letter = f"WAIT {len(self.router_sequence_buffer)}/{ROUTER_WINDOW_FRAMES}"
                            self.raw_prediction = raw_prediction

                        else:
                            # Jika router melihat static/diam, token ini dianggap SPELLING.
                            # Setelah masuk SPELLING, gerakan transisi antar huruf tidak akan memicu sequence.
                            if is_alphabet_label(predicted_letter) or is_alphabet_label(raw_prediction):
                                self.enter_spelling_mode()

                            self.handle_static_letter_candidate(
                                label=predicted_letter,
                                confidence=confidence,
                                margin=margin,
                                hand_count=hand_count,
                            )

            predicted_letter_for_overlay = self.predicted_letter
            raw_prediction_for_overlay = self.raw_prediction
            confidence_for_overlay = self.confidence
            margin_for_overlay = self.margin
            hand_count_for_overlay = self.hand_count

        image = draw_scanner_overlay(
            image=image,
            predicted_letter=predicted_letter_for_overlay,
            raw_prediction=raw_prediction_for_overlay,
            confidence=confidence_for_overlay,
            margin=margin_for_overlay,
            hand_count=hand_count_for_overlay,
        )

        return av.VideoFrame.from_ndarray(image, format="bgr24")


