from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf

from .config import ROUTER_MODEL_PATH, SEQUENCE_MODEL_PATH, STATIC_MODEL_PATH
from .utils import load_labels

@st.cache_resource
def load_cached_static_model():
    model = tf.keras.models.load_model(STATIC_MODEL_PATH, compile=False)
    print("STATIC INPUT SHAPE :", model.input_shape)
    print("STATIC OUTPUT SHAPE:", model.output_shape)
    return model


@st.cache_resource
def load_cached_sequence_model():
    model = tf.keras.models.load_model(SEQUENCE_MODEL_PATH, compile=False)
    print("SEQUENCE INPUT SHAPE :", model.input_shape)
    print("SEQUENCE OUTPUT SHAPE:", model.output_shape)
    return model


@st.cache_resource
def load_cached_router_model():
    model = tf.keras.models.load_model(ROUTER_MODEL_PATH, compile=False)
    print("ROUTER INPUT SHAPE :", model.input_shape)
    print("ROUTER OUTPUT SHAPE:", model.output_shape)
    return model


def load_router_labels_safe(labels_path: Path):
    if labels_path.exists():
        return load_labels(labels_path)
    # Fallback jika training pertama abort sebelum router_labels.json tersimpan.
    return {0: "IDLE", 1: "STATIC", 2: "DYNAMIC"}



def predict_with_model(model, input_data: np.ndarray, lock: threading.Lock):
    with lock:
        input_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)
        probs = model(input_tensor, training=False).numpy()[0]

    ranked = np.argsort(probs)[::-1]
    predicted_idx = int(ranked[0])
    confidence = float(probs[predicted_idx])
    second_confidence = float(probs[ranked[1]]) if len(ranked) > 1 else 0.0
    margin = confidence - second_confidence
    return predicted_idx, confidence, margin


