"""
models/model_loader.py
Load the Keras model ONCE at app startup and share across all detectors.
"""

import tensorflow as tf
import numpy as np
from config import MODEL_PATH, INPUT_SIZE

_model = None   # module-level singleton


def load_model():
    """Load and return the Keras model. Called once at Flask startup."""
    global _model
    if _model is None:
        print(f"[ModelLoader] Loading model from: {MODEL_PATH}")
        _model = tf.keras.models.load_model(MODEL_PATH)
        print("[ModelLoader] Model loaded successfully.")
        # Warm-up pass so first inference isn't slow
        dummy = np.zeros((1, INPUT_SIZE[0], INPUT_SIZE[1], 3), dtype=np.float32)
        _model.predict(dummy, verbose=0)
        print("[ModelLoader] Warm-up complete.")
    return _model


def get_model():
    """Get the already-loaded model (must call load_model first)."""
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")
    return _model
