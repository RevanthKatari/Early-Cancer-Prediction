from __future__ import annotations

from functools import lru_cache
from io import BytesIO
from typing import Dict, Tuple

import joblib
import numpy as np
import tensorflow as tf

from .config import (
    BRAIN_CLASSES,
    BRAIN_MODEL_PATH,
    CERVICAL_MODEL_PATH,
    CERVICAL_SCALER_PATH,
    MODELS_DIR,
    ORAL_LABELS,
    ORAL_MODEL_PATH,
)
from .preprocessing import (
    BRAIN_FIELDS,
    CERVICAL_FIELDS,
    ORAL_FIELDS,
    build_cervical_vector,
    build_default_cervical_payload,
    prepare_brain_batch,
    prepare_oral_vector,
    prepare_oral_batch,
)


@lru_cache(maxsize=1)
def _brain_model():
    return tf.keras.models.load_model(BRAIN_MODEL_PATH, compile=False)


@lru_cache(maxsize=1)
def _oral_model():
    """Load oral cancer model - supports both Keras (.keras) and joblib formats."""
    # Try loading as Keras model first (new format)
    keras_path = MODELS_DIR / "oc_rf_model.keras"
    joblib_path = MODELS_DIR / "oc_rf_model"
    
    if keras_path.exists():
        return tf.keras.models.load_model(str(keras_path), compile=False)
    elif joblib_path.exists():
        # Fallback to old joblib format for backward compatibility
        model = joblib.load(str(joblib_path))
        if not hasattr(model, "monotonic_cst"):
            setattr(model, "monotonic_cst", None)
        return model
    else:
        raise FileNotFoundError(f"Oral model not found at {keras_path} or {joblib_path}")


@lru_cache(maxsize=1)
def _cervical_model():
    return joblib.load(CERVICAL_MODEL_PATH)


@lru_cache(maxsize=1)
def _cervical_scaler_bundle():
    bundle = joblib.load(CERVICAL_SCALER_PATH)
    return bundle


def load_cervical_defaults():
    bundle = _cervical_scaler_bundle()
    return build_default_cervical_payload(bundle["defaults"])


def infer_brain(image_bytes: bytes):
    batch, preview = prepare_brain_batch(BytesIO(image_bytes))
    probs = _brain_model().predict(batch, verbose=0)[0]
    idx = int(np.argmax(probs))
    return preview, probs, BRAIN_CLASSES[idx], float(probs[idx])


def infer_oral(image_bytes: bytes):
    """Infer oral cancer - supports both Keras and sklearn models."""
    model = _oral_model()
    
    # Check if it's a Keras model
    if isinstance(model, tf.keras.Model):
        # Keras model expects batched input
        batch, preview = prepare_oral_batch(BytesIO(image_bytes))
        probs_raw = model.predict(batch, verbose=0)[0]
        # Convert sigmoid output to [non-cancer, cancer] probabilities
        probs = np.array([1 - probs_raw[0], probs_raw[0]])
        idx = 1 if probs_raw[0] > 0.5 else 0
        label = ORAL_LABELS.get(idx, "Non-cancer" if idx == 0 else "Cancer")
        confidence = float(probs[idx])
    else:
        # Old sklearn model (backward compatibility)
        vector, preview = prepare_oral_vector(BytesIO(image_bytes))
        probs = model.predict_proba(vector)[0]
        idx = int(np.argmax(probs))
        label = ORAL_LABELS.get(model.classes_[idx], str(model.classes_[idx]))
        confidence = float(probs[idx])
    
    return preview, probs, label, confidence


def infer_cervical(feature_payload: Dict[str, float]):
    bundle = _cervical_scaler_bundle()
    vector = build_cervical_vector(feature_payload, bundle["feature_names"], bundle["scaler"])
    model = _cervical_model()
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(vector)[0]
    else:
        pred = model.predict(vector)[0]
        probs = np.array([1 - pred, pred])
    risk = float(probs[1])
    return probs, risk


def get_cervical_fieldset():
    return CERVICAL_FIELDS


def get_brain_fieldset():
    return BRAIN_FIELDS


def get_oral_fieldset():
    return ORAL_FIELDS
