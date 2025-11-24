from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "Models"

BRAIN_MODEL_PATH = MODELS_DIR / "bt-cnn2.keras"
CERVICAL_MODEL_PATH = MODELS_DIR / "cc_bagging_model"
CERVICAL_SCALER_PATH = MODELS_DIR / "cc_scaler.joblib"
ORAL_MODEL_PATH = MODELS_DIR / "oc_rf_model"

BRAIN_CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
ORAL_LABELS = {0: "Non-cancer", 1: "Cancer"}

BRAIN_IMAGE_SIZE = (180, 180)
ORAL_IMAGE_SIZE = (64, 64)

MODEL_HIGHLIGHTS = [
    {
        "title": "Brain MRI CNN",
        "metric": "Test accuracy",
        "value": "93%",
        "caption": "bt-cnn2 (3-layer CNN + augmentation)",
    },
    {
        "title": "Cervical Bagging",
        "metric": "F1 (positive)",
        "value": "0.77",
        "caption": "Decision-tree bagging on risk factors",
    },
    {
        "title": "Oral RF",
        "metric": "Cross-val accuracy",
        "value": "76%",
        "caption": "Random forest on 64×64 oral images",
    },
]

APP_TITLE = "Early Cancer Screening Studio"
APP_DESCRIPTION = (
    "Upload medical images or risk-factor profiles to validate the existing "
    "models and experiment with different diagnostic scenarios. "
    "All predictions run locally using the trained artifacts in `Models/`."
)
