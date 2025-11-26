from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

from .config import BRAIN_IMAGE_SIZE, ORAL_IMAGE_SIZE


class InvalidImageError(ValueError):
    """Raised when an uploaded asset is not a valid RGB image."""


def _load_image(file, size: Tuple[int, int]) -> Image.Image:
    try:
        img = Image.open(file).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImageError("The provided file is not a supported image.") from exc
    if size:
        img = img.resize(size)
    return img


def prepare_brain_batch(file) -> Tuple[np.ndarray, Image.Image]:
    """Return (batch, preview_image) for the brain CNN.
    Images are normalized to [0, 1] by dividing by 255.
    """
    img = _load_image(file, BRAIN_IMAGE_SIZE)
    arr = np.array(img, dtype=np.float32)
    # Normalize to [0, 1] range (model expects this)
    arr = arr / 255.0
    batch = np.expand_dims(arr, axis=0)
    return batch, img


def prepare_oral_vector(file) -> Tuple[np.ndarray, Image.Image]:
    """Return (vector, preview) for the oral cancer classifier.
    Images are normalized to [0, 1] by dividing by 255.
    Supports both old (flattened) and new (batched) formats.
    """
    img = _load_image(file, ORAL_IMAGE_SIZE)
    arr = np.array(img, dtype=np.float32)
    # Normalize to [0, 1] range (model expects this)
    arr = arr / 255.0
    vector = arr.flatten().reshape(1, -1)
    return vector, img


def prepare_oral_batch(file) -> Tuple[np.ndarray, Image.Image]:
    """Return (batch, preview) for the oral cancer Keras model.
    Images are normalized to [0, 1] and returned as batched array.
    """
    img = _load_image(file, ORAL_IMAGE_SIZE)
    arr = np.array(img, dtype=np.float32)
    # Normalize to [0, 1] range (model expects this)
    arr = arr / 255.0
    batch = np.expand_dims(arr, axis=0)
    return batch, img


def build_cervical_vector(
    features: Dict[str, float],
    feature_names: List[str],
    scaler,
) -> np.ndarray:
    ordered = {name: [float(features.get(name, 0.0))] for name in feature_names}
    frame = pd.DataFrame(ordered)[feature_names]
    return scaler.transform(frame)


@dataclass(frozen=True)
class CervicalField:
    key: str
    label: str
    section: str
    input_type: str
    min_value: float | None = None
    max_value: float | None = None
    step: float = 1.0
    help_text: str | None = None


CERVICAL_FIELDS: Tuple[CervicalField, ...] = (
    CervicalField("Age", "Age (years)", "Demographics", "number", 13, 84, 1),
    CervicalField(
        "Number of sexual partners",
        "# sexual partners",
        "Demographics",
        "number",
        0,
        30,
        1,
    ),
    CervicalField(
        "First sexual intercourse",
        "Age at first intercourse",
        "Demographics",
        "number",
        10,
        40,
        1,
    ),
    CervicalField(
        "Num of pregnancies",
        "# pregnancies",
        "Demographics",
        "number",
        0,
        16,
        1,
    ),
    CervicalField("Smokes", "Currently smokes", "Lifestyle", "binary"),
    CervicalField("Smokes (years)", "Smoking history (years)", "Lifestyle", "number", 0, 37, 1),
    CervicalField("Smokes (packs/year)", "Smoking intensity (packs/yr)", "Lifestyle", "number", 0, 60, 1),
    CervicalField("Hormonal Contraceptives", "Uses hormonal contraceptives", "Lifestyle", "binary"),
    CervicalField(
        "Hormonal Contraceptives (years)",
        "Hormonal contraceptives (years)",
        "Lifestyle",
        "number",
        0,
        30,
        1,
    ),
    CervicalField("IUD", "Uses IUD", "Lifestyle", "binary"),
    CervicalField("IUD (years)", "IUD usage (years)", "Lifestyle", "number", 0, 25, 1),
    CervicalField("STDs", "History of any STD", "STD History", "binary"),
    CervicalField("STDs (number)", "# of STDs diagnosed", "STD History", "number", 0, 10, 1),
    CervicalField("STDs:condylomatosis", "STDs: condylomatosis", "STD History", "binary"),
    CervicalField("STDs:cervical condylomatosis", "STDs: cervical condylomatosis", "STD History", "binary"),
    CervicalField("STDs:vaginal condylomatosis", "STDs: vaginal condylomatosis", "STD History", "binary"),
    CervicalField("STDs:vulvo-perineal condylomatosis", "STDs: vulvo-perineal condylomatosis", "STD History", "binary"),
    CervicalField("STDs:syphilis", "STDs: syphilis", "STD History", "binary"),
    CervicalField(
        "STDs:pelvic inflammatory disease",
        "STDs: pelvic inflammatory disease",
        "STD History",
        "binary",
    ),
    CervicalField("STDs:genital herpes", "STDs: genital herpes", "STD History", "binary"),
    CervicalField("STDs:molluscum contagiosum", "STDs: molluscum contagiosum", "STD History", "binary"),
    CervicalField("STDs:AIDS", "STDs: AIDS", "STD History", "binary"),
    CervicalField("STDs:HIV", "STDs: HIV", "STD History", "binary"),
    CervicalField("STDs:Hepatitis B", "STDs: Hepatitis B", "STD History", "binary"),
    CervicalField("STDs:HPV", "STDs: HPV", "STD History", "binary"),
    CervicalField("STDs: Number of diagnosis", "Total STD diagnoses", "STD History", "number", 0, 10, 1),
    CervicalField("Dx:Cancer", "Doctor Dx: Cancer", "Diagnostics", "binary"),
    CervicalField("Dx:CIN", "Doctor Dx: CIN", "Diagnostics", "binary"),
    CervicalField("Dx:HPV", "Doctor Dx: HPV", "Diagnostics", "binary"),
    CervicalField("Dx", "Doctor Dx: other", "Diagnostics", "binary"),
    CervicalField("Hinselmann", "Hinselmann test positive", "Diagnostics", "binary"),
    CervicalField("Schiller", "Schiller test positive", "Diagnostics", "binary"),
    CervicalField("Citology", "Cytology positive", "Diagnostics", "binary"),
)


def build_default_cervical_payload(metadata: Dict[str, float]) -> Dict[str, float]:
    return {field.key: float(metadata.get(field.key, 0.0)) for field in CERVICAL_FIELDS}


# Brain Tumor Risk Factor Fields
BRAIN_FIELDS: Tuple[CervicalField, ...] = (
    CervicalField("Age", "Age (years)", "Demographics", "number", 0, 100, 1.0),
    CervicalField("Gender", "Gender (0=Female, 1=Male)", "Demographics", "binary"),
    CervicalField("Headache_Frequency", "Headache frequency per week", "Symptoms", "number", 0, 7, 1.0),
    CervicalField("Seizures", "History of seizures", "Symptoms", "binary"),
    CervicalField("Vision_Problems", "Vision problems or blurriness", "Symptoms", "binary"),
    CervicalField("Nausea_Vomiting", "Nausea or vomiting", "Symptoms", "binary"),
    CervicalField("Memory_Issues", "Memory or cognitive issues", "Symptoms", "binary"),
    CervicalField("Weakness", "Muscle weakness or numbness", "Symptoms", "binary"),
    CervicalField("Balance_Problems", "Balance or coordination problems", "Symptoms", "binary"),
    CervicalField("Speech_Difficulties", "Speech difficulties", "Symptoms", "binary"),
    CervicalField("Family_History", "Family history of brain tumors", "Family History", "binary"),
    CervicalField("Radiation_Exposure", "Previous radiation exposure", "Risk Factors", "binary"),
    CervicalField("Smoking", "Smoking history", "Lifestyle", "binary"),
    CervicalField("Alcohol_Consumption", "Regular alcohol consumption", "Lifestyle", "binary"),
)


# Oral Cancer Risk Factor Fields
ORAL_FIELDS: Tuple[CervicalField, ...] = (
    CervicalField("Age", "Age (years)", "Demographics", "number", 18, 100, 1.0),
    CervicalField("Gender", "Gender (0=Female, 1=Male)", "Demographics", "binary"),
    CervicalField("Smoking", "Current or past smoking", "Lifestyle", "binary"),
    CervicalField("Smoking_Years", "Years of smoking", "Lifestyle", "number", 0, 60, 1.0),
    CervicalField("Alcohol_Consumption", "Regular alcohol consumption", "Lifestyle", "binary"),
    CervicalField("Alcohol_Years", "Years of alcohol use", "Lifestyle", "number", 0, 50, 1.0),
    CervicalField("Betel_Quid", "Betel quid or paan use", "Lifestyle", "binary"),
    CervicalField("Poor_Oral_Hygiene", "Poor oral hygiene", "Oral Health", "binary"),
    CervicalField("Dental_Problems", "Frequent dental problems", "Oral Health", "binary"),
    CervicalField("Mouth_Sores", "Persistent mouth sores or ulcers", "Symptoms", "binary"),
    CervicalField("Red_White_Patches", "Red or white patches in mouth", "Symptoms", "binary"),
    CervicalField("Difficulty_Swallowing", "Difficulty swallowing", "Symptoms", "binary"),
    CervicalField("Lump_Throat", "Lump or thickening in mouth/throat", "Symptoms", "binary"),
    CervicalField("Pain_Mouth", "Persistent pain in mouth", "Symptoms", "binary"),
    CervicalField("HPV_History", "History of HPV infection", "Medical History", "binary"),
    CervicalField("Family_History", "Family history of oral cancer", "Family History", "binary"),
)
