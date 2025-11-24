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
    """Return (batch, preview_image) for the brain CNN."""
    img = _load_image(file, BRAIN_IMAGE_SIZE)
    arr = np.array(img, dtype=np.float32)
    batch = np.expand_dims(arr, axis=0)
    return batch, img


def prepare_oral_vector(file) -> Tuple[np.ndarray, Image.Image]:
    """Return (vector, preview) for the oral cancer RF classifier."""
    img = _load_image(file, ORAL_IMAGE_SIZE)
    arr = np.array(img, dtype=np.float32)
    vector = arr.flatten().reshape(1, -1)
    return vector, img


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
