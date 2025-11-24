from io import BytesIO

import numpy as np
from PIL import Image
import pytest

from app.preprocessing import (
    BRAIN_IMAGE_SIZE,
    ORAL_IMAGE_SIZE,
    InvalidImageError,
    prepare_brain_batch,
    prepare_oral_vector,
)


def _make_image_bytes(size=(256, 256)):
    buf = BytesIO()
    Image.fromarray(np.full((*size, 3), 128, dtype=np.uint8)).save(buf, format="PNG")
    buf.seek(0)
    return buf


def test_prepare_brain_batch_shapes_and_values():
    batch, preview = prepare_brain_batch(_make_image_bytes(BRAIN_IMAGE_SIZE))
    assert batch.shape == (1, *BRAIN_IMAGE_SIZE, 3)
    assert preview.size == BRAIN_IMAGE_SIZE


def test_prepare_oral_vector_shapes():
    vector, preview = prepare_oral_vector(_make_image_bytes(ORAL_IMAGE_SIZE))
    expected_features = ORAL_IMAGE_SIZE[0] * ORAL_IMAGE_SIZE[1] * 3
    assert vector.shape == (1, expected_features)
    assert preview.size == ORAL_IMAGE_SIZE


def test_invalid_image_raises():
    with pytest.raises(InvalidImageError):
        prepare_brain_batch(BytesIO(b"not-an-image"))

