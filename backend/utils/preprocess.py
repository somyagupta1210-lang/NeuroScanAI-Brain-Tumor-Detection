"""Image preprocessing - MUST match what train_model.py uses."""
import io
import numpy as np
from PIL import Image

IMG_SIZE = 224


def preprocess_image(source) -> np.ndarray:
    """source: file path or raw bytes. Returns array shape (1, 224, 224, 3), values 0-1."""
    img = Image.open(io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else source)
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img, dtype="float32") / 255.0
    return np.expand_dims(arr, axis=0)
