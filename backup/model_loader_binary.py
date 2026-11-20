"""Loads the trained CNN once. Falls back to DEMO mode if no model file exists."""
import json
import os

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE, "model", "brain_tumor_model.h5")
METRICS_PATH = os.path.join(BASE, "model", "metrics.json")

_model = None
_loaded = False


def _load():
    global _model, _loaded
    if _loaded:
        return
    _loaded = True
    if os.path.exists(MODEL_PATH):
        from tensorflow.keras.models import load_model
        _model = load_model(MODEL_PATH)


def is_demo_mode() -> bool:
    _load()
    return _model is None


def predict(batch) -> float:
    """Returns probability (0-1) that the scan contains a tumor."""
    _load()
    if _model is None:
        # DEMO ONLY: pseudo-score from pixel statistics. NOT a real prediction.
        m = float(batch.mean())
        return float(min(max((m - 0.15) * 2.2, 0.03), 0.97))
    return float(_model.predict(batch, verbose=0)[0][0])


def model_info() -> dict:
    _load()
    info = {
        "name": "Custom CNN (Conv2D + MaxPooling + Dense)",
        "input": "224x224 RGB",
        "output": "Binary - Tumor / No Tumor (sigmoid)",
        "mode": "demo" if _model is None else "trained",
        "metrics": None,
    }
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            info["metrics"] = json.load(f)
    return info
