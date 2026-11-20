"""
Loads the trained 4-class CNN model.
Provides predictions and Grad-CAM heatmaps.
"""

import os
import json
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model, Model


# ==========================================
# 1. MODEL PATHS AND CONFIGURATION
# ==========================================

BASE = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

MODEL_PATH = os.path.join(
    BASE, "model", "brain_tumor_model.h5"
)

METRICS_PATH = os.path.join(
    BASE, "model", "metrics.json"
)

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

_model = None
_grad_model = None
_loaded = False


# ==========================================
# 2. LOAD TRAINED MODEL
# ==========================================


def _load():

    global _model, _grad_model, _loaded

    if _loaded:
        return

    if not os.path.exists(MODEL_PATH):
        _loaded = True
        return

    print("Loading trained CNN model...")

    _model = load_model(MODEL_PATH)

    # Build the model's computational graph
    # using a symbolic input tensor.
    inputs = tf.keras.Input(
        shape=(224, 224, 3)
    )

    x = inputs
    conv_outputs = None

    # Pass input through each existing model layer.
    for layer in _model.layers:

        x = layer(x)

        if layer.name == "last_conv_layer":
            conv_outputs = x

    if conv_outputs is None:
        raise RuntimeError(
            "last_conv_layer not found in trained model."
        )

    # Create a functional model for Grad-CAM.
    _grad_model = Model(
        inputs=inputs,
        outputs=[
            conv_outputs,
            x
        ]
    )

    _loaded = True

    print("Trained CNN model loaded successfully!")


# ==========================================
# 3. CHECK DEMO MODE
# ==========================================

def is_demo_mode():

    _load()

    return _model is None


# ==========================================
# 4. FOUR-CLASS PREDICTION
# ==========================================

def predict(batch):
    """
    Predicts one of four brain MRI categories.

    Returns:
        predicted_class
        confidence
        probabilities
    """

    _load()

    if _model is None:
        raise RuntimeError(
            "Trained model not found. "
            "Please train the CNN first."
        )

    predictions = _model.predict(
        batch,
        verbose=0
    )[0]

    predicted_index = int(
        np.argmax(predictions)
    )

    predicted_class = CLASS_NAMES[predicted_index]

    probabilities = {
        CLASS_NAMES[i]: float(predictions[i])
        for i in range(len(CLASS_NAMES))
    }

    return {
        "predicted_class": predicted_class,
        "confidence": float(predictions[predicted_index]),
        "probabilities": probabilities
    }


# ==========================================
# 5. GRAD-CAM GENERATION
# ==========================================

def generate_gradcam(batch, class_index):
    """
    Generates Grad-CAM heatmap for a selected class.

    batch shape: (1, 224, 224, 3)
    class_index: 0, 1, 2 or 3
    """

    _load()

    if _model is None:
        raise RuntimeError(
            "Grad-CAM requires a trained model."
        )

    if not 0 <= class_index < len(CLASS_NAMES):
        raise ValueError("Invalid class index.")

    # Convert input into TensorFlow tensor
    image = tf.cast(batch, tf.float32)

    with tf.GradientTape() as tape:

        conv_outputs, predictions = _grad_model(
            image,
            training=False
        )

        # Select the predicted class score
        class_score = predictions[:, class_index]

    # Calculate gradients
    gradients = tape.gradient(
        class_score,
        conv_outputs
    )

    if gradients is None:
        raise RuntimeError(
            "Could not calculate Grad-CAM gradients."
        )

    # Calculate channel importance weights
    weights = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )

    # Weighted combination of feature maps
    heatmap = tf.reduce_sum(
        conv_outputs[0] * weights[0],
        axis=-1
    )

    # Keep positive activations
    heatmap = tf.maximum(heatmap, 0)

    # Normalize between 0 and 1
    heatmap = heatmap / (
        tf.reduce_max(heatmap) + tf.keras.backend.epsilon()
    )

    return heatmap.numpy().astype(np.float32)


# ==========================================
# 6. MODEL INFORMATION
# ==========================================

def model_info():

    _load()

    info = {

        "name": "Custom CNN (4-Class Classification)",

        "input": "224x224 RGB",

        "output": "4-Class Softmax Classification",

        "classes": CLASS_NAMES,

        "mode": "demo" if _model is None else "trained",

        "metrics": None

    }

    if os.path.exists(METRICS_PATH):

        with open(METRICS_PATH) as file:

            info["metrics"] = json.load(file)

    return info