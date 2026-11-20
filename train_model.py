"""
Brain Tumor Detection - 4 Class Classification

Classes:
1. Glioma
2. Meningioma
3. No Tumor
4. Pituitary

Training dataset is divided into:
80% Training
20% Validation

Testing dataset is kept separate for final evaluation.
"""

import os
import json
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ==========================================
# 1. CONFIGURATION
# ==========================================

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20

TRAIN_DIR = "dataset/Training"
TEST_DIR = "dataset/Testing"

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "brain_tumor_model.h5")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

os.makedirs(MODEL_DIR, exist_ok=True)

tf.random.set_seed(42)
np.random.seed(42)


# ==========================================
# 2. LOAD TRAINING DATASET
# ==========================================

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="int",
    validation_split=0.2,
    subset="training",
    seed=42,
    shuffle=True
)

# Save class names in the correct order
class_names = train_ds.class_names

print("\nClasses:", class_names)


# ==========================================
# 3. LOAD VALIDATION DATASET
# ==========================================

val_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="int",
    validation_split=0.2,
    subset="validation",
    seed=42,
    shuffle=False
)


# ==========================================
# 4. LOAD TESTING DATASET
# ==========================================

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

# Verify that all datasets use the same class order
assert class_names == val_ds.class_names
assert class_names == test_ds.class_names

# Normalize image pixels between 0 and 1
def normalize_images(images, labels):
    images = tf.cast(images, tf.float32) / 255.0
    return images, labels


train_ds = train_ds.map(normalize_images)
val_ds = val_ds.map(normalize_images)
test_ds = test_ds.map(normalize_images)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)


# ==========================================
# 5. CNN MODEL ARCHITECTURE
# ==========================================

model = models.Sequential([

    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

    # Data augmentation
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),

    # Convolution Block 1
    layers.Conv2D(
        32, 3, activation="relu", padding="same",
        name="conv1"
    ),
    layers.MaxPooling2D(),

    # Convolution Block 2
    layers.Conv2D(
        64, 3, activation="relu", padding="same",
        name="conv2"
    ),
    layers.MaxPooling2D(),

    # Convolution Block 3
    layers.Conv2D(
        128, 3, activation="relu", padding="same",
        name="conv3"
    ),
    layers.MaxPooling2D(),

    # Convolution Block 4
    # Grad-CAM will use this layer
    layers.Conv2D(
        128, 3, activation="relu", padding="same",
        name="last_conv_layer"
    ),
    layers.MaxPooling2D(),

    # Classification layers
    layers.GlobalAveragePooling2D(),

    layers.Dropout(0.4),

    layers.Dense(128, activation="relu"),

    # Four output classes
    layers.Dense(4, activation="softmax", name="predictions")

])


# ==========================================
# 6. COMPILE MODEL
# ==========================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)

print("\nCNN Model Architecture:")
model.summary()


# ==========================================
# 7. CALLBACKS
# ==========================================

callbacks = [

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=0.00001
    )

]


# ==========================================
# 8. TRAIN MODEL
# ==========================================

print("\nStarting 4-Class CNN Training...\n")

history = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=EPOCHS,

    callbacks=callbacks

)


# ==========================================
# 9. SAVE TRAINED MODEL
# ==========================================

model.save(MODEL_PATH)

print("\nModel saved successfully!")
print("Location:", MODEL_PATH)


# ==========================================
# 10. FINAL MODEL EVALUATION
# ==========================================

print("\nEvaluating model on unseen testing data...")

y_true = np.concatenate([
    labels.numpy()
    for _, labels in test_ds
])

predictions = model.predict(test_ds)

y_pred = np.argmax(predictions, axis=1)


# ==========================================
# 11. CALCULATE PERFORMANCE METRICS
# ==========================================

accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(len(class_names)))
)

report = classification_report(
    y_true,
    y_pred,
    labels=list(range(len(class_names))),
    target_names=class_names,
    output_dict=True,
    zero_division=0
)


# ==========================================
# 12. SAVE METRICS
# ==========================================

metrics = {

    "accuracy": float(accuracy),

    "precision": float(precision),

    "recall": float(recall),

    "f1_score": float(f1),

    "class_names": class_names,

    "confusion_matrix": cm.tolist(),

    "classification_report": report

}

with open(METRICS_PATH, "w") as file:

    json.dump(metrics, file, indent=4)


# ==========================================
# 13. DISPLAY RESULTS
# ==========================================

print("\n========== FINAL RESULTS ==========")

print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1 Score  : {f1 * 100:.2f}%")

print("\nClassification Report:")

print(
    classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(cm)

print("\nMetrics saved at:", METRICS_PATH)

print("\n4-Class Training Completed Successfully!")