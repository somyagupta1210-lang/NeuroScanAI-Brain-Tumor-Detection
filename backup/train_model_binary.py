"""
Train the binary CNN (Tumor / No Tumor) and save model + metrics.
Dataset layout (Kaggle: "Brain Tumor MRI Dataset" by Masoud Nickparvar):
  dataset/Training/{glioma,meningioma,notumor,pituitary}
  dataset/Testing/{glioma,meningioma,notumor,pituitary}
All tumor classes are merged into one "Tumor" class.
Run:  python train_model.py
"""
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

IMG, BATCH, EPOCHS = 224, 32, 15
NO_TUMOR = "notumor"


def load(path, shuffle):
    ds = tf.keras.utils.image_dataset_from_directory(
        path, image_size=(IMG, IMG), batch_size=BATCH, shuffle=shuffle, label_mode="int")
    nt = ds.class_names.index(NO_TUMOR)
    to_bin = lambda x, y: (x / 255.0, tf.cast(tf.not_equal(y, nt), tf.float32))  # same scaling as preprocess.py
    return ds.map(to_bin).prefetch(tf.data.AUTOTUNE)


train, test = load("dataset/Training", True), load("dataset/Testing", False)

model = models.Sequential([
    layers.Input((IMG, IMG, 3)),
    layers.RandomFlip("horizontal"), layers.RandomRotation(0.05),
    layers.Conv2D(32, 3, activation="relu"), layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu"), layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation="relu"), layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation="relu"), layers.MaxPooling2D(),
    layers.Flatten(), layers.Dropout(0.5),
    layers.Dense(128, activation="relu"),
    layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(train, validation_data=test, epochs=EPOCHS, callbacks=[
    tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)])
model.save("model/brain_tumor_model.h5")

y_true = np.concatenate([y.numpy() for _, y in test])
y_pred = (model.predict(test).ravel() >= 0.5).astype(int)
metrics = {
    "accuracy": float(accuracy_score(y_true, y_pred)),
    "precision": float(precision_score(y_true, y_pred)),
    "recall": float(recall_score(y_true, y_pred)),
    "f1": float(f1_score(y_true, y_pred)),
    "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
}
json.dump(metrics, open("model/metrics.json", "w"), indent=2)
print(json.dumps(metrics, indent=2))
