"""
train_model.py
--------------
Trains a 1D Convolutional Neural Network (CNN) on synthetic forensic
DNA STR profile pairs to classify: Match (1) vs No Match (0).

Architecture:
  Input (60 features) → Reshape → Conv1D × 2 → GlobalMaxPool → Dense → Output
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, roc_auc_score, roc_curve
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, GlobalMaxPooling1D,
    Dense, Dropout, BatchNormalization, Reshape
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# ─── Config ────────────────────────────────────────────────────────────────
DATA_PATH   = "data/dna_str_dataset.csv"
MODEL_PATH  = "models/cnn_dna_model.h5"
RESULTS_DIR = "results"
EPOCHS      = 50
BATCH_SIZE  = 32
LEARNING_RATE = 0.001
os.makedirs("models", exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─── Load & Preprocess Data ─────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

X = df.drop("label", axis=1).values
y = df["label"].values

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Reshape for Conv1D: (samples, timesteps, channels)
X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

X_train, X_test, y_train, y_test = train_test_split(
    X_reshaped, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples : {X_train.shape[0]}")
print(f"Testing samples  : {X_test.shape[0]}")
print(f"Input shape      : {X_train.shape[1:]}")

# ─── Build CNN Model ─────────────────────────────────────────────────────────
def build_cnn(input_shape):
    model = Sequential([
        # Block 1
        Conv1D(filters=64, kernel_size=3, activation='relu',
               padding='same', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        # Block 2
        Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        # Block 3
        Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
        GlobalMaxPooling1D(),

        # Fully Connected
        Dense(64, activation='relu'),
        Dropout(0.4),
        Dense(32, activation='relu'),

        # Output
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

model = build_cnn((X_train.shape[1], 1))
model.summary()

# ─── Train ───────────────────────────────────────────────────────────────────
callbacks = [
    EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
    ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_accuracy')
]

print("\nTraining CNN model...")
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# ─── Evaluate ────────────────────────────────────────────────────────────────
print("\nEvaluating on test set...")
y_pred_prob = model.predict(X_test).flatten()
y_pred = (y_pred_prob >= 0.5).astype(int)

acc    = accuracy_score(y_test, y_pred)
auc    = roc_auc_score(y_test, y_pred_prob)
report = classification_report(y_test, y_pred, target_names=["No Match", "Match"])
cm     = confusion_matrix(y_test, y_pred)

print(f"\nAccuracy : {acc * 100:.2f}%")
print(f"AUC-ROC  : {auc:.4f}")
print(f"\nClassification Report:\n{report}")
print(f"Confusion Matrix:\n{cm}")

# Save metrics
metrics = {"accuracy": round(acc * 100, 2), "auc_roc": round(auc, 4)}
with open(f"{RESULTS_DIR}/metrics.json", "w") as f:
    json.dump(metrics, f)

# ─── Plot: Accuracy & Loss ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("CNN Training Results — Forensic DNA Profile Matching", fontsize=14, fontweight='bold')

axes[0].plot(history.history['accuracy'], label='Train', color='#2196F3')
axes[0].plot(history.history['val_accuracy'], label='Validation', color='#FF5722')
axes[0].set_title('Model Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(history.history['loss'], label='Train', color='#2196F3')
axes[1].plot(history.history['val_loss'], label='Validation', color='#FF5722')
axes[1].set_title('Model Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/training_curves.png", dpi=150)
plt.close()

# ─── Plot: Confusion Matrix ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(['No Match', 'Match'])
ax.set_yticklabels(['No Match', 'Match'])
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('Actual', fontsize=12)
ax.set_title('Confusion Matrix', fontsize=13, fontweight='bold')
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i][j]), ha='center', va='center',
                fontsize=16, color='white' if cm[i][j] > cm.max()/2 else 'black')
plt.colorbar(im)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/confusion_matrix.png", dpi=150)
plt.close()

# ─── Plot: ROC Curve ────────────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, color='#2196F3', lw=2, label=f'AUC = {auc:.4f}')
ax.plot([0, 1], [0, 1], 'k--', lw=1)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve — DNA Profile Matching', fontweight='bold')
ax.legend(loc='lower right')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/roc_curve.png", dpi=150)
plt.close()

print(f"\nAll results saved to '{RESULTS_DIR}/' folder.")
print(f"Model saved to '{MODEL_PATH}'")
print("\nProject complete!")
