"""
predict.py
----------
Load the trained CNN model and predict whether two DNA STR profiles match.
Can be used from command line or imported as a module.

Usage:
    python predict.py
"""

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# ─── STR Loci (must match training order) ────────────────────────────────────
STR_LOCI = [
    "D3S1358", "vWA", "FGA", "D8S1179", "D21S11",
    "D18S51", "D5S818", "D13S317", "D7S820", "D16S539",
    "TH01", "TPOX", "CSF1PO", "D2S1338", "D19S433"
]


def load_model(model_path="models/cnn_dna_model.h5"):
    model = tf.keras.models.load_model(model_path)
    print(f"Model loaded from {model_path}")
    return model


def preprocess_profiles(profile1, profile2, data_path="data/dna_str_dataset.csv"):
    """
    Normalize two profiles using the training data scaler.
    profile1, profile2: lists of 30 integer allele values each.
    """
    df = pd.read_csv(data_path)
    X = df.drop("label", axis=1).values

    scaler = MinMaxScaler()
    scaler.fit(X)

    combined = np.array(profile1 + profile2).reshape(1, -1)
    scaled = scaler.transform(combined)
    return scaled.reshape(1, scaled.shape[1], 1)


def predict_match(profile1, profile2, model, data_path="data/dna_str_dataset.csv"):
    """
    Returns:
        result  : 'MATCH' or 'NO MATCH'
        confidence: float (0–100%)
    """
    input_data = preprocess_profiles(profile1, profile2, data_path)
    prob = model.predict(input_data, verbose=0)[0][0]
    result = "MATCH" if prob >= 0.5 else "NO MATCH"
    confidence = prob * 100 if prob >= 0.5 else (1 - prob) * 100
    return result, round(confidence, 2), round(float(prob), 4)


# ─── Demo ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(99)

    model = load_model()

    # Simulate: same person (should MATCH)
    person_a_profile1 = [np.random.randint(5, 35) for _ in range(30)]
    person_a_profile2 = [v + np.random.choice([-1, 0, 0, 1]) for v in person_a_profile1]

    # Simulate: different people (should NOT MATCH)
    person_b_profile  = [np.random.randint(5, 35) for _ in range(30)]

    print("\n" + "="*55)
    print("  FORENSIC DNA PROFILE MATCHING — PREDICTION DEMO")
    print("="*55)

    result, conf, prob = predict_match(person_a_profile1, person_a_profile2, model)
    print(f"\nTest 1 (Same person)")
    print(f"  Prediction  : {result}")
    print(f"  Confidence  : {conf}%")
    print(f"  Raw Score   : {prob}")

    result, conf, prob = predict_match(person_a_profile1, person_b_profile, model)
    print(f"\nTest 2 (Different person)")
    print(f"  Prediction  : {result}")
    print(f"  Confidence  : {conf}%")
    print(f"  Raw Score   : {prob}")

    print("\n" + "="*55)
