"""Synthetic rule-derived classification demo. Not for health decisions."""
from __future__ import annotations

import argparse
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score

FEATURES = ["age", "fever", "tiredness", "cough"]
RANDOM_STATE = 42


def generate_synthetic_data(n_samples: int = 200, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Create random features and a target derived from a deliberately simple rule."""
    if n_samples < 10:
        raise ValueError("n_samples must be at least 10")
    rng = np.random.default_rng(random_state)
    data = pd.DataFrame({
        "age": rng.integers(20, 70, n_samples),
        "fever": rng.integers(0, 2, n_samples),
        "tiredness": rng.integers(0, 2, n_samples),
        "cough": rng.integers(0, 2, n_samples),
    })
    data["rule_label"] = (
        (data["age"] > 50).astype(int) + data["fever"] +
        data["tiredness"] + data["cough"]
    >= 2).astype(int)
    return data


def train_baseline(data: pd.DataFrame):
    """Fit a reproducible LogisticRegression baseline and return model/metrics."""
    missing = set(FEATURES + ["rule_label"]) - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if data[FEATURES + ["rule_label"]].isna().any().any():
        raise ValueError("Training data contains missing values")
    if data["rule_label"].nunique() != 2:
        raise ValueError("Training requires both rule-label classes")
    x_train, x_test, y_train, y_test = train_test_split(
        data[FEATURES], data["rule_label"],
        test_size=0.2, random_state=RANDOM_STATE, stratify=data["rule_label"],
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, pred),
        "test_samples": len(y_test),
    }
    return model, metrics


def validate_input(age: int, fever: int, tiredness: int, cough: int) -> pd.DataFrame:
    """Validate demo inputs and create a one-row frame in model feature order."""
    if not 0 <= age <= 120:
        raise ValueError("Age must be between 0 and 120")
    if any(value not in (0, 1) for value in (fever, tiredness, cough)):
        raise ValueError("Symptom flags must be 0 (no) or 1 (yes)")
    return pd.DataFrame([[age, fever, tiredness, cough]], columns=FEATURES)


def predict_demo(model, input_frame: pd.DataFrame) -> int:
    """Return the model's synthetic rule-label prediction; no clinical meaning."""
    missing = set(FEATURES) - set(input_frame.columns)
    if missing:
        raise ValueError(f"Missing input columns: {sorted(missing)}")
    return int(model.predict(input_frame[FEATURES])[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=200)
    args = parser.parse_args()
    data = generate_synthetic_data(args.samples)
    model, metrics = train_baseline(data)
    print("Synthetic rule-label classifier demo")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Balanced accuracy: {metrics['balanced_accuracy']:.3f}")
    print(f"Test rows: {metrics['test_samples']}")
    print("\nThis metric measures reproduction of synthetic rule labels only.")
    print("It is not a medical-risk estimate and is not clinically validated.")

    print("\nEnter example features:")
    try:
        age = int(input("Age (0–120): "))
        fever = int(input("Fever flag (0/1): "))
        tiredness = int(input("Tiredness flag (0/1): "))
        cough = int(input("Cough flag (0/1): "))
        sample = validate_input(age, fever, tiredness, cough)
        prediction = predict_demo(model, sample)
    except (ValueError, EOFError) as exc:
        parser.error(str(exc))
    print(f"Predicted synthetic rule label: {prediction}")
    print("Educational output only — do not use for health decisions.")


if __name__ == "__main__":
    main()
