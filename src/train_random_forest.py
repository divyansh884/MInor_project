"""
PermaWatch - Random Forest Classifier Training Module
Trains an ensemble Random Forest model on satellite and environmental features
using spatial-block partitioned training data.
"""

import os
import logging
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocessing import get_scaled_datasets, FEATURE_COLUMNS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrainRandomForest")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODELS_DIR, "random_forest.pkl")


def train_random_forest(datasets=None) -> RandomForestClassifier:
    """Train and serialize the Random Forest Classifier."""
    if datasets is None:
        datasets = get_scaled_datasets()

    X_train = datasets["X_train"]
    y_train = datasets["y_train"]
    X_test = datasets["X_test"]
    y_test = datasets["y_test"]

    logger.info("Initializing Random Forest Classifier...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=16,
        min_samples_split=6,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )

    logger.info(f"Fitting Random Forest on {len(X_train)} samples with 13 features...")
    rf_model.fit(X_train, y_train)

    # Quick test set check
    y_pred = rf_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    logger.info(f"Random Forest Spatial Validation Results:")
    logger.info(f"  Test Accuracy: {acc:.4f}")
    logger.info(f"  Test F1-Score: {f1:.4f}")

    joblib.dump(rf_model, MODEL_PATH)
    logger.info(f"Saved Random Forest model to: {MODEL_PATH} ({os.path.getsize(MODEL_PATH)} bytes)")

    return rf_model


if __name__ == "__main__":
    train_random_forest()
