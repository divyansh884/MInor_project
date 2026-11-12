"""
PermaWatch - XGBoost Classifier Training Module
Trains a gradient-boosted decision tree ensemble using XGBoost
with class imbalance compensation and spatial block validation.
"""

import os
import logging
import joblib
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, classification_report
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocessing import get_scaled_datasets, FEATURE_COLUMNS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrainXGBoost")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODELS_DIR, "xgboost.pkl")


def train_xgboost(datasets=None) -> xgb.XGBClassifier:
    """Train and serialize the XGBoost Classifier."""
    if datasets is None:
        datasets = get_scaled_datasets()

    X_train = datasets["X_train"]
    y_train = datasets["y_train"]
    X_test = datasets["X_test"]
    y_test = datasets["y_test"]

    # Calculate class weight ratio
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / max(1, pos_count)

    logger.info("Initializing XGBoost Classifier...")
    logger.info(f"  Class balance: Negatives={neg_count}, Positives={pos_count}, ScalePosWeight={scale_pos_weight:.3f}")

    xgb_model = xgb.XGBClassifier(
        n_estimators=250,
        learning_rate=0.08,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    logger.info(f"Fitting XGBoost on {len(X_train)} samples...")
    xgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    y_pred = xgb_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    logger.info(f"XGBoost Spatial Validation Results:")
    logger.info(f"  Test Accuracy: {acc:.4f}")
    logger.info(f"  Test F1-Score: {f1:.4f}")

    joblib.dump(xgb_model, MODEL_PATH)
    logger.info(f"Saved XGBoost model to: {MODEL_PATH} ({os.path.getsize(MODEL_PATH)} bytes)")

    return xgb_model


if __name__ == "__main__":
    train_xgboost()
