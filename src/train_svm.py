"""
PermaWatch - Support Vector Machine (SVM) Training Module
Trains a Support Vector Classifier with an RBF kernel on standardized features,
with Platt scaling probability calibration and spatial block evaluation.
"""

import os
import logging
import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocessing import get_scaled_datasets, FEATURE_COLUMNS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrainSVM")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODELS_DIR, "svm.pkl")


def train_svm(datasets=None, subsample_size: int = 12000) -> SVC:
    """
    Train and serialize the Support Vector Machine Classifier.
    Applies StandardScaler features and RBF kernel with Platt probability estimation.
    """
    if datasets is None:
        datasets = get_scaled_datasets()

    X_train_scaled = datasets["X_train_scaled"]
    y_train = datasets["y_train"]
    X_test_scaled = datasets["X_test_scaled"]
    y_test = datasets["y_test"]

    # Subsample training data if large to keep training within lightweight laptop budget
    if len(X_train_scaled) > subsample_size:
        logger.info(f"Subsampling {subsample_size} stratified points from {len(X_train_scaled)} for SVM training efficiency...")
        rng = np.random.RandomState(42)
        idx_pos = np.where(y_train == 1)[0]
        idx_neg = np.where(y_train == 0)[0]
        n_pos = int(subsample_size * (len(idx_pos) / len(y_train)))
        n_neg = subsample_size - n_pos

        sub_pos = rng.choice(idx_pos, size=n_pos, replace=False)
        sub_neg = rng.choice(idx_neg, size=n_neg, replace=False)
        sub_idx = np.concatenate([sub_pos, sub_neg])
        rng.shuffle(sub_idx)

        X_fit = X_train_scaled[sub_idx]
        y_fit = y_train[sub_idx]
    else:
        X_fit = X_train_scaled
        y_fit = y_train

    logger.info(f"Initializing Support Vector Classifier (RBF kernel, C=1.5, gamma='scale')...")
    svm_model = SVC(
        kernel="rbf",
        C=1.5,
        gamma="scale",
        class_weight="balanced",
        probability=True,
        cache_size=1000,
        random_state=42,
    )

    logger.info(f"Fitting SVM model on {len(X_fit)} standardized samples...")
    svm_model.fit(X_fit, y_fit)

    y_pred = svm_model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    logger.info(f"SVM Spatial Validation Results:")
    logger.info(f"  Test Accuracy: {acc:.4f}")
    logger.info(f"  Test F1-Score: {f1:.4f}")

    joblib.dump(svm_model, MODEL_PATH)
    logger.info(f"Saved SVM model to: {MODEL_PATH} ({os.path.getsize(MODEL_PATH)} bytes)")

    return svm_model


if __name__ == "__main__":
    train_svm()
