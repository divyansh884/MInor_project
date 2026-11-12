"""
PermaWatch - Preprocessing & Spatial Validation Module
Prepares structured feature matrices, handles missing values, applies scaling,
and performs geographically separated spatial-block splitting (GroupShuffleSplit)
to prevent spatial autocorrelation leakage.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List, Optional
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Preprocessing")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "permafrost_training_dataset.parquet")
CSV_PATH = os.path.join(BASE_DIR, "data", "processed", "permafrost_training_dataset.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

FEATURE_COLUMNS = [
    "latitude",
    "longitude",
    "elevation",
    "slope",
    "blue",
    "green",
    "red",
    "nir",
    "swir1",
    "swir2",
    "ndvi",
    "ndwi",
    "ndsi",
]

TARGET_COLUMN = "permafrost_class"
GROUP_COLUMN = "spatial_block_id"


def load_dataset() -> pd.DataFrame:
    """Load the processed training dataset from Parquet or CSV."""
    if os.path.exists(DATA_PATH):
        df = pd.read_parquet(DATA_PATH)
    elif os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
    else:
        raise FileNotFoundError(f"Training dataset not found. Looked in {DATA_PATH} and {CSV_PATH}")

    if "spatial_block_id" not in df.columns and "region_name" in df.columns:
        df["spatial_block_id"] = df["region_name"]

    # Missing value cleaning
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).copy()
    return df


def prepare_spatial_split(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """
    Perform spatial block splitting using GroupShuffleSplit.
    Guarantees train and test samples are from disjoint geographic blocks.
    """
    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMN].values.astype(int)
    groups = df[GROUP_COLUMN].values

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    df_train = df.iloc[train_idx].copy()
    df_test = df.iloc[test_idx].copy()

    X_train = df_train[FEATURE_COLUMNS].values
    y_train = df_train[TARGET_COLUMN].values
    X_test = df_test[FEATURE_COLUMNS].values
    y_test = df_test[TARGET_COLUMN].values

    logger.info("Spatial-Block Train/Test Partition Complete:")
    logger.info(f"  Train samples: {len(X_train)} across blocks: {sorted(df_train[GROUP_COLUMN].unique())}")
    logger.info(f"  Test samples:  {len(X_test)} across blocks:  {sorted(df_test[GROUP_COLUMN].unique())}")
    logger.info(f"  Train class distribution: {np.bincount(y_train)}")
    logger.info(f"  Test class distribution:  {np.bincount(y_test)}")

    return X_train, X_test, y_train, y_test, df_train, df_test


def fit_and_save_scaler(X_train: np.ndarray) -> StandardScaler:
    """Fit StandardScaler on training features and save to models/scaler.pkl."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    logger.info(f"Saved feature scaler to: {scaler_path}")
    return scaler


def get_scaled_datasets(
    df: Optional[pd.DataFrame] = None,
    test_size: float = 0.20,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    High-level preprocessor helper that returns raw and scaled datasets
    along with metadata for modeling and evaluation.
    """
    if df is None:
        df = load_dataset()

    X_train, X_test, y_train, y_test, df_train, df_test = prepare_spatial_split(
        df, test_size=test_size, random_state=random_state
    )

    scaler = fit_and_save_scaler(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "df_train": df_train,
        "df_test": df_test,
        "scaler": scaler,
        "feature_names": FEATURE_COLUMNS,
    }


if __name__ == "__main__":
    data = get_scaled_datasets()
    print("Feature dimensions:", len(data["feature_names"]), data["feature_names"])
    print("Train shape:", data["X_train"].shape, "Test shape:", data["X_test"].shape)
