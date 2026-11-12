"""
PermaWatch - Unified Model Evaluation & Comparison Module
Evaluates Random Forest, XGBoost, and SVM on the identical spatial test partition.
Calculates Accuracy, Precision, Recall, F1-Score, Balanced Accuracy, and ROC-AUC.
Generates publication-quality comparison tables, ROC curves, confusion matrices,
and feature importance visualizations.
"""

import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    classification_report,
)
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocessing import get_scaled_datasets, FEATURE_COLUMNS
from src.train_random_forest import train_random_forest
from src.train_xgboost import train_xgboost
from src.train_svm import train_svm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EvaluateModels")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
METRICS_DIR = os.path.join(OUTPUTS_DIR, "metrics")
PLOTS_DIR = os.path.join(OUTPUTS_DIR, "plots")

os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Set clean scientific plotting style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10


def load_or_train_models(datasets):
    """Ensure all three models exist, training any that are missing."""
    models = {}
    rf_path = os.path.join(MODELS_DIR, "random_forest.pkl")
    xgb_path = os.path.join(MODELS_DIR, "xgboost.pkl")
    svm_path = os.path.join(MODELS_DIR, "svm.pkl")

    if os.path.exists(rf_path):
        logger.info(f"Loading cached Random Forest from {rf_path}")
        models["Random Forest"] = joblib.load(rf_path)
    else:
        logger.info("Training Random Forest...")
        models["Random Forest"] = train_random_forest(datasets)

    if os.path.exists(xgb_path):
        logger.info(f"Loading cached XGBoost from {xgb_path}")
        models["XGBoost"] = joblib.load(xgb_path)
    else:
        logger.info("Training XGBoost...")
        models["XGBoost"] = train_xgboost(datasets)

    if os.path.exists(svm_path):
        logger.info(f"Loading cached SVM from {svm_path}")
        models["SVM"] = joblib.load(svm_path)
    else:
        logger.info("Training Support Vector Machine...")
        models["SVM"] = train_svm(datasets)

    return models


def evaluate_all_models():
    """
    Evaluates all three models on the same spatial-block test split.
    Generates comparison tables, plots, and saves all outputs.
    """
    logger.info("Starting comprehensive model evaluation...")
    datasets = get_scaled_datasets()
    models = load_or_train_models(datasets)

    X_test_raw = datasets["X_test"]
    X_test_scaled = datasets["X_test_scaled"]
    y_test = datasets["y_test"]

    metrics_records = []
    roc_data = {}
    cm_dict = {}
    reports_text = []

    for name, model in models.items():
        logger.info(f"Evaluating {name}...")
        # SVM requires standardized inputs; tree models use raw features
        X_eval = X_test_scaled if name == "SVM" else X_test_raw

        y_pred = model.predict(X_eval)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_eval)[:, 1]
        else:
            y_prob = y_pred.astype(float)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        try:
            auc = roc_auc_score(y_test, y_prob)
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_data[name] = (fpr, tpr, auc)
        except Exception as e:
            auc = np.nan
            logger.warning(f"Could not compute ROC-AUC for {name}: {e}")

        cm = confusion_matrix(y_test, y_pred)
        cm_dict[name] = cm

        rep = classification_report(y_test, y_pred, target_names=["Non-Permafrost", "Permafrost-Dominant"])
        reports_text.append(f"==================================================\nMODEL: {name}\n==================================================\n{rep}\n")

        metrics_records.append({
            "Model": name,
            "Accuracy": np.round(acc, 4),
            "Balanced Accuracy": np.round(bal_acc, 4),
            "Precision": np.round(prec, 4),
            "Recall": np.round(rec, 4),
            "F1-Score": np.round(f1, 4),
            "ROC-AUC": np.round(auc, 4) if not np.isnan(auc) else None,
        })

    # Create Comparison DataFrame
    df_metrics = pd.DataFrame(metrics_records)
    # Determine best model based on weighted rank of F1-Score and ROC-AUC
    df_metrics["Rank_Score"] = df_metrics["F1-Score"] + df_metrics["ROC-AUC"].fillna(0)
    best_row = df_metrics.loc[df_metrics["Rank_Score"].idxmax()]
    best_model_name = best_row["Model"]
    df_metrics = df_metrics.drop(columns=["Rank_Score"])

    logger.info(f"Model Evaluation Summary:\n{df_metrics.to_string(index=False)}")
    logger.info(f"Best Performing Model: {best_model_name} (F1: {best_row['F1-Score']}, ROC-AUC: {best_row['ROC-AUC']})")

    # Save metrics table to CSV and JSON
    csv_metrics_path = os.path.join(METRICS_DIR, "model_comparison.csv")
    json_metrics_path = os.path.join(METRICS_DIR, "metrics.json")
    reports_path = os.path.join(METRICS_DIR, "classification_reports.txt")

    df_metrics.to_csv(csv_metrics_path, index=False)
    with open(json_metrics_path, "w") as f:
        json.dump({
            "models": df_metrics.to_dict(orient="records"),
            "best_model": best_model_name,
            "best_metrics": best_row.to_dict(),
            "test_samples": len(y_test),
            "features_used": FEATURE_COLUMNS,
        }, f, indent=2)

    with open(reports_path, "w") as f:
        f.write("\n".join(reports_text))

    # Plot 1: Side-by-side Confusion Matrices
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300)
    for ax, (m_name, cm) in zip(axes, cm_dict.items()):
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues" if m_name != best_model_name else "Greens",
            cbar=False,
            ax=ax,
            xticklabels=["Non-Permafrost", "Permafrost"],
            yticklabels=["Non-Permafrost", "Permafrost"],
            annot_kws={"size": 11, "weight": "bold"}
        )
        title_suffix = " (Best Model)" if m_name == best_model_name else ""
        ax.set_title(f"{m_name}{title_suffix}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("True Ground Reference", fontsize=10)
    plt.tight_layout()
    cm_plot_path = os.path.join(PLOTS_DIR, "confusion_matrices.png")
    plt.savefig(cm_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved Confusion Matrices plot: {cm_plot_path}")

    # Plot 2: ROC Curves Comparison
    plt.figure(figsize=(8, 6), dpi=300)
    colors = {"Random Forest": "#2b5c8f", "XGBoost": "#107c41", "SVM": "#d83b01"}
    for m_name, (fpr, tpr, auc_val) in roc_data.items():
        lw = 2.5 if m_name == best_model_name else 1.8
        ls = "-" if m_name == best_model_name else "--"
        label_text = f"{m_name} (AUC = {auc_val:.3f})"
        if m_name == best_model_name:
            label_text += " [BEST]"
        plt.plot(fpr, tpr, label=label_text, color=colors.get(m_name, "black"), lw=lw, linestyle=ls)
    plt.plot([0, 1], [0, 1], "k:", lw=1.2, label="Random Chance (AUC = 0.500)")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    plt.title("Receiver Operating Characteristic (ROC) Comparison\nSpatial Block Cross-Validation (Alaska Transect)", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right", frameon=True, fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    roc_plot_path = os.path.join(PLOTS_DIR, "roc_curves.png")
    plt.savefig(roc_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved ROC Curves plot: {roc_plot_path}")

    # Plot 3: Feature Importances for Tree Models (Random Forest & XGBoost)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Random Forest Importance (MDI)
    rf_imp = pd.Series(models["Random Forest"].feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=True)
    rf_imp.plot(kind="barh", ax=ax1, color="#2b5c8f", edgecolor="none")
    ax1.set_title("Random Forest - Gini Feature Importance", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Relative Importance Score", fontsize=10)

    # XGBoost Importance (Gain)
    xgb_imp = pd.Series(models["XGBoost"].feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=True)
    xgb_imp.plot(kind="barh", ax=ax2, color="#107c41", edgecolor="none")
    ax2.set_title("XGBoost - Feature Importance (Gain)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Relative Importance Score", fontsize=10)

    plt.suptitle("Environmental Feature Importance in Permafrost Detection", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fi_plot_path = os.path.join(PLOTS_DIR, "feature_importance.png")
    plt.savefig(fi_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved Feature Importance plot: {fi_plot_path}")

    # Plot 4: Correlation Matrix of Environmental Predictors
    plt.figure(figsize=(10, 8), dpi=300)
    df_raw = datasets["df_train"][FEATURE_COLUMNS]
    corr = df_raw.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8},
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )
    plt.title("Correlation Matrix of Geospatial & Satellite Predictors", fontsize=12, fontweight="bold")
    corr_plot_path = os.path.join(PLOTS_DIR, "correlation_matrix.png")
    plt.savefig(corr_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved Correlation Matrix plot: {corr_plot_path}")

    logger.info("Evaluation Complete. All artifacts saved successfully.")
    return df_metrics, best_model_name


if __name__ == "__main__":
    evaluate_all_models()
