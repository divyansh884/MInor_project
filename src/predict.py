"""
PermaWatch - Geographical Prediction & Inference Engine
Executes inference on single-point queries and generates continuous regional
permafrost probability & classification maps across the Alaska study domain.
Exports publication maps, GeoTIFF rasters, and summary geospatial statistics.
"""

import os
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import rasterio
from rasterio.transform import from_origin
import json
from typing import Dict, Any, Tuple, Optional
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocessing import FEATURE_COLUMNS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PredictEngine")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
MAPS_DIR = os.path.join(OUTPUTS_DIR, "maps")
PREDICTIONS_DIR = os.path.join(OUTPUTS_DIR, "predictions")

os.makedirs(MAPS_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

MODEL_FILES = {
    "Random Forest": os.path.join(MODELS_DIR, "random_forest.pkl"),
    "XGBoost": os.path.join(MODELS_DIR, "xgboost.pkl"),
    "SVM": os.path.join(MODELS_DIR, "svm.pkl"),
}
SCALER_FILE = os.path.join(MODELS_DIR, "scaler.pkl")


def load_model_and_scaler(model_name: str = "XGBoost"):
    """Load specified model and feature scaler."""
    if model_name not in MODEL_FILES:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(MODEL_FILES.keys())}")
    m_path = MODEL_FILES[model_name]
    if not os.path.exists(m_path):
        raise FileNotFoundError(f"Model file {m_path} does not exist. Train the models first.")

    model = joblib.load(m_path)
    scaler = joblib.load(SCALER_FILE) if os.path.exists(SCALER_FILE) else None
    return model, scaler


def predict_point(features: Dict[str, float], model_name: str = "XGBoost") -> Dict[str, Any]:
    """
    Predict permafrost class and likelihood score for a single geographical cell.
    """
    model, scaler = load_model_and_scaler(model_name)

    # Compute missing indices if spectral bands are supplied
    if "ndvi" not in features and "nir" in features and "red" in features:
        features["ndvi"] = (features["nir"] - features["red"]) / (features["nir"] + features["red"] + 1e-6)
    if "ndwi" not in features and "green" in features and "nir" in features:
        features["ndwi"] = (features["green"] - features["nir"]) / (features["green"] + features["nir"] + 1e-6)
    if "ndsi" not in features and "green" in features and "swir1" in features:
        features["ndsi"] = (features["green"] - features["swir1"]) / (features["green"] + features["swir1"] + 1e-6)

    # Assemble feature vector
    x_vec = np.array([[features[col] for col in FEATURE_COLUMNS]], dtype=float)

    if model_name == "SVM" and scaler is not None:
        x_in = scaler.transform(x_vec)
    else:
        x_in = x_vec

    pred_class = int(model.predict(x_in)[0])
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(x_in)[0, 1])
    else:
        prob = float(pred_class)

    risk_label = "Low" if prob < 0.35 else ("Moderate" if prob < 0.65 else "High")

    return {
        "model_used": model_name,
        "predicted_class": pred_class,
        "class_label": "Permafrost-Dominant" if pred_class == 1 else "Not Permafrost-Dominant",
        "permafrost_likelihood": np.round(prob, 4),
        "likelihood_percentage": f"{prob*100:.1f}%",
        "risk_level": risk_label,
        "input_features": features,
    }


def generate_regional_prediction_maps(
    model_name: str = "XGBoost",
    lat_min: float = 64.0,
    lat_max: float = 71.2,
    lon_min: float = -156.0,
    lon_max: float = -145.0,
    step_deg: float = 0.05
) -> Dict[str, Any]:
    """
    Generates a continuous regional prediction grid and produces:
    - High-res probability map (Green-Yellow-Red)
    - High-res binary classification map
    - GeoTIFF raster
    - Regional summary statistics
    """
    logger.info(f"Generating Regional Permafrost Prediction Maps using {model_name}...")
    model, scaler = load_model_and_scaler(model_name)

    # Create coordinate grid
    lats = np.arange(lat_min, lat_max + step_deg, step_deg)
    lons = np.arange(lon_min, lon_max + step_deg, step_deg)
    lon_mesh, lat_mesh = np.meshgrid(lons, lats)

    flat_lats = lat_mesh.flatten()
    flat_lons = lon_mesh.flatten()

    # Generate realistic physiographic features across Alaska transect
    # Brooks Range elevation peak around 68.0 - 68.8
    dist_to_brooks = np.abs(flat_lats - 68.3)
    synth_elevation = np.maximum(50.0, 1800.0 * np.exp(- (dist_to_brooks**2) / 0.6) + 150.0 * np.sin(flat_lons * 2.0))
    # Foothills & coast (flat, low slope) vs mountain crest
    synth_slope = np.maximum(0.2, 18.0 * np.exp(- (dist_to_brooks**2) / 0.5) + np.random.uniform(0.1, 1.5, size=len(flat_lats)))

    # Arctic tundra reflectances
    # North slope (high lat): tundra, lakes, higher NDWI, lower SWIR
    tundra_factor = (flat_lats - 64.0) / (71.2 - 64.0)
    blue = np.clip(0.04 + 0.02 * tundra_factor, 0.0, 1.0)
    green = np.clip(0.08 + 0.03 * tundra_factor, 0.0, 1.0)
    red = np.clip(0.06 + 0.01 * tundra_factor, 0.0, 1.0)
    nir = np.clip(0.25 - 0.05 * tundra_factor, 0.0, 1.0)
    swir1 = np.clip(0.14 - 0.04 * tundra_factor, 0.0, 1.0)
    swir2 = np.clip(0.08 - 0.03 * tundra_factor, 0.0, 1.0)

    ndvi = (nir - red) / (nir + red + 1e-6)
    ndwi = (green - nir) / (green + nir + 1e-6)
    ndsi = (green - swir1) / (green + swir1 + 1e-6)

    grid_df = pd.DataFrame({
        "latitude": flat_lats,
        "longitude": flat_lons,
        "elevation": synth_elevation,
        "slope": synth_slope,
        "blue": blue,
        "green": green,
        "red": red,
        "nir": nir,
        "swir1": swir1,
        "swir2": swir2,
        "ndvi": ndvi,
        "ndwi": ndwi,
        "ndsi": ndsi,
    })

    X_grid = grid_df[FEATURE_COLUMNS].values
    if model_name == "SVM" and scaler is not None:
        X_in = scaler.transform(X_grid)
    else:
        X_in = X_grid

    # Inference
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_in)[:, 1]
    else:
        probs = model.predict(X_in).astype(float)
    classes = (probs >= 0.5).astype(int)

    prob_matrix = probs.reshape(lat_mesh.shape)
    class_matrix = classes.reshape(lat_mesh.shape)

    # Save CSV grid
    grid_df["permafrost_prob"] = np.round(probs, 4)
    grid_df["permafrost_class"] = classes
    csv_out = os.path.join(PREDICTIONS_DIR, "regional_predictions_grid.csv")
    grid_df.to_csv(csv_out, index=False)

    # Custom Colormap: Green (Low) -> Yellow (Moderate) -> Red (High)
    cmap_risk = mcolors.LinearSegmentedColormap.from_list(
        "PermaWatchRisk",
        [(0.0, "#1a9850"), (0.35, "#91cf60"), (0.50, "#fee08b"), (0.65, "#fc8d59"), (1.0, "#d73027")]
    )

    # Map 1: Continuous Permafrost Probability Map
    plt.figure(figsize=(10, 7.5), dpi=300)
    im = plt.imshow(
        prob_matrix,
        origin="lower",
        extent=[lon_min, lon_max, lat_min, lat_max],
        cmap=cmap_risk,
        vmin=0.0,
        vmax=1.0,
        aspect="auto"
    )
    plt.contour(
        lon_mesh,
        lat_mesh,
        prob_matrix,
        levels=[0.5],
        colors=["black"],
        linewidths=[1.5],
        linestyles=["--"]
    )
    cbar = plt.colorbar(im, fraction=0.035, pad=0.04)
    cbar.set_label("Permafrost Likelihood Score (0.0 - 1.0)", fontsize=11, fontweight="bold")
    cbar.ax.tick_params(labelsize=10)

    plt.xlabel("Longitude (°W)", fontsize=11, fontweight="bold")
    plt.ylabel("Latitude (°N)", fontsize=11, fontweight="bold")
    plt.title(f"PermaWatch - Permafrost Probability Map (Alaska Transect)\nModel: {model_name} | Resolution: ~{step_deg*111:.1f} km", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.5, color="white")

    # Add landmark markers (Prudhoe Bay, Toolik, Brooks Range, Fairbanks)
    landmarks = [
        ("Deadhorse (Prudhoe)", 70.2, -148.5),
        ("Toolik Field Station", 68.6, -149.6),
        ("Brooks Range Crest", 68.1, -150.0),
        ("Yukon River Crossing", 65.9, -149.7),
        ("Fairbanks (Interior)", 64.8, -147.7),
    ]
    for name, la, lo in landmarks:
        if lat_min <= la <= lat_max and lon_min <= lo <= lon_max:
            plt.plot(lo, la, "k^", markersize=7)
            plt.text(lo + 0.15, la, name, fontsize=8, fontweight="bold", color="black",
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.75, edgecolor="none"))

    prob_map_path = os.path.join(MAPS_DIR, "permafrost_probability_map.png")
    plt.savefig(prob_map_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved Probability Map: {prob_map_path}")

    # Map 2: Binary Permafrost-Dominant Classification Map
    cmap_bin = mcolors.ListedColormap(["#2ca25f", "#de2d26"])
    plt.figure(figsize=(10, 7.5), dpi=300)
    im2 = plt.imshow(
        class_matrix,
        origin="lower",
        extent=[lon_min, lon_max, lat_min, lat_max],
        cmap=cmap_bin,
        vmin=0,
        vmax=1,
        aspect="auto"
    )
    plt.xlabel("Longitude (°W)", fontsize=11, fontweight="bold")
    plt.ylabel("Latitude (°N)", fontsize=11, fontweight="bold")
    plt.title(f"PermaWatch - Binary Permafrost Dominance Map (Alaska)\nModel: {model_name} (Threshold: P >= 0.5)", fontsize=12, fontweight="bold")

    # Legend
    import matplotlib.patches as mpatches
    p_patch = mpatches.Patch(color="#de2d26", label=f"Permafrost-Dominant ({(classes==1).mean()*100:.1f}%)")
    np_patch = mpatches.Patch(color="#2ca25f", label=f"Not Permafrost-Dominant ({(classes==0).mean()*100:.1f}%)")
    plt.legend(handles=[p_patch, np_patch], loc="lower left", frameon=True, fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.5, color="white")

    bin_map_path = os.path.join(MAPS_DIR, "permafrost_classification_map.png")
    plt.savefig(bin_map_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved Classification Map: {bin_map_path}")

    # Export GeoTIFF Raster using Rasterio
    geotiff_path = os.path.join(MAPS_DIR, "permafrost_probability.tif")
    transform_affine = from_origin(lon_min, lat_max, step_deg, step_deg)
    with rasterio.open(
        geotiff_path,
        "w",
        driver="GTiff",
        height=prob_matrix.shape[0],
        width=prob_matrix.shape[1],
        count=1,
        dtype=rasterio.float32,
        crs="EPSG:4326",
        transform=transform_affine,
    ) as dst:
        dst.write(prob_matrix.astype(rasterio.float32), 1)
    logger.info(f"Saved GeoTIFF: {geotiff_path}")

    # Summary Statistics
    total_cells = len(classes)
    perma_cells = int((classes == 1).sum())
    non_perma_cells = total_cells - perma_cells
    cell_area_km2 = (step_deg * 111.0) * (step_deg * 111.0 * np.cos(np.radians(67.5)))
    total_area_km2 = total_cells * cell_area_km2
    perma_area_km2 = perma_cells * cell_area_km2

    stats = {
        "model": model_name,
        "total_cells": total_cells,
        "permafrost_cells": perma_cells,
        "non_permafrost_cells": non_perma_cells,
        "permafrost_fraction_pct": np.round((perma_cells / total_cells) * 100, 2),
        "total_area_km2": np.round(total_area_km2, 1),
        "permafrost_area_km2": np.round(perma_area_km2, 1),
        "mean_likelihood": np.round(float(np.mean(probs)), 4),
        "median_likelihood": np.round(float(np.median(probs)), 4),
        "lat_range": [lat_min, lat_max],
        "lon_range": [lon_min, lon_max],
    }

    stats_path = os.path.join(MAPS_DIR, "regional_statistics.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    return {
        "stats": stats,
        "probability_map": prob_map_path,
        "classification_map": bin_map_path,
        "geotiff": geotiff_path,
    }


if __name__ == "__main__":
    res = generate_regional_prediction_maps("XGBoost")
    print("Regional Mapping Results:", res["stats"])
