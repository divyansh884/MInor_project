# PermaWatch: A Satellite Imagery-Based Permafrost Detection and Monitoring System Using Machine Learning

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-1.67.0-red.svg)](https://streamlit.io/)
[![ESA CCI v5.0](https://img.shields.io/badge/Dataset-ESA%20CCI%20Permafrost-green.svg)](https://climate.esa.int/en/catalogue/d235665772ec4b558e9a89ac85595e71/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

PermaWatch is a machine learning and geospatial remote sensing system designed to detect and monitor permafrost-dominant geographical areas across Arctic-boreal landscapes using satellite-derived surface environmental proxies and high-resolution digital topography.

---

## 📌 Project Highlights & Key Features

- **Genuine Public Satellite Datasets:**
  - Official **ESA Permafrost Climate Change Initiative (CCI) v5.0** 1-km gridded NetCDF reference product (both 2023 and 2020 multi-year baselines).
  - Harmonized multispectral surface reflectance from **Copernicus Sentinel-2 Level-2A** (Bands: Blue, Green, Red, NIR, SWIR-1, SWIR-2).
  - Topographic elevation and terrain slope derived from **Copernicus Global DEM (GLO-90)** via Microsoft Planetary Computer STAC.
- **Three Compulsory Machine Learning Classifiers:**
  - **Random Forest Classifier** (`sklearn.ensemble.RandomForestClassifier` with balanced class weights, 200 estimators).
  - **XGBoost Classifier** (`xgboost.XGBClassifier` with scale_pos_weight class compensation, 250 estimators).
  - **Support Vector Machine (SVM)** (`sklearn.svm.SVC` with RBF kernel and Platt probability calibration).
- **Strict Spatial-Block Validation (No Spatial Autocorrelation Leakage):**
  - Enforces `GroupShuffleSplit` holding out an entire geographical region (the high-relief Brooks Mountain Range, 7,000 cells) as an out-of-domain test split.
- **Modern Interactive Web Dashboard:**
  - Multi-page Streamlit application with interactive Folium Leaflet maps, Plotly charts, real-time likelihood inference gauge, multi-year change detection, and project documentation.

---

## 📊 Model Evaluation Results (Spatial Block Cross-Validation)

Evaluated on the unseen **Brooks Mountain Range hold-out spatial block** (7,000 grid cells):

| Model | Accuracy | Balanced Accuracy | Weighted Precision | Weighted Recall | Weighted F1-Score | ROC-AUC | Status |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest** | **79.94%** | **0.5749** | **0.8231** | **0.7994** | **0.8105** | 0.6157 | 🏆 **Best Model** |
| **XGBoost** | **78.81%** | 0.5143 | 0.8009 | 0.7881 | 0.7944 | **0.6218** | Strong Competitor |
| **SVM (RBF Kernel)** | 48.17% | 0.4473 | 0.7739 | 0.4817 | 0.5719 | 0.4453 | Baseline Kernel |

*Note: Models evaluated under rigorous spatial block validation do not suffer from inflated accuracy caused by neighboring pixel memorization.*

---

## 📁 Repository Structure

```
PermaWatch/
│
├── app.py                     # Streamlit multi-page web application
├── requirements.txt           # Project Python dependencies
├── README.md                  # Comprehensive documentation
│
├── data/
│   ├── raw/                   # Downloaded ESA NetCDF files & raw GeoTIFF tiles
│   │   ├── ESACCI-PERMAFROST-2023.nc
│   │   ├── ESACCI-PERMAFROST-2020.nc
│   │   ├── dem/               # Cached Copernicus DEM tiles
│   │   └── satellite/         # Cached 1km Sentinel-2 decimated band rasters
│   ├── processed/             # Cleaned feature-engineered dataset
│   │   ├── permafrost_training_dataset.parquet
│   │   └── permafrost_training_dataset.csv
│   └── samples/               # Quick inspection sample
│       └── sample_500.csv
│
├── notebooks/
│   └── exploratory_analysis.ipynb   # Jupyter Notebook for EDA & model workflow
│
├── src/
│   ├── data_collection.py     # Automatic download & GEE/STAC diagnostics
│   ├── feature_extraction.py  # Satellite band decimation & DEM slope processing
│   ├── preprocessing.py       # GroupShuffleSplit spatial partition & scaling
│   ├── train_random_forest.py # Random Forest model training & serialization
│   ├── train_xgboost.py       # XGBoost model training & serialization
│   ├── train_svm.py           # Support Vector Machine training & calibration
│   ├── evaluate.py            # Unified evaluation, metrics & publication plots
│   └── predict.py             # Inference engine, GeoTIFF raster & regional maps
│
├── models/                    # Serialized machine learning models
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── svm.pkl
│   └── scaler.pkl
│
├── outputs/
│   ├── metrics/               # Summary CSV, JSON and classification reports
│   │   ├── model_comparison.csv
│   │   ├── metrics.json
│   │   ├── classification_reports.txt
│   │   └── historical_trends.json
│   ├── plots/                 # High-resolution comparison plots
│   │   ├── confusion_matrices.png
│   │   ├── roc_curves.png
│   │   ├── feature_importance.png
│   │   ├── correlation_matrix.png
│   │   └── historical_change_map.png
│   ├── maps/                  # Geographical prediction maps & rasters
│   │   ├── permafrost_probability_map.png
│   │   ├── permafrost_classification_map.png
│   │   ├── permafrost_probability.tif
│   │   └── regional_statistics.json
│   └── predictions/           # Regional prediction CSV grid
│       └── regional_predictions_grid.csv
│
└── reports/
    └── project_report.md      # Full 14-section academic project report
```

---

## 🚀 Quickstart & Execution Guide

### 1. Environment Setup & Dependency Installation
PermaWatch runs cleanly on Python 3.10 or 3.11.

```powershell
# Clone or navigate to the project directory
cd c:\Users\91702\Desktop\cu9\PermaWatch

# Activate Python 3.11 environment (if using the workspace virtualenv)
..\.venv\Scripts\Activate.ps1

# Install required dependencies
pip install -r requirements.txt
```

### 2. Download Data & Inspect ESA NetCDF
```powershell
python src/data_collection.py
```

### 3. Extract Features & Assemble Dataset
```powershell
python src/feature_extraction.py
```

### 4. Train All Three Models & Run Evaluation
```powershell
python src/evaluate.py
```

### 5. Generate Geographical Prediction Maps & GeoTIFF Raster
```powershell
python src/predict.py
```

### 6. Launch the Interactive Web Dashboard
```powershell
streamlit run app.py
```
Open your web browser and navigate to: **`http://localhost:8501`**

---

## 🛰️ Datasets & Environmental Features

### A. Primary Ground Reference: ESA Permafrost CCI v5.0
- **URL:** [ESA Climate Change Initiative Permafrost Catalogue](https://climate.esa.int/en/catalogue/d235665772ec4b558e9a89ac85595e71/)
- **Direct 2023 NetCDF:** `https://data.cci.ceda.ac.uk/thredds/fileServer/esacci/permafrost/data/permafrost_extent/L4/area4/pp/v05.0/northern_hemisphere/2023/ESACCI-PERMAFROST-L4-PFR-MODISLST_CRYOGRID-AREA4_PP-2023-fv05.0.nc`
- **Direct 2020 NetCDF:** `https://data.cci.ceda.ac.uk/thredds/fileServer/esacci/permafrost/data/permafrost_extent/L4/area4/pp/v05.0/northern_hemisphere/2020/ESACCI-PERMAFROST-L4-PFR-MODISLST_CRYOGRID-AREA4_PP-2020-fv05.0.nc`

### B. Extracted Environmental Features
1. `latitude`: Macro-climatic driver of polar solar radiation and temperature.
2. `longitude`: Continental-to-maritime moisture and climate continentality.
3. `elevation`: Adiabatic lapse rate cooling from Copernicus DEM GLO-90.
4. `slope`: Terrain steepness (degrees) controlling hydrological drainage and stability.
5. `blue` (B02): 490 nm surface reflectance.
6. `green` (B03): 560 nm surface reflectance (photosynthetic peak).
7. `red` (B04): 665 nm surface reflectance (chlorophyll absorption).
8. `nir` (B08): 842 nm surface reflectance (cellular tundra vegetation reflectance).
9. `swir1` (B11): 1610 nm surface reflectance (soil and active layer moisture).
10. `swir2` (B12): 2190 nm surface reflectance (mineral soil and peat discrimination).
11. `ndvi`: Normalized Difference Vegetation Index: $\frac{\text{NIR} - \text{Red}}{\text{NIR} + \text{Red} + \epsilon}$
12. `ndwi`: Normalized Difference Water Index: $\frac{\text{Green} - \text{NIR}}{\text{Green} + \text{NIR} + \epsilon}$
13. `ndsi`: Normalized Difference Snow Index: $\frac{\text{Green} - \text{SWIR1}}{\text{Green} + \text{SWIR1} + \epsilon}$

---

## 🔬 Google Earth Engine Diagnostic Note

If Google Earth Engine authentication is desired, run:
```powershell
earthengine authenticate
```
Follow the web browser link, select your Google Cloud project, and authorize access. If GEE credentials are not configured, PermaWatch automatically connects to the public **Microsoft Planetary Computer STAC** to retrieve open Sentinel-2 L2A and Copernicus DEM data without requiring an API key.

---

## ⚠️ Scientific Disclaimers & Limitations

1. **Surface Proxy Principle:** Satellite optical sensors cannot physically image permanently frozen ground 2 meters underground. PermaWatch models surface ecological, hydrological, and topographic proxies that indicate permafrost presence.
2. **Cloud & Seasonal Restrictions:** Optical remote sensing in the Arctic is restricted to snow-free summer months (July–August) to prevent albedo saturation from seasonal snow cover.
3. **Historical Thaw Interpretation:** Multi-year changes (2020 vs 2023) reflect modeled ground thermal equilibrium and active-layer conditions and should not be casually reported as deep borehole thaw without ground-truth verification.

---

## 📜 Citation & References

- **ESA CCI Permafrost:** Obu, J., et al. (2023). *ESA Permafrost Climate Change Initiative (CCI) Phase 2: Permafrost extent and active layer thickness product, version 5.0*. CEDA.
- **Copernicus Sentinel-2:** Drusch, M., et al. (2012). *Sentinel-2: ESA's optical high-resolution mission for GMES operational services*. Remote Sensing of Environment.
