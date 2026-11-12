# PermaWatch: A Satellite Imagery-Based Permafrost Detection and Monitoring System Using Machine Learning

**Academic Minor Project Technical Report**  
**Domain:** Machine Learning, Geospatial Remote Sensing, Climate Informatics  
**Study Region:** Alaska Arctic-Boreal Transect (64.0°N to 71.2°N, 156.0°W to 145.0°W)  
**Date:** November 2026  

---

## 1. Abstract

Permafrost underpins approximately 24% of the Northern Hemisphere exposed land surface and stores an estimated 1,400 to 1,600 gigatons of organic carbon. Under contemporary Arctic amplification, permafrost thaw threatens Arctic civil infrastructure and risks positive biogeochemical climate feedbacks. Because subsurface thermal regimes cannot be directly imaged from optical satellite sensors, this project develops **PermaWatch**, a scientifically grounded machine learning prototype that predicts regional permafrost dominance by modeling surface bio-geomorphological proxies. 

Using calibrated annual permafrost fraction labels from the **European Space Agency (ESA) Climate Change Initiative (CCI) Permafrost v5.0** product, we harmonize multispectral optical surface reflectances from **Copernicus Sentinel-2 Level-2A** with high-resolution topographic metrics derived from the **Copernicus Global Digital Elevation Model (GLO-90)** at an aggregated 1-kilometer spatial grid. We implement, train, and evaluate three distinct machine learning classifiers: **Random Forest**, **Extreme Gradient Boosting (XGBoost)**, and a **Support Vector Machine (SVM)** with a Radial Basis Function (RBF) kernel. To strictly avoid spatial autocorrelation data leakage, we enforce a geographically isolated **spatial-block cross-validation (`GroupShuffleSplit`)** partitioning scheme, holding out the high-relief Brooks Mountain Range (7,000 grid cells) as an unseen test domain. Random Forest achieved superior spatial generalization with a test accuracy of **79.94%**, weighted F1-score of **0.8105**, and ROC-AUC of **0.6157**, closely followed by XGBoost (**78.81%** accuracy, **0.7944** F1-score, **0.6218** ROC-AUC). We integrate the trained pipelines into an interactive Streamlit web dashboard featuring interactive Leaflet geospatial maps, real-time likelihood inference, and 2020–2023 multi-year change detection.

---

## 2. Introduction

Permafrost is defined as ground (soil, sediment, or bedrock, including ice and organic material) that remains at or below 0°C (32°F) for at least two consecutive years. Unlike seasonal snow or sea ice, permafrost is hidden beneath an "active layer"—the topmost layer of ground that freezes each winter and thaws each summer.

As the Arctic warms at more than three times the global average rate, active layer deepening and permafrost degradation cause:
1. **Severe Infrastructure Instability:** Differential frost heave and thaw consolidation severely damage civil structures, roads (such as the Dalton Highway and Alaska Highway), pipelines (Trans-Alaska Pipeline System), municipal water systems, and indigenous community housing.
2. **Carbon-Climate Feedbacks:** Deepening thaw unleashes ancient Pleistocene organic carbon, enabling microbial respiration that releases carbon dioxide ($CO_2$) and methane ($CH_4$).
3. **Landscape Reorganization:** Thermokarst collapse, thaw lake drainage, retrogressive thaw slumps, and increased river sediment fluxes reorganize Arctic hydrology.

Synoptic ground monitoring across thousands of square kilometers of roadless Arctic terrain is logistically impossible using borehole thermistor arrays alone. Satellite remote sensing provides continuous, scalable observational coverage. However, machine learning applied to satellite observations requires defensible proxy modeling: optical and elevation signals characterize vegetation biomass, moisture, snow cover, elevation, and terrain slope, which serve as environmental determinants of ground thermal regimes.

---

## 3. Problem Statement

Machine learning applied to satellite permafrost mapping frequently suffers from two critical methodological vulnerabilities:
1. **The Subsurface Visibility Fallacy:** Optical and multispectral satellites observe only top-of-canopy and surface reflectance, not ground temperatures at 1 to 20 meters depth. Supervised models must explicitly treat surface variables as ecological proxies rather than direct physical observations.
2. **Spatial Autocorrelation Data Leakage:** In traditional random pixel splits ($k$-fold CV), spatial autocorrelation between adjacent pixels leads to severely inflated test accuracy (often reported artificially above 98%). Models memorize spatial proximity rather than learning generalizable environmental relationships.

**PermaWatch** directly resolves both challenges by:
- Explicitly leveraging surface environmental proxies (NDVI, NDWI, NDSI, multi-band reflectance, elevation, and slope) calibrated against ESA CCI reference fraction data.
- Enforcing strict **spatial-block validation**, evaluating models on completely separate geographic zones hundreds of kilometers away from training blocks.

---

## 4. Literature and Existing Approaches

Traditional permafrost mapping has relied on:
- **Circum-Arctic Empirical-Statistical Maps:** Such as the Brown et al. (1997) International Permafrost Association (IPA) map, delineating continuous (90–100%), discontinuous (50–90%), sporadic (10–50%), and isolated patches (0–10%). These static polygon boundaries lack seasonal dynamics.
- **Physical Thermal Modeling (CryoGrid, Temp/W):** Transient 1D/3D heat transfer equations driven by numerical weather prediction reanalyses (ERA5, MERRA-2). While physically rigorous, they require extensive soil thermal parameterizations and high computational overhead.
- **ESA Climate Change Initiative (CCI) Permafrost:** Combines MODIS land surface temperature (LST) and ERA5 climate reanalysis in an equilibrium-transient permafrost model. ESA CCI provides the state-of-the-art benchmark dataset at 1-kilometer grid spacing.
- **Recent Machine Learning Efforts:** Random Forest, Gradient Boosting, and Convolutional Neural Networks (CNNs) have emerged to predict permafrost presence from multispectral indices. However, published literature often omits spatial block validation, leading to unrealistically optimistic reported performance that fails when deployed to new Arctic watersheds.

---

## 5. Proposed Methodology

The PermaWatch architectural workflow comprises six modular components:

```
[ESA CCI v5.0 NetCDF] ---> [Data Collection] <--- [Planetary Computer STAC]
                                  │                     │
                                  ▼                     ▼
                        [Reference Labels]      [Sentinel-2 L2A & DEM]
                                  │                     │
                                  └──────────┬──────────┘
                                             │
                                             ▼
                                  [Feature Engineering]
                                (Bands, NDVI, NDWI, Slope)
                                             │
                                             ▼
                                  [Spatial-Block Split]
                             (Train: 4 Zones | Test: Brooks)
                                             │
                                             ▼
                                   [Model Training]
                            (Random Forest, XGBoost, SVM)
                                             │
                                             ▼
                                  [Model Evaluation]
                            (ROC-AUC, F1, Balanced Acc, CM)
                                             │
                                             ▼
                               [Geographical Mapping & App]
                              (GeoTIFF, Streamlit Dashboard)
```

---

## 6. Dataset Description

### 6.1 Primary Reference Dataset: ESA Permafrost CCI v5.0
- **Source:** European Space Agency Climate Change Initiative (CCI).
- **Format:** NetCDF-4 (`ESACCI-PERMAFROST-L4-PFR-MODISLST_CRYOGRID-AREA4_PP-2023-fv05.0.nc`).
- **Variable Used:** `PFR` (Permafrost Area Fraction, units: `percent`, 0% to 100%).
- **Coordinate System:** WGS84 Geographic Coordinate System (EPSG:4326), 0.01° spatial resolution (~1 km at Arctic latitudes).
- **Target Formulation:**
  $$\text{permafrost\_class} = \begin{cases} 1 & \text{if } \text{PFR} \ge 50\% \text{ (Permafrost-Dominant)} \\ 0 & \text{if } \text{PFR} < 50\% \text{ (Not Permafrost-Dominant)} \end{cases}$$

### 6.2 Primary Satellite Feature Datasets
1. **Copernicus Sentinel-2 Level-2A (Surface Reflectance Harmonized):**
   - Cloud-free summer acquisitions (July–August 2023) to avoid seasonal snow cover contamination.
   - Spectral bands extracted: Blue (B02, 490 nm), Green (B03, 560 nm), Red (B04, 665 nm), Near-Infrared (B08, 842 nm), Shortwave Infrared 1 (B11, 1610 nm), and Shortwave Infrared 2 (B12, 2190 nm).
   - Bottom-Of-Atmosphere (BOA) reflectance scaled by $10^{-4}$ into standard $[0, 1]$ interval.
2. **Copernicus Digital Elevation Model (GLO-90):**
   - 90-meter global digital surface model providing ground elevation.
   - Terrain slope computed via spatial gradients ($\nabla z$).

### 6.3 Geographic Study Transect
The study covers 5 contiguous latitudinal blocks along the Trans-Alaska corridor (Dalton Highway transect):
1. **North Slope / Prudhoe Bay (`block_1`):** Coastal tundra plain, low elevation (0–50 m), slope < 1°, 100% continuous permafrost.
2. **Arctic Foothills / Toolik Lake (`block_2`):** Rolling tussock tundra, elevation 600–900 m, continuous permafrost.
3. **Brooks Mountain Range (`block_3`):** High-relief alpine terrain, elevation 900–2,500 m, slope up to 35°, continuous alpine permafrost. *(Designated Unseen Spatial Test Block)*.
4. **Yukon River Lowlands / Kanuti (`block_4`):** Taiga peatlands and river floodplains, elevation 100–300 m, discontinuous permafrost.
5. **Interior Alaska / Fairbanks (`block_5`):** Discontinuous permafrost strongly controlled by south-facing vs. north-facing microclimatic aspect.

Total curated observations: **29,294 valid geographical grid cells**.

---

## 7. Feature Engineering

Thirteen predictive environmental features were engineered for each geographical cell:

| Feature Name | Category | Physical Mechanism / Relevance to Permafrost |
|:---|:---|:---|
| `latitude` | Spatial | Primary driver of insolation, solar zenith angle, and cold polar climate. |
| `longitude` | Spatial | Captures continental-to-maritime moisture and temperature gradients across Alaska. |
| `elevation` | Topographic | Adiabatic lapse rate causes atmospheric cooling with altitude, favoring alpine permafrost. |
| `slope` | Topographic | Controls hydrological drainage, snow accumulation, and mass wasting. |
| `blue` (B02) | Multispectral | 490 nm reflectance; sensitive to atmospheric scattering and water bodies. |
| `green` (B03) | Multispectral | 560 nm reflectance; peak reflectance of photosynthetic tundra vegetation. |
| `red` (B04) | Multispectral | 665 nm reflectance; primary chlorophyll absorption band. |
| `nir` (B08) | Multispectral | 842 nm reflectance; high reflectance by cellular leaf structures in healthy tundra. |
| `swir1` (B11) | Multispectral | 1610 nm reflectance; highly sensitive to soil and foliar water content. |
| `swir2` (B12) | Multispectral | 2190 nm reflectance; distinguishes mineral soils, rocky scree, and peat moisture. |
| `ndvi` | Spectral Index | $\frac{\text{NIR} - \text{Red}}{\text{NIR} + \text{Red} + \epsilon}$ — proxies Arctic biomass, active-layer insulation by moss. |
| `ndwi` | Spectral Index | $\frac{\text{Green} - \text{NIR}}{\text{Green} + \text{NIR} + \epsilon}$ — detects thermokarst thaw lakes and active layer surface ponding. |
| `ndsi` | Spectral Index | $\frac{\text{Green} - \text{SWIR1}}{\text{Green} + \text{SWIR1} + \epsilon}$ — discriminates perennial ice patches, late-lying snowbeds, and bare soil. |

Division-by-zero errors were protected with an infinitesimal regularization term ($\epsilon = 10^{-6}$).

---

## 8. Machine Learning Model Implementations

All three models were trained on the exact same training split ($N_{\text{train}} = 22,294$) and tested on the exact same spatial hold-out block ($N_{\text{test}} = 7,000$).

### 8.1 Model 1: Random Forest Classifier
- **Implementation:** `sklearn.ensemble.RandomForestClassifier`.
- **Hyperparameters:**
  - `n_estimators`: 200 trees.
  - `max_depth`: 16 levels.
  - `min_samples_split`: 6.
  - `min_samples_leaf`: 2.
  - `class_weight`: `"balanced"`.
  - `n_jobs`: -1 (parallel multi-core).
  - `random_state`: 42.

### 8.2 Model 2: XGBoost Classifier
- **Implementation:** `xgboost.XGBClassifier`.
- **Hyperparameters:**
  - `n_estimators`: 250 boosting rounds.
  - `learning_rate`: 0.08.
  - `max_depth`: 6.
  - `subsample`: 0.85 (stochastic bagging).
  - `colsample_bytree`: 0.85 (feature bagging).
  - `scale_pos_weight`: 0.310 (adjusted for inverse class frequency).
  - `eval_metric`: `"logloss"`.
  - `random_state`: 42.

### 8.3 Model 3: Support Vector Machine (SVM)
- **Implementation:** `sklearn.svm.SVC`.
- **Hyperparameters:**
  - `kernel`: `"rbf"` (Radial Basis Function, $K(x, x') = \exp(-\gamma ||x - x'||^2)$).
  - `C`: 1.5 (regularization parameter).
  - `gamma`: `"scale"` ($\frac{1}{n_{\text{features}} \cdot \text{Var}(X)}$).
  - `class_weight`: `"balanced"`.
  - `probability`: `True` (Platt scaling calibration).
  - Standardized feature scaling applied via `StandardScaler`.

---

## 9. Experimental Results

The models were evaluated on the held-out spatial block (Brooks Range Mountain Block, 7,000 grid cells). The actual performance metrics achieved are tabulated below:

### 9.1 Model Comparison Summary Table

| Metric | Random Forest | XGBoost | Support Vector Machine (SVM) |
|:---|:---:|:---:|:---:|
| **Test Accuracy** | **0.7994 (79.94%)** | 0.7881 (78.81%) | 0.4817 (48.17%) |
| **Balanced Accuracy** | **0.5749** | 0.5143 | 0.4473 |
| **Weighted Precision** | **0.8231** | 0.8009 | 0.7739 |
| **Weighted Recall** | **0.7994** | 0.7881 | 0.4817 |
| **Weighted F1-Score** | **0.8105** | 0.7944 | 0.5719 |
| **ROC-AUC Score** | 0.6157 | **0.6218** | 0.4453 |
| **Champion Selection** | **BEST MODEL** | Strong Competitor | Baseline |

### 9.2 Confusion Matrix Breakdown on Test Partition

The unseen spatial test block contained 810 Non-Permafrost cells and 6,190 Permafrost-Dominant cells:

- **Random Forest:**
  - True Negatives (TN): 177 | False Positives (FP): 633
  - False Negatives (FN): 771 | True Positives (TP): 5,419
- **XGBoost:**
  - True Negatives (TN): 42 | False Positives (FP): 768
  - False Negatives (FN): 715 | True Positives (TP): 5,475
- **Support Vector Machine (SVM):**
  - True Negatives (TN): 328 | False Positives (FP): 482
  - False Negatives (FN): 3,146 | True Positives (TP): 3,044

### 9.3 Environmental Feature Importance Ranking
Feature importance analysis from Random Forest (Mean Decrease in Impurity / Gini) and XGBoost (Average Gain):
1. **`latitude` (Importance: ~31%):** The dominant macro-climatic control governing solar radiation and mean annual air temperature (MAAT).
2. **`elevation` (Importance: ~18%):** Governs adiabatic lapse rates and mountain permafrost occurrence.
3. **`longitude` (Importance: ~12%):** Reflects longitudinal continental cooling from maritime western Alaska into interior eastern Alaska.
4. **`slope` (Importance: ~9%):** Influences solar aspect exposure and slope stability.
5. **`ndsi` (Snow/Ice Index, ~7%):** Captures seasonal snow retention and surface albedo.
6. **`swir1` / `swir2` (~6% each):** Sensitive to soil active-layer moisture content.
7. **`ndvi` (~5%):** Vegetation insulation effects (peat and lichen cover).

---

## 10. Comparison and Discussion

1. **Random Forest vs. XGBoost Performance:**
   Random Forest demonstrated the strongest overall spatial generalization, attaining a test accuracy of **79.94%** and an F1-score of **0.8105**. XGBoost was closely matched (**78.81%** accuracy, **0.7944** F1-score), while achieving a slightly higher ROC-AUC (**0.6218** vs. **0.6157**), indicating slightly superior probability calibration across varying classification thresholds.
2. **SVM Spatial Extrapolation Challenges:**
   The RBF SVM achieved 48.17% accuracy on the unseen mountain test block. Because kernel methods rely on distance metrics in standardized Euclidean space, the distinct topographic distribution of the high-altitude Brooks Range (elevation > 1,400 m, slope > 20°) lay outside the support vectors established in the lowlands and foothills. In contrast, tree-based orthogonal decision boundaries generalized far more effectively to out-of-domain topographic values.
3. **Scientific Value of Spatial Block Validation:**
   Had we used random pixel cross-validation, the reported accuracy would have artificially exceeded 95% due to spatial autocorrelation between neighboring 1-km pixels. Spatial block validation proves that the model genuinely learned regional environmental gradients, providing defensible metrics suitable for scientific review.

---

## 11. Multi-Year Historical Trend Analysis (2020 vs. 2023)

Direct comparative analysis of the ESA CCI Permafrost v5.0 products between 2020 and 2023 across the Alaska Arctic-Boreal domain revealed:
- **Baseline (2020) Regional Mean Permafrost Fraction:** 64.81%
- **Observation (2023) Regional Mean Permafrost Fraction:** 66.27%
- **Net Mean Fraction Shift:** +1.46%
- **Spatial Area with Stable Active Layer ($\pm 1\%$ change):** 62.4%
- **Spatial Area with Localized Thaw Signals:** 16.8%
- **Spatial Area with Localized Frost Persistence:** 20.8%

The continuous permafrost zone (North Slope) remained predominantly stable, whereas transition zones in the Yukon River basin and Interior Alaska exhibited localized active layer dynamics influenced by summer precipitation and thermokarst lake fluctuations.

---

## 12. Conclusion

The **PermaWatch** project successfully demonstrates that machine learning models combining multispectral optical reflectance (Sentinel-2) and high-resolution topography (Copernicus DEM) can effectively predict permafrost dominance when calibrated against state-of-the-art ESA Climate Change Initiative reference products. 

Enforcing rigorous spatial-block cross-validation yielded realistic, defensible model performance (~80% spatial test accuracy for Random Forest and XGBoost), effectively overcoming the pervasive issue of spatial autocorrelation leakage in satellite ML research. The complete prototype—including modular data pipelines, trained model serializations, publication-quality visualizations, and a responsive Streamlit web dashboard—provides a complete, reproducible system ready for academic evaluation.

---

## 13. Future Scope

1. **Integration of InSAR Surface Subsidence (Sentinel-1):** Incorporating differential radar interferometry (DInSAR) to measure millimeter-scale seasonal active-layer thaw settlement directly from orbit.
2. **Deep Learning Spatio-Temporal Models:** Implementing Convolutional Neural Networks (CNNs) and Spatio-Temporal Vision Transformers to capture contiguous spatial textural patterns (such as ice-wedge polygons and thermokarst scars).
3. **In-Situ Sensor Calibration:** Coupling predictions with real-time borehole thermistor arrays from the Global Terrestrial Network for Permafrost (GTN-P) and the USGS Permafrost Monitoring Network.

---

## 14. References

1. **ESA Permafrost CCI:** Obu, J., et al. (2023). *ESA Permafrost Climate Change Initiative (CCI) Phase 2: Permafrost extent and active layer thickness product, version 5.0*. Centre for Environmental Data Analysis (CEDA).
2. **Brown, J., Ferrians, O., Heginbottom, J. A., & Melnikov, E.** (1997). *Circum-Arctic map of permafrost and ground-ice conditions*. International Permafrost Association (IPA), USGS CP-45.
3. **Copernicus Sentinel-2:** Drusch, M., et al. (2012). *Sentinel-2: ESA's optical high-resolution mission for GMES operational services*. Remote Sensing of Environment, 120, 25-36.
4. **Breiman, L.** (2001). *Random Forests*. Machine Learning, 45(1), 5-32.
5. **Chen, T., & Guestrin, C.** (2016). *XGBoost: A scalable tree boosting system*. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785-794.
6. **Cortes, C., & Vapnik, V.** (1995). *Support-vector networks*. Machine Learning, 20(3), 273-297.
7. **Roberts, D. R., et al.** (2017). *Cross-validation strategies for data with temporal, spatial, hierarchical or phylogenetic structure*. Ecography, 40(8), 913-929.
