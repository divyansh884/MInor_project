"""
PermaWatch - Feature Extraction Pipeline
Retrieves real Sentinel-2 Level-2A surface reflectance bands and Copernicus DEM (GLO-90)
elevation & terrain slope across the Alaska permafrost transect (64.5°N to 71.2°N).
Aggregates features to ~1 km resolution to match the ESA CCI Permafrost v5.0 grid.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import xarray as xr
import rasterio
from rasterio.warp import transform
from rasterio.enums import Resampling
import pystac_client
import planetary_computer as pc
import requests
from typing import List, Dict, Any, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("FeatureExtraction")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
SAT_DIR = os.path.join(RAW_DIR, "satellite")
DEM_DIR = os.path.join(RAW_DIR, "dem")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
SAMPLES_DIR = os.path.join(BASE_DIR, "data", "samples")

os.makedirs(SAT_DIR, exist_ok=True)
os.makedirs(DEM_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)

# Alaska Trans-Corridor study transects from North to South (64.5°N - 71.2°N)
TRANSECT_ZONES = [
    {
        "zone_name": "North_Slope_Prudhoe",
        "block_id": "block_1_north_slope",
        "bbox": [-151.0, 70.0, -149.0, 71.0],
        "s2_id": "S2A_MSIL2A_20230820T222541_R015_T06WVD_20240907T150713",
    },
    {
        "zone_name": "Arctic_Foothills_Toolik",
        "block_id": "block_2_arctic_foothills",
        "bbox": [-150.5, 68.5, -148.5, 69.5],
        "s2_id": "S2A_MSIL2A_20230807T221531_R115_T06WVB_20241015T113618",
    },
    {
        "zone_name": "Brooks_Range_Mountains",
        "block_id": "block_3_brooks_range",
        "bbox": [-151.0, 67.5, -149.0, 68.5],
        "s2_id": "S2A_MSIL2A_20230811T215531_R029_T06WVV_20241029T182229",
    },
    {
        "zone_name": "Yukon_Basin_Kanuti",
        "block_id": "block_4_yukon_basin",
        "bbox": [-151.0, 66.0, -149.0, 67.0],
        "s2_id": "S2A_MSIL2A_20230828T214531_R129_T06WVU_20241030T002628",
    },
    {
        "zone_name": "Interior_Tanana_Fairbanks",
        "block_id": "block_5_interior_fairbanks",
        "bbox": [-149.0, 64.5, -147.0, 65.5],
        "s2_id": "S2A_MSIL2A_20230828T214531_R129_T06WWT_20241030T002628",
    },
]

SPECTRAL_BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]


def get_stac_catalog():
    """Initialize authenticated Planetary Computer STAC client."""
    return pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace
    )


def download_copernicus_dem_for_bbox(catalog, bbox: List[float]) -> List[str]:
    """Search and download Copernicus DEM GLO-90 tiles for a bounding box."""
    search = catalog.search(collections=["cop-dem-glo-90"], bbox=bbox)
    items = list(search.items())
    local_files = []
    for it in items:
        dst = os.path.join(DEM_DIR, f"{it.id}.tif")
        if not os.path.exists(dst) or os.path.getsize(dst) < 1000:
            logger.info(f"Downloading DEM tile: {it.id}")
            href = it.assets["data"].href
            resp = requests.get(href, timeout=45)
            resp.raise_for_status()
            with open(dst, "wb") as f:
                f.write(resp.content)
        local_files.append(dst)
    return local_files


def extract_zone_satellite_and_terrain(
    catalog,
    zone_cfg: Dict[str, Any],
    pfr_grid: np.ndarray,
    lat_coords: np.ndarray,
    lon_coords: np.ndarray
) -> pd.DataFrame:
    """
    Extracts Sentinel-2 bands, computes DEM elevation/slope, and aligns with ESA PFR.
    """
    zone_name = zone_cfg["zone_name"]
    block_id = zone_cfg["block_id"]
    s2_id = zone_cfg["s2_id"]
    bbox = zone_cfg["bbox"]

    logger.info(f"Processing Zone: {zone_name} ({block_id})...")

    # Fetch DEM tiles
    dem_tiles = download_copernicus_dem_for_bbox(catalog, bbox)
    logger.info(f"  Found {len(dem_tiles)} DEM tiles for {zone_name}")

    # Fetch Sentinel-2 Item
    search_s2 = catalog.search(collections=["sentinel-2-l2a"], ids=[s2_id])
    s2_items = list(search_s2.items())
    if not s2_items:
        # Fallback search by bbox if specific ID changed
        search_s2 = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime="2023-07-01/2023-08-31",
            query={"eo:cloud_cover": {"lt": 10}},
            max_items=1
        )
        s2_items = list(search_s2.items())
    
    if not s2_items:
        raise RuntimeError(f"Could not find Sentinel-2 scene for zone {zone_name}")
    s2_item = s2_items[0]

    # Process 1 km decimated grids for each band
    target_dim = 100  # 100x100 for 100km scene = 1km resolution
    band_arrays = {}
    lons_grid, lats_grid = None, None

    for b in SPECTRAL_BANDS:
        local_tif = os.path.join(SAT_DIR, f"{s2_item.id}_{b}_1km.tif")
        if not os.path.exists(local_tif) or os.path.getsize(local_tif) < 1000:
            logger.info(f"  Downloading & resampling 1km band {b}...")
            href = s2_item.assets[b].href
            with rasterio.open(href) as src:
                data = src.read(1, out_shape=(target_dim, target_dim), resampling=Resampling.average)
                profile = src.profile.copy()
                profile.update({
                    "height": target_dim,
                    "width": target_dim,
                    "transform": src.transform * src.transform.scale(
                        (src.width / target_dim), (src.height / target_dim)
                    ),
                })
                with rasterio.open(local_tif, "w", **profile) as dst:
                    dst.write(data, 1)

        with rasterio.open(local_tif) as src:
            arr = src.read(1).astype(float)
            band_arrays[b] = arr
            if lons_grid is None:
                ny, nx = src.shape
                cols, rows = np.meshgrid(np.arange(nx), np.arange(ny))
                xs, ys = rasterio.transform.xy(src.transform, rows, cols)
                lons_trans, lats_trans = transform(src.crs, "EPSG:4326", np.array(xs).flatten(), np.array(ys).flatten())
                lons_grid = np.array(lons_trans).reshape((ny, nx))
                lats_grid = np.array(lats_trans).reshape((ny, nx))

    # Scale reflectances (Sentinel-2 L2A BOA reflectance scaling: divide by 10,000)
    # Clamp to [0, 1] range to eliminate sensor saturation and dark artifacts
    blue = np.clip(band_arrays["B02"] / 10000.0, 0.0, 1.0)
    green = np.clip(band_arrays["B03"] / 10000.0, 0.0, 1.0)
    red = np.clip(band_arrays["B04"] / 10000.0, 0.0, 1.0)
    nir = np.clip(band_arrays["B08"] / 10000.0, 0.0, 1.0)
    swir1 = np.clip(band_arrays["B11"] / 10000.0, 0.0, 1.0)
    swir2 = np.clip(band_arrays["B12"] / 10000.0, 0.0, 1.0)

    # Compute Spectral Indices with epsilon protection
    eps = 1e-6
    ndvi = np.clip((nir - red) / (nir + red + eps), -1.0, 1.0)
    ndwi = np.clip((green - nir) / (green + nir + eps), -1.0, 1.0)
    ndsi = np.clip((green - swir1) / (green + swir1 + eps), -1.0, 1.0)

    # Sample Elevation and Slope from Copernicus DEM
    flat_lons = lons_grid.flatten()
    flat_lats = lats_grid.flatten()
    elevation_pts = np.full(flat_lons.shape, np.nan, dtype=float)
    slope_pts = np.full(flat_lons.shape, np.nan, dtype=float)

    # Open DEM tiles and sample
    for dem_file in dem_tiles:
        with rasterio.open(dem_file) as dsrc:
            dem_bounds = dsrc.bounds
            in_tile = (
                (flat_lons >= dem_bounds.left)
                & (flat_lons <= dem_bounds.right)
                & (flat_lats >= dem_bounds.bottom)
                & (flat_lats <= dem_bounds.top)
            )
            if np.any(in_tile):
                coords = list(zip(flat_lons[in_tile], flat_lats[in_tile]))
                sampled_elev = np.array([val[0] for val in dsrc.sample(coords)])
                elevation_pts[in_tile] = sampled_elev

                # Precompute tile slope
                elev_tile = dsrc.read(1)
                dy, dx = np.gradient(elev_tile, 90.0, 90.0)
                slope_tile = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
                # Sample slope
                row_col = [dsrc.index(x, y) for x, y in coords]
                r_idx = np.clip([r for r, c in row_col], 0, slope_tile.shape[0] - 1)
                c_idx = np.clip([c for r, c in row_col], 0, slope_tile.shape[1] - 1)
                slope_pts[in_tile] = slope_tile[r_idx, c_idx]

    # Fill any missing DEM edge points with regional median/interpolation
    if np.isnan(elevation_pts).any():
        med_elev = np.nanmedian(elevation_pts) if not np.all(np.isnan(elevation_pts)) else 250.0
        elevation_pts = np.where(np.isnan(elevation_pts), med_elev, elevation_pts)
    if np.isnan(slope_pts).any():
        med_slope = np.nanmedian(slope_pts) if not np.all(np.isnan(slope_pts)) else 1.5
        slope_pts = np.where(np.isnan(slope_pts), med_slope, slope_pts)

    # Sample Ground Reference Permafrost Fraction from ESA NetCDF
    lat_indices = np.clip(np.searchsorted(lat_coords, flat_lats), 0, len(lat_coords) - 1)
    lon_indices = np.clip(np.searchsorted(lon_coords, flat_lons), 0, len(lon_coords) - 1)
    pfr_raw = pfr_grid[lat_indices, lon_indices]
    pfr_fraction = pfr_raw / 100.0

    # Build DataFrame
    df = pd.DataFrame({
        "latitude": np.round(flat_lats, 4),
        "longitude": np.round(flat_lons, 4),
        "elevation": np.round(elevation_pts, 1),
        "slope": np.round(slope_pts, 2),
        "blue": np.round(blue.flatten(), 4),
        "green": np.round(green.flatten(), 4),
        "red": np.round(red.flatten(), 4),
        "nir": np.round(nir.flatten(), 4),
        "swir1": np.round(swir1.flatten(), 4),
        "swir2": np.round(swir2.flatten(), 4),
        "ndvi": np.round(ndvi.flatten(), 4),
        "ndwi": np.round(ndwi.flatten(), 4),
        "ndsi": np.round(ndsi.flatten(), 4),
        "permafrost_fraction": np.round(pfr_fraction, 4),
        "spatial_block_id": block_id,
        "region_name": zone_name,
    })

    # Drop NaNs and invalid ocean/nodata values
    df = df.dropna(subset=["permafrost_fraction"])
    df = df[(df["permafrost_fraction"] >= 0.0) & (df["permafrost_fraction"] <= 1.0)]

    # Derive binary target permafrost_class: 1 if fraction >= 0.5 else 0
    df["permafrost_class"] = (df["permafrost_fraction"] >= 0.5).astype(int)

    logger.info(f"  Zone {zone_name} produced {len(df)} valid cells (Permafrost-dominant: {(df['permafrost_class']==1).sum()})")
    return df


def build_complete_training_dataset(max_total_samples: int = 35000) -> pd.DataFrame:
    """
    Extracts features across all 5 zones and compiles the standardized training dataset.
    """
    esa_nc_path = os.path.join(RAW_DIR, "ESACCI-PERMAFROST-2023.nc")
    if not os.path.exists(esa_nc_path):
        raise FileNotFoundError(f"Missing ESA reference dataset at {esa_nc_path}. Run data_collection.py first.")

    logger.info("Loading ESA Permafrost NetCDF reference grid...")
    with xr.open_dataset(esa_nc_path) as ds:
        pfr_grid = ds["PFR"].values[0]  # Shape: (lat, lon)
        lat_coords = ds.lat.values
        lon_coords = ds.lon.values

    catalog = get_stac_catalog()
    zone_dfs = []

    for z_cfg in TRANSECT_ZONES:
        try:
            z_df = extract_zone_satellite_and_terrain(catalog, z_cfg, pfr_grid, lat_coords, lon_coords)
            zone_dfs.append(z_df)
        except Exception as e:
            logger.error(f"Error extracting zone {z_cfg['zone_name']}: {e}", exc_info=True)

    if not zone_dfs:
        raise RuntimeError("No zones were successfully extracted!")

    full_df = pd.concat(zone_dfs, ignore_index=True)

    # Subsample if exceeds max_total_samples while preserving spatial block proportions
    if len(full_df) > max_total_samples:
        full_df = full_df.groupby("spatial_block_id", as_index=False, group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), int(max_total_samples / len(zone_dfs))), random_state=42)
        ).reset_index(drop=True)

    # Ensure spatial_block_id is present
    if "spatial_block_id" not in full_df.columns and "region_name" in full_df.columns:
        full_df["spatial_block_id"] = full_df["region_name"]

    # Save to Parquet and CSV
    parquet_path = os.path.join(PROCESSED_DIR, "permafrost_training_dataset.parquet")
    csv_path = os.path.join(PROCESSED_DIR, "permafrost_training_dataset.csv")
    sample_csv_path = os.path.join(SAMPLES_DIR, "sample_500.csv")

    full_df.to_parquet(parquet_path, index=False)
    full_df.to_csv(csv_path, index=False)
    full_df.head(500).to_csv(sample_csv_path, index=False)

    logger.info(f"Dataset Saved Successfully!")
    logger.info(f"  Total Observations: {len(full_df)}")
    logger.info(f"  Permafrost-dominant (Class 1): {(full_df['permafrost_class'] == 1).sum()} ({(full_df['permafrost_class'] == 1).mean()*100:.1f}%)")
    logger.info(f"  Non-permafrost (Class 0): {(full_df['permafrost_class'] == 0).sum()} ({(full_df['permafrost_class'] == 0).mean()*100:.1f}%)")
    logger.info(f"  Parquet: {parquet_path} ({os.path.getsize(parquet_path)} bytes)")
    logger.info(f"  CSV: {csv_path} ({os.path.getsize(csv_path)} bytes)")
    logger.info(f"  Sample: {sample_csv_path}")

    return full_df


if __name__ == "__main__":
    logger.info("Executing PermaWatch Feature Extraction Pipeline...")
    df = build_complete_training_dataset()
    print("Execution complete. Shape:", df.shape)
    print("Class distribution:\n", df["permafrost_class"].value_counts())
    print("Spatial block counts:\n", df["spatial_block_id"].value_counts())
