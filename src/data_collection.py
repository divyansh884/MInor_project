"""
PermaWatch - Data Collection Module
Collects ESA Permafrost Climate Change Initiative (CCI) v5.0 NetCDF datasets,
tests Google Earth Engine authentication, and connects to open satellite archives (Planetary Computer / STAC).
"""

import os
import sys
import logging
import requests
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DataCollection")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")

ESA_CCI_URLS = {
    2023: "https://data.cci.ceda.ac.uk/thredds/fileServer/esacci/permafrost/data/permafrost_extent/L4/area4/pp/v05.0/northern_hemisphere/2023/ESACCI-PERMAFROST-L4-PFR-MODISLST_CRYOGRID-AREA4_PP-2023-fv05.0.nc",
    2020: "https://data.cci.ceda.ac.uk/thredds/fileServer/esacci/permafrost/data/permafrost_extent/L4/area4/pp/v05.0/northern_hemisphere/2020/ESACCI-PERMAFROST-L4-PFR-MODISLST_CRYOGRID-AREA4_PP-2020-fv05.0.nc",
}


def download_file(url: str, dest_path: str, chunk_size: int = 1024 * 1024) -> str:
    """Download a file with streaming chunks and progress display."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 10 * 1024 * 1024:
        logger.info(f"File already exists and is valid: {dest_path} ({os.path.getsize(dest_path)} bytes)")
        return dest_path

    logger.info(f"Downloading {url} to {dest_path}...")
    temp_path = dest_path + ".tmp"
    with requests.get(url, stream=True, timeout=90) as r:
        r.raise_for_status()
        total_len = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(temp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_len > 0:
                        pct = (downloaded / total_len) * 100
                        logger.info(f"Downloaded {downloaded}/{total_len} bytes ({pct:.1f}%)")
    if os.path.exists(dest_path):
        os.remove(dest_path)
    os.rename(temp_path, dest_path)
    logger.info(f"Successfully saved {dest_path} ({os.path.getsize(dest_path)} bytes)")
    return dest_path


def download_esa_permafrost_dataset(year: int = 2023) -> str:
    """Download the official ESA Permafrost Climate Change Initiative dataset."""
    if year not in ESA_CCI_URLS:
        raise ValueError(f"Year {year} not in supported years: {list(ESA_CCI_URLS.keys())}")
    dest = os.path.join(RAW_DATA_DIR, f"ESACCI-PERMAFROST-{year}.nc")
    return download_file(ESA_CCI_URLS[year], dest)


def check_earth_engine_status() -> Dict[str, Any]:
    """
    Attempt Earth Engine initialization and diagnose authentication state.
    Provides actionable guidance if authentication is missing.
    """
    status = {"installed": False, "authenticated": False, "details": ""}
    try:
        import ee
        status["installed"] = True
        try:
            ee.Initialize()
            status["authenticated"] = True
            status["details"] = "Google Earth Engine initialized successfully with active credentials."
            logger.info("Google Earth Engine: AUTHENTICATED.")
        except Exception as e:
            status["authenticated"] = False
            status["details"] = (
                f"Google Earth Engine is installed but unauthenticated: {str(e)}\n"
                "To authenticate, run: 'earthengine authenticate' in the terminal,\n"
                "follow the browser link, select your Google Cloud project, and paste the authorization token."
            )
            logger.warning(status["details"])
    except ImportError:
        status["installed"] = False
        status["details"] = "earthengine-api package is not installed."
        logger.warning(status["details"])
    return status


def inspect_netcdf(file_path: str) -> Dict[str, Any]:
    """Inspect NetCDF structure, coordinates, and metadata using xarray."""
    import xarray as xr
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"NetCDF file does not exist: {file_path}")

    with xr.open_dataset(file_path) as ds:
        info = {
            "title": ds.attrs.get("title", "Unknown"),
            "institution": ds.attrs.get("institution", "Unknown"),
            "dimensions": dict(ds.sizes),
            "data_variables": list(ds.data_vars.keys()),
            "coordinates": list(ds.coords.keys()),
            "lat_range": [float(ds.lat.min()), float(ds.lat.max())] if "lat" in ds else None,
            "lon_range": [float(ds.lon.min()), float(ds.lon.max())] if "lon" in ds else None,
            "pfr_attrs": dict(ds["PFR"].attrs) if "PFR" in ds else {},
        }
    logger.info(f"NetCDF Inspection for {os.path.basename(file_path)}:")
    logger.info(f"  Dimensions: {info['dimensions']}")
    logger.info(f"  Variables: {info['data_variables']}")
    logger.info(f"  Lat bounds: {info['lat_range']}, Lon bounds: {info['lon_range']}")
    return info


if __name__ == "__main__":
    logger.info("Starting PermaWatch Data Collection Pipeline...")
    ee_diag = check_earth_engine_status()
    print("GEE Status:", ee_diag["authenticated"], "|", ee_diag["details"][:100])

    path_2023 = download_esa_permafrost_dataset(2023)
    info_2023 = inspect_netcdf(path_2023)
    print("2023 Dataset Ready:", os.path.basename(path_2023), info_2023["dimensions"])
