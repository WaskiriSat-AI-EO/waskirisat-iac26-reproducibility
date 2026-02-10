#!/usr/bin/env python3
"""
Landsat 9 Thermal Image Processing Pipeline for Wildfire Detection Research

This script converts Landsat 9 Band 10 thermal infrared images (Digital Numbers) to
calibrated brightness temperature and simulates FLIR Lepton 3.5 thermal camera characteristics
for academic wildfire detection research.

Processing Pipeline:
    Phase 1 - Radiometric Calibration:
        1. Load Landsat 9 Band 10 GeoTIFF (DN values)
        2. Apply geometric normalization (warp to square, remove padding)
        3. Convert DN → Radiance (linear scaling)
        4. Convert Radiance → Brightness Temperature (Planck inversion)
        5. Apply AGC normalization (Global/Auto/Fixed modes)
        6. Convert to RGB using thermal colormap

    Phase 2 - FLIR Lepton 3.5 Sensor Simulation:
        1. Crop valid regions and split into quadrants
        2. Downsample to 160×120 (Lepton resolution)
        3. Add thermal noise (calibrated to NETD spec)
        4. Apply optical blur (lens MTF approximation)
        5. Add fixed-pattern noise (FPN)
        6. Generate degradation profiles A/B/C

Usage:
    python process_landsat_thermal.py \\
        --input_dir ./landsat_band10/ \\
        --metadata_geojson ./metadata/scenes.geojson \\
        --output_dir ./dataset_thermal/ \\
        --seed 42 \\
        --validation
"""

import argparse
import sys
import json
import re
import gc
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import cv2
import matplotlib.pyplot as plt
import rasterio
import rasterio.enums
import geopandas as gpd
from tqdm import tqdm


# =============================================================================
# PHYSICAL CONSTANTS & SENSOR SPECIFICATIONS
# =============================================================================

# --- Landsat 9 Band 10 Thermal Infrared Specifications ---
LANDSAT_BAND = "Band 10 (10.6-11.19 µm)"
LANDSAT_WAVELENGTH_UM = (10.6, 11.19)  # Thermal infrared wavelength range

# --- FLIR Lepton 3.5 Specifications ---
LEPTON_NETD_MK = 50.0  # Noise Equivalent Temperature Difference [mK]
LEPTON_RESOLUTION = (160, 120)  # Width × Height [pixels]
LEPTON_PIXEL_PITCH_UM = 12.0  # Pixel pitch [µm]
LEPTON_SPECTRAL_RANGE_UM = (8.0, 14.0)  # Long-wave infrared [µm]
LEPTON_SCENE_RANGE_MAX_C = 400.0  # Maximum scene temperature [°C]
LEPTON_DETECTOR_TYPE = "VOx microbolometer (uncooled)"
LEPTON_FRAME_RATE_HZ = 8.6  # Nominal frame rate

# --- Fixed AGC Temperature Range ---
FIXED_TEMP_MIN_C = -15.0
FIXED_TEMP_MAX_C = 85.0

# --- Degradation Profile Specifications ---
DEGRADATION_PROFILES = {
    "A": {
        "thermal_noise_factor": 1.0,  # 1× base NETD
        "optical_blur_sigma_px": 0.9,
        "fpn_strength": 0.075,
        "fpn_factor": None,  # No FPN for profile A
        "description": "Minimal degradation (best case sensor performance)"
    },
    "B": {
        "thermal_noise_factor": 2.0,  # 2× base NETD
        "optical_blur_sigma_px": 1.0,
        "fpn_strength": 0.075,
        "fpn_factor": 0.8,
        "description": "Moderate degradation (typical sensor performance)"
    },
    "C": {
        "thermal_noise_factor": 4.0,  # 4× base NETD
        "optical_blur_sigma_px": 1.5,
        "fpn_strength": 0.075,
        "fpn_factor": 1.5,
        "description": "High degradation (worst case sensor performance)"
    }
}

# Script version for metadata tracking
SCRIPT_VERSION = "1.0.0"


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def log_progress(message: str, level: str = "info"):
    """
    Log progress messages with consistent formatting.

    Args:
        message: Message to log
        level: Log level ('info', 'warning', 'error', 'success')
    """
    prefixes = {
        "info": "[*]",
        "warning": "[WARNING]",
        "error": "[ERROR]",
        "success": "[+]"
    }
    prefix = prefixes.get(level, "[*]")

    if level == "error":
        print(f"{prefix} {message}", file=sys.stderr)
    else:
        print(f"{prefix} {message}")


def setup_output_structure(output_dir: Path, profiles: List[str]) -> Dict[str, Path]:
    """
    Create organized output directory structure.

    Args:
        output_dir: Root output directory
        profiles: List of profile names (e.g., ['A', 'B', 'C'])

    Returns:
        Dictionary mapping directory names to paths
    """
    dirs = {
        "metadata": output_dir / "metadata",
        "validation": output_dir / "validation",
    }

    for profile in profiles:
        for agc_mode in ["global", "auto", "fixed"]:
            key = f"images_{profile}_{agc_mode}"
            dirs[key] = output_dir / "images" / f"profile_{profile}" / agc_mode

    # Create all directories
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    log_progress(f"Created output directory structure at {output_dir}", "success")
    return dirs


def validate_inputs(input_dir: Path, metadata_geojson: Path) -> None:
    """
    Pre-flight validation of input files and metadata.

    Args:
        input_dir: Directory containing Landsat GeoTIFF files
        metadata_geojson: Path to GeoJSON metadata file

    Raises:
        SystemExit: If validation fails
    """
    errors = []

    # Check input directory exists
    if not input_dir.exists():
        errors.append(f"Input directory not found: {input_dir}")
    else:
        # Check for TIF files
        tif_files = list(input_dir.glob("*.tif")) + list(input_dir.glob("*.TIF"))
        if len(tif_files) == 0:
            errors.append(f"No .tif files found in {input_dir}")
        else:
            log_progress(f"Found {len(tif_files)} TIF files in input directory")

    # Check metadata file
    if not metadata_geojson.exists():
        errors.append(f"Metadata file not found: {metadata_geojson}")
    else:
        try:
            gdf = gpd.read_file(metadata_geojson)
            required_fields = [
                "RADIANCE_MULT_BAND_10",
                "RADIANCE_ADD_BAND_10",
                "K1_CONSTANT_BAND_10",
                "K2_CONSTANT_BAND_10"
            ]
            missing = [f for f in required_fields if f not in gdf.columns]
            if missing:
                errors.append(f"Missing required metadata fields: {missing}")
            else:
                log_progress(f"Metadata file valid with {len(gdf)} entries")
        except Exception as e:
            errors.append(f"Cannot read metadata file: {e}")

    # Report errors and exit if any
    if errors:
        log_progress("Input validation failed:", "error")
        for err in errors:
            log_progress(f"  - {err}", "error")
        sys.exit(1)

    log_progress("Input validation passed", "success")


# =============================================================================
# METADATA & PROVENANCE FUNCTIONS
# =============================================================================

def get_library_versions() -> Dict[str, str]:
    """
    Capture versions of key libraries for reproducibility.

    Returns:
        Dictionary of library names and versions
    """
    import numpy
    import cv2 as opencv
    import rasterio

    return {
        "numpy": numpy.__version__,
        "opencv": opencv.__version__,
        "rasterio": rasterio.__version__,
        "python": sys.version.split()[0]
    }


def get_git_commit() -> Optional[str]:
    """
    Get current git commit hash if in a git repository.

    Returns:
        Commit hash string or None if not in git repo
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None


def build_metadata(
    scene_id: str,
    scene_data: Dict[str, Any],
    agc_mode: str,
    profile: str,
    temp_min_k: float,
    temp_max_k: float,
    quadrant_idx: int,
    quality_metrics: Optional[Dict[str, Any]],
    seed: int
) -> Dict[str, Any]:
    """
    Build comprehensive metadata dictionary for output image.

    Args:
        scene_id: Landsat product ID
        scene_data: Scene metadata from GeoJSON
        agc_mode: AGC normalization mode (GLOBAL/AUTO/FIXED)
        profile: Degradation profile (A/B/C)
        temp_min_k: AGC minimum temperature [K]
        temp_max_k: AGC maximum temperature [K]
        quadrant_idx: Quadrant index (0-3)
        quality_metrics: Optional quality metrics dictionary
        seed: Random seed used

    Returns:
        Complete metadata dictionary
    """
    profile_spec = DEGRADATION_PROFILES[profile]

    # Calculate AGC range justification
    agc_range_k = temp_max_k - temp_min_k
    if agc_mode == "GLOBAL":
        justification = f"Global percentiles from multiple scenes"
    elif agc_mode == "AUTO":
        justification = f"Per-scene automatic percentiles (2%, 98%)"
    else:  # FIXED
        justification = f"Fixed range for wildfire monitoring: {FIXED_TEMP_MIN_C}°C to {FIXED_TEMP_MAX_C}°C"

    metadata = {
        "processing_metadata": {
            "script_version": SCRIPT_VERSION,
            "execution_timestamp": datetime.now(timezone.utc).isoformat(),
            "random_seed": seed,
            "git_commit_hash": get_git_commit(),
            "library_versions": get_library_versions()
        },
        "input_data": {
            "source_image": scene_data.get("tif_path"),
            "product_id": scene_id,
            "acquisition_date": scene_data.get("date"),
            "cloud_cover_percent": scene_data.get("cloud_cover"),
            "quadrant_index": quadrant_idx
        },
        "radiometric_calibration": {
            "landsat_band": LANDSAT_BAND,
            "wavelength_range_um": LANDSAT_WAVELENGTH_UM,
            "radiance_mult_band_10": scene_data.get("R_mult"),
            "radiance_add_band_10": scene_data.get("R_add"),
            "k1_constant_band_10": scene_data.get("K1"),
            "k2_constant_band_10": scene_data.get("K2"),
            "calibration_equation": "BT = K2 / ln((K1 / L) + 1)",
            "note": "Brightness Temperature (at-sensor), not surface temperature"
        },
        "agc_normalization": {
            "mode": agc_mode,
            "temp_min_k": float(temp_min_k),
            "temp_max_k": float(temp_max_k),
            "temp_min_c": float(temp_min_k - 273.15),
            "temp_max_c": float(temp_max_k - 273.15),
            "temp_range_k": float(agc_range_k),
            "justification": justification
        },
        "sensor_simulation": {
            "sensor_model": "FLIR Lepton 3.5",
            "netd_mk": LEPTON_NETD_MK,
            "resolution": list(LEPTON_RESOLUTION),
            "pixel_pitch_um": LEPTON_PIXEL_PITCH_UM,
            "spectral_range_um": list(LEPTON_SPECTRAL_RANGE_UM),
            "detector_type": LEPTON_DETECTOR_TYPE,
            "frame_rate_hz": LEPTON_FRAME_RATE_HZ,
            "datasheet_reference": "FLIR Lepton Engineering Datasheet Rev 200"
        },
        "degradation_profile": {
            "profile": profile,
            "description": profile_spec["description"],
            "thermal_noise_factor": profile_spec["thermal_noise_factor"],
            "thermal_noise_netd_equivalent_mk": LEPTON_NETD_MK * profile_spec["thermal_noise_factor"],
            "optical_blur_sigma_px": profile_spec["optical_blur_sigma_px"],
            "fpn_strength": profile_spec["fpn_strength"],
            "fpn_factor": profile_spec["fpn_factor"]
        },
        "provenance_chain": [
            "Landsat 9 Band 10 GeoTIFF (Digital Numbers)",
            "Geometric normalization (warp to square, mask padding)",
            "DN → Radiance (linear scaling with RADIANCE_MULT/ADD)",
            "Radiance → Brightness Temperature (Planck inversion)",
            f"AGC normalization ({agc_mode} mode)",
            "Thermal colormap (inferno) → RGB",
            "Quadrant splitting (4 sub-regions)",
            f"FLIR Lepton 3.5 simulation (Profile {profile})"
        ]
    }

    # Add quality metrics if available
    if quality_metrics:
        metadata["quality_metrics"] = quality_metrics

    return metadata


def save_metadata_json(metadata: Dict[str, Any], output_path: Path) -> None:
    """
    Save metadata to JSON file with pretty formatting.

    Args:
        metadata: Metadata dictionary
        output_path: Output JSON file path
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def generate_batch_summary(
    output_dir: Path,
    total_scenes: int,
    total_images: int,
    processing_time_s: float,
    global_temp_range: Tuple[float, float],
    args: argparse.Namespace
) -> None:
    """
    Generate batch-level summary JSON.

    Args:
        output_dir: Root output directory
        total_scenes: Number of scenes processed
        total_images: Total output images generated
        processing_time_s: Total processing time in seconds
        global_temp_range: Global (T_low, T_high) in Kelvin
        args: Command-line arguments
    """
    summary = {
        "batch_metadata": {
            "processing_date": datetime.now(timezone.utc).isoformat(),
            "script_version": SCRIPT_VERSION,
            "git_commit": get_git_commit(),
            "total_processing_time_s": round(processing_time_s, 2)
        },
        "dataset_statistics": {
            "total_scenes_processed": total_scenes,
            "total_images_generated": total_images,
            "profiles_generated": args.profiles.split(","),
            "agc_modes": ["global", "auto", "fixed"],
            "quadrants_per_scene": 4
        },
        "global_temperature_statistics": {
            "temp_min_k": float(global_temp_range[0]),
            "temp_max_k": float(global_temp_range[1]),
            "temp_min_c": float(global_temp_range[0] - 273.15),
            "temp_max_c": float(global_temp_range[1] - 273.15),
            "percentiles_used": args.agc_percentiles
        },
        "processing_parameters": {
            "random_seed": args.seed,
            "downscale_factor": args.downscale_factor,
            "lepton_resolution": list(args.lepton_res),
            "fixed_temp_range_c": [args.fixed_temp_range[0], args.fixed_temp_range[1]],
            "jpg_quality": args.jpg_quality,
            "validation_enabled": args.validation
        }
    }

    summary_path = output_dir / "batch_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    log_progress(f"Batch summary saved to {summary_path}", "success")


# =============================================================================
# PHASE 1: RADIOMETRIC CALIBRATION FUNCTIONS
# =============================================================================

def load_geojson_properties(geojson_path: Path) -> List[Dict[str, Any]]:
    """
    Load properties from GeoJSON features.

    Args:
        geojson_path: Path to GeoJSON metadata file

    Returns:
        List of property dictionaries
    """
    with open(geojson_path, 'r', encoding='utf-8') as f:
        gj = json.load(f)

    features = gj.get("features", [])
    properties = [ft.get("properties", {}) for ft in features]

    return properties


def extract_product_id(filename: str) -> Optional[str]:
    """
    Extract Landsat product ID from filename.

    Args:
        filename: Landsat filename (e.g., "L9_Banda10_LC09_L1GT_230070_20250110_02_T2.tif")

    Returns:
        Product ID or None if not found
    """
    # Match pattern: LC09_L1GT_XXXXXX_YYYYMMDD_YYYYMMDD_02_TX
    match = re.search(r"(LC09_L1(?:GT|TP)_\d{6}_\d{8}_\d{8}_02_T\d)", filename)
    return match.group(1) if match else None


def build_scene_index(
    input_dir: Path,
    metadata_geojson: Path
) -> Dict[str, Dict[str, Any]]:
    """
    Build scene index matching TIF files with metadata.

    Args:
        input_dir: Directory containing Landsat GeoTIFF files
        metadata_geojson: Path to GeoJSON metadata file

    Returns:
        Dictionary mapping product IDs to scene data
    """
    # Index TIF files by product ID
    tif_files = list(input_dir.glob("*.tif")) + list(input_dir.glob("*.TIF"))
    image_index = {}

    for tif_path in tif_files:
        pid = extract_product_id(tif_path.name)
        if pid:
            image_index[pid] = {"tif_path": str(tif_path)}

    log_progress(f"Indexed {len(image_index)} TIF files by product ID")

    # Load metadata and build index
    properties = load_geojson_properties(metadata_geojson)
    meta_index = {}

    for props in properties:
        pid = props.get("LANDSAT_PRODUCT_ID")
        if pid:
            meta_index[pid] = props

    log_progress(f"Loaded metadata for {len(meta_index)} scenes")

    # Build final scene index with radiometric parameters
    required_fields = [
        "RADIANCE_MULT_BAND_10",
        "RADIANCE_ADD_BAND_10",
        "K1_CONSTANT_BAND_10",
        "K2_CONSTANT_BAND_10"
    ]

    scene_index = {}
    for pid, img_data in image_index.items():
        if pid not in meta_index:
            log_progress(f"No metadata found for {pid}", "warning")
            continue

        meta = meta_index[pid]

        # Check for required radiometric fields
        if any(k not in meta or meta[k] is None for k in required_fields):
            log_progress(f"Missing radiometric parameters for {pid}", "warning")
            continue

        scene_index[pid] = {
            "tif_path": img_data["tif_path"],
            "date": meta.get("DATE_ACQUIRED"),
            "cloud_cover": meta.get("CLOUD_COVER"),
            "R_mult": float(meta["RADIANCE_MULT_BAND_10"]),
            "R_add": float(meta["RADIANCE_ADD_BAND_10"]),
            "K1": float(meta["K1_CONSTANT_BAND_10"]),
            "K2": float(meta["K2_CONSTANT_BAND_10"])
        }

    log_progress(f"Built scene index with {len(scene_index)} valid scenes", "success")
    return scene_index


def preprocess_thermal_scene(tif_path: str, downscale: int = 4) -> Optional[np.ndarray]:
    """
    Preprocess thermal scene: downscale, mask nodata, geometric normalization.

    This function performs geometric correction to handle Landsat scenes that may
    have black padding or rotation. It warps the valid data region to a square
    format suitable for further processing.

    Args:
        tif_path: Path to Landsat Band 10 GeoTIFF
        downscale: Downscale factor for memory efficiency (default: 4)

    Returns:
        Square float32 array with DN values, or None if processing fails
    """
    try:
        with rasterio.open(tif_path) as src:
            # Read with downscaling for memory efficiency
            dn = src.read(
                1,
                out_shape=(1, src.height // downscale, src.width // downscale),
                resampling=rasterio.enums.Resampling.average
            ).astype(np.float32)

            nodata = src.nodata

        # Mask nodata values
        if nodata is not None:
            dn = np.where(dn == nodata, np.nan, dn)

        # Mask invalid DN values
        dn[dn <= 0] = np.nan

        # Find valid data region using contours
        mask = (~np.isnan(dn)).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Get largest contour (main data region)
        cnt = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect).astype(np.int32)

        w, h = map(int, rect[1])
        if w == 0 or h == 0:
            return None

        # Warp to axis-aligned rectangle
        src_pts = box.astype('float32')
        dst_pts = np.array([[0, h-1], [0, 0], [w-1, 0], [w-1, h-1]], dtype='float32')

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(dn, M, (w, h))

        # Pad to square for consistent processing
        side = max(w, h)
        square = np.full((side, side), np.nan, dtype=np.float32)
        square[(side-h)//2:(side-h)//2+h, (side-w)//2:(side-w)//2+w] = warped

        # Clean up memory
        del dn, warped, mask
        gc.collect()

        return square

    except Exception as e:
        log_progress(f"Failed to preprocess {tif_path}: {e}", "error")
        return None


def dn_to_radiance(dn: np.ndarray, R_mult: float, R_add: float) -> np.ndarray:
    """
    Convert Digital Numbers to Top-of-Atmosphere Radiance.

    Equation: L = R_mult × DN + R_add

    Args:
        dn: Digital Number array
        R_mult: Radiance multiplicative rescaling factor
        R_add: Radiance additive rescaling factor

    Returns:
        Radiance in W/(m² sr µm)
    """
    return R_mult * dn + R_add


def radiance_to_bt(radiance: np.ndarray, K1: float, K2: float) -> np.ndarray:
    """
    Convert Radiance to Brightness Temperature using Planck's Law inversion.

    Equation: BT = K2 / ln((K1 / L) + 1)

    Args:
        radiance: Top-of-atmosphere radiance [W/(m² sr µm)]
        K1: Thermal constant K1 [W/(m² sr µm)]
        K2: Thermal constant K2 [K]

    Returns:
        Brightness temperature in Kelvin
    """
    # Mask invalid radiance values
    radiance_safe = np.where(np.isfinite(radiance) & (radiance > 0), radiance, np.nan)

    # Apply Planck inversion
    bt = K2 / np.log((K1 / radiance_safe) + 1.0)

    return bt


def compute_global_percentiles(
    scene_index: Dict[str, Dict[str, Any]],
    downscale: int,
    percentiles: Tuple[float, float],
    max_samples_per_scene: int = 20000
) -> Tuple[float, float]:
    """
    Compute global temperature percentiles across all scenes.

    This function samples brightness temperatures from all scenes to determine
    global AGC range for consistent normalization.

    Args:
        scene_index: Dictionary of scene data
        downscale: Downscale factor for preprocessing
        percentiles: Low and high percentile values (e.g., (2, 98))
        max_samples_per_scene: Maximum pixels to sample per scene

    Returns:
        Tuple of (T_low, T_high) in Kelvin
    """
    log_progress(f"Computing global percentiles ({percentiles[0]}%, {percentiles[1]}%)...")

    samples = []

    for _, scene_data in tqdm(scene_index.items(), desc="Sampling scenes"):
        # Preprocess scene
        square_dn = preprocess_thermal_scene(scene_data["tif_path"], downscale=downscale)
        if square_dn is None:
            continue

        # Convert to brightness temperature
        radiance = dn_to_radiance(square_dn, scene_data["R_mult"], scene_data["R_add"])
        bt = radiance_to_bt(radiance, scene_data["K1"], scene_data["K2"])

        # Sample valid pixels
        valid = bt[np.isfinite(bt)]
        if valid.size == 0:
            continue

        # Random sampling to limit memory usage
        n_samples = min(max_samples_per_scene, valid.size)
        idx = np.random.choice(valid.size, n_samples, replace=False)
        samples.append(valid[idx])

        # Clean up
        del square_dn, radiance, bt
        gc.collect()

    if not samples:
        raise RuntimeError("No valid temperature samples found across scenes")

    # Concatenate all samples
    all_samples = np.concatenate(samples)

    # Compute percentiles
    T_low, T_high = np.percentile(all_samples, percentiles)

    log_progress(f"Global BT range: {T_low:.2f}K to {T_high:.2f}K "
                f"({T_low-273.15:.2f}°C to {T_high-273.15:.2f}°C)", "success")

    return float(T_low), float(T_high)


def apply_agc(
    bt: np.ndarray,
    T_min: float,
    T_max: float
) -> np.ndarray:
    """
    Apply Automatic Gain Control (AGC) normalization to [0, 1] range.

    Args:
        bt: Brightness temperature array [K]
        T_min: Minimum temperature for normalization [K]
        T_max: Maximum temperature for normalization [K]

    Returns:
        Normalized array in [0, 1] range
    """
    return np.clip((bt - T_min) / (T_max - T_min), 0.0, 1.0)


def thermal_to_rgb(normalized: np.ndarray, cmap: str = "inferno") -> np.ndarray:
    """
    Convert normalized thermal data to RGB using colormap.

    Args:
        normalized: Normalized temperature array [0, 1]
        cmap: Matplotlib colormap name

    Returns:
        8-bit RGB image
    """
    colormap = plt.get_cmap(cmap)
    rgb = colormap(normalized)[..., :3]  # Drop alpha channel
    return (rgb * 255).astype(np.uint8)


# =============================================================================
# PHASE 2: SENSOR SIMULATION FUNCTIONS
# =============================================================================

def crop_valid_region(img: np.ndarray, threshold: int = 10) -> np.ndarray:
    """
    Crop image to remove black padding/borders.

    Args:
        img: RGB image
        threshold: Intensity threshold for detecting valid pixels

    Returns:
        Cropped image
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > threshold

    coords = np.column_stack(np.where(mask))
    if coords.size == 0:
        return img

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)

    return img[y0:y1+1, x0:x1+1]


def split_into_quadrants(img: np.ndarray) -> List[np.ndarray]:
    """
    Split image into 4 equal quadrants.

    Args:
        img: Input image

    Returns:
        List of 4 quadrant images [top-left, top-right, bottom-left, bottom-right]
    """
    h, w = img.shape[:2]
    h2, w2 = h // 2, w // 2

    return [
        img[0:h2, 0:w2],      # Top-left
        img[0:h2, w2:w],      # Top-right
        img[h2:h, 0:w2],      # Bottom-left
        img[h2:h, w2:w]       # Bottom-right
    ]


def resize_to_lepton(img: np.ndarray, resolution: Tuple[int, int]) -> np.ndarray:
    """
    Resize image to FLIR Lepton resolution.

    Args:
        img: Input image
        resolution: Target (width, height)

    Returns:
        Resized image
    """
    return cv2.resize(img, resolution, interpolation=cv2.INTER_AREA)


def calculate_thermal_noise_sigma(netd_mk: float, agc_range_k: float) -> float:
    """
    Convert NETD specification to normalized noise sigma.

    The Noise Equivalent Temperature Difference (NETD) specifies the minimum
    temperature difference detectable by the sensor. This function converts
    the NETD from millikelvin to normalized sigma in [0,1] space based on
    the AGC temperature range.

    Args:
        netd_mk: NETD in millikelvin (e.g., 50 mK for Lepton 3.5)
        agc_range_k: AGC temperature range in Kelvin

    Returns:
        Noise sigma in normalized [0,1] space

    References:
        FLIR Lepton 3.5 Engineering Datasheet Rev 200: NETD < 50 mK
    """
    netd_k = netd_mk / 1000.0  # Convert mK → K
    sigma_normalized = netd_k / agc_range_k
    return sigma_normalized


def add_thermal_noise(
    img_rgb: np.ndarray,
    sigma_normalized: float,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Add Gaussian thermal noise to image.

    Simulates thermal sensor noise based on NETD specification.

    Args:
        img_rgb: Input RGB image (8-bit)
        sigma_normalized: Noise sigma in normalized [0,1] space
        rng: Random number generator for reproducibility

    Returns:
        Noisy 8-bit RGB image
    """
    img_float = img_rgb.astype(np.float32) / 255.0

    # Generate Gaussian noise
    noise = rng.normal(0.0, sigma_normalized, size=img_float.shape).astype(np.float32)

    # Add noise and clip
    noisy = np.clip(img_float + noise, 0.0, 1.0)

    return (noisy * 255).astype(np.uint8)


def apply_optical_blur(img_rgb: np.ndarray, sigma_px: float) -> np.ndarray:
    """
    Apply Gaussian blur to simulate lens MTF (Modulation Transfer Function).

    Args:
        img_rgb: Input RGB image
        sigma_px: Gaussian blur sigma in pixels

    Returns:
        Blurred image
    """
    if sigma_px <= 0:
        return img_rgb

    # Calculate kernel size (must be odd)
    ksize = int(2 * round(2 * sigma_px) + 1)

    return cv2.GaussianBlur(img_rgb, (ksize, ksize), sigmaX=sigma_px)


def generate_fpn(
    shape: Tuple[int, int, int],
    strength: float,
    seed: int
) -> np.ndarray:
    """
    Generate Fixed-Pattern Noise (FPN) map with deterministic seed.

    FPN represents non-uniformity in sensor pixel responses. This function
    generates a reproducible FPN pattern that can be applied consistently
    across all images.

    Args:
        shape: FPN shape (height, width, channels)
        strength: FPN strength (standard deviation)
        seed: Random seed for reproducibility

    Returns:
        FPN map as float32 array
    """
    rng = np.random.default_rng(seed)
    h, w, _ = shape  # channels not used

    # Generate per-pixel FPN (same across channels)
    fpn = rng.normal(0.0, strength, size=(h, w, 1)).astype(np.float32)

    return fpn


def apply_fpn(
    img_rgb: np.ndarray,
    fpn_map: np.ndarray,
    factor: float
) -> np.ndarray:
    """
    Apply Fixed-Pattern Noise to image.

    Args:
        img_rgb: Input RGB image (8-bit)
        fpn_map: FPN map (same size as image)
        factor: FPN scaling factor

    Returns:
        Image with FPN applied (8-bit)
    """
    img_float = img_rgb.astype(np.float32) / 255.0

    # Apply scaled FPN
    img_float = img_float + (factor * fpn_map)

    # Clip to valid range
    img_float = np.clip(img_float, 0.0, 1.0)

    return (img_float * 255).astype(np.uint8)


def simulate_profile(
    img_rgb: np.ndarray,
    profile: str,
    agc_range_k: float,
    fpn_map: Optional[np.ndarray],
    lepton_resolution: Tuple[int, int],
    rng: np.random.Generator
) -> np.ndarray:
    """
    Apply complete FLIR Lepton simulation for specified degradation profile.

    Args:
        img_rgb: Input RGB image (preprocessed thermal)
        profile: Profile name ('A', 'B', or 'C')
        agc_range_k: AGC temperature range in Kelvin
        fpn_map: Fixed-pattern noise map (or None)
        lepton_resolution: Target Lepton resolution (width, height)
        rng: Random number generator

    Returns:
        Degraded 8-bit RGB image at Lepton resolution
    """
    spec = DEGRADATION_PROFILES[profile]

    # 1. Resize to Lepton resolution
    img = resize_to_lepton(img_rgb, lepton_resolution)

    # 2. Calculate and add thermal noise
    base_sigma = calculate_thermal_noise_sigma(LEPTON_NETD_MK, agc_range_k)
    thermal_sigma = base_sigma * spec["thermal_noise_factor"]
    img = add_thermal_noise(img, thermal_sigma, rng)

    # 3. Apply optical blur
    img = apply_optical_blur(img, spec["optical_blur_sigma_px"])

    # 4. Apply FPN if specified
    if spec["fpn_factor"] is not None and fpn_map is not None:
        img = apply_fpn(img, fpn_map, spec["fpn_factor"])

    # Ensure valid 8-bit range
    return np.clip(img, 0, 255).astype(np.uint8)


# =============================================================================
# VALIDATION & QUALITY METRICS FUNCTIONS
# =============================================================================

def calculate_snr_db(img: np.ndarray) -> float:
    """
    Estimate Signal-to-Noise Ratio using Laplacian-based noise estimator.

    This method is more robust than simple standard deviation as it focuses
    on high-frequency noise components rather than image structure.

    Args:
        img: Input image (grayscale or RGB)

    Returns:
        SNR in decibels
    """
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    # Estimate signal
    signal_mean = np.mean(gray)

    # Estimate noise using Laplacian
    laplacian = cv2.Laplacian(gray.astype(float), cv2.CV_64F)
    noise_std = np.std(laplacian) / np.sqrt(2)

    if noise_std == 0:
        return float('inf')

    # Calculate SNR
    snr = signal_mean / noise_std
    snr_db = 20 * np.log10(snr) if snr > 0 else -np.inf

    return float(snr_db)


def calculate_histogram_stats(img: np.ndarray) -> Dict[str, float]:
    """
    Calculate histogram-based statistics.

    Args:
        img: Input image (grayscale or RGB)

    Returns:
        Dictionary with statistics
    """
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    # Calculate percentiles
    percentiles = [1, 5, 25, 50, 75, 95, 99]
    pct_values = np.percentile(gray, percentiles)

    # Calculate entropy
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    hist = hist[hist > 0]  # Remove zero bins
    prob = hist / hist.sum()
    entropy = -np.sum(prob * np.log2(prob))

    stats = {
        "mean": float(np.mean(gray)),
        "std": float(np.std(gray)),
        "min": float(np.min(gray)),
        "max": float(np.max(gray)),
        "entropy_bits": float(entropy)
    }

    # Add percentiles
    for pct, val in zip(percentiles, pct_values):
        stats[f"p{pct:02d}"] = float(val)

    # Saturation check
    stats["saturation_low_pct"] = float(np.sum(gray <= 5) / gray.size * 100)
    stats["saturation_high_pct"] = float(np.sum(gray >= 250) / gray.size * 100)

    return stats


def validate_profile_consistency(images_dict: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """
    Validate that degradation profiles have expected noise ordering.

    Profile A should have highest SNR (least noise)
    Profile C should have lowest SNR (most noise)

    Args:
        images_dict: Dictionary mapping profile names to images

    Returns:
        Validation results dictionary
    """
    snrs = {prof: calculate_snr_db(img) for prof, img in images_dict.items()}

    # Check ordering
    valid = True
    if "A" in snrs and "B" in snrs:
        valid = valid and (snrs["A"] > snrs["B"])
    if "B" in snrs and "C" in snrs:
        valid = valid and (snrs["B"] > snrs["C"])
    if "A" in snrs and "C" in snrs:
        valid = valid and (snrs["A"] > snrs["C"])

    return {
        "snr_ordering_valid": valid,
        "snr_values_db": snrs,
        "note": "Expected: SNR_A > SNR_B > SNR_C"
    }


def generate_validation_report(
    output_dir: Path,
    validation_data: List[Dict[str, Any]]
) -> None:
    """
    Generate validation report from collected validation data.

    Args:
        output_dir: Output directory
        validation_data: List of validation data dictionaries
    """
    report_path = output_dir / "validation" / "validation_report.json"

    report = {
        "validation_timestamp": datetime.now(timezone.utc).isoformat(),
        "total_images_validated": len(validation_data),
        "validation_results": validation_data
    }

    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    log_progress(f"Validation report saved to {report_path}", "success")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Process Landsat 9 thermal images for wildfire detection research",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default parameters
  python process_landsat_thermal.py --input_dir ./landsat/ --metadata_geojson ./meta.geojson --output_dir ./output/

  # With validation and custom parameters
  python process_landsat_thermal.py \\
      --input_dir ./landsat_band10/ \\
      --metadata_geojson ./metadata.geojson \\
      --output_dir ./dataset/ \\
      --seed 42 \\
      --profiles A,B,C \\
      --validation
        """
    )

    # Required arguments
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing Landsat 9 Band 10 GeoTIFF files"
    )
    parser.add_argument(
        "--metadata_geojson",
        type=str,
        required=True,
        help="GeoJSON file with Landsat metadata (RADIANCE_MULT_BAND_10, K1/K2 constants, etc.)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Root output directory for processed images and metadata"
    )

    # Processing parameters
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--downscale_factor",
        type=int,
        default=4,
        help="Preprocessing downscale factor (default: 4)"
    )
    parser.add_argument(
        "--agc_percentiles",
        type=float,
        nargs=2,
        default=[2.0, 98.0],
        metavar=("LOW", "HIGH"),
        help="Global AGC percentiles (default: 2 98)"
    )
    parser.add_argument(
        "--fixed_temp_range",
        type=float,
        nargs=2,
        default=[FIXED_TEMP_MIN_C, FIXED_TEMP_MAX_C],
        metavar=("MIN_C", "MAX_C"),
        help=f"Fixed AGC range in Celsius (default: {FIXED_TEMP_MIN_C} {FIXED_TEMP_MAX_C})"
    )
    parser.add_argument(
        "--lepton_res",
        type=int,
        nargs=2,
        default=list(LEPTON_RESOLUTION),
        metavar=("WIDTH", "HEIGHT"),
        help=f"Lepton resolution W×H (default: {LEPTON_RESOLUTION[0]} {LEPTON_RESOLUTION[1]})"
    )
    parser.add_argument(
        "--profiles",
        type=str,
        default="A,B,C",
        help="Comma-separated profiles to generate (default: A,B,C)"
    )
    parser.add_argument(
        "--validation",
        action="store_true",
        help="Enable validation metrics generation"
    )
    parser.add_argument(
        "--jpg_quality",
        type=int,
        default=90,
        help="JPEG quality for outputs (default: 90)"
    )

    args = parser.parse_args()

    # Convert paths
    input_dir = Path(args.input_dir)
    metadata_geojson = Path(args.metadata_geojson)
    output_dir = Path(args.output_dir)

    # Parse profiles
    profiles_list = [p.strip().upper() for p in args.profiles.split(",")]

    # Print header
    print("=" * 70)
    print("Landsat 9 Thermal Image Processing Pipeline")
    print(f"Version {SCRIPT_VERSION}")
    print("=" * 70)

    # Validate inputs
    validate_inputs(input_dir, metadata_geojson)

    # Setup output structure
    dirs = setup_output_structure(output_dir, profiles_list)

    # Build scene index
    log_progress("Building scene index...")
    scene_index = build_scene_index(input_dir, metadata_geojson)

    if len(scene_index) == 0:
        log_progress("No valid scenes found. Exiting.", "error")
        sys.exit(1)

    # Compute global percentiles
    log_progress("Computing global temperature statistics...")
    T_global_low, T_global_high = compute_global_percentiles(
        scene_index,
        args.downscale_factor,
        tuple(args.agc_percentiles)
    )

    # Convert fixed temp range to Kelvin
    T_fixed_min = args.fixed_temp_range[0] + 273.15
    T_fixed_max = args.fixed_temp_range[1] + 273.15

    # Generate FPN map (deterministic with seed)
    lepton_res_tuple = tuple(args.lepton_res)
    fpn_map = generate_fpn(
        (lepton_res_tuple[1], lepton_res_tuple[0], 3),  # (H, W, C)
        strength=0.075,
        seed=args.seed
    )
    log_progress("Generated deterministic FPN map", "success")

    # Initialize random generator
    rng = np.random.default_rng(args.seed)

    # Process scenes
    log_progress(f"Processing {len(scene_index)} scenes...")
    total_images = 0
    validation_data = []
    start_time = datetime.now()

    for scene_id, scene_data in tqdm(scene_index.items(), desc="Processing scenes"):
        try:
            # Preprocess scene
            square_dn = preprocess_thermal_scene(
                scene_data["tif_path"],
                downscale=args.downscale_factor
            )
            if square_dn is None:
                log_progress(f"Skipping {scene_id}: preprocessing failed", "warning")
                continue

            # Convert to brightness temperature
            radiance = dn_to_radiance(square_dn, scene_data["R_mult"], scene_data["R_add"])
            bt = radiance_to_bt(radiance, scene_data["K1"], scene_data["K2"])

            # Compute auto percentiles for this scene
            valid_bt = bt[np.isfinite(bt)]
            if valid_bt.size == 0:
                log_progress(f"Skipping {scene_id}: no valid temperature data", "warning")
                continue

            T_auto_low, T_auto_high = np.percentile(valid_bt, args.agc_percentiles)

            # Process each AGC mode
            agc_modes = {
                "global": (T_global_low, T_global_high, "GLOBAL"),
                "auto": (T_auto_low, T_auto_high, "AUTO"),
                "fixed": (T_fixed_min, T_fixed_max, "FIXED")
            }

            for agc_name, (T_min, T_max, agc_label) in agc_modes.items():
                # Apply AGC normalization
                bt_normalized = apply_agc(bt, T_min, T_max)

                # Convert to RGB
                rgb_full = thermal_to_rgb(bt_normalized, cmap="inferno")

                # Crop valid region
                rgb_cropped = crop_valid_region(rgb_full)

                # Check if image is too small
                if rgb_cropped.shape[0] < 20 or rgb_cropped.shape[1] < 20:
                    continue

                # Split into quadrants
                quadrants = split_into_quadrants(rgb_cropped)

                # Process each quadrant with each profile
                for quad_idx, quadrant in enumerate(quadrants):
                    for profile in profiles_list:
                        # Simulate sensor
                        degraded = simulate_profile(
                            quadrant,
                            profile,
                            T_max - T_min,  # AGC range
                            fpn_map,
                            lepton_res_tuple,
                            rng
                        )

                        # Calculate quality metrics
                        quality_metrics = None
                        if args.validation:
                            quality_metrics = {
                                "snr_db": calculate_snr_db(degraded),
                                **calculate_histogram_stats(degraded)
                            }

                        # Build metadata
                        metadata = build_metadata(
                            scene_id,
                            scene_data,
                            agc_label,
                            profile,
                            T_min,
                            T_max,
                            quad_idx,
                            quality_metrics,
                            args.seed
                        )

                        # Generate output filename
                        output_name = f"{scene_id}_r{quad_idx}_{agc_name}_{profile}"

                        # Save image
                        img_dir = dirs[f"images_{profile}_{agc_name}"]
                        img_path = img_dir / f"{output_name}.png"
                        cv2.imwrite(
                            str(img_path),
                            cv2.cvtColor(degraded, cv2.COLOR_RGB2BGR)
                        )

                        # Save metadata
                        meta_path = dirs["metadata"] / f"{output_name}.json"
                        save_metadata_json(metadata, meta_path)

                        total_images += 1

                        # Collect validation data
                        if args.validation and quality_metrics:
                            validation_data.append({
                                "image": output_name,
                                "scene_id": scene_id,
                                "profile": profile,
                                "agc_mode": agc_label,
                                "quadrant": quad_idx,
                                **quality_metrics
                            })

            # Clean up memory
            del square_dn, radiance, bt, rgb_full
            gc.collect()

        except Exception as e:
            log_progress(f"Error processing {scene_id}: {e}", "error")
            continue

    # Calculate processing time
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()

    # Generate validation report
    if args.validation and validation_data:
        generate_validation_report(output_dir, validation_data)

    # Generate batch summary
    generate_batch_summary(
        output_dir,
        len(scene_index),
        total_images,
        processing_time,
        (T_global_low, T_global_high),
        args
    )

    # Print summary
    print("\n" + "=" * 70)
    log_progress(f"Processing complete!", "success")
    log_progress(f"Scenes processed: {len(scene_index)}")
    log_progress(f"Total images generated: {total_images}")
    log_progress(f"Processing time: {processing_time:.1f}s")
    log_progress(f"Output directory: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
