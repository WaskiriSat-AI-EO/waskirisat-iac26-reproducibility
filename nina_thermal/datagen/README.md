# `nina_thermal` — Sentinel-3 Thermal Fire Dataset Generator

End-to-end pipeline for generating machine learning datasets of wildfire hotspot detection from Copernicus Sentinel-3 SLSTR Level 1B imagery. Designed for the WaskiriSat NINA mission — a nanosatellite carrying a FLIR Lepton 3.5 thermal sensor at 525 km SSO altitude over South America (Bolivia focus area).

---

## What It Does

The package:

1. **Downloads** Sentinel-3 SLSTR granules from the Copernicus Data Space Ecosystem
2. **Extracts** the thermal brightness-temperature channel (F2, the low-gain fire channel at 10.85 µm; S8 is its co-located standard-gain twin) from NetCDF
3. **Processes** them with physically correct radiance-domain aggregation to match the NINA Lepton GSD (~3,563 m/pixel)
4. **Simulates** the full Lepton sensor degradation chain (thermal noise, AGC quantization, optical PSF, FPN)
5. **Labels** patches from NASA FIRMS active-fire detections (confidence- and FRP-gated), matched to the SLSTR grid — the brightness-temperature mask is a QA cross-check only, never a label source
6. **Splits** at granule level (not patch level) to prevent spatial data leakage
7. **Balances** fire/no-fire class ratio (target: 20% fire) by undersampling
8. **Reports** comprehensive statistics for dataset publication

---

## Key Mission Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| NINA altitude | 525 km | Sun-synchronous orbit |
| Lepton GSD | 3,563 m/pixel | Operational ground sampling distance |
| SLSTR pixel size | 1,000 m | Sentinel-3 nadir thermal channel |
| Aggregation factor | 4× | SLSTR pixels aggregated to match Lepton GSD (~4,000 m) |
| F2 wavelength | 10.85 µm | Sentinel-3 fire channel (S8 shares its spectral response) |
| FIRMS min confidence | `nominal` | Minimum FIRMS active-fire confidence tier accepted |
| FIRMS min FRP | 10 MW | Minimum Fire Radiative Power for a labelled detection |
| BT QA threshold | 310 K | F2_BT cross-check only — quantifies warm-surface FP rate, not a label |
| AGC range (flight) | 258.15–358.15 K | Fixed quantization window (−15°C to +85°C) |
| Fire ratio target | 20% | Target fire-to-no-fire ratio after balancing |
| Lepton NETD | 50 mK | Noise Equivalent Temperature Difference |
| Lepton resolution | 160×120 px | Native sensor resolution → patch size |

---

## Directory Structure

```
nina_thermal/
├── config.py                   # All pipeline constants
├── acquisition/
│   ├── cdse_auth.py            # Copernicus OAuth2 token management
│   ├── cdse_search.py          # OData catalog search by geometry + date
│   └── cdse_download.py        # ZIP download + .SEN3 extraction
├── ingestion/
│   ├── extract_channels.py     # Batch NetCDF → .npy extraction
│   ├── load_granule.py         # Fast (.npy) or fallback (NetCDF) loader
│   └── validate_granule.py     # Pre-flight shape/dtype/NaN QA
├── processing/
│   ├── radiometry.py           # Planck's law BT ↔ radiance conversions
│   ├── spatial.py              # Radiance-domain aggregation + patch tiling
│   ├── spectral.py             # Narrowband F2 → broadband Lepton correction
│   └── agc.py                  # Automatic Gain Control (fixed/global/auto)
├── simulation/
│   ├── noise.py                # Thermal noise (NETD) and FPN generation
│   ├── optics.py               # Optical PSF Gaussian blur
│   └── sensor.py               # Full degradation chain orchestrator
├── labeling/
│   ├── fire_label.py           # Dual-criterion fire labeling + surface classification
│   └── split.py                # Granule-level stratified train/val/test splitting
├── dataset/
│   ├── balance.py              # Class undersampling to target fire ratio
│   ├── export.py               # PNG + JSON metadata I/O (only module that writes files)
│   └── report.py               # Aggregate dataset statistics → dataset_summary.json
├── validation/
│   ├── metrics.py              # SNR (Laplacian estimator), histogram stats
│   └── noise_validation.py     # Validates measured noise against theoretical NETD sigma
├── visualization/
│   └── colormap.py             # Grayscale uint8 → false-colour RGB (output only, never in pipeline)
├── pipelines/
│   ├── run_acquisition.py      # Stage 1: download
│   ├── run_extraction.py       # Stage 2: NetCDF → .npy
│   ├── run_processing.py       # Stage 3: aggregate + tile patches
│   ├── run_simulation.py       # Stage 4: simulate + label
│   ├── run_split.py            # Stage 5: split at granule level
│   ├── run_balance.py          # Stage 6: balance fire ratio
│   └── run_report.py           # Report: build dataset_summary.json
└── tests/
    ├── test_fire_label.py
    ├── test_agc.py
    ├── test_radiometry.py
    ├── test_spatial.py
    ├── test_spectral.py
    ├── test_noise.py
    ├── test_split.py
    ├── test_balance.py
    └── test_integration.py
```

---

## Prerequisites

### Python Dependencies

```
numpy
scipy
netCDF4
requests
python-dotenv
Pillow
matplotlib
opencv-python
```

### Copernicus Credentials

Register at [Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu/) and set credentials as environment variables (e.g. in a `.env` file):

```
CDSE_USER=your@email.com
CDSE_PASSWORD=yourpassword
```

---

## Usage: End-to-End Pipeline

### Stage 1 — Download from Copernicus

```bash
python -m nina_thermal.pipelines.run_acquisition \
  --roi resources/bo_nina_roi_simplified.geojson \
  --output_dir data/thermal/01_raw \
  --max_products 50
```

Downloads Sentinel-3 SLSTR L1B granules intersecting the AOI, extracts them as `.SEN3` directories.

**Outputs:**
- `data/01_raw/*.SEN3/` — NetCDF granule directories
- `data/01_raw/acquisition_manifest.json`

---

### Stage 2 — Extract Thermal Channels

```bash
python -m nina_thermal.pipelines.run_extraction \
  --input_dir data/thermal/01_raw \
  --output_dir data/thermal/02_extracted
```

Reads F2_BT, S8_BT, lat, lon from NetCDF; applies cloud and exception bitmasks (sets masked pixels to NaN); saves as `.npy`.

**Outputs per granule:**
- `*_F2_BT.npy` — brightness temperature at 10.85 µm [K]
- `*_S8_BT.npy` — brightness temperature at 10.85 µm [K] (standard-gain twin of F2)
- `*_lat.npy`, `*_lon.npy` — geolocation [degrees]
- `data/02_extracted/extraction_manifest.json`

---

### Stage 3 — Process Patches

```bash
python -m nina_thermal.pipelines.run_processing \
  --input_dir data/thermal/02_extracted \
  --output_dir data/thermal/03_patches
```

For each granule:
1. Validates shape, dtype, and minimum finite fraction (>20%)
2. Applies broadband spectral correction (narrowband 10.85 µm → broadband 8–14 µm Lepton)
3. Aggregates 4×4 SLSTR pixels via radiance-domain averaging (Planck law; not temperature averaging)
4. Tiles the result into 160×120 patches with 50% stride overlap
5. Rejects patches with >70% NaN; fills residual NaN with patch median

**Outputs:**
- `data/03_patches/{granule_id}_p{idx:04d}_bt.npy` — BT patches [K]
- `data/03_patches/patches_manifest.json` — centroid lat/lon, NaN fraction, BT range per patch

---

### Stage 4 — Simulate Sensor + Label

```bash
python -m nina_thermal.pipelines.run_simulation \
  --input_dir data/thermal/03_patches \
  --output_dir data/thermal/04_dataset \
  --seed 42
```

This stage is the **split authority**: it computes the granule-level train/val/test
split itself (writing `split_manifest.json`) *before* generating images, so it can
apply a per-split degradation-profile / AGC policy instead of generating every
profile × AGC combination for every patch regardless of split:

| Split | Profiles | AGC modes |
|-------|----------|-----------|
| train | B only | `fixed`, `global` |
| val | B only | `fixed` |
| test | A, B, C | `fixed` |

> **Why per-split?** Profiles A/B/C are fixed-seed deterministic noise variants of
> the *same* underlying patch, not stochastic augmentation — training on all three
> teaches the model to memorize specific noise patterns rather than generalize, and
> destroys their value as a held-out cross-degradation robustness measure. `fixed`
> AGC is the confirmed flight inference configuration, so val/test metrics are
> restricted to it; `global` is a train-only contrast augmentation, and its
> percentiles are computed from **train patches only** to avoid leaking test BT
> statistics into a training-time transform.

Pass `--profiles A,B,C --agc_modes fixed,global` to override the policy uniformly
across all splits (debugging / ad-hoc full generation) — this is not the
recommended default and reintroduces the profile/AGC leakage the per-split policy
avoids.

For each patch × (effective) AGC mode × degradation profile:
1. Labels the physical patch (fire/no-fire, surface type)
2. Applies AGC normalization → [0, 1]
3. Simulates the Lepton degradation chain:
   - Thermal noise in BT space (NETD-calibrated Gaussian)
   - AGC quantization → uint8
   - Optical PSF blur (f/1.1 Gaussian)
   - Fixed Pattern Noise (column-level + per-pixel readout artifact)
4. Saves grayscale PNG + JSON metadata

**Degradation Profiles:**

| Profile | Thermal Noise | Blur | FPN Strength | Scenario |
|---------|---|---|---|---|
| A | 1.0× NETD | 0.5 px | low | Best case (post-FFC) |
| B | 2.0× NETD | 0.8 px | medium | Typical on-orbit |
| C | 4.0× NETD | 1.2 px | high | Worst case (pre-FFC, SAA passage) |

**AGC Modes:**

| Mode | t_min / t_max | Notes |
|------|---|---|
| `fixed` | 258.15 K / 358.15 K | Flight configuration (matches planned inference) |
| `global` | 2nd–98th percentile across train patches only | Train-only contrast augmentation |
| `auto` | 2nd–98th percentile per patch | Per-image normalization |

**Outputs:**
- `data/04_dataset/images/{granule}_p{idx}_{agc}_{profile}.png` — grayscale uint8
- `data/04_dataset/metadata/{granule}_p{idx}_{agc}_{profile}.json` — full provenance chain
- `data/04_dataset/label_manifest.json`, `simulation_manifest.json`, `split_manifest.json`

---

### Stage 5 — Split Train/Val/Test

```bash
python -m nina_thermal.pipelines.run_split \
  --dataset_dir data/thermal/04_dataset \
  --seed 42
```

Splits at **granule level** (never patch level) using stratified sampling: high-fire granules (>5% fire patches) and low-fire granules are each independently shuffled and distributed proportionally into 70/15/15 train/val/test.

Stage 4 already wrote `split_manifest.json` (it's the split authority). This stage
reuses it verbatim when its granule set still matches `label_manifest.json`,
guaranteeing the split stamped into metadata matches the policy already baked into
which images were generated. If the manifest is missing, or stale (granules were
added/removed since it was written), the split is recomputed from scratch instead
of silently trusting a mismatched file.

> **Why granule-level?** Patches from the same granule share atmospheric conditions, sun angle, and surface state. Patch-level splitting would create validation metrics that are artificially optimistic.

**Outputs:**
- `data/04_dataset/split_manifest.json`
- Metadata JSONs updated with `"split"` field

---

### Stage 6 — Balance Classes

```bash
python -m nina_thermal.pipelines.run_balance \
  --dataset_dir data/thermal/04_dataset \
  --fire_ratio 0.20 \
  --seed 42
```

Undersamples no-fire patches within each split independently to reach the target fire ratio (20%). Fire patches are never removed. Image files are untouched — balance is tracked at manifest level via `"excluded_by_balance": true`.

**Output:**
- `data/04_dataset/balanced_manifest.json`

---

### Report

```bash
python -m nina_thermal.pipelines.run_report \
  --dataset_dir data/thermal/04_dataset
```

**Output:** `data/04_dataset/dataset_summary.json` — comprehensive statistics including granule/patch counts per split, fire fraction, surface type breakdown, BT statistics, geographic and temporal coverage, and all processing parameters.

---

## Fire Labelling

**Labels come from NASA FIRMS active-fire detections, not from the imagery.**
FIRMS detections are matched to the SLSTR grid within a ±1-day overpass window
and accepted only when they clear both quality gates:

```
confidence tier >= nominal        (VIIRS string / MODIS 0-100 mapped per C6.1)
fire radiative power >= 10 MW      (energetic enough for a detectable LWIR anomaly at 3563 m)
```

A patch receives **label = 1** when an accepted FIRMS detection falls inside it.

A two-channel brightness-temperature discriminant over the F2/S8 pair was
implemented and evaluated and carries **no fire information**: the two channels
share a spectral response function, so their BT difference is identically zero
wherever both are unsaturated. The `compute_patch_label` BT mask
(`F2_BT > 310 K`, `delta_T` disabled) is retained only as a QA cross-check to
quantify warm-surface false-positive rates against FIRMS, and to label
synthetic test fixtures.

Patches are further classified by surface type:

| Type | Condition | Typical origin |
|------|-----------|----------------|
| `fire` | label=1 | Active fire / hotspot |
| `warm_surface` | label=0, F2_BT > 310 K | Desert, bare rock, urban heat |
| `nominal` | label=0, F2_BT ≤ 310 K | Vegetation, water, cloud-covered |

---

## Key Design Decisions

### Radiance-Domain Aggregation

Spatial aggregation is performed in radiance space, not temperature space:

```
L = Planck(F2_BT)          # convert each SLSTR pixel to radiance
L_mean = nanmean(L, 4×4)   # average radiances over the aggregation block
BT_agg = Planck_inv(L_mean) # invert back to brightness temperature
```

Averaging temperatures directly underestimates hot sub-pixel fires. Radiance averaging correctly preserves the contribution of fire pixels within a mixed-composition block.

### Sensor Simulation Order

The degradation chain follows physical causality:

```
1. Thermal noise (before quantization — NETD is a temperature uncertainty)
2. AGC quantization to uint8
3. Optical PSF blur (acts on the focal plane signal)
4. FPN (readout electronics artifact, after the optical path)
```

### I/O Isolation

Only `dataset/export.py` writes files. All processing functions are pure (data in → data out), making them straightforward to test in isolation and reuse.

---

## Testing

```bash
pytest datagen/nina_thermal/tests/ -v
```

Tests cover: fire labeling, AGC, radiometric conversions, spatial aggregation, spectral correction, noise, splitting, balancing, and a full end-to-end integration.

---

## Data Formats

| Stage | Format | Shape | Description |
|-------|--------|-------|-------------|
| Raw | NetCDF (`.SEN3/`) | ~4000×4000 | Sentinel-3 SLSTR L1B granule |
| Extracted | `.npy` float32 | (H, W) | Per-channel brightness temperature / lat / lon |
| Patches (BT) | `.npy` float32 | (120, 160) | Aggregated BT patches [K] |
| Images | PNG uint8 | (120, 160) | Single-channel grayscale sensor output |
| Metadata | JSON | — | Per-image provenance, bounds, AGC, split |
| Manifests | JSON | — | Per-stage record index |

---

## ROI Files

Example GeoJSON regions of interest are provided in `resources/`:

| File | Area |
|------|------|
| `bolivia_roi_broad.geojson` | Broad Bolivia bounding box |
| `bolivia_roi_chiquitania_beni.geojson` | Chiquitanía and Beni fire-prone regions |
| `bolivia_roi_fire_zones.geojson` | High-priority fire zone polygons |

---

## Output Metadata Example

Each image ships with a full provenance record:

```json
{
  "granule_id": "S3A_SL_1_RBT____20230601T120000_...",
  "patch_index": 42,
  "label": 1,
  "surface_type": "fire",
  "agc_mode": "fixed",
  "profile": "B",
  "t_min": 258.15,
  "t_max": 358.15,
  "nan_fraction": 0.023,
  "centroid_lat": -15.3,
  "centroid_lon": -62.7,
  "split": "train",
  "seed": 42,
  "flight_config": true,
  "provenance": [
    "SLSTR L1B NetCDF",
    "cloud+exception bitmask",
    "broadband spectral correction",
    "radiance-consistent spatial aggregation (factor=4)",
    "AGC normalisation (fixed, 258.15–358.15 K)",
    "sensor simulation profile B",
    "agc_to_uint8"
  ]
}
```
