# NINA Thermal Dataset Pipeline

Pre-launch training dataset generator for the NINA (Neural Infrared Anomaly Observer) wildfire
detection payload on WaskiriSat — Bolivia's first university-developed CubeSat (UCB San Pablo /
BIRDS / KyuTech). The pipeline constructs a physically rigorous proxy dataset by downscaling
Sentinel-3 SLSTR L1B calibrated brightness temperature to synthetic FLIR Lepton 3.5 images
at 3,563 m/px GSD from a 525 km SSO.

---

## Installation

```bash
pip install -r requirements.txt
```

**Required dependencies** (see `requirements.txt`):
- `numpy`, `opencv-python`, `matplotlib`, `pillow`
- `rasterio`, `netCDF4`, `geopandas`
- `requests`, `python-dotenv`, `tqdm`
- `pytest` (tests only)

Copy `.env.example` to `.env` and set CDSE credentials before running acquisition:

```
CDSE_USER=your_email@example.com
CDSE_PASSWORD=your_password
```

---

## Seven-Step Execution Sequence

Each step reads from the previous step's output directory and writes to its own output directory.
All steps are checkpoint-resumable — re-running a step does not require re-running upstream steps.

### Step 1 — Acquire SLSTR granules from CDSE

```bash
python -m nina_thermal.pipelines.run_acquisition \
    --roi          resources/bolivia_roi.geojson \
    --start        20230601 \
    --end          20230901 \
    --output_dir   data/thermal/01_raw \
    --max_products 50
```

Authenticates with Copernicus Data Space Ecosystem, searches for Sentinel-3 SL_1_RBT___
products over Bolivia, downloads and extracts .zip archives. Writes `acquisition_manifest.json`.

### Step 2 — Extract thermal channels from .SEN3 archives

```bash
python -m nina_thermal.pipelines.run_extraction \
    --input_dir  data/thermal/01_raw \
    --output_dir data/thermal/02_extracted
```

Reads NetCDF files from each .SEN3 directory, applies cloud and exception masks, saves
`{granule_id}_F2_BT.npy`, `{granule_id}_S8_BT.npy`, `{granule_id}_lat.npy`,
`{granule_id}_lon.npy`. Writes `extraction_manifest.json`.

### Step 3 — Process granules: correct, aggregate, tile (Pass 1)

```bash
python -m nina_thermal.pipelines.run_processing \
    --input_dir  data/thermal/02_extracted \
    --output_dir data/thermal/03_patches
```

For each granule: applies broadband spectral correction, performs radiance-consistent
aggregation 4×4 (1,000 m → 4,000 m synthetic GSD), tiles into 160×120 px patches with
NaN handling. Saves patch `.npy` files and `patches_manifest.json`.

### Step 4 — Simulate Lepton images, label patches, export dataset (Pass 2)

```bash
python -m nina_thermal.pipelines.run_simulation \
    --input_dir  data/thermal/03_patches \
    --output_dir data/thermal/04_dataset \
    --profiles   A,B,C \
    --agc_modes  fixed,global \
    --seed       42
```

Applies AGC normalisation, simulates FLIR Lepton sensor (thermal noise → uint8 cast →
PSF blur → FPN) for each profile × AGC mode combination. Computes fire labels using dual
criterion (ΔT > 8 K AND F2_BT > 320 K). Saves grayscale PNGs, per-image JSON metadata,
and `label_manifest.json`.

### Step 5 — Assign granule-level train/val/test splits

```bash
python -m nina_thermal.pipelines.run_split \
    --dataset_dir data/thermal/04_dataset \
    --seed        42
```

Assigns splits at granule level (never patch level) using stratified sampling by fire
fraction tier. Prevents spatial autocorrelation leakage. Writes `split_manifest.json`.

### Step 6 — Balance class ratio

```bash
python -m nina_thermal.pipelines.run_balance \
    --dataset_dir data/thermal/04_dataset \
    --fire_ratio  0.20 \
    --seed        42
```

Undersamples no-fire patches per split to reach 1:4 fire:no-fire ratio. Fire patches are
never removed. Writes `balanced_manifest.json` (manifest filter only — no image files
are deleted).

### Step 7 — Generate dataset report

```bash
python -m nina_thermal.pipelines.run_report \
    --dataset_dir data/thermal/04_dataset \
    --output      data/thermal/04_dataset/dataset_summary.json
```

Aggregates statistics across all metadata JSON files. Prints summary table to stdout.
Writes `dataset_summary.json` — the primary deliverable for the paper's dataset table.

---

## Data Directory Structure

```
data/
└── thermal/
    ├── 01_raw/
    │   ├── S3A_SL_1_RBT____20230601T...SEN3/   # extracted granule directories
    │   │   ├── F2_BT.nc
    │   │   ├── S8_BT.nc
    │   │   ├── geodetic_in.nc
    │   │   └── flags_in.nc
    │   └── acquisition_manifest.json
    │
    ├── 02_extracted/
    │   ├── S3A_SL_1_RBT____20230601T..._F2_BT.npy
    │   ├── S3A_SL_1_RBT____20230601T..._S8_BT.npy
    │   ├── S3A_SL_1_RBT____20230601T..._lat.npy
    │   ├── S3A_SL_1_RBT____20230601T..._lon.npy
    │   └── extraction_manifest.json
    │
    ├── 03_patches/
    │   ├── S3A_SL_1_RBT____20230601T..._p0000_bt.npy   # 160×120 BT patches [K]
    │   ├── S3A_SL_1_RBT____20230601T..._p0001_bt.npy
    │   └── patches_manifest.json
    │
    └── 04_dataset/
        ├── images/
        │   ├── S3A_SL_1_RBT____20230601T..._p0000_fixed_A.png   # uint8 grayscale
        │   └── ...                                                # one per patch × mode × profile
        ├── metadata/
        │   ├── S3A_SL_1_RBT____20230601T..._p0000_fixed_A.json
        │   └── ...
        ├── viz/                                                   # false-colour PNGs (optional)
        ├── label_manifest.json       # per-patch: label, granule_id, surface_type, split
        ├── balanced_manifest.json    # filtered to balanced subset
        ├── split_manifest.json       # {"train": [...], "val": [...], "test": [...]}
        ├── simulation_manifest.json
        └── dataset_summary.json      # paper dataset table source
```

**Image naming convention:** `{granule_id}_p{patch_index:04d}_{agc_mode}_{profile}.png`

---

## Reproducing the Dataset with a Given Seed

All stochastic operations (split assignment, class balancing, noise generation, FPN
generation) accept a `--seed` argument. To reproduce the full dataset exactly:

```bash
# Steps 1–3 are deterministic (no RNG involved)
python -m nina_thermal.pipelines.run_simulation --seed 42 ...
python -m nina_thermal.pipelines.run_split      --seed 42 ...
python -m nina_thermal.pipelines.run_balance    --seed 42 ...
```

The default seed is `42` (defined as `DEFAULT_SEED` in `nina_thermal/config.py`).
The pipeline version is recorded in every metadata JSON as `script_version` and in
`dataset_summary.json` under `processing_parameters.script_version`.

To verify reproducibility:

```bash
python -m pytest nina_thermal/tests/ -v
```

All 74 tests must pass. The test suite includes determinism assertions for split
assignment, class balancing, noise, and AGC.

---

## Package Architecture

Physical constants and sensor parameters are defined exclusively in `nina_thermal/config.py`.
Processing functions (`processing/`, `simulation/`, `labeling/`) are pure functions with
no file I/O or side effects. File I/O is confined to `dataset/` and `pipelines/`.
The `visualization/` module is isolated and never imported by any processing or simulation code.

See `docs/CLAUDE_CODE_BRIEF.md` for the complete scientific rationale and architectural
specification.
