# NINA Thermal Dataset Pipeline — Claude Code Implementation Brief

## How to use this document

Read this entire document before writing a single line of code. It contains
the complete scientific rationale, architectural decisions, implementation
specifications, and build order for the `nina_thermal` package. Every decision
recorded here has been reviewed and confirmed. Do not deviate from the
architecture without flagging the reason as a code comment.

When a decision is marked **CONFIRMED**, it is final. When a value is marked
**from TN-WKS-GSD-001** or **from datasheet Rev 200**, it is traceable to a
project document and must not be changed without updating that reference.

---

## 1. Mission context

**Mission:** WaskiriSat — Bolivia's first university-developed CubeSat.
**Institution:** Universidad Católica Boliviana "San Pablo" (UCB San Pablo)
**Program:** BIRDS / Kyushu Institute of Technology (KyuTech)
**Payload:** NINA — Neural Infrared Anomaly Observer

NINA is the AI+EO payload subsystem. It carries a FLIR Lepton 3.5 LWIR
microbolometer and a PAG7936 RGB sensor. The primary detection objective is
autonomous thermal anomaly detection for wildfire early warning over Bolivian
territory from 525 km SSO.

The pipeline in this repository constructs the pre-launch training dataset for
NINA's fire detection classifier. Because no prior flight data exists for this
instrument in this operational domain, the dataset is itself a scientific
contribution. The methodology must be defensible against expert peer review at
IAC 2026 and in subsequent journal publication.

---

## 2. The core scientific argument

The training dataset bridges two resolution domains:

- **Source:** Sentinel-3 SLSTR L1B thermal channels at 1,000 m/px nadir
  resolution. Calibrated brightness temperature, ESA-processed, full
  radiometric traceability.
- **Target:** FLIR Lepton 3.5 at 3,563 m/px GSD (TN-WKS-GSD-001),
  160×120 px, 8–14 µm LWIR broadband, NETD < 50 mK.

The methodology claim is: **we construct the most physically rigorous
pre-launch proxy training dataset achievable given the available calibrated
ground-truth sources, and we bound the simulation uncertainty explicitly.**

This is not a claim of flight equivalence. It is a claim of methodological
rigor and bounded uncertainty. Every pipeline design decision must support
one of these three properties:

1. Physical correctness — the operation is the right thing to do physically
2. Conservative bias — where approximations exist, errors favor harder
   detection (weaker signals in training than in flight)
3. Bounded uncertainty — where gaps exist, they are quantified and stated

---

## 3. Package structure

Create this exact directory and file structure. Do not add files not listed
here without a comment explaining why.

```
nina_thermal/
│
├── config.py                    # ALL physical constants and parameters
│
├── acquisition/
│   ├── __init__.py
│   ├── cdse_auth.py             # OAuth2 token management
│   ├── cdse_search.py           # CDSE OData catalog search
│   └── cdse_download.py         # download + .zip extraction
│
├── ingestion/
│   ├── __init__.py
│   ├── load_granule.py          # load_slstr_granule() — npy and NetCDF paths
│   ├── extract_channels.py      # extract_thermal_channels() from .SEN3
│   └── validate_granule.py      # pre-flight data quality checks
│
├── processing/
│   ├── __init__.py
│   ├── radiometry.py            # bt_to_radiance(), radiance_to_bt()
│   ├── spectral.py              # apply_broadband_correction()
│   ├── spatial.py               # aggregate_bt_to_lepton_gsd(),
│   │                            # tile_granule_to_lepton_patches()
│   └── agc.py                   # apply_agc(), compute_global_percentiles()
│
├── simulation/
│   ├── __init__.py
│   ├── noise.py                 # thermal noise, FPN generation + application
│   ├── optics.py                # PSF blur
│   └── sensor.py                # simulate_lepton() — orchestrates full sim
│
├── labeling/
│   ├── __init__.py
│   ├── fire_label.py            # compute_patch_label(),
│   │                            # classify_patch_surface_type()
│   └── split.py                 # granule_level_split(),
│                                # filter_records_by_split()
│
├── dataset/
│   ├── __init__.py
│   ├── balance.py               # build_balanced_dataset()
│   ├── export.py                # save images + metadata + manifests
│   └── report.py                # build_dataset_report()
│
├── visualization/
│   ├── __init__.py
│   └── colormap.py              # apply_colormap() — display only, never
│                                # imported by processing or simulation
│
├── validation/
│   ├── __init__.py
│   ├── metrics.py               # SNR, histogram stats
│   └── noise_validation.py      # PSD validation vs theoretical NETD
│
├── pipelines/
│   ├── __init__.py
│   ├── run_acquisition.py       # CLI: acquire SLSTR granules from CDSE
│   ├── run_extraction.py        # CLI: extract .npy from .SEN3
│   ├── run_processing.py        # CLI: Pass 1 — load, correct, aggregate, tile
│   ├── run_simulation.py        # CLI: Pass 2 — simulate, label, export
│   ├── run_split.py             # CLI: granule-level split assignment
│   ├── run_balance.py           # CLI: class balance
│   └── run_report.py            # CLI: dataset summary generation
│
└── tests/
    ├── __init__.py
    ├── test_radiometry.py
    ├── test_spatial.py
    ├── test_spectral.py
    ├── test_agc.py
    ├── test_fire_label.py
    ├── test_split.py
    ├── test_noise.py
    └── test_balance.py
```

---

## 4. Architectural rules — non-negotiable

**Rule 1 — config.py is the single source of truth.**
No module defines physical constants locally. Every constant is imported
from `nina_thermal.config`. If a value appears in two places, it is wrong.

**Rule 2 — Processing functions are pure.**
Functions in `processing/`, `simulation/`, and `labeling/` accept arrays
and parameters, return arrays, and have zero side effects. No file I/O,
no printing, no global state mutation. This makes every function
independently unit-testable.

**Rule 3 — visualization/ is isolated.**
`visualization/colormap.py` must never be imported by any module in
`processing/`, `simulation/`, `labeling/`, or `dataset/`. It imports
only from `config` and standard libraries. The false-color rendering
path is for human communication only — it is never in the model
training data path.

**Rule 4 — File I/O lives in dataset/ and pipelines/ only.**
`export.py` and the `run_*.py` scripts handle all disk reads and writes.
Processing modules never open files.

**Rule 5 — Metadata is the contract between pipeline stages.**
Each pipeline stage reads a manifest JSON from the previous stage's
output directory and writes an updated manifest to its own output
directory. No stage reads files written by a non-adjacent stage.

**Rule 6 — The pipeline is checkpoint-resumable.**
Each `run_*.py` script can be re-run independently without re-running
upstream stages. This requires that each stage reads only from its
`--input_dir` and writes only to its `--output_dir`, never modifying
its inputs.

**Rule 7 — All scripts are CLI-executable and importable.**
Every `pipelines/run_*.py` uses `argparse` and a `main()` function
protected by `if __name__ == "__main__"`. They can be run as:
`python -m nina_thermal.pipelines.run_processing --input_dir ...`

---

## 5. config.py — complete specification

This is the authoritative values table. Implement config.py exactly as
specified. All values are confirmed and traceable.

```python
# ── Physical constants ────────────────────────────────────────────────────
H_PLANCK    = 6.626e-34    # Planck constant [J·s]
C_LIGHT     = 3.0e8        # Speed of light [m/s]
K_BOLTZMANN = 1.381e-23    # Boltzmann constant [J/K]

# ── Sensor: FLIR Lepton 3.5 ───────────────────────────────────────────────
# Source: FLIR Lepton Engineering Datasheet Rev 200
LEPTON_NETD_MK           = 50.0        # Noise Equivalent Temperature Diff [mK]
LEPTON_RESOLUTION        = (160, 120)  # (width, height) [px]
LEPTON_PIXEL_PITCH_UM    = 12.0        # pixel pitch [µm]
LEPTON_FOCAL_LENGTH_MM   = 1.77        # derived focal length [mm]
LEPTON_FNUM              = 1.1         # f/number
LEPTON_SPECTRAL_MIN_UM   = 8.0         # spectral response lower bound [µm]
LEPTON_SPECTRAL_MAX_UM   = 14.0        # spectral response upper bound [µm]
LEPTON_SCENE_MAX_C       = 400.0       # maximum scene temperature [°C]
LEPTON_FRAME_RATE_HZ     = 8.6         # nominal frame rate [Hz]
LEPTON_FFC_INTERVAL_S    = 180.0       # approximate FFC cycle interval [s]
LEPTON_FFC_DURATION_S    = 45.0        # FFC event duration [s]

# ── Mission: NINA / WaskiriSat ────────────────────────────────────────────
# Source: TN-WKS-GSD-001 (full mathematical traceability in that document)
NINA_ALTITUDE_KM         = 525.0       # nominal SSO altitude [km]
NINA_LEPTON_GSD_M        = 3563.0      # operational GSD [m/px] — DO NOT ROUND
# Derivation: (12e-6 m × 525000 m) / 1.77e-3 m = 3,559 m/px
# Documented as 3,563 m/px in TN-WKS-GSD-001 after full traceability review

# ── Source: Sentinel-3 SLSTR L1B ─────────────────────────────────────────
# Source: ESA SLSTR Product Data Format Specification
SLSTR_PIXEL_SIZE_M       = 1000.0      # nadir thermal channel pixel size [m]
SLSTR_F2_WAVELENGTH_UM   = 10.85       # F2/S8 center wavelength [µm]
SLSTR_F2_WAVELENGTH_M    = SLSTR_F2_WAVELENGTH_UM * 1e-6  # [m] for Planck
SLSTR_VIEW               = "nadir"     # nadir view only — oblique not used

# Aggregation: SLSTR px per synthetic Lepton px
# 4 × 1000 m = 4000 m synthetic GSD
# vs 3563 m actual → +12.3% overestimation, conservative direction
SLSTR_AGG_FACTOR         = 4
SLSTR_SYNTHETIC_GSD_M    = SLSTR_PIXEL_SIZE_M * SLSTR_AGG_FACTOR  # 4000.0 m
SLSTR_GSD_OVERESTIMATION = (SLSTR_SYNTHETIC_GSD_M - NINA_LEPTON_GSD_M) \
                            / NINA_LEPTON_GSD_M  # ~0.123 = 12.3%

# ── Spectral broadband correction ─────────────────────────────────────────
# First-order linear correction from 10.85 µm narrowband to 8-14 µm broadband
# alpha = 0.015: ~1.5% BT increase per 100 K above ambient
# Effect: <2 K at ambient temperatures, ~6-8 K at 400 K fire pixels
# Acknowledged as approximation in paper — full SRF unavailable
BROADBAND_ALPHA          = 0.015
AMBIENT_REFERENCE_K      = 295.0       # reference ambient temperature [K]

# ── AGC: Fixed mode — CONFIRMED as flight inference configuration ──────────
# Physical justification in paper:
#   Lower bound 258.15 K: below min Bolivian surface temp at 525 km
#   Upper bound 358.15 K: covers full ambient range incl. hot dry season
#   100 K range → quantization step 100/255 ≈ 0.39 K/uint8 step
#   Fire pixels saturate near 1.0 → correct behavior for detection
FLIGHT_AGC_MODE          = "fixed"
FIXED_TEMP_MIN_C         = -15.0
FIXED_TEMP_MAX_C         =  85.0
FIXED_TEMP_MIN_K         = FIXED_TEMP_MIN_C + 273.15   # 258.15 K
FIXED_TEMP_MAX_K         = FIXED_TEMP_MAX_C + 273.15   # 358.15 K
FIXED_TEMP_RANGE_K       = FIXED_TEMP_MAX_K - FIXED_TEMP_MIN_K  # 100.0 K
AGC_QUANTIZATION_K       = FIXED_TEMP_RANGE_K / 255.0  # ~0.392 K/step

# ── Degradation profiles ──────────────────────────────────────────────────
# Profile A: nominal NETD, minimal blur — approximates post-FFC state
# Profile B: 2× NETD, moderate — approximates mid-FFC-cycle state
# Profile C: 4× NETD, heavy — approximates pre-FFC / SAA transit worst case
# The FFC temporal framing is documented in the paper methodology section
DEGRADATION_PROFILES = {
    "A": {
        "thermal_noise_factor":  1.0,
        "optical_blur_sigma_px": 0.5,
        "fpn_column_strength":   0.008,
        "fpn_pixel_strength":    0.012,
        "fpn_factor":            None,
        "ffc_phase":             "post-FFC",
        "description":           "Best case — nominal NETD, minimal blur"
    },
    "B": {
        "thermal_noise_factor":  2.0,
        "optical_blur_sigma_px": 0.8,
        "fpn_column_strength":   0.012,
        "fpn_pixel_strength":    0.020,
        "fpn_factor":            0.8,
        "ffc_phase":             "mid-cycle",
        "description":           "Typical — 2x NETD, moderate blur and FPN"
    },
    "C": {
        "thermal_noise_factor":  4.0,
        "optical_blur_sigma_px": 1.2,
        "fpn_column_strength":   0.018,
        "fpn_pixel_strength":    0.035,
        "fpn_factor":            1.5,
        "ffc_phase":             "pre-FFC / SAA transit",
        "description":           "SAA worst case — 4x NETD, heavy blur, FPN"
    }
}

# ── Dataset construction ──────────────────────────────────────────────────
# Fire ratio 1:4 justified by asymmetric detection cost:
# missed fire >> false alarm in disaster response context
FIRE_RATIO               = 0.20        # fire fraction = 1/(1+4)
TRAIN_FRAC               = 0.70
VAL_FRAC                 = 0.15
TEST_FRAC                = 0.15
MAX_NAN_FRACTION         = 0.15        # max masked px fraction to accept patch
HIGH_FIRE_GRANULE_THRESH = 0.05        # fire fraction tier for stratified split

# ── Fire labeling thresholds ──────────────────────────────────────────────
# Dual criterion: pixel must satisfy BOTH to count as fire
# min_fire_fraction: minimum fraction of fire pixels for patch label = 1
# Allows subpixel fires at 3563 m GSD to produce positive labels
FIRE_DELTA_T_THRESHOLD_K = 8.0         # delta_T = F2_BT - S8_BT [K]
FIRE_F2_ABS_THRESHOLD_K  = 320.0       # absolute F2_BT floor [K]
MIN_FIRE_FRACTION        = 0.005       # 0.5% of patch pixels
WARM_SURFACE_THRESH_K    = 310.0       # threshold for warm_surface label

# ── Visualization ─────────────────────────────────────────────────────────
# For STEM communication and paper figures ONLY
# Never used in model training data path
DEFAULT_COLORMAP         = "inferno"
COLORMAPS_AVAILABLE      = ["inferno", "iron", "rainbow", "grayscale"]

# ── Pipeline paths (defaults, overridable via CLI) ────────────────────────
DEFAULT_DATA_ROOT        = "./data"
DIR_RAW                  = "01_raw"
DIR_EXTRACTED            = "02_extracted"
DIR_PATCHES              = "03_patches"
DIR_DATASET              = "04_dataset"

# ── Reproducibility ───────────────────────────────────────────────────────
DEFAULT_SEED             = 42
SCRIPT_VERSION           = "1.0.0"
```

---

## 6. Module specifications

### 6.1 processing/radiometry.py

Two pure functions. No imports from any other nina_thermal module except config.

**`bt_to_radiance(bt_k, wavelength_m) -> np.ndarray`**
- Converts brightness temperature array [K] to spectral radiance
  [W sr⁻¹ m⁻² m⁻¹] using Planck's law
- NaN inputs must produce NaN outputs
- Uses H_PLANCK, C_LIGHT, K_BOLTZMANN from config
- No clipping — caller is responsible for valid input range

**`radiance_to_bt(L, wavelength_m) -> np.ndarray`**
- Inverse of bt_to_radiance
- NaN and non-positive L must produce NaN outputs
- Must handle overflow in exp() gracefully using np.errstate

**Test requirement:** `test_radiometry.py` must verify:
1. Round-trip: `radiance_to_bt(bt_to_radiance(T, lam), lam) ≈ T` for
   T in [250, 400] K at SLSTR_F2_WAVELENGTH_M
2. Known value: at T=300 K, λ=10.85 µm, L ≈ 9.88 W sr⁻¹ m⁻² µm⁻¹
   (verify against published Planck tables)
3. NaN propagation: NaN input → NaN output

---

### 6.2 processing/spectral.py

**`apply_broadband_correction(bt_k, ambient_k=None, alpha=None) -> np.ndarray`**
- Defaults: ambient_k=AMBIENT_REFERENCE_K, alpha=BROADBAND_ALPHA
- Formula: BT_corrected = BT + alpha × max(BT − ambient_k, 0)
- NaN inputs → NaN outputs
- Effect is zero for BT <= ambient_k (ambient pixels unchanged)
- Effect grows linearly for BT > ambient_k (fire pixels upward adjusted)

**Test requirement:** verify correction is zero at ambient, positive for
fire temperatures, NaN-safe.

---

### 6.3 processing/spatial.py

**`aggregate_bt_to_lepton_gsd(bt_array, agg_factor=None, wavelength_m=None) -> np.ndarray`**

CRITICAL: This function must use radiance-consistent aggregation,
NOT direct temperature averaging. Reason: sensors integrate radiance
over their footprint; temperature averaging underestimates the
contribution of hot sub-pixel fire targets.

Algorithm:
1. Crop bt_array to exact multiple of agg_factor in both dimensions
2. Convert entire BT array to radiance via bt_to_radiance()
3. Reshape to (H//agg, agg, W//agg, agg)
4. nanmean over axes (1, 3) → shape (H//agg, W//agg)
5. Convert mean radiance back to BT via radiance_to_bt()
6. Return float32

Defaults: agg_factor=SLSTR_AGG_FACTOR, wavelength_m=SLSTR_F2_WAVELENGTH_M

**`tile_granule_to_lepton_patches(bt_array, agg_factor=None, lepton_w=None, lepton_h=None, max_nan_fraction=None) -> List[np.ndarray]`**

Algorithm:
1. Call aggregate_bt_to_lepton_gsd() first
2. Tile the aggregated array into (lepton_h, lepton_w) patches
   with stride = full patch size (no overlap)
3. For each patch:
   a. Compute nan_fraction = np.isnan(patch).mean()
   b. Reject if nan_fraction > max_nan_fraction
   c. If nan_fraction > 0 (but <= threshold):
      fill NaN with patch nanmedian
4. Return list of float32 patches

Defaults from config: lepton_w=LEPTON_RESOLUTION[0]=160,
lepton_h=LEPTON_RESOLUTION[1]=120, max_nan_fraction=MAX_NAN_FRACTION

**Test requirement:** verify radiance-consistent aggregation produces
higher output BT than temperature averaging for a synthetic fire pixel
(one hot pixel surrounded by ambient). Quantify difference and assert
it is > 0.

---

### 6.4 processing/agc.py

**`apply_agc(bt_patch, t_min, t_max) -> np.ndarray`**
- Returns float32 array clipped to [0, 1]
- Formula: clip((BT - t_min) / (t_max - t_min), 0, 1)

**`compute_global_percentiles(all_patches, percentiles=(2.0, 98.0), max_samples_per_patch=5000, seed=None) -> Tuple[float, float]`**
- Samples up to max_samples_per_patch values per patch
- Returns (T_low, T_high) in Kelvin

**`agc_to_uint8(normalized) -> np.ndarray`**
- Converts float32 [0,1] to uint8 [0,255]
- Formula: np.clip(np.round(normalized * 255), 0, 255).astype(np.uint8)
- This is the final step before model input — must be lossless relative
  to the normalized representation at uint8 precision

---

### 6.5 ingestion/load_granule.py

**`load_slstr_granule(granule_dir) -> Optional[Dict[str, np.ndarray]]`**

Two execution paths — select automatically:

**Path A (fast) — .npy files present:**
- Detect via: `list(granule_dir.glob("*_F2_BT.npy"))`
- Extract prefix from filename stem
- Load: {prefix}_F2_BT.npy, {prefix}_S8_BT.npy,
        {prefix}_lat.npy, {prefix}_lon.npy
- Derive: delta_T = F2_BT - S8_BT
- No masking step — masks applied during extraction

**Path B (fallback) — .SEN3 NetCDF:**
- Locate .SEN3 directory (may be granule_dir itself or a subdirectory)
- Open F2_BT.nc, S8_BT.nc: read variable, cast float32,
  replace _FillValue with NaN
- Open geodetic_in.nc: read latitude_in, longitude_in
- Open flags_in.nc: build combined mask:
    cloud_mask = (cloud_in & 1).astype(bool)
    exception_mask = (confidence_in & 2).astype(bool)
    combined = cloud_mask | exception_mask
- Apply: F2_BT[combined] = NaN, S8_BT[combined] = NaN
- Derive: delta_T = F2_BT - S8_BT

**Output dict (both paths identical):**
```python
{
    "F2_BT":   np.ndarray float32 (rows, cols) [K],
    "S8_BT":   np.ndarray float32 (rows, cols) [K],
    "delta_T": np.ndarray float32 (rows, cols) [K],
    "lat":     np.ndarray float32 (rows, cols) [degrees],
    "lon":     np.ndarray float32 (rows, cols) [degrees],
}
```

Return None on any failure. Log the specific failure reason.

---

### 6.6 labeling/fire_label.py

**`compute_patch_label(f2_bt_patch, delta_t_patch, fire_delta_t_threshold_k=None, fire_f2_abs_threshold_k=None, min_fire_fraction=None) -> int`**

Dual criterion — pixel counts as fire if BOTH:
- delta_T > FIRE_DELTA_T_THRESHOLD_K
- F2_BT > FIRE_F2_ABS_THRESHOLD_K
- pixel is finite (np.isfinite)

Label = 1 if fire_pixels.mean() >= MIN_FIRE_FRACTION else 0

Physical justification: dual criterion prevents warm-but-uniform
surfaces (hot desert, bare rock) from triggering fire labels.
The fraction threshold allows subpixel fires at 3563 m GSD to produce
positive labels.

**`classify_patch_surface_type(f2_bt_patch, delta_t_patch, label) -> str`**

Returns one of: "fire", "warm_surface", "nominal"
- "fire": label == 1
- "warm_surface": label == 0 AND f2_bt_patch.max() > WARM_SURFACE_THRESH_K
- "nominal": label == 0 AND no anomalously warm pixels

Purpose: enables reporting of label ambiguity in paper methodology section.
Warm surface patches represent the primary source of false positive risk
and must be tracked separately in the dataset report.

---

### 6.7 labeling/split.py

**`granule_level_split(granule_ids_with_labels, train_frac=None, val_frac=None, test_frac=None, seed=None) -> Dict[str, List[str]]`**

Input: dict mapping granule_id -> list of int labels (0/1) for that granule

CRITICAL: Split must be at GRANULE level, never at patch level.
Patches from the same granule share atmospheric conditions, sun angle,
and land surface state. Patch-level splitting produces optimistic
validation metrics due to spatial autocorrelation.

Algorithm:
1. Compute fire fraction per granule
2. Stratify granules into high-fire (> HIGH_FIRE_GRANULE_THRESH)
   and low-fire tiers
3. Shuffle each tier with seeded RNG
4. Assign whole granules to train/val/test in target fractions
5. Ensure each split receives granules from both tiers

Returns: {"train": [granule_ids], "val": [granule_ids], "test": [granule_ids]}

Print summary: granule count, patch count, fire fraction per split.

---

### 6.8 dataset/balance.py

**`build_balanced_dataset(records, fire_ratio=None, seed=None) -> List[Dict]`**

Input: list of dicts with minimum keys: 'label' (int), 'granule_id' (str),
       'patch' (str)

Strategy: undersample no-fire to reach target ratio. Fire patches are
never removed. Applied AFTER split assignment so balance is applied
within each split independently (caller is responsible for calling
this per split, not on the full dataset).

Formula: n_nofire_target = round(n_fire × (1 - fire_ratio) / fire_ratio)
         capped at available no-fire supply

Prints: original counts, target counts, actual ratio achieved.

---

### 6.9 simulation/noise.py

**`calculate_thermal_noise_sigma(netd_mk=None, agc_range_k=None) -> float`**
- sigma = (netd_mk / 1000.0) / agc_range_k
- Returns noise sigma in normalized [0,1] BT space

**`add_thermal_noise(bt_normalized, sigma, rng) -> np.ndarray`**
- Add Gaussian noise to float32 normalized BT array
- clip output to [0, 1]
- Must be called BEFORE agc_to_uint8() and BEFORE colormap rendering
  Physical reason: microbolometer NETD is a temperature uncertainty,
  not a pixel value uncertainty. Adding noise after colormap creates
  noise that varies with colormap gradient — physically incorrect.

**`generate_lepton_fpn(resolution=None, column_strength=0.0, pixel_strength=0.0, seed=42) -> Dict[str, np.ndarray]`**
- Two-component FPN model matching VOx microbolometer readout architecture
- column_fpn: shape (H, W) — one offset per column, broadcast over rows
  Models column-level readout amplifier variation (dominant artifact)
- pixel_fpn: shape (H, W) — independent per-pixel random offset
  Models residual non-uniformity after factory NUC
- Returns {"column_fpn": ndarray, "pixel_fpn": ndarray}
- Seed must be deterministic for reproducibility

**`apply_fpn(img_float, fpn, factor) -> np.ndarray`**
- Apply combined column + pixel FPN to float32 [0,1] image
- combined = (fpn["column_fpn"] + fpn["pixel_fpn"]) × factor
- clip output to [0, 1]
- Must be called AFTER agc_to_uint8() cast (applied to float32 then re-cast)

---

### 6.10 simulation/optics.py

**`apply_psf_blur(img, sigma_px) -> np.ndarray`**
- Gaussian blur approximating Lepton f/1.1 optical PSF
- ksize = int(2 × round(2 × sigma_px) + 1)  — must be odd
- Return same dtype as input
- If sigma_px <= 0: return input unchanged

---

### 6.11 simulation/sensor.py

**`simulate_lepton(bt_normalized, profile, agc_range_k, fpn, rng) -> np.ndarray`**

Orchestrates the complete sensor simulation. ORDER OF OPERATIONS IS
PHYSICALLY MOTIVATED AND MUST NOT BE CHANGED:

```
Step 1: add thermal noise in BT space (float32 [0,1])
        → noise is a temperature uncertainty, must precede rendering
Step 2: agc_to_uint8() cast → uint8 [0,255]
Step 3: apply_psf_blur() → uint8
        → optical blur applied to rendered pixel values
Step 4: if profile has fpn_factor:
            cast to float32, apply_fpn(), cast back to uint8
        → FPN is a readout artifact, applied after optical path
```

Returns uint8 (H, W) single-channel grayscale image.
Profile parameters come from DEGRADATION_PROFILES[profile].

---

### 6.12 visualization/colormap.py

**`apply_colormap(uint8_image, colormap_name=None) -> np.ndarray`**

Converts uint8 grayscale (H, W) to uint8 RGB (H, W, 3) via matplotlib
colormap. Returns 8-bit RGB for display, paper figures, and STEM
communication.

ARCHITECTURE NOTE: This function is the ONLY place false-color
rendering occurs. It is called:
- By pipeline scripts for generating sample visualization images
- By paper figure generation scripts
- NEVER by simulate_lepton(), NEVER by any training data export

colormap_name defaults to DEFAULT_COLORMAP from config.
Supports all values in COLORMAPS_AVAILABLE.

---

### 6.13 dataset/export.py

**`save_patch_image(img_uint8, path) -> None`**
- Save (H, W) uint8 grayscale as PNG
- No color conversion — single channel saved directly

**`save_patch_metadata(metadata_dict, path) -> None`**
- Save JSON with indent=2

**`build_image_metadata(granule_id, patch_index, label, surface_type, agc_mode, profile, t_min, t_max, agc_range_k, nan_fraction, centroid_lat, centroid_lon, split, seed) -> Dict`**
- Assembles the complete per-image metadata dict
- Must include "provenance" list documenting the transformation chain
- Must include "flight_config": bool (True when agc_mode == FLIGHT_AGC_MODE)
- Must include "script_version" from config.SCRIPT_VERSION

**`save_label_manifest(records, path) -> None`**
- Save list of {"patch": str, "label": int, "granule_id": str,
  "surface_type": str, "split": str} as JSON

---

### 6.14 dataset/report.py

**`build_dataset_report(dataset_dir) -> Dict`**

Reads all metadata JSON files and label_manifest.json.
Produces dataset_summary.json containing:

```python
{
    "total_granules": int,
    "total_patches_before_rejection": int,
    "total_patches_after_rejection": int,
    "rejection_rate": float,
    "splits": {
        "train": {"granules": int, "patches": int, "fire_fraction": float},
        "val":   {"granules": int, "patches": int, "fire_fraction": float},
        "test":  {"granules": int, "patches": int, "fire_fraction": float},
    },
    "label_distribution": {
        "fire": int, "no_fire": int, "fire_fraction": float
    },
    "surface_types": {
        "fire": int, "warm_surface": int, "nominal": int
    },
    "agc_modes": {"fixed": int, "global": int, "auto": int},
    "profiles": {"A": int, "B": int, "C": int},
    "bt_statistics": {
        "fire_patches":    {"mean_k": float, "std_k": float,
                            "p2_k": float, "p98_k": float},
        "nofire_patches":  {"mean_k": float, "std_k": float,
                            "p2_k": float, "p98_k": float},
    },
    "geographic_coverage": {
        "lat_min": float, "lat_max": float,
        "lon_min": float, "lon_max": float,
        "patch_centroids": [{"lat": float, "lon": float, "label": int}]
    },
    "temporal_coverage": {
        "granule_dates": [str],
        "date_min": str,
        "date_max": str,
    },
    "processing_parameters": {
        "agg_factor": int,
        "synthetic_gsd_m": float,
        "nina_lepton_gsd_m": float,
        "gsd_overestimation_pct": float,
        "fixed_agc_range_k": float,
        "fire_ratio_target": float,
        "seed": int,
        "script_version": str
    }
}
```

This report is a primary deliverable for the paper's dataset contribution
section. Every number in the paper's dataset table comes from this file.

---

## 7. Pipeline scripts specification

### 7.1 run_acquisition.py

```
CLI args:
  --roi          Path to Bolivia AOI GeoJSON
  --start        Start date YYYYMMDD
  --end          End date YYYYMMDD
  --output_dir   Directory for raw .zip/.SEN3 files
  --max_products Max products to download (default 50)
  --product_type SLSTR L1B product type (default SL_1_RBT___)

Actions:
  1. Authenticate with CDSE (reads CDSE_USER, CDSE_PASSWORD from .env)
  2. Search catalog with SENTINEL-3 + SL_1_RBT___ + AOI + date range
  3. Download and extract each product
  4. Write acquisition_manifest.json to output_dir listing all
     downloaded granule directories
```

### 7.2 run_extraction.py

```
CLI args:
  --input_dir   Directory containing .SEN3 directories
  --output_dir  Directory for extracted .npy files

Actions:
  1. For each .SEN3 directory:
     a. Call extract_thermal_channels()
     b. Save: {prefix}_F2_BT.npy, {prefix}_S8_BT.npy,
              {prefix}_lat.npy, {prefix}_lon.npy
  2. Write extraction_manifest.json listing all extracted granules
```

### 7.3 run_processing.py  — Pass 1

```
CLI args:
  --input_dir   Directory with .npy files (output of run_extraction)
  --output_dir  Directory for patch data

Actions:
  1. For each granule:
     a. load_slstr_granule()
     b. apply_broadband_correction() on F2_BT
     c. aggregate_bt_to_lepton_gsd() — radiance-consistent
     d. tile_granule_to_lepton_patches() — with NaN handling
     e. For each valid patch:
        - record granule_id, patch_index, nan_fraction,
          centroid_lat, centroid_lon, bt_min, bt_max
  2. Write patches_manifest.json with one record per patch
  3. Save aggregated BT patches as .npy files in output_dir
     naming: {granule_id}_p{patch_index:04d}_bt.npy
```

### 7.4 run_simulation.py  — Pass 2

```
CLI args:
  --input_dir   Directory with patch .npy files + patches_manifest.json
  --output_dir  Root dataset directory
  --profiles    Comma-separated e.g. "A,B,C"
  --agc_modes   Comma-separated e.g. "fixed,global,auto"
  --seed        Random seed (default 42)
  --validation  Flag: generate validation metrics

Actions:
  1. Load patches_manifest.json
  2. Compute global BT percentiles across all patches (for global AGC)
  3. Generate FPN maps per profile (deterministic, from seed)
  4. For each patch:
     a. compute_patch_label() on raw BT
     b. classify_patch_surface_type()
     c. For each agc_mode:
        - apply_agc()
        - For each profile:
          * simulate_lepton()
          * save_patch_image() as grayscale PNG
          * save_patch_metadata()
          * optionally: save_patch_image(apply_colormap(...)) as
            visualization PNG in separate viz/ subdirectory
  5. save_label_manifest()
  6. Write simulation_manifest.json
```

### 7.5 run_split.py

```
CLI args:
  --dataset_dir  Root dataset directory (output of run_simulation)
  --seed         Random seed (default 42)

Actions:
  1. Read label_manifest.json
  2. Build granule -> labels mapping
  3. granule_level_split()
  4. Update each patch's metadata JSON with "split" field
  5. Write split_manifest.json:
     {"train": [granule_ids], "val": [granule_ids], "test": [granule_ids]}
```

### 7.6 run_balance.py

```
CLI args:
  --dataset_dir  Root dataset directory
  --fire_ratio   Target fire fraction (default 0.20)
  --seed         Random seed (default 42)

Actions:
  1. Read split_manifest.json and label_manifest.json
  2. For each split independently:
     a. Filter records to this split's granules
     b. build_balanced_dataset()
     c. Mark excluded records in metadata as "excluded_by_balance": true
  3. Write balanced_manifest.json with only the retained records
  NOTE: never deletes image files — balance is a manifest filter only
```

### 7.7 run_report.py

```
CLI args:
  --dataset_dir   Root dataset directory
  --output        Output path for dataset_summary.json

Actions:
  1. build_dataset_report()
  2. Print summary table to stdout
  3. Save dataset_summary.json
```

---

## 8. Test specifications

All tests in `tests/` use pytest. No test writes to disk except to a
`tmp_path` fixture. All tests use synthetic numpy arrays — no real
satellite data required.

**test_radiometry.py:**
- Round-trip BT → radiance → BT within 0.01 K for T in [250, 400]
- Known Planck value at 300 K, 10.85 µm
- NaN propagation in both directions

**test_spatial.py:**
- Radiance-consistent aggregation > temperature averaging for fire pixel
  Synthetic: 4×4 block, one pixel at 380 K, 15 at 295 K
  Assert: radiance_agg_BT > temperature_avg_BT
  Assert: difference > 1.0 K (physically expected ~1.8 K)
- Output shape correct: (H//4, W//4) for H=2400, W=1500
- NaN handling: block with all NaN → NaN output pixel

**test_spectral.py:**
- Correction is zero at ambient temperature (295 K)
- Correction is positive for fire temperature (380 K)
- Correction is NaN-safe

**test_agc.py:**
- Values within range map to [0, 1]
- Values below T_min clip to 0.0
- Values above T_max clip to 1.0
- uint8 cast is lossless at uint8 precision

**test_fire_label.py:**
- All ambient patch → label 0
- Patch with >0.5% fire pixels meeting dual criterion → label 1
- Patch with warm pixels below delta_T threshold → label 0, surface_type warm_surface
- NaN-safe: patch with NaN pixels does not crash

**test_split.py:**
- No granule appears in more than one split
- Each split contains granules from both fire-fraction tiers
- Fractions sum to 1.0 within rounding
- Deterministic with same seed

**test_noise.py:**
- Thermal noise sigma scales correctly with NETD and AGC range
- FPN column structure: std along rows > std along columns
  (column FPN produces vertical stripes — std across rows should be
  dominated by inter-column variation)
- apply_fpn clips to [0, 1]

**test_balance.py:**
- Fire patches never removed
- Output fire fraction within 1% of target
- Deterministic with same seed

---

## 9. Implementation order

Implement in this exact order. Each layer depends only on layers below.
Run tests for each module before implementing the next.

```
1.  nina_thermal/config.py
2.  nina_thermal/processing/radiometry.py  + tests/test_radiometry.py
3.  nina_thermal/processing/spectral.py    + tests/test_spectral.py
4.  nina_thermal/processing/spatial.py     + tests/test_spatial.py
5.  nina_thermal/processing/agc.py         + tests/test_agc.py
6.  nina_thermal/ingestion/load_granule.py
7.  nina_thermal/ingestion/extract_channels.py
8.  nina_thermal/ingestion/validate_granule.py
9.  nina_thermal/labeling/fire_label.py    + tests/test_fire_label.py
10. nina_thermal/labeling/split.py         + tests/test_split.py
11. nina_thermal/simulation/noise.py       + tests/test_noise.py
12. nina_thermal/simulation/optics.py
13. nina_thermal/simulation/sensor.py
14. nina_thermal/dataset/balance.py        + tests/test_balance.py
15. nina_thermal/dataset/export.py
16. nina_thermal/dataset/report.py
17. nina_thermal/visualization/colormap.py
18. nina_thermal/validation/metrics.py
19. nina_thermal/validation/noise_validation.py
20. nina_thermal/acquisition/cdse_auth.py
21. nina_thermal/acquisition/cdse_search.py
22. nina_thermal/acquisition/cdse_download.py
23. nina_thermal/pipelines/run_acquisition.py
24. nina_thermal/pipelines/run_extraction.py
25. nina_thermal/pipelines/run_processing.py
26. nina_thermal/pipelines/run_simulation.py
27. nina_thermal/pipelines/run_split.py
28. nina_thermal/pipelines/run_balance.py
29. nina_thermal/pipelines/run_report.py
```

---

## 10. Dependencies

```
# requirements.txt
numpy>=1.24
scipy>=1.10
opencv-python>=4.8
matplotlib>=3.7
rasterio>=1.3
netCDF4>=1.6
tqdm>=4.65
python-dotenv>=1.0
requests>=2.31
sentinelsat>=1.3    # used only for geojson_to_wkt utility
pytest>=7.4
geopandas>=0.13     # optional, for spatial validation
```

---

## 11. What NOT to implement

These items are explicitly out of scope for this codebase:

- Any Landsat processing (superseded — referenced in paper as prior art only)
- RGB three-channel model inputs (confirmed decision: uint8 grayscale only)
- Model training code (separate repository)
- Inference pipeline (flight software, MicroPython on STM32N6)
- Any hardcoded file paths (all paths via CLI args or config defaults)
- Any internet requests outside acquisition/ modules
- Any matplotlib display calls (plt.show()) — save to file or return array only

---

## 12. Key paper claims this pipeline must support

Every pipeline decision maps to a paper claim. These are the claims
that must be verifiable from the code and the dataset_summary.json:

1. "Spatial aggregation uses radiance-consistent averaging"
   → aggregate_bt_to_lepton_gsd() in spatial.py

2. "Synthetic GSD of 4,000 m represents a conservative 12.3%
   overestimation of the 3,563 m operational pixel footprint"
   → SLSTR_SYNTHETIC_GSD_M and SLSTR_GSD_OVERESTIMATION in config.py

3. "Thermal noise is added in brightness temperature space prior
   to rendering, not in pixel value space"
   → simulate_lepton() order of operations in sensor.py

4. "Fixed AGC range of 258.15–358.15 K is the designated flight
   inference configuration"
   → FLIGHT_AGC_MODE and FIXED_TEMP_* in config.py
   → "flight_config": true field in image metadata

5. "Dataset splits are assigned at granule level to prevent
   spatial autocorrelation leakage"
   → granule_level_split() in split.py

6. "Class balance of 1:4 fire:no-fire reflects asymmetric
   detection cost structure"
   → build_balanced_dataset() in balance.py

7. "Label ambiguity from warm non-fire surfaces is characterized
   and reported"
   → classify_patch_surface_type() in fire_label.py
   → surface_types field in dataset_summary.json

8. "FPN model includes column and pixel components consistent with
   VOx microbolometer readout architecture"
   → generate_lepton_fpn() in noise.py

9. "Three degradation profiles bound the simulation uncertainty
   envelope from post-FFC best case to SAA transit worst case"
   → DEGRADATION_PROFILES in config.py with ffc_phase documentation

10. "Dataset provenance is fully traceable from SLSTR granule ID
    to individual training sample"
    → "provenance" chain in per-image metadata from export.py
```
