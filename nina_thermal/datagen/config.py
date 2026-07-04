from typing import Any, Dict, List

# ── Physical constants ────────────────────────────────────────────────────
H_PLANCK = 6.626e-34  # Planck constant [J·s]
C_LIGHT = 3.0e8  # Speed of light [m/s]
K_BOLTZMANN = 1.381e-23  # Boltzmann constant [J/K]

# ── Sensor: FLIR Lepton 3.5 ───────────────────────────────────────────────
# Source: FLIR Lepton Engineering Datasheet Rev 200
LEPTON_NETD_MK = 50.0  # Noise Equivalent Temperature Diff [mK]
LEPTON_RESOLUTION = (160, 120)  # (width, height) [px]
LEPTON_PIXEL_PITCH_UM = 12.0  # pixel pitch [µm]
LEPTON_FOCAL_LENGTH_MM = 1.77  # derived focal length [mm]
LEPTON_FNUM = 1.1  # f/number
LEPTON_SPECTRAL_MIN_UM = 8.0  # spectral response lower bound [µm]
LEPTON_SPECTRAL_MAX_UM = 14.0  # spectral response upper bound [µm]
LEPTON_SCENE_MAX_C = 400.0  # maximum scene temperature [°C]
LEPTON_FRAME_RATE_HZ = 8.6  # nominal frame rate [Hz]
LEPTON_FFC_INTERVAL_S = 180.0  # approximate FFC cycle interval [s]
LEPTON_FFC_DURATION_S = 45.0  # FFC event duration [s]

# ── Mission: NINA / WaskiriSat ────────────────────────────────────────────
# Source: TN-WKS-GSD-001 (full mathematical traceability in that document)
NINA_ALTITUDE_KM = 525.0  # nominal SSO altitude [km]
NINA_LEPTON_GSD_M = 3563.0  # operational GSD [m/px] — DO NOT ROUND
# Derivation: (12e-6 m × 525000 m) / 1.77e-3 m = 3,559 m/px
# Documented as 3,563 m/px in TN-WKS-GSD-001 after full traceability review

# ── Source: Sentinel-3 SLSTR L1B ─────────────────────────────────────────
# Source: ESA SLSTR Product Data Format Specification
SLSTR_PIXEL_SIZE_M = 1000.0  # nadir thermal channel pixel size [m]
SLSTR_F2_WAVELENGTH_UM = 10.85  # F2/S8 center wavelength [µm]
SLSTR_F2_WAVELENGTH_M = SLSTR_F2_WAVELENGTH_UM * 1e-6  # [m] for Planck
SLSTR_VIEW = "nadir"  # nadir view only — oblique not used

# Aggregation: SLSTR px per synthetic Lepton px
# 4 × 1000 m = 4000 m synthetic GSD
# vs 3563 m actual → +12.3% overestimation, conservative direction
SLSTR_AGG_FACTOR = 4
SLSTR_SYNTHETIC_GSD_M = SLSTR_PIXEL_SIZE_M * SLSTR_AGG_FACTOR  # 4000.0 m
SLSTR_GSD_OVERESTIMATION = (
    SLSTR_SYNTHETIC_GSD_M - NINA_LEPTON_GSD_M
) / NINA_LEPTON_GSD_M  # ~0.123 = 12.3%

# ── Spectral broadband correction ─────────────────────────────────────────
# First-order linear correction from 10.85 µm narrowband to 8-14 µm broadband
# alpha = 0.015: ~1.5% BT increase per 100 K above ambient
# Effect: <2 K at ambient temperatures, ~6-8 K at 400 K fire pixels
# Acknowledged as approximation in paper — full SRF unavailable
BROADBAND_ALPHA = 0.015
AMBIENT_REFERENCE_K = 295.0  # reference ambient temperature [K]

# ── AGC: Fixed mode — CONFIRMED as flight inference configuration ──────────
# Physical justification in paper:
#   Lower bound 258.15 K: below min Bolivian surface temp at 525 km
#   Upper bound 358.15 K: covers full ambient range incl. hot dry season
#   100 K range → quantization step 100/255 ≈ 0.39 K/uint8 step
#   Fire pixels saturate near 1.0 → correct behavior for detection
FLIGHT_AGC_MODE = "fixed"
VALID_AGC_MODES: List[str] = ["fixed", "global", "auto"]
FIXED_TEMP_MIN_C = -15.0
FIXED_TEMP_MAX_C = 85.0
FIXED_TEMP_MIN_K = FIXED_TEMP_MIN_C + 273.15  # 258.15 K
FIXED_TEMP_MAX_K = FIXED_TEMP_MAX_C + 273.15  # 358.15 K
FIXED_TEMP_RANGE_K = FIXED_TEMP_MAX_K - FIXED_TEMP_MIN_K  # 100.0 K
AGC_QUANTIZATION_K = FIXED_TEMP_RANGE_K / 255.0  # ~0.392 K/step

# ── Degradation profiles ──────────────────────────────────────────────────
# Profile A: nominal NETD, minimal blur — approximates post-FFC state
# Profile B: 2× NETD, moderate — approximates mid-FFC-cycle state
# Profile C: 4× NETD, heavy — approximates pre-FFC / SAA transit worst case
# The FFC temporal framing is documented in the paper methodology section
DEGRADATION_PROFILES: Dict[str, Dict[str, Any]] = {
    "A": {
        "thermal_noise_factor": 1.0,
        "optical_blur_sigma_px": 0.5,
        "fpn_column_strength": 0.008,
        "fpn_pixel_strength": 0.012,
        "fpn_factor": None,
        "ffc_phase": "post-FFC",
        "description": "Best case — nominal NETD, minimal blur",
    },
    "B": {
        "thermal_noise_factor": 2.0,
        "optical_blur_sigma_px": 0.8,
        "fpn_column_strength": 0.012,
        "fpn_pixel_strength": 0.020,
        "fpn_factor": 0.8,
        "ffc_phase": "mid-cycle",
        "description": "Typical — 2x NETD, moderate blur and FPN",
    },
    "C": {
        "thermal_noise_factor": 4.0,
        "optical_blur_sigma_px": 1.2,
        "fpn_column_strength": 0.018,
        "fpn_pixel_strength": 0.035,
        "fpn_factor": 1.5,
        "ffc_phase": "pre-FFC / SAA transit",
        "description": "SAA worst case — 4x NETD, heavy blur, FPN",
    },
}

# ── Profile / AGC policy per split ───────────────────────────────────────
# Profiles A/B/C are fixed-seed deterministic noise variants of the SAME
# underlying BT patch — not stochastic augmentation. Training on all three
# teaches the model to memorise specific FPN/blur patterns rather than learning
# to generalise across noise levels. Their methodological value is a test-set
# cross-degradation robustness table ("how much does accuracy drop in the
# SAA/pre-FFC worst case?"), which is destroyed if the model trains on C.
#
# AGC 'fixed' is the confirmed flight inference configuration (FLIGHT_AGC_MODE);
# it is the only mode val/test may use, so reported metrics correspond to real
# deployment. 'global' is a legitimate train-only contrast augmentation (shifts
# the effective BT mapping without duplicating spatial content).
#
# Policy enforced by run_simulation (the split authority):
#   train → profile B,     AGC {fixed, global}
#   val   → profile B,     AGC {fixed}
#   test  → profiles A,B,C, AGC {fixed}
TRAIN_PROFILES: List[str] = ["B"]
TEST_PROFILES: List[str] = list(DEGRADATION_PROFILES.keys())  # ["A", "B", "C"]
TRAIN_AGC_MODES: List[str] = ["fixed", "global"]
EVAL_AGC_MODES: List[str] = ["fixed"]  # val + test

assert set(TRAIN_PROFILES) <= set(
    DEGRADATION_PROFILES
), "TRAIN_PROFILES must be a subset of DEGRADATION_PROFILES keys"
assert set(TEST_PROFILES) <= set(
    DEGRADATION_PROFILES
), "TEST_PROFILES must be a subset of DEGRADATION_PROFILES keys"
assert set(TRAIN_AGC_MODES) <= set(
    VALID_AGC_MODES
), "TRAIN_AGC_MODES must be a subset of VALID_AGC_MODES"
assert set(EVAL_AGC_MODES) <= set(
    VALID_AGC_MODES
), "EVAL_AGC_MODES must be a subset of VALID_AGC_MODES"

# ── Dataset construction ──────────────────────────────────────────────────
# Fire ratio 1:4 justified by asymmetric detection cost:
# missed fire >> false alarm in disaster response context
FIRE_RATIO = 0.20  # fire fraction = 1/(1+4)
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
MAX_NAN_FRACTION = 0.15  # max masked px fraction to accept patch
HIGH_FIRE_GRANULE_THRESH = 0.05  # fire fraction tier for stratified split

# ── Fire labeling thresholds ──────────────────────────────────────────────
# Positive fire labels are derived SOLELY from NASA FIRMS active-fire
# detections (VIIRS 375 m / MODIS 1 km).  The BT-threshold mask below is
# retained only as a QA cross-check to quantify warm-surface false-positive
# rates relative to FIRMS — it is NEVER used as a label source.
#
# NOTE: F2 (10.85 µm) and S8 (11.45 µm) are both LWIR channels.
# Their BT difference is physically bounded to ~0–3 K for any surface,
# including fire pixels. A delta_T discriminant is only meaningful for
# MWIR vs LWIR channel pairs (e.g. 3.7 µm vs 11 µm). Since WaskiriSat
# Lepton is a broadband LWIR sensor (8–14 µm) with no MWIR channel,
# the BT check alone cannot discriminate fire from warm bare rock/sand.
FIRE_DELTA_T_THRESHOLD_K = float("-inf")  # disabled — F2/S8 both LWIR
FIRE_F2_ABS_THRESHOLD_K = 310.0  # QA cross-check threshold [K] — NOT a label source
FIRE_EXEMPT_THRESHOLD_K = (
    308.0  # pixels above this are never cloud-masked in extraction
)
MIN_FIRE_FRACTION = 0.0005  # 0.05% of patch pixels — sub-pixel fires at 3563 m GSD
WARM_SURFACE_THRESH_K = 310.0  # threshold for warm_surface label (reporting only)

# ── FIRMS authoritative-label parameters ──────────────────────────────────
# FIRMS is the sole positive-label source.  Two quality gates are applied:
#   1. Confidence tier: VIIRS string ("low"/"nominal"/"high") or MODIS numeric
#      (0–100, mapped per MODIS C6/6.1 spec: low 0–29, nominal 30–79, high 80–100).
#   2. Fire Radiative Power floor: ensures each labeled detection is energetic
#      enough to produce a detectable LWIR thermal anomaly at the 3563 m Lepton
#      GSD given NETD=50 mK and quantization step ~0.39 K/uint8.
#      Justification: a 10 MW fire covering ~0.1% of a 3.5 km pixel raises the
#      pixel BT by ~0.5 K above background — at the 1-sigma detection limit.
#      Detections below this floor produce contradictory (unlearnable) labels.
FIRMS_MIN_CONFIDENCE = "nominal"  # minimum FIRMS confidence tier to accept
FIRMS_MIN_FRP_MW = 10.0  # minimum Fire Radiative Power [MW] to label as fire

# ── Visualization ─────────────────────────────────────────────────────────
# For STEM communication and paper figures ONLY
# Never used in model training data path
DEFAULT_COLORMAP = "inferno"
COLORMAPS_AVAILABLE = ["inferno", "iron", "rainbow", "grayscale"]

# ── Pipeline paths (defaults, overridable via CLI) ────────────────────────
DEFAULT_DATA_ROOT = "./data/thermal"
DIR_RAW = "01_raw"
DIR_EXTRACTED = "02_extracted"
DIR_PATCHES = "03_patches"
DIR_DATASET = "04_dataset"

# ── Acquisition date window (defaults, overridable via CLI) ───────────────
DEFAULT_START_DATE = "20230801"  # YYYYMMDD — Bolivia fire season peak
DEFAULT_END_DATE = "20230930"  # YYYYMMDD — Bolivia fire season peak

# ── Reproducibility ───────────────────────────────────────────────────────
DEFAULT_SEED = 42
SCRIPT_VERSION = "1.0.0"
