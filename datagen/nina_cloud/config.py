# ── PAG7936 sensor (confirmed from OpenMV N6 product page + lens datasheet) ──
# Source: https://openmv.io/products/openmv-n6  (PAG7936 1MP Color Global Shutter)
PAG7936_PIXEL_PITCH_UM    = 3.0        # pixel pitch [µm] (square)
PAG7936_WIDTH_PX          = 1280       # nominal mission resolution width [px]
PAG7936_HEIGHT_PX         = 800        # nominal mission resolution height [px]
PAG7936_FOCAL_LENGTH_MM   = 2.8        # lens focal length [mm]
PAG7936_FNUM              = 2.0        # lens f-number
PAG7936_SNR_DB            = 36.0       # signal-to-noise ratio [dB] (datasheet)
PAG7936_DYNAMIC_RANGE_DB  = 64.0       # dynamic range [dB] (datasheet)
PAG7936_MAX_FPS           = 120        # full-resolution frame rate [Hz]
PAG7936_SHUTTER           = "global"   # global shutter (no rolling artefacts)
PAG7936_IR_CUT_NM         = 650        # IR cut filter wavelength [nm]

# ── Mission: NINA / WaskiriSat ────────────────────────────────────────────────
NINA_ALTITUDE_KM          = 525.0      # nominal SSO altitude [km]
LEO_GROUND_SPEED_M_S      = 7600.0     # approx. ground speed at 525 km SSO
PAG7936_EXPOSURE_S        = 0.005      # assumed exposure (5ms; << frame period 8.33ms)

# ── GSD and FOV (derived from sensor specs + mission altitude) ────────────────
# GSD = pixel_pitch_µm × 1e-6 × altitude_m / focal_length_m
PAG7936_GSD_M = (PAG7936_PIXEL_PITCH_UM * 1e-6 * NINA_ALTITUDE_KM * 1e3) / \
                (PAG7936_FOCAL_LENGTH_MM * 1e-3)
# = (3.0e-6 × 525000) / 2.8e-3 = 562.5 m/px

# Swath extents
PAG7936_SWATH_W_KM = (PAG7936_WIDTH_PX  * PAG7936_GSD_M) / 1000.0  # 720.0 km
PAG7936_SWATH_H_KM = (PAG7936_HEIGHT_PX * PAG7936_GSD_M) / 1000.0  # 450.0 km

# Confirmed from datasheet: H FOV = 68.8°, V FOV = 46.4°
PAG7936_HFOV_DEG = 68.8
PAG7936_VFOV_DEG = 46.4

# Motion blur: smear = ground_speed × exposure / GSD
# = 7600 × 0.005 / 562.5 = 0.067 px — effectively zero at 562.5m GSD
PAG7936_MOTION_BLUR_PX = (LEO_GROUND_SPEED_M_S * PAG7936_EXPOSURE_S) / PAG7936_GSD_M

# ── Source: Sentinel-2 MSI L2A ────────────────────────────────────────────────
S2_GSD_M                  = 10.0       # BOA reflectance RGB bands (R10m/) [m]
S2_SCL_GSD_M              = 20.0       # SCL band native resolution (R20m/) [m]
S2_GRANULE_SIZE_PX        = 10980      # standard MGRS tile width/height at 10m
S2_COLLECTION             = "SENTINEL-2"
S2_PRODUCT_TYPE           = "S2MSI2A"  # Level-2A (BOA reflectance + SCL)
S2_BOA_SCALE              = 10000.0    # DN to reflectance scale factor (ESA standard)

# ── Scale gap and PSF simulation ──────────────────────────────────────────────
# PAG7936 GSD = 562.5m  vs  S2 = 10m  →  ratio = 56.25
# One full PAG7936 frame (720km × 450km) requires ~27 S2 granules;
# spatial aggregation is infeasible — Gaussian PSF blur is the correct approach.
S2_TO_PAG_GSD_RATIO       = PAG7936_GSD_M / S2_GSD_M   # 56.25
# Gaussian sigma suppressing frequencies above PAG7936 Nyquist at S2 scale:
# sigma ≈ GSD_ratio / (2 × sqrt(2)) ≈ 19.9 S2 pixels
# Physical: Gaussian-equivalent lens MTF at 562.5m GSD projected onto S2 10m grid
PAG7936_PSF_SIGMA_S2_PX   = S2_TO_PAG_GSD_RATIO / (2.0 * (2.0 ** 0.5))  # ≈ 19.9 px

# ── Training tile dimensions ──────────────────────────────────────────────────
TILE_WIDTH_PX             = PAG7936_WIDTH_PX   # 1280 S2 pixels = 12.8 km
TILE_HEIGHT_PX            = PAG7936_HEIGHT_PX  # 800  S2 pixels = 8.0 km
TILE_STRIDE_FRACTION      = 0.5                # 50% overlap between tiles

# ── Cloud labeling thresholds ────────────────────────────────────────────────
# SCL cloud classes (S2 L2A Scene Classification Layer, 20m resolution)
SCL_CLOUD_CLASSES         = frozenset({8, 9, 10})   # medium prob, high prob, thin cirrus
SCL_INVALID_CLASSES       = frozenset({0, 1})        # nodata, saturated — excluded from denominator
CLOUD_LABEL_HIGH_THRESH   = 0.70   # cloud_frac > 0.70 → label = 1 (cloud)
CLOUD_LABEL_LOW_THRESH    = 0.30   # cloud_frac < 0.30 → label = 0 (clear)
# Tiles with cloud_frac in [0.30, 0.70] → ambiguous → EXCLUDED from dataset

# ── Acquisition cloud cover search bounds ────────────────────────────────────
ACQUISITION_CLEAR_MAX_PCT  = 30    # CDSE cloudCover ≤ 30 → "clear" acquisition pass
ACQUISITION_CLOUDY_MIN_PCT = 60    # CDSE cloudCover ≥ 60 → "cloudy" acquisition pass

# ── Degradation profiles (CMOS dark-current / noise envelope) ─────────────────
# A: nominal (cool sensor, low radiation dose)
# B: moderate on-orbit (mid-mission, nominal temperature)
# C: worst case (SAA transit, radiation accumulation, warm focal plane)
DEGRADATION_PROFILES = {
    "A": {
        "psf_sigma_px":        PAG7936_PSF_SIGMA_S2_PX,
        "shot_noise_factor":   1.0,
        "read_noise_factor":   1.0,
        "fpn_column_strength": 0.004,
        "fpn_pixel_strength":  0.006,
        "fpn_factor":          None,
        "jpeg_quality":        92,
        "description":         "Best case — nominal noise, cool sensor",
    },
    "B": {
        "psf_sigma_px":        PAG7936_PSF_SIGMA_S2_PX,
        "shot_noise_factor":   2.0,
        "read_noise_factor":   1.5,
        "fpn_column_strength": 0.008,
        "fpn_pixel_strength":  0.012,
        "fpn_factor":          0.5,
        "jpeg_quality":        85,
        "description":         "Typical on-orbit — moderate dark current, standard JPEG",
    },
    "C": {
        "psf_sigma_px":        PAG7936_PSF_SIGMA_S2_PX,
        "shot_noise_factor":   4.0,
        "read_noise_factor":   3.0,
        "fpn_column_strength": 0.014,
        "fpn_pixel_strength":  0.022,
        "fpn_factor":          1.0,
        "jpeg_quality":        75,
        "description":         "Worst case — SAA transit, elevated dark current, lower JPEG quality",
    },
}

# ── Shot noise model (derived from PAG7936 SNR specification) ─────────────────
# SNR = 36 dB → linear SNR ≈ 63.1
# At full well: SNR = sqrt(N_photons) → N_photons ≈ SNR² ≈ 3981
# Used to scale Poisson sampling in add_shot_noise()
PAG7936_FULL_WELL_ELECTRONS = round(10 ** (PAG7936_SNR_DB / 10))   # ≈ 3981 e⁻

# ── Dataset construction ──────────────────────────────────────────────────────
CLOUD_RATIO               = 0.50   # 50/50 cloud:clear (symmetric binary task)
TRAIN_FRAC                = 0.70
VAL_FRAC                  = 0.15
TEST_FRAC                 = 0.15
MAX_NODATA_FRACTION       = 0.10   # reject tile if SCL nodata fraction > 10%
HIGH_CLOUD_SCENE_THRESH   = 0.50   # scene cloud fraction tier boundary for stratified split

# ── Visualization (overlay.py only — never in training data path) ─────────────
SCL_CLASS_COLORS = {
    0:  (0,   0,   0),     # nodata
    1:  (64,  64,  64),    # saturated
    2:  (100, 100, 100),   # dark area
    3:  (80,  40,  40),    # cloud shadow
    4:  (34,  139, 34),    # vegetation
    5:  (210, 180, 140),   # bare soil/rock
    6:  (0,   100, 255),   # water
    7:  (200, 200, 100),   # unclassified
    8:  (160, 160, 230),   # cloud medium probability
    9:  (220, 220, 255),   # cloud high probability
    10: (180, 210, 255),   # thin cirrus
    11: (255, 255, 255),   # snow/ice
}

# ── Pipeline defaults (overridable via CLI) ───────────────────────────────────
DEFAULT_DATA_ROOT         = "./data/cloud"
DIR_RAW                   = "01_raw"
DIR_EXTRACTED             = "02_extracted"
DIR_TILES                 = "03_tiles"
DIR_DATASET               = "04_dataset"
DEFAULT_START_DATE        = "20230601"
DEFAULT_END_DATE          = "20230831"
DEFAULT_SEED              = 42
SCRIPT_VERSION            = "1.0.0"
