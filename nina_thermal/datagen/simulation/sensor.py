"""Full sensor simulation pipeline for the FLIR Lepton 3.5 on WaskiriSat.

Orchestrates the complete degradation chain from normalised BT space to
a uint8 grayscale image that matches the expected on-orbit output.

Order of operations is physically motivated and must not be changed:

1. **Thermal noise** (float32 BT space) — NETD is a temperature
   uncertainty; it must precede rendering.  Adding noise after colormap
   would create artefacts whose magnitude depends on the colormap
   gradient — physically incorrect.
2. **AGC quantisation** (→ uint8) — locks in the 8-bit representation.
3. **PSF blur** (uint8) — optical blur acts on pixel values in the focal
   plane; it belongs after quantisation and before readout electronics.
4. **FPN** (float32 round-trip) — Fixed Pattern Noise is a readout
   amplifier artifact; it occurs after the optical path.
"""

from typing import Dict, Optional

import numpy as np

from nina_thermal.datagen import config
from nina_thermal.datagen.processing.agc import agc_to_uint8
from nina_thermal.datagen.simulation.noise import add_thermal_noise, apply_fpn
from shared.optics import apply_psf_blur


def simulate_lepton(
    bt_normalized: np.ndarray,
    profile: str,
    agc_range_k: float,
    fpn: Dict[str, np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate a FLIR Lepton 3.5 image from a normalised BT patch.

    Applies thermal noise, AGC quantisation, optical PSF blur, and Fixed
    Pattern Noise in the physically correct order (see module docstring).

    Args:
        bt_normalized: float32 (H, W) array in [0, 1] representing the
            scene after AGC normalisation.
        profile: Degradation profile key — one of ``"A"``, ``"B"``,
            ``"C"`` (see ``config.DEGRADATION_PROFILES``).
        agc_range_k: Temperature range of the AGC window [K], used to
            convert NETD to normalised sigma.
        fpn: FPN maps dict from ``generate_lepton_fpn``, containing
            ``"column_fpn"`` and ``"pixel_fpn"`` arrays (H, W).
        rng: ``numpy.random.Generator`` for reproducible stochastic steps.

    Returns:
        uint8 ``(H, W)`` single-channel grayscale image.

    Example::

        >>> import numpy as np
        >>> from nina_thermal.datagen.simulation.noise import generate_lepton_fpn
        >>> from nina_thermal.datagen.simulation.sensor import simulate_lepton
        >>> rng = np.random.default_rng(0)
        >>> bt = np.full((120, 160), 0.5, dtype=np.float32)
        >>> fpn = generate_lepton_fpn(seed=0)
        >>> img = simulate_lepton(bt, profile="A", agc_range_k=100.0, fpn=fpn, rng=rng)
        >>> img.dtype
        dtype('uint8')
        >>> img.shape
        (120, 160)
    """
    prof = config.DEGRADATION_PROFILES[profile]

    # ── Step 1: thermal noise in BT space ─────────────────────────────────────
    netd_mk = config.LEPTON_NETD_MK * prof["thermal_noise_factor"]
    sigma = (netd_mk / 1000.0) / agc_range_k
    img = add_thermal_noise(bt_normalized, sigma, rng)

    # ── Step 2: AGC quantisation → uint8 ──────────────────────────────────────
    img_u8 = agc_to_uint8(img)

    # ── Step 3: optical PSF blur (uint8) ──────────────────────────────────────
    img_u8 = apply_psf_blur(img_u8, sigma_px=prof["optical_blur_sigma_px"])

    # ── Step 4: FPN (after optical path) ──────────────────────────────────────
    fpn_factor = prof.get("fpn_factor")
    if fpn_factor is not None:
        img_f = img_u8.astype(np.float32) / 255.0
        img_f = apply_fpn(img_f, fpn, factor=fpn_factor)
        img_u8 = (img_f * 255.0).round().clip(0, 255).astype(np.uint8)

    return img_u8
