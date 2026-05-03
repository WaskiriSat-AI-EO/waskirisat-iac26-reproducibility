"""Full sensor simulation pipeline for the PAG7936 on WaskiriSat.

Orchestrates the complete degradation chain from S2 BOA reflectance to a
uint8 RGB image that matches expected on-orbit PAG7936 output.

Order of operations is physically motivated and must not be changed:

1. **PSF blur** (float32 reflectance) — optical averaging integrates scene
   radiance over the PAG7936 GSD footprint; must precede electronic noise.
2. **Motion blur** (float32) — sensor integration over relative displacement
   during exposure; ~0.067 px for PAG7936 at 5ms, effectively negligible.
3. **Shot noise + read noise** (float32) — photoelectron statistics in
   normalised [0,1] space, before quantisation.
4. **Quantise to uint8** — locks in the 8-bit representation.
5. **FPN** (float32 round-trip) — readout electronics artefact; occurs after
   photon capture and ADC.
6. **JPEG compression** — on-board compression step; final output artefact.
"""

import io
from typing import Dict, Optional

import cv2
import numpy as np

from nina_cloud import config
from nina_cloud.processing.reflectance import reflectance_to_uint8
from nina_cloud.simulation.motion import calculate_motion_blur_px, motion_blur_kernel
from nina_cloud.simulation.noise import add_read_noise, add_shot_noise, apply_fpn
from nina_cloud.simulation.optics import apply_pag7936_psf


def simulate_pag7936(
    rgb_float: np.ndarray,
    profile: str,
    fpn: Dict[str, np.ndarray],
    rng: np.random.Generator,
    jpeg_encode: bool = True,
) -> np.ndarray:
    """Simulate a PAG7936 image from a normalised S2 reflectance patch.

    Applies the full degradation chain in physically correct order (see
    module docstring): PSF blur → motion blur → shot/read noise → uint8
    quantisation → FPN → JPEG compression.

    Args:
        rgb_float: float32 (H, W, 3) BOA reflectance in [0, 1] at S2 10m.
        profile: Degradation profile key — one of ``"A"``, ``"B"``, ``"C"``
            (see ``config.DEGRADATION_PROFILES``).
        fpn: FPN maps dict from ``generate_pag7936_fpn``, containing
            ``"column_fpn"`` and ``"pixel_fpn"`` arrays of shape (H, W).
        rng: ``numpy.random.Generator`` for reproducible stochastic steps.
        jpeg_encode: If True, apply JPEG compression at the profile quality
            setting.  Set False for validation / lossless reference output.

    Returns:
        uint8 (H, W, 3) RGB image.

    Example::

        >>> import numpy as np
        >>> from nina_cloud.simulation.noise import generate_pag7936_fpn
        >>> from nina_cloud.simulation.sensor import simulate_pag7936
        >>> rng = np.random.default_rng(0)
        >>> rgb = np.full((800, 1280, 3), 0.3, dtype=np.float32)
        >>> fpn = generate_pag7936_fpn((800, 1280), seed=0)
        >>> img = simulate_pag7936(rgb, profile="A", fpn=fpn, rng=rng)
        >>> img.dtype
        dtype('uint8')
        >>> img.shape
        (800, 1280, 3)
    """
    prof = config.DEGRADATION_PROFILES[profile]

    # ── Step 1: optical PSF blur ───────────────────────────────────────────────
    img = apply_pag7936_psf(rgb_float, sigma_px=prof["psf_sigma_px"])

    # ── Step 2: motion blur (~0.067 px — negligible but physically complete) ───
    blur_px = calculate_motion_blur_px()
    if blur_px >= 1.0:
        kernel = motion_blur_kernel(blur_px, angle_deg=0.0)
        for ch in range(img.shape[2]):
            img[:, :, ch] = cv2.filter2D(img[:, :, ch], -1, kernel)

    # ── Step 3: shot noise + read noise (float32 space before quantisation) ────
    img = add_shot_noise(img, factor=prof["shot_noise_factor"], rng=rng)
    img = add_read_noise(img, sigma_dn=prof["read_noise_factor"], rng=rng)

    # ── Step 4: quantise to uint8 ─────────────────────────────────────────────
    img_u8 = reflectance_to_uint8(img)

    # ── Step 5: FPN (readout artefact, after ADC) ─────────────────────────────
    fpn_factor: Optional[float] = prof.get("fpn_factor")
    if fpn_factor is not None:
        img_f = img_u8.astype(np.float32) / 255.0
        img_f = apply_fpn(img_f, fpn, factor=fpn_factor)
        img_u8 = (img_f * 255.0).round().clip(0, 255).astype(np.uint8)

    # ── Step 6: JPEG compression (on-board compression) ──────────────────────
    if jpeg_encode:
        quality = prof["jpeg_quality"]
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        # cv2 uses BGR; convert → encode → decode → convert back
        bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)
        _, encoded = cv2.imencode(".jpg", bgr, encode_param)
        bgr_decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        img_u8 = cv2.cvtColor(bgr_decoded, cv2.COLOR_BGR2RGB)

    return img_u8
