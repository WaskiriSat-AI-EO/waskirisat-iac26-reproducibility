# ARCHITECTURAL CONSTRAINT (Rule 3 — Section 4 of CLAUDE_CODE_BRIEF.md):
# This module is the ONLY place false-colour rendering occurs in the entire
# pipeline.  It must NEVER be imported by any module in processing/,
# simulation/, labeling/, or dataset/.  It imports only from config and
# standard/third-party libraries.  The false-colour rendering path is for
# human communication only — it is never in the model training data path.
"""False-colour rendering for visualisation, paper figures, and STEM output.

See architectural constraint at the top of this file.
"""

from typing import Optional

import numpy as np

from nina_thermal import config


def apply_colormap(
    uint8_image: np.ndarray,
    colormap_name: Optional[str] = None,
) -> np.ndarray:
    """Convert a uint8 grayscale image to uint8 RGB via a matplotlib colormap.

    This is the sole entry point for false-colour rendering in the pipeline.
    Called only by visualisation and paper-figure scripts — never by
    ``simulate_lepton()`` or any training-data export path.

    Args:
        uint8_image: uint8 (H, W) grayscale array with values in [0, 255].
        colormap_name: Matplotlib colormap name.  Must be one of
            ``config.COLORMAPS_AVAILABLE`` (``"inferno"``, ``"iron"``,
            ``"rainbow"``, ``"grayscale"``).
            Defaults to ``config.DEFAULT_COLORMAP`` (``"inferno"``).

    Returns:
        uint8 (H, W, 3) RGB array.

    Raises:
        ValueError: If *colormap_name* is not in ``COLORMAPS_AVAILABLE``.
    """
    if colormap_name is None:
        colormap_name = config.DEFAULT_COLORMAP

    if colormap_name not in config.COLORMAPS_AVAILABLE:
        raise ValueError(
            f"Unknown colormap '{colormap_name}'. "
            f"Choose from: {config.COLORMAPS_AVAILABLE}"
        )

    import matplotlib.cm as cm  # type: ignore
    import matplotlib.colors as mcolors  # type: ignore

    # "grayscale" is not a standard matplotlib name; map it.
    mpl_name = "gray" if colormap_name == "grayscale" else colormap_name

    try:
        cmap = cm.get_cmap(mpl_name)
    except ValueError:
        raise ValueError(
            f"Matplotlib does not recognise colormap '{mpl_name}'"
        )

    normalised = uint8_image.astype(np.float32) / 255.0
    rgba = cmap(normalised)           # (H, W, 4) float64 in [0, 1]
    rgb  = (rgba[:, :, :3] * 255.0).round().astype(np.uint8)
    return rgb
