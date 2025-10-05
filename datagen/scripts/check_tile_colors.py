#!/usr/bin/env python3
"""
Diagnostic script to check the color composition of generated tiles
"""

import numpy as np
from PIL import Image
import sys


def analyze_tile(tile_path):
    """
    Analyze a tile's color composition
    """
    img = Image.open(tile_path)
    arr = np.array(img)

    print(f"Analyzing: {tile_path}")
    print(f"Shape: {arr.shape}")
    print(f"Dtype: {arr.dtype}")
    print(f"\nChannel Statistics:")
    print(
        f"  Channel 0 (Red):   min={arr[:,:,0].min()}, max={arr[:,:,0].max()}, mean={arr[:,:,0].mean():.1f}"
    )
    print(
        f"  Channel 1 (Green): min={arr[:,:,1].min()}, max={arr[:,:,1].max()}, mean={arr[:,:,1].mean():.1f}"
    )
    print(
        f"  Channel 2 (Blue):  min={arr[:,:,2].min()}, max={arr[:,:,2].max()}, mean={arr[:,:,2].mean():.1f}"
    )

    # Check for unusual patterns
    if arr[:, :, 0].mean() < arr[:, :, 2].mean():
        print("\n⚠️  WARNING: Red channel has lower mean than Blue channel!")
        print("    This suggests the bands might be in BGR order instead of RGB.")

    if arr.mean() < 50:
        print("\n⚠️  WARNING: Overall image is very dark (mean < 50)")
        print("    This might indicate under-exposure or incorrect processing.")

    return arr


if __name__ == "__main__":
    tile_path = "datagen/sentinel/image_tiles/GRANULE_tile_1024_1024.png"
    analyze_tile(tile_path)
