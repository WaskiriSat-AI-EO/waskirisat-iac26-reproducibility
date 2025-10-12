import os
import zipfile
import shutil
from pathlib import Path
import rasterio
import numpy as np
from PIL import Image
from tqdm import tqdm


def unzip_product(zip_path, extract_dir):
    """
    Unzips a Sentinel-2 product and finds the .SAFE directory.

    Args:
        zip_path (Path): Path to the .zip file.
        extract_dir (Path): Directory to extract the contents to.

    Returns:
        Path: The path to the extracted .SAFE directory, or None if not found.
    """
    safe_dir_name = zip_path.stem
    if not safe_dir_name.endswith(".SAFE"):
        safe_dir_name = safe_dir_name + ".SAFE"

    safe_dir_path = extract_dir / safe_dir_name

    # If the .SAFE directory already exists, skip unzipping
    if safe_dir_path.exists():
        print(f"[*] Found existing extracted directory: {safe_dir_path}")
        return safe_dir_path

    print(f"[*] Unzipping {zip_path.name}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    if safe_dir_path.exists():
        return safe_dir_path
    else:
        print(f"[ERROR] Could not find {safe_dir_name} after extraction.")
        return None


def find_rgb_band_files(safe_dir):
    """
    Finds the file paths for the Red (B04), Green (B03), and Blue (B02) bands.
    Supports both L1C (IMG_DATA/) and L2A (IMG_DATA/R10m/) product structures.

    Args:
        safe_dir (Path): Path to the .SAFE product directory.

    Returns:
        dict: A dictionary mapping band names to their file paths.
    """
    band_files = {}

    r10m_paths = list(safe_dir.glob("GRANULE/*/IMG_DATA/R10m/*_B0[234]_10m.jp2"))

    # If not found, try L1C structure (IMG_DATA/ directly)
    if not r10m_paths:
        r10m_paths = list(safe_dir.glob("GRANULE/*/IMG_DATA/*_B0[234].jp2"))
        print(f"[*] Detected L1C product structure")
    else:
        print(f"[*] Detected L2A product structure")

    for path in r10m_paths:
        # Handle both L1C format (B02.jp2) and L2A format (B02_10m.jp2)
        if "_B02" in path.name:
            band_files["blue"] = path
        elif "_B03" in path.name:
            band_files["green"] = path
        elif "_B04" in path.name:
            band_files["red"] = path

    if len(band_files) != 3:
        print(
            f"[WARNING] Could not find all RGB bands in {safe_dir}. Found: {list(band_files.keys())}"
        )
        return None

    return band_files


def create_rgb_tiles(band_files, output_dir, tile_size=512):
    """
    Reads band files, stacks them into an RGB image, and saves it as tiles.

    Args:
        band_files (dict): Dictionary with paths to red, green, and blue band files.
        output_dir (Path): Directory to save the output PNG tiles.
        tile_size (int): The width and height of the tiles in pixels.
    """
    try:
        with rasterio.open(band_files["red"]) as src_r, rasterio.open(
            band_files["green"]
        ) as src_g, rasterio.open(band_files["blue"]) as src_b:

            red = src_r.read(1)
            green = src_g.read(1)
            blue = src_b.read(1)

            # --- Balanced Contrast Stretch for Natural Colors ---
            # Sentinel-2 L1C data is 12-bit (0-4095 range typically)
            # Use a moderate percentile stretch that preserves color relationships

            # Stack all bands to get global statistics
            all_values = np.concatenate(
                [red.flatten(), green.flatten(), blue.flatten()]
            )

            # Use 2nd and 98th percentile, but apply more conservative stretch
            p2 = np.percentile(all_values, 2)
            p98 = np.percentile(all_values, 98)

            # Add a bit of padding to avoid over-saturation
            value_range = p98 - p2
            p2_adjusted = max(0, p2 - 0.05 * value_range)
            p98_adjusted = p98 + 0.05 * value_range

            def stretch_band(band, p_low, p_high):
                """Apply linear stretch with clipping"""
                stretched = (band - p_low) / (p_high - p_low) * 255
                return np.clip(stretched, 0, 255).astype(np.uint8)

            red_stretched = stretch_band(red, p2_adjusted, p98_adjusted)
            green_stretched = stretch_band(green, p2_adjusted, p98_adjusted)
            blue_stretched = stretch_band(blue, p2_adjusted, p98_adjusted)

            # Stack the bands into an RGB image (Height, Width, Channels)
            rgb_image = np.dstack((red_stretched, green_stretched, blue_stretched))

            height, width, _ = rgb_image.shape
            print(
                f"[*] Tiling image of size {width}x{height} into {tile_size}x{tile_size} tiles..."
            )

            # Calculate total number of tiles
            num_tiles_y = height // tile_size
            num_tiles_x = width // tile_size
            total_tiles = num_tiles_y * num_tiles_x
            print(f"[*] Will create approximately {total_tiles} tiles...")

            # Iterate over the image and create tiles
            tile_count = 0
            for y in range(0, height, tile_size):
                for x in range(0, width, tile_size):
                    # Ensure the tile doesn't go out of bounds
                    if y + tile_size > height or x + tile_size > width:
                        continue

                    tile = rgb_image[y : y + tile_size, x : x + tile_size, :]

                    # Create a unique filename for each tile
                    scene_name = band_files["red"].parts[-5]  # Gets the product name
                    tile_name = f"{scene_name}_tile_{y}_{x}.png"
                    output_path = output_dir / tile_name

                    # Save the tile as a PNG image
                    Image.fromarray(tile).save(output_path)
                    tile_count += 1

                    # Progress update every 50 tiles
                    if tile_count % 50 == 0:
                        print(
                            f"[*] Created {tile_count}/{total_tiles} tiles...", end="\r"
                        )

            print(f"\n[+] Created {tile_count} tiles from this product")

    except Exception as e:
        print(f"[ERROR] Failed during tiling process: {e}")
        import traceback

        traceback.print_exc()


def main():
    """
    Main function to orchestrate the processing of all .zip files in a directory.
    """
    input_zip_dir = Path("./datagen/sentinel/sentinel_data_raw")

    temp_extract_dir = Path("./datagen/sentinel/temp_unzipped")

    output_tiles_dir = Path("./datagen/sentinel/image_tiles")

    input_zip_dir.mkdir(exist_ok=True)
    temp_extract_dir.mkdir(exist_ok=True)
    output_tiles_dir.mkdir(exist_ok=True)

    # Find all .zip files in the input directory
    zip_files = list(input_zip_dir.glob("*.zip"))
    if not zip_files:
        print(f"[ERROR] No .zip files found in {input_zip_dir}")
        print("        Please run the acquisition script first.")
        return

    print(f"[+] Found {len(zip_files)} Sentinel-2 products to process.")

    # Process each zip file
    for zip_path in tqdm(zip_files, desc="Processing Products"):
        safe_dir = unzip_product(zip_path, temp_extract_dir)
        if not safe_dir:
            continue

        band_files = find_rgb_band_files(safe_dir)
        if not band_files:
            continue

        # Using 1280×1280 tiles for better match with SXGA (1280×1024) aspect ratio
        create_rgb_tiles(band_files, output_tiles_dir, tile_size=1280)

    print("[*] Cleaning up temporary extracted files...")
    shutil.rmtree(temp_extract_dir)

    print("\n[SUCCESS] All products have been processed.")
    print(f"[*] Your ML-ready image tiles are located in: {output_tiles_dir}")


if __name__ == "__main__":
    main()
