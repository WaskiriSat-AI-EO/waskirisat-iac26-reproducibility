import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

# --- Physical Constants & Sensor Data ---
# From OV5640 Datasheet and OpenMV H7 Plus Mission Specs
OV5640_PIXEL_SIZE_UM = 1.4  # Native 1.4 µm pixel size (no binning in SXGA)
PIXEL_BINNING = 1  # SXGA mode uses full resolution (no binning)
OV5640_EFFECTIVE_PIXEL_SIZE_UM = (
    OV5640_PIXEL_SIZE_UM * PIXEL_BINNING
)  # 1.4 µm in SXGA mode

# Camera Resolution (SXGA mode - OpenMV H7 Plus)
CAMERA_WIDTH_PX = 1280  # SXGA width
CAMERA_HEIGHT_PX = 1024  # SXGA height
CAMERA_FORMAT = "RGB565"  # 16-bit color format (5-6-5)

# LEO orbital mechanics
LEO_GROUND_SPEED_M_S = 7600  # Approx. ground speed at 500-550 km

# --- Simulation Functions ---


def calculate_gsd(pixel_size_um, altitude_km, focal_length_mm):
    """Calculates the Ground Sampling Distance (GSD) in meters per pixel."""
    return (pixel_size_um * altitude_km) / focal_length_mm


def calculate_motion_blur_px(ground_speed_ms, exposure_time_s, gsd_m):
    """Calculates motion blur length in pixels based on physical parameters."""
    smear_distance_m = ground_speed_ms * exposure_time_s
    return smear_distance_m / gsd_m


def motion_blur_kernel(length, angle_deg):
    """Creates a motion blur kernel."""
    if length < 1:
        return np.array([[1]])  # Return a 1x1 identity kernel
    length = int(np.ceil(length))  # Ensure integer length
    # Make kernel size odd
    kernel_size = length if length % 2 else length + 1

    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = kernel_size // 2
    kernel[center, center - length // 2 : center + length // 2 + 1] = 1.0 / length

    rot_mat = cv2.getRotationMatrix2D((center, center), angle_deg, 1.0)
    return cv2.warpAffine(kernel, rot_mat, (kernel_size, kernel_size))


def add_sensor_noise(img_float, shot_strength, read_sigma, fpn_gain, rng):
    """Applies a simple sensor noise model."""
    # Fixed Pattern Noise (multiplicative)
    if fpn_gain > 0:
        h, w, c = img_float.shape
        pattern = rng.normal(1.0, fpn_gain, (h, w, 1)).astype(np.float32)
        img_float = np.clip(img_float * pattern, 0, 1)

    # Shot Noise (Poisson-like)
    if shot_strength > 0:
        N = 1.0 / shot_strength
        img_float = rng.poisson(img_float * N).astype(np.float32) / N

    # Read Noise (Gaussian)
    if read_sigma > 0:
        noise = rng.normal(0, read_sigma, img_float.shape).astype(np.float32)
        img_float = img_float + noise

    return np.clip(img_float, 0, 1)


def degrade_tile(img_bgr, params):
    """
    Applies the full degradation pipeline to a single image tile.

    Args:
        img_bgr (np.array): Input BGR image from OpenCV.
        params (dict): Dictionary containing all degradation parameters.

    Returns:
        np.array: Degraded 8-bit BGR image.
    """
    rng = np.random.default_rng(params["seed"])

    # 1. Downscale to match the target GSD and camera resolution
    h, w = img_bgr.shape[:2]

    # Calculate target size based on GSD or use camera resolution if specified
    if "target_width" in params and "target_height" in params:
        # Match camera resolution exactly
        new_w = params["target_width"]
        new_h = params["target_height"]
    else:
        # Calculate based on scale factor
        new_w = max(1, int(round(w / params["scale_factor"])))
        new_h = max(1, int(round(h / params["scale_factor"])))

    lowres = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    img_float = lowres.astype(np.float32) / 255.0

    # 2. Apply optical blur (PSF) and motion blur
    if params["gauss_sigma"] > 0:
        img_float = cv2.GaussianBlur(img_float, (0, 0), params["gauss_sigma"])

    if params["motion_len"] > 0:
        kernel = motion_blur_kernel(params["motion_len"], params["motion_angle"])
        img_float = cv2.filter2D(img_float, -1, kernel)

    # 3. Add sensor noise
    img_float = add_sensor_noise(
        img_float,
        params["shot_strength"],
        params["read_noise"],
        params["fpn_gain"],
        rng,
    )

    # 4. Convert back to 8-bit for saving
    img_uint8 = np.clip(img_float * 255.0, 0, 255).astype(np.uint8)

    return img_uint8


# --- Main Execution ---


def main():
    parser = argparse.ArgumentParser(
        description="Degrade high-res Sentinel tiles to simulate OpenMV camera."
    )
    parser.add_argument(
        "input_dir", type=str, help="Directory containing the clean PNG tiles."
    )
    parser.add_argument(
        "output_dir", type=str, help="Directory to save the degraded JPEG tiles."
    )
    parser.add_argument(
        "--altitude_km", type=float, default=525.0, help="Orbit altitude in km."
    )
    parser.add_argument(
        "--focal_length_mm",
        type=float,
        default=8.0,
        help="Lens focal length in mm (default: 8mm).",
    )
    parser.add_argument(
        "--exposure_s",
        type=float,
        default=0.0778,
        help="Camera exposure time in seconds (SXGA: 77.8ms).",
    )
    parser.add_argument(
        "--s2_gsd",
        type=float,
        default=10.0,
        help="GSD of source Sentinel-2 images in meters.",
    )
    parser.add_argument(
        "--target_width",
        type=int,
        default=CAMERA_WIDTH_PX,
        help="Target output width in pixels.",
    )
    parser.add_argument(
        "--target_height",
        type=int,
        default=CAMERA_HEIGHT_PX,
        help="Target output height in pixels.",
    )
    parser.add_argument(
        "--jpg_quality",
        type=int,
        default=85,
        help="JPEG quality for output images (0-100). Lower for SXGA to control file size.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # --- Calculate Physical Parameters ---
    target_gsd = calculate_gsd(
        OV5640_EFFECTIVE_PIXEL_SIZE_UM, args.altitude_km, args.focal_length_mm
    )
    scale_factor = target_gsd / args.s2_gsd
    motion_len = calculate_motion_blur_px(
        LEO_GROUND_SPEED_M_S, args.exposure_s, target_gsd
    )

    # Calculate swath width
    sensor_width_mm = (OV5640_EFFECTIVE_PIXEL_SIZE_UM / 1000.0) * CAMERA_WIDTH_PX
    swath_width_km = (sensor_width_mm / args.focal_length_mm) * args.altitude_km

    print("--- Simulation Parameters ---")
    print(
        f"Camera: OpenMV H7 Plus - OV5640 (SXGA {CAMERA_WIDTH_PX}×{CAMERA_HEIGHT_PX})"
    )
    print(f"Pixel Size: {OV5640_EFFECTIVE_PIXEL_SIZE_UM:.1f} µm (no binning in SXGA)")
    print(f"Color Format: {CAMERA_FORMAT}")
    print(f"Focal Length: {args.focal_length_mm:.1f} mm")
    print(f"Orbit Altitude: {args.altitude_km:.1f} km")
    print(f"Target GSD: {target_gsd:.2f} m/pixel")
    print(f"Swath Width: {swath_width_km:.2f} km")
    print(f"Scale Factor: {scale_factor:.2f}×")
    print(f"Motion Blur: {motion_len:.2f} pixels")
    print(f"Exposure Time: {args.exposure_s*1000:.1f} ms")
    print(f"Output Size: {args.target_width}×{args.target_height} pixels")
    print("-----------------------------")

    # Define the set of degradation parameters
    degradation_params = {
        "scale_factor": scale_factor,
        "target_width": args.target_width,
        "target_height": args.target_height,
        "gauss_sigma": 0.6,  # PSF/lens blur. 8mm lens has good quality.
        "motion_len": motion_len,
        "motion_angle": 15.0,  # Angle can be randomized later
        "shot_strength": 0.004,  # Photon noise. Lower due to longer exposure (77.8ms).
        "read_noise": 0.018,  # Sensor electronics noise. Higher due to more pixels.
        "fpn_gain": 0.010,  # Fixed-pattern noise. Slightly higher for SXGA.
        "seed": 42,
    }

    image_files = list(input_path.glob("*.png"))
    if not image_files:
        print(f"[ERROR] No PNG files found in {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[*] Found {len(image_files)} tiles to process")

    for file_path in tqdm(image_files, desc="Degrading Tiles"):
        img = cv2.imread(str(file_path))
        if img is None:
            print(f"[WARNING] Could not read {file_path.name}", file=sys.stderr)
            continue

        degraded_img = degrade_tile(img, degradation_params)

        output_file = output_path / f"{file_path.stem}_degraded.jpg"
        cv2.imwrite(
            str(output_file), degraded_img, [cv2.IMWRITE_JPEG_QUALITY, args.jpg_quality]
        )

    print(f"\n[SUCCESS] Processing complete. Degraded images saved to {output_path}")


if __name__ == "__main__":
    main()
