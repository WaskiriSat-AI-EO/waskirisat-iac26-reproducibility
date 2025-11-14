# Data Generation & Acquisition

This folder handles all aspects of satellite data acquisition, processing, and preparation for machine learning.

## Folder Structure

```
datagen/
├── README.md                      # This file
├── resources/
│   └── bolivia_roi.geojson       # Region of Interest (ROI) geometries
├── scripts/
│   ├── acquire_sentinel_data_cdse.py    # Download Sentinel data from CDSE
│   ├── process_sentinel_data.py         # Pre-process and clean raw data
│   ├── degrade_sentinel_tiles.py        # Degrade image resolution
│   ├── check_tile_colors.py             # Validate image quality/colors
│   └── __pycache__/                     # Python cache (ignored in git)
└── sentinel/
    ├── image_tiles/             # Processed, tiled sentinel imagery
    ├── degraded_tiles_sxga/     # Degraded resolution tiles (SXGA)
    └── sentinel_data_raw/       # Raw downloaded sentinel data (large)
```

## Purpose of Each Component

### Scripts

- **`acquire_sentinel_data_cdse.py`**: Downloads Sentinel satellite imagery from Copernicus Data Space Ecosystem (CDSE)
  - Uses ROI from `bolivia_roi.geojson`
  - Configurable date ranges, cloud cover thresholds
  - Outputs to `sentinel/sentinel_data_raw/`

- **`process_sentinel_data.py`**: Transforms raw satellite data into usable format
  - Applies radiometric corrections
  - Normalizes data ranges
  - Outputs to `sentinel/image_tiles/`

- **`degrade_sentinel_tiles.py`**: Reduces image resolution for specific use cases
  - Creates SXGA (1280×1024) versions for edge devices
  - Outputs to `sentinel/degraded_tiles_sxga/`

- **`check_tile_colors.py`**: Validates image quality and color consistency
  - Checks for artifacts, nodata regions
  - Generates quality reports

### Resources

- **`bolivia_roi.geojson`**: GeoJSON file defining Region of Interest (ROI)
  - Used by acquisition scripts to limit downloads
  - Modify this to change acquisition area

### Sentinel Data Storage

- **`sentinel_data_raw/`**: Raw downloaded files (typically large, often 100GB+)
  - `.SAFE` format (Sentinel original format)
  - Should be backed up separately
  
- **`image_tiles/`**: Processed, analyzed tiles ready for ML
  - GeoTIFF or PNG format
  - Organized by date/location
  
- **`degraded_tiles_sxga/`**: Resolution-reduced versions
  - For on-board processing on satellite payload
  - Lower bandwidth, lower compute requirements

## Workflow

1. **Acquisition**: Run `acquire_sentinel_data_cdse.py`
   ```bash
   python datagen/scripts/acquire_sentinel_data_cdse.py
   ```
   Output: Data in `sentinel/sentinel_data_raw/`

2. **Processing**: Run `process_sentinel_data.py`
   ```bash
   python datagen/scripts/process_sentinel_data.py
   ```
   Output: Data in `sentinel/image_tiles/`

3. **Quality Check**: Run `check_tile_colors.py`
   ```bash
   python datagen/scripts/check_tile_colors.py
   ```
   Output: Quality reports

4. **Degradation** (optional): Run `degrade_sentinel_tiles.py` for on-board use
   ```bash
   python datagen/scripts/degrade_sentinel_tiles.py
   ```
   Output: Data in `sentinel/degraded_tiles_sxga/`

## Configuration

Key environment variables (in `.env`):
- `CDSE_USERNAME`: Copernicus credentials
- `CDSE_PASSWORD`: Copernicus password
- `DATA_DIR`: Path to store sentinel data
- `ROI_PATH`: Path to ROI GeoJSON file

## Data Size Considerations

- Raw Sentinel-2 data: ~100-500 MB per tile per date
- Processed tiles: ~30-50 MB per tile (compressed)
- SXGA degraded: ~5-10 MB per tile

**Note**: Git-ignore large data files. Use `.gitignore` for `sentinel/sentinel_data_raw/` and similar directories.

## Next Steps

After processing, use the data from `sentinel/image_tiles/` for:
- Dataset splitting (train/val/test) → `training/`
- Model training → `training/scripts/train.py`

## Dependencies

See `requirements.txt` for dependencies:
- `rasterio`: GeoTIFF handling
- `sentinelsat`: CDSE API interaction
- `opencv-python`: Image processing
- `numpy`, `pillow`: Array/image manipulation
