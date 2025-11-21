# Satellite Image Object Detection Pipeline

This repository contains a modular pipeline for inferencing satellite imagery (TIFF files) using MMDetection. The pipeline handles fetching satellite data from KML coordinates, tiling images with overlap for robust detection, and georeferencing the results into standard GIS formats (GPKG).

## Workflow

The pipeline is split into three sequential steps:

1.  **`01_fetch_satellite_data.py`**: Parses KML points, fetches WMS satellite imagery, and saves them as GeoTIFFs.
2.  **`02_detect_objects.py`**: Tiles the GeoTIFFs (with overlap), runs MMDetection inference, and saves raw detection results.
3.  **`03_georeference_results.py`**: Converts pixel-based detections to global map coordinates, applies Global Non-Maximum Suppression (NMS), and exports results to GeoPackage (GPKG) and annotated GeoTIFFs.

## Requirements

*   Python 3.8+
*   `gdal` / `osgeo`
*   `mmdet` (MMDetection) & `mmcv`
*   `numpy`, `cv2`, `requests`, `pyproj`, `tqdm`

## Usage

You can run these scripts from a terminal or a Jupyter Notebook (using `!python`).

### 1. Fetch Data
Fetch satellite imagery for points defined in a KML file.

```bash
python 01_fetch_satellite_data.py \
  --kml /path/to/locations.kml \
  --out_dir ./data/satellite_images \
  --buffer_m 250 \
  --res_m 0.15
```

*   `--kml`: Input KML file containing Placemarks.
*   `--out_dir`: Directory to save the downloaded GeoTIFFs.
*   `--buffer_m`: Radius in meters around the point to fetch (default: 250).
*   `--res_m`: Pixel resolution in meters (default: 0.15).

### 2. Detect Objects
Run MMDetection on a specific GeoTIFF. This process handles tiling (sliding window) automatically.

```bash
python 02_detect_objects.py \
  --img ./data/satellite_images/loc_0.tif \
  --config /path/to/mmdet_config.py \
  --checkpoint /path/to/checkpoint.pth \
  --out_dir ./work_dirs/results \
  --device cuda:0 \
  --patch_size 256 \
  --overlap 0.2
```

*   `--img`: Path to the source GeoTIFF.
*   `--config`: MMDetection config file path.
*   `--checkpoint`: Model weights (.pth) path.
*   `--out_dir`: Root directory for intermediate outputs (grids, pickles).
*   `--overlap`: Fraction of overlap between tiles (0.0 - 1.0). Recommended: 0.2 (20%).

### 3. Georeference Results
Process the raw detections, handle overlaps using Global NMS, and export to GIS formats.

```bash
python 03_georeference_results.py \
  --pickle ./work_dirs/results/loc_0/loc_0_tile_dets.pkl \
  --out_dir ./final_outputs \
  --nms_thr 0.5
```

*   `--pickle`: The `.pkl` file generated in Step 2.
*   `--out_dir`: Directory to save the final GPKG and annotated tiles.
*   `--nms_thr`: IoU threshold for Global NMS (default: 0.5).

## Jupyter Notebook Example

If running in a Jupyter Notebook, simply prefix commands with `!`:

```python
# Step 1: Get Images
!python 01_fetch_satellite_data.py --kml data/pin.kml --out_dir data/images

# Step 2: Inference (Example for one image)
!python 02_detect_objects.py \
    --img data/images/location_1.tif \
    --config configs/my_config.py \
    --checkpoint weights/epoch_100.pth \
    --out_dir works/temp_res \
    --overlap 0.2

# Step 3: Georeference
!python 03_georeference_results.py \
    --pickle works/temp_res/location_1/location_1_tile_dets.pkl \
    --out_dir results/final
```

## Key Features

*   **Sliding Window Inference:** Supports overlapping tiles (`stride` logic) to prevent missing objects at tile edges.
*   **Global NMS:** Performs Non-Maximum Suppression in global map coordinates (meters) to correctly merge duplicate detections from overlapping tiles.
*   **Geospatial Output:** Exports directly to `.gpkg` (GeoPackage) for easy viewing in QGIS/ArcGIS.
