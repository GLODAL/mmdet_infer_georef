import os, time, importlib, pickle, argparse

import geospatial_process
import geo_det_utils
from mmdet.apis import init_detector

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tile and inference on a GeoTIFF.")
    parser.add_argument("--img", required=True, help="Path to input GeoTIFF image.")
    parser.add_argument("--config", required=True, help="Path to MMDetection config file.")
    parser.add_argument("--checkpoint", required=True, help="Path to MMDetection checkpoint file.")
    parser.add_argument("--out_dir", required=True, help="Root output directory for tiles and results.")
    parser.add_argument("--device", default="cuda:0", help="Device to use for inference (default: cuda:0).")
    parser.add_argument("--patch_size", type=int, default=256, help="Patch size (default: 256).")
    parser.add_argument("--overlap", type=float, default=0.2, help="Overlap fraction (0.0 - 1.0) (default: 0.2).")
    parser.add_argument("--score_thr", type=float, default=0.8, help="Score threshold for inference (default: 0.8).")

    args = parser.parse_args()

    extent_tif = args.img
    extent_name = os.path.splitext(os.path.basename(extent_tif))[0]
    patch_size = args.patch_size

    # Sub-folder for this specific image
    grid_output_root = os.path.join(args.out_dir, extent_name)
    os.makedirs(grid_output_root, exist_ok=True)

    config_path = args.config
    checkpoint = args.checkpoint
    device = args.device

    start_time = time.time()
    stride = int(patch_size * (1 - args.overlap))
    geospatial_process.generate_test_images(
        extent = extent_tif,
        dimension = patch_size,
        stride = stride,
        output_path = grid_output_root,
        isGuide = True,
    )
    print("Tile generation time (s):", time.time() - start_time)

    extent_base = os.path.splitext(os.path.basename(extent_tif))[0]
    tile_folder = os.path.join(grid_output_root, f"{extent_base}_{patch_size}x{patch_size}")
    print("Tile folder:", tile_folder)

    model = init_detector(config_path, checkpoint, device=device)
    print(geo_det_utils.msg("[OK] Model loaded", geo_det_utils.Colors.GREEN))

    tile_dets = geo_det_utils.infer_folder_pixels(
        model = model,
        folder_path = tile_folder,
        score_thr = args.score_thr,
    )

    print("Tiles with detections:", len(tile_dets))

    # Save results for process_3.py
    pickle_path = os.path.join(grid_output_root, f"{extent_base}_tile_dets.pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump(tile_dets, f)
    print(f"Saved detection results to: {pickle_path}")
