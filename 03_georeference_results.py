import os, importlib, pickle, argparse
import geo_det_utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Georeference detections and save results.")
    parser.add_argument("--pickle", required=True, help="Path to detection pickle file from step 2.")
    parser.add_argument("--out_dir", required=True, help="Root directory to save final results.")
    parser.add_argument("--nms_thr", type=float, default=0.5, help="NMS IoU threshold (default: 0.5).")

    args = parser.parse_args()

    pickle_path = args.pickle
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"Could not find {pickle_path}")

    with open(pickle_path, "rb") as f:
        tile_dets = pickle.load(f)
    print(f"Loaded detections from: {pickle_path}")

    # Derive names/sizes (assuming strict naming convention or just use generic output)
    # The previous code parsed extent_tif path, but we only have the pickle now.
    # We can construct an output folder based on the input pickle name.

    pickle_name = os.path.splitext(os.path.basename(pickle_path))[0]
    # pickle_name like "{extent_base}_tile_dets"

    result_root = os.path.join(args.out_dir, pickle_name)
    os.makedirs(result_root, exist_ok=True)

    annotated_dir = os.path.join(result_root, "annotated_tiles")
    os.makedirs(annotated_dir, exist_ok=True)

    det_dict = geo_det_utils.georeference_detections(
        tile_dets = tile_dets,
        score_thr = 0.0,
        apply_nms = args.nms_thr,
        annotated_out_dir = annotated_dir,
    )

    print("Total georeferenced detections:", len(det_dict))
    print("Annotated GeoTIFF tiles saved in:", annotated_dir)

    gpkg_path = os.path.join(result_root, "detections.gpkg")
    geo_det_utils.detections_to_gpkg(
        det_dict = det_dict,
        gpkg_path = gpkg_path,
        layer_name = "detections",
    )
