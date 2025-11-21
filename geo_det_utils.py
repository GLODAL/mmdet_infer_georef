from __future__ import annotations

import os
from typing import Dict, Any

import cv2
import numpy as np
from tqdm import tqdm
from osgeo import gdal, ogr, osr

from mmdet.apis import inference_detector


class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    CYAN = '\033[96m'
    END = '\033[0m'


def msg(text, color=Colors.CYAN):
    return f"{color}{text}{Colors.END}"


# ---------------------------------------------------------
# Hard-coded WKT for EPSG:3857 (WGS 84 / Pseudo-Mercator)
# -> avoids needing proj.db on the server
# ---------------------------------------------------------
WKT_3857 = (
    'PROJCS["WGS 84 / Pseudo-Mercator",'
    'GEOGCS["WGS 84",'
    'DATUM["WGS_1984",'
    'SPHEROID["WGS 84",6378137,298.257223563]],'
    'PRIMEM["Greenwich",0],'
    'UNIT["degree",0.0174532925199433]],'
    'PROJECTION["Mercator_1SP"],'
    'PARAMETER["central_meridian",0],'
    'PARAMETER["scale_factor",1],'
    'PARAMETER["false_easting",0],'
    'PARAMETER["false_northing",0],'
    'UNIT["metre",1],'
    'AXIS["X",EAST],AXIS["Y",NORTH],'
    'AUTHORITY["EPSG","3857"]]'
)


# ---------------------------------------------------------
# 1. Extract bboxes/scores/labels from MMDet result
# ---------------------------------------------------------
def _extract_bboxes(result) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize MMDet result into (bboxes, scores, labels).
    """
    # mmdet 3.x: usually list with one element
    if isinstance(result, (list, tuple)):
        if len(result) == 0:
            return np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=int)
        result = result[0]

    # InstanceData style
    if hasattr(result, "pred_instances"):
        inst = result.pred_instances
    elif isinstance(result, dict) and "pred_instances" in result:
        inst = result["pred_instances"]
    else:
        raise ValueError(f"Unsupported result type: {type(result)}")

    bboxes = inst.bboxes.cpu().numpy()
    scores = inst.scores.cpu().numpy()
    labels = inst.labels.cpu().numpy()
    return bboxes, scores, labels


# ---------------------------------------------------------
# 2. Inference on a folder of tiles (pixel space only)
# -> returns per-tile detections in pixel coords
# ---------------------------------------------------------
def infer_folder_pixels(
    model,
    folder_path: str,
    score_thr: float = 0.8,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Run MMDet model on all .tif tiles in folder and return
    per-tile detections (pixel space).

    Returns:
    tile_dets: {
        tile_path: {
            "bboxes": (N,4),
            "scores": (N,),
            "labels": (N,)
        }, ...
    }
    """
    assert os.path.isdir(folder_path), f"{folder_path} is not a folder."

    tif_files = sorted(
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".tif", ".tiff"))
    )
    if not tif_files:
        raise RuntimeError(f"No TIF/TIFF files found in {folder_path}")

    print(msg(f"[DEBUG] Inferencing folder (tiles only): {folder_path} ({len(tif_files)} tiles)"))

    tile_dets: Dict[str, Dict[str, np.ndarray]] = {}

    for fname in tqdm(tif_files, total=len(tif_files), desc="Inferencing tiles"):
        tif_path = os.path.join(folder_path, fname)

        img = cv2.imread(tif_path, cv2.IMREAD_COLOR)
        if img is None:
            print(msg(f"[WARN] cv2 could not read {tif_path}", Colors.RED))
            continue

        result = inference_detector(model, img)
        bboxes, scores, labels = _extract_bboxes(result)

        mask = scores >= score_thr
        bboxes_f = bboxes[mask]
        scores_f = scores[mask]
        labels_f = labels[mask]

        tile_dets[tif_path] = {
            "bboxes": bboxes_f,
            "scores": scores_f,
            "labels": labels_f,
        }

    print(msg("[OK] Tile inference complete", Colors.GREEN))
    return tile_dets


# ---------------------------------------------------------
# 3. Global NMS (optional)
# ---------------------------------------------------------
def non_max_suppression(data_dict: Dict[int, Dict[str, Any]], iou_thr: float):
    """
    NMS in global space to reduce overlapping detections.
    Assumes 'global_bbox' contains [min_x, min_y, max_x, max_y].
    """
    if not data_dict:
        return data_dict

    order = dict(sorted(data_dict.items(), key=lambda kv: kv[1]["score"], reverse=True))
    keep: Dict[int, Dict[str, Any]] = {}

    while order:
        idx = next(iter(order.keys()))
        base = order.pop(idx)
        keep[idx] = base

        # Use global bbox for NMS
        x1, y1, x2, y2 = base["global_bbox"]
        base_area = (x2 - x1) * (y2 - y1)

        to_delete = []
        for j, cand in order.items():
            cx1, cy1, cx2, cy2 = cand["global_bbox"]
            inter_w = max(0.0, min(x2, cx2) - max(x1, cx1))
            inter_h = max(0.0, min(y2, cy2) - max(y1, cy1))
            inter = inter_w * inter_h

            cand_area = (cx2 - cx1) * (cy2 - cy1)
            union = base_area + cand_area - inter
            iou = inter / union if union > 0 else 0.0

            if iou > iou_thr:
                to_delete.append(j)

        for j in to_delete:
            order.pop(j, None)

    return keep


# ---------------------------------------------------------
# 4. Draw bboxes on a GeoTIFF and save new GeoTIFF
# -> keeps GeoTransform, hard-codes CRS (EPSG:3857)
# ---------------------------------------------------------
def draw_bboxes_on_geotiff(
    src_tif: str,
    out_tif: str,
    det_list_xy: list[list[float]],
    color=(0, 0, 255),
    thickness=2,
):
    ds = gdal.Open(src_tif, gdal.GA_ReadOnly)
    if ds is None:
        print(msg(f"[WARN] Could not open {src_tif}", Colors.RED))
        return

    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()  # may be empty, we’ll override anyway
    arr = ds.ReadAsArray()  # (bands, H, W) or (H, W)
    ds = None

    # ---- Normalize to H x W x 3 ----
    if arr.ndim == 2:
        base = arr
        img = np.stack([base, base, base], axis=-1)
    elif arr.ndim == 3:
        bands_in = arr.shape[0]
        if bands_in >= 3:
            img = np.transpose(arr[:3, ...], (1, 2, 0))  # -> H, W, 3
        else:
            base = arr[0]
            img = np.stack([base, base, base], axis=-1)
    else:
        raise ValueError(f"Unexpected array shape from GDAL: {arr.shape}")

    # Ensure uint8 + C-contiguous for OpenCV
    img = np.array(img, dtype=np.uint8, copy=True)
    img = np.ascontiguousarray(img)

    # ---- Draw each bbox ----
    for (x1, y1, x2, y2) in det_list_xy:
        p1 = (int(x1), int(y1))
        p2 = (int(x2), int(y2))
        cv2.rectangle(img, p1, p2, color, thickness)

    # ---- Write back to GeoTIFF with same georef + hard-coded CRS ----
    h, w, _ = img.shape
    img_t = np.transpose(img, (2, 0, 1))  # (3, H, W)

    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(out_tif, w, h, 3, gdal.GDT_Byte)
    out_ds.SetGeoTransform(gt)

    # If original proj is empty, or we just want consistency, override:
    out_ds.SetProjection(WKT_3857)

    for i in range(3):
        out_ds.GetRasterBand(i + 1).WriteArray(img_t[i])

    out_ds.FlushCache()
    out_ds = None

    print(msg(f"[OK] Annotated GeoTIFF saved: {out_tif}", Colors.GREEN))


# ---------------------------------------------------------
# 5. Georeference detections (tile_dets -> global coords)
# + optionally write annotated GeoTIFF tiles
# ---------------------------------------------------------
def georeference_detections(
    tile_dets: Dict[str, Dict[str, np.ndarray]],
    score_thr: float = 0.0,
    apply_nms: float | bool = False,
    annotated_out_dir: str | None = None,
) -> Dict[int, Dict[str, Any]]:
    """
    Convert per-tile pixel detections to map coordinates using each tile's
    GeoTransform, and optionally write annotated GeoTIFFs.

    tile_dets structure (from infer_tiles):
    {
        tile_path: {
            "bboxes": (N,4),
            "scores": (N,),
            "labels": (N,)
        }, ...
    }

    Returns:
    det_dict: {
        idx: {
            "coordinate": [(X,Y), ...], # map coords
            "xy": [x1,y1,x2,y2], # pixel coords
            "label": int,
            "score": float,
            "tile_path": str,
        }, ...
    }
    """
    if annotated_out_dir is not None:
        os.makedirs(annotated_out_dir, exist_ok=True)

    all_dets: Dict[int, Dict[str, Any]] = {}

    for tile_path, det in tile_dets.items():
        bboxes = det["bboxes"]
        scores = det["scores"]
        labels = det["labels"]

        if bboxes.size == 0:
            continue

        ds = gdal.Open(tile_path, gdal.GA_ReadOnly)
        if ds is None:
            print(msg(f"[WARN] Could not open {tile_path}", Colors.RED))
            continue
        gt = ds.GetGeoTransform()
        ds = None

        mask = scores >= score_thr
        bboxes_f = bboxes[mask]
        scores_f = scores[mask]
        labels_f = labels[mask]

        if bboxes_f.shape[0] == 0:
            continue

        # 1) build global detection dict
        for i, box in enumerate(bboxes_f):
            score = float(scores_f[i])
            label = int(labels_f[i])

            x1, y1, x2, y2 = [float(v) for v in box]

            # Calculate Geo Coords
            # Note: gt[5] is usually negative (north-up image)
            # We calculate 4 corners or just min/max to be safe if there's rotation (though simple affine usually implies 0 rotation here)

            # Top-Left pixel (x1, y1) -> Geo
            geo_x_tl = gt[0] + x1 * gt[1] + y1 * gt[2]
            geo_y_tl = gt[3] + x1 * gt[4] + y1 * gt[5]

            # Bottom-Right pixel (x2, y2) -> Geo
            geo_x_br = gt[0] + x2 * gt[1] + y2 * gt[2]
            geo_y_br = gt[3] + x2 * gt[4] + y2 * gt[5]

            # Determine min/max for global bbox (axis-aligned)
            g_xmin = min(geo_x_tl, geo_x_br)
            g_xmax = max(geo_x_tl, geo_x_br)
            g_ymin = min(geo_y_tl, geo_y_br)
            g_ymax = max(geo_y_tl, geo_y_br)

            # Polygon for GPKG (can be more complex if needed, but here we stick to box)
            # Note: Keeping original logic of 4 corners, but derived from g_min/g_max to be safe
            poly_coords = [
                (g_xmin, g_ymax),
                (g_xmax, g_ymax),
                (g_xmax, g_ymin),
                (g_xmin, g_ymin),
                (g_xmin, g_ymax),
            ]

            all_dets[len(all_dets)] = {
                "coordinate": poly_coords,
                "xy": [x1, y1, x2, y2], # Local pixel coords
                "global_bbox": [g_xmin, g_ymin, g_xmax, g_ymax], # Global map coords for NMS
                "label": label,
                "score": score,
                "tile_path": tile_path,
            }

        # 2) annotated tiles
        if annotated_out_dir is not None:
            det_list_xy = [list(map(float, bb)) for bb in bboxes_f]
            out_tif = os.path.join(annotated_out_dir, os.path.basename(tile_path))
            draw_bboxes_on_geotiff(
                src_tif=tile_path,
                out_tif=out_tif,
                det_list_xy=det_list_xy,
                color=(0, 0, 255),
                thickness=2,
            )

    iou_thr = None
    if isinstance(apply_nms, bool) and apply_nms:
        iou_thr = 0.5 # Default
    elif isinstance(apply_nms, float) and apply_nms > 0:
        iou_thr = apply_nms

    if iou_thr is not None:
        print(msg(f"[INFO] Applying global NMS with IoU={iou_thr}", Colors.CYAN))
        all_dets = non_max_suppression(all_dets, iou_thr=iou_thr)

    print(msg("[OK] Georeferencing complete", Colors.GREEN))
    return all_dets


# ---------------------------------------------------------
# 6. Export detections to GPKG (CRS from WKT_3857)
# ---------------------------------------------------------
def detections_to_gpkg(
    det_dict: Dict[int, Dict[str, Any]],
    gpkg_path: str,
    layer_name: str = "detections",
):
    drv = ogr.GetDriverByName("GPKG")
    if os.path.exists(gpkg_path):
        drv.DeleteDataSource(gpkg_path)

    ds = drv.CreateDataSource(gpkg_path)
    if ds is None:
        raise RuntimeError(f"Cannot create GPKG: {gpkg_path}")

    srs = osr.SpatialReference()
    srs.ImportFromWkt(WKT_3857)

    layer = ds.CreateLayer(layer_name, srs=srs, geom_type=ogr.wkbPolygon)

    fld_score = ogr.FieldDefn("score", ogr.OFTReal)
    layer.CreateField(fld_score)
    fld_label = ogr.FieldDefn("label", ogr.OFTInteger)
    layer.CreateField(fld_label)

    defn = layer.GetLayerDefn()

    for det in det_dict.values():
        coords = det["coordinate"]  # list of (x, y)
        ring = ogr.Geometry(ogr.wkbLinearRing)
        for x, y in coords:
            ring.AddPoint(float(x), float(y))

        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(ring)

        feat = ogr.Feature(defn)
        feat.SetGeometry(poly)
        feat.SetField("score", float(det["score"]))
        feat.SetField("label", int(det["label"]))
        layer.CreateFeature(feat)
        feat = None

    ds = None  # close file
    print(msg(f"[OK] GPKG saved: {gpkg_path}", Colors.GREEN))
