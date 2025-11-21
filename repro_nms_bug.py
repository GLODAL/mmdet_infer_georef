from typing import Dict, Any

# Mock logic from the updated code
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

def test_nms_bug():
    # Simulate two detections from TWO DIFFERENT TILES
    # Tile 1: Top Left Tile (0,0)
    # Tile 2: Bottom Right Tile (1,1) - far away in reality!

    det1 = {
        "coordinate": [(0,10), (10,10), (10,0), (0,0)],
        "xy": [50, 50, 100, 100], # Local Pixel coords
        "global_bbox": [0, 0, 10, 10], # Global coords
        "label": 1,
        "score": 0.9,
        "tile_path": "tile_1.tif"
    }

    det2 = {
        "coordinate": [(1000,1010), (1010,1010), (1010,1000), (1000,1000)],
        "xy": [50, 50, 100, 100], # Local Pixel coords (SAME AS DET1)
        "global_bbox": [1000, 1000, 1010, 1010], # Global coords (FAR AWAY)
        "label": 1,
        "score": 0.85,
        "tile_path": "tile_2.tif"
    }

    all_dets = {
        0: det1,
        1: det2
    }

    print(f"Before NMS: {len(all_dets)} detections")

    # Apply NMS with low threshold
    result = non_max_suppression(all_dets, iou_thr=0.5)

    print(f"After NMS: {len(result)} detections")

    if len(result) == 2:
        print("PASS: NMS kept both detections because they are far apart globally.")
    else:
        print("FAIL: NMS merged detections!")

if __name__ == "__main__":
    test_nms_bug()
