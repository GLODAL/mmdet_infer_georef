import xml.etree.ElementTree as ET
from pyproj import Transformer
import re
import math
import requests
import os
import numpy as np
import argparse
from osgeo import gdal

# Import WKT from geo_det_utils to avoid duplication
from geo_det_utils import WKT_3857

def parse_kml_points(kml_path):
    tree = ET.parse(kml_path)
    root = tree.getroot()
    ns = {"kml": "http://www.opengis.net/kml/2.2"}

    points = []
    for pm in root.findall(".//kml:Placemark", ns):
        name_el = pm.find("kml:name", ns)
        name_raw = name_el.text if name_el is not None else "noname"

        coord_el = pm.find(".//kml:Point/kml:coordinates", ns)
        if coord_el is None or not coord_el.text:
            continue

        coord_text = coord_el.text.strip()
        parts = coord_text.split(",")
        if len(parts) < 2:
            continue

        lon = float(parts[0])
        lat = float(parts[1])
        points.append({"name": name_raw, "lat": lat, "lon": lon})

    return points

def create_bbox(lat, lon, buffer_m=250):
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x, y = transformer.transform(lon, lat)

    min_x, min_y = x - buffer_m, y - buffer_m
    max_x, max_y = x + buffer_m, y + buffer_m

    return (min_x, min_y, max_x, max_y)

def generate_image(bbox_3857, name, res_m=0.15):
    """
    bbox_3857: (min_x, min_y, max_x, max_y) in EPSG:3857 meters
    name: output GeoTIFF path (.tif)
    res_m: pixel size in meters (e.g. 0.15 = 15cm)
    """
    min_x, min_y, max_x, max_y = bbox_3857

    width = math.ceil((max_x - min_x) / res_m)
    height = math.ceil((max_y - min_y) / res_m)

    url = "http://mapproxy:3857/service"

    params = {
        "SERVICE": "WMS",
        "VERSION": "1.3.0",
        "REQUEST": "GetMap",
        "LAYERS": "GoogleMapsSatellite200712",
        "STYLES": "",
        "FORMAT": "image/tiff",
        "CRS": "EPSG:3857",
        "BBOX": f"{min_x},{min_y},{max_x},{max_y}",
        "WIDTH": width,
        "HEIGHT": height,
    }

    r = requests.get(url, params=params)
    print("Response code:", r.status_code)

    tmp_path = name + ".tmp.tif"
    os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
    with open(tmp_path, "wb") as f:
        f.write(r.content)

    src_ds = gdal.Open(tmp_path, gdal.GA_ReadOnly)
    if src_ds is None:
        raise RuntimeError(f"GDAL could not open WMS result: {tmp_path}")

    bands = src_ds.RasterCount
    arr = src_ds.ReadAsArray()
    src_ds = None

    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
        bands = 1

    _, h, w = arr.shape

    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(name, w, h, bands, gdal.GDT_Byte)

    gt = (min_x, res_m, 0.0, max_y, 0.0, -res_m)
    out_ds.SetGeoTransform(gt)
    out_ds.SetProjection(WKT_3857)

    for i in range(bands):
        out_ds.GetRasterBand(i + 1).WriteArray(arr[i])

    out_ds.FlushCache()
    out_ds = None

    os.remove(tmp_path)
    print("✅ Saved clean GeoTIFF with CRS:", name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch satellite imagery from WMS based on KML points.")
    parser.add_argument("--kml", required=True, help="Path to the input KML file.")
    parser.add_argument("--out_dir", required=True, help="Directory to save the output GeoTIFFs.")
    parser.add_argument("--buffer_m", type=float, default=250, help="Buffer distance in meters around the point.")
    parser.add_argument("--res_m", type=float, default=0.15, help="Pixel resolution in meters.")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    pts = parse_kml_points(args.kml)
    print("Loaded features:", len(pts))

    for i, p in enumerate(pts):
        name_raw = p["name"]
        lat, lon = p["lat"], p["lon"]

        fname = re.sub(r"[^a-zA-Z0-9]", "", name_raw) or f"loc_{i}"
        print("Location:", fname)
        print("Lat/Lon:", lat, lon)

        bbox_3857 = create_bbox(lat, lon, buffer_m=args.buffer_m)
        tif_path = os.path.join(args.out_dir, f"{fname}.tif")
        generate_image(bbox_3857, tif_path, res_m=args.res_m)
