from osgeo import gdal, osr
from tqdm import tqdm
import simplekml
import math
import os
import numpy as np

class Colors:
    BLACK = '\033[30m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def msg(output, color):
    return color + f"{output} " + Colors.END


def get_bounding_box(dataset):
    """Return (min_x, min_y, max_x, max_y) from a GDAL dataset GeoTransform."""
    geotransform = dataset.GetGeoTransform()
    min_x = geotransform[0]
    max_y = geotransform[3]
    max_x = min_x + geotransform[1] * dataset.RasterXSize
    min_y = max_y + geotransform[5] * dataset.RasterYSize
    return min_x, min_y, max_x, max_y


def generate_guide(tif_path, name, output_path):
    """
    Create a KML guide with one polygon per tile in `tif_path`.
    """
    kml = simplekml.Kml()
    output_path = os.path.join(output_path, f"{name}.kml")

    tif_list = [f for f in os.listdir(tif_path) if f.lower().endswith(".tif")]

    for x in tqdm(tif_list, total=len(tif_list), desc="Generating Guide"):
        full_path = os.path.join(tif_path, x)
        dataset = gdal.Open(full_path, gdal.GA_ReadOnly)
        if dataset is None:
            print(msg(f"Could not open {full_path}", Colors.RED))
            continue

        min_x, min_y, max_x, max_y = get_bounding_box(dataset)

        bbox_polygon = kml.newpolygon()
        bbox_polygon.name = os.path.basename(x)
        bbox_polygon.outerboundaryis = [
            (min_x, min_y),
            (max_x, min_y),
            (max_x, max_y),
            (min_x, max_y),
            (min_x, min_y),
        ]

    kml.save(output_path)


def generate_test_images(extent, dimension=1024, stride=None, output_path="", isGuide=True):
    """
    Split a GeoTIFF into fixed-size tiles (dimension x dimension) preserving
    GeoTransform + Projection. Optionally generate a KML guide.

    extent: path to source .tif
    dimension: tile size in pixels (int)
    stride: step size for sliding window. Defaults to dimension (no overlap).
    output_path: directory where the tile folder + KML will be created
    isGuide: if True, writes a KML with tile footprints
    """
    assert isinstance(dimension, int), "Value must be integer."
    assert os.path.basename(extent).rsplit(".", 1)[-1].lower() == "tif", \
        "File is not a TIFF (.tif) file"

    if stride is None:
        stride = dimension

    try:
        path = extent
        tif_data = gdal.Open(path, gdal.GA_ReadOnly)
        if tif_data is None:
            raise RuntimeError(f"GDAL could not open {path}")

        extent_info = tif_data.GetGeoTransform()
        tif_name = os.path.basename(path).rsplit(".", 1)[0]

        size = dimension
        print("Test image dimension is set:", msg(f"{dimension}x{dimension}", Colors.GREEN))
        print("Stride is set to:", msg(f"{stride}", Colors.GREEN))

        output_kml = output_path
        output_path = os.path.join(output_path, f"{tif_name}_{size}x{size}")
        os.makedirs(output_path, exist_ok=True)
        print("Test image output path:", msg(f"{output_path}", Colors.GREEN))

        # Calculate steps
        x_steps = range(0, tif_data.RasterXSize, stride)
        y_steps = range(0, tif_data.RasterYSize, stride)

        print(
            "Output images grid approximate size:",
            msg(f"{len(x_steps)}x{len(y_steps)}", Colors.CYAN),
            "using dimension:",
            msg(f"{int(size)}x{int(size)}", Colors.CYAN),
        )
        print("Generating Images using extent:", msg(f"{tif_name}", Colors.GREEN))

        for idx, x in enumerate(x_steps):
            xmin_coordinate = x
            xmax_coordinate = x + size

            for idy, y in enumerate(
                tqdm(
                    y_steps,
                    total=len(y_steps),
                    desc=f"Generating grid {len(x_steps)}x{len(y_steps)} - {int(idx+1)}x{len(y_steps)}"
                )
            ):
                ymin_coordinate = y
                ymax_coordinate = y + size

                out_tif = os.path.join(output_path, f"{tif_name}_{idx+1}x{idy+1}.tif")

                # Handle edges
                # If window goes OOB, we can clamp or pad.
                # Current logic: read whatever is available (up to image edge) and write into 'size x size' buffer?
                # Or if read size < 'size', the Create() call makes a black-filled TIF of 'size x size'.

                # Dimensions to read from source
                read_x = min(size, tif_data.RasterXSize - xmin_coordinate)
                read_y = min(size, tif_data.RasterYSize - ymin_coordinate)

                # If stride logic pushes us completely out (shouldn't happen with range), we skip
                if read_x <= 0 or read_y <= 0:
                    continue

                out_dr = gdal.GetDriverByName('GTiff')
                out_ds = out_dr.Create(
                    out_tif,
                    size,
                    size,
                    tif_data.RasterCount,
                    tif_data.GetRasterBand(1).DataType
                )

                # Preserve projection + geotransform
                out_ds.SetProjection(tif_data.GetProjection())
                out_ds.SetGeoTransform((
                    extent_info[0] + xmin_coordinate * extent_info[1],
                    extent_info[1],
                    0,
                    extent_info[3] + ymin_coordinate * extent_info[5],
                    0,
                    extent_info[5]
                ))

                # Initialize with 0 (black) if we are on edge and have partial data
                # GDAL Create() does not guarantee init to 0, but GDT_Byte usually starts clean or we can force it.
                # Better to write the data into a buffer of zeros.

                for band_idx in range(1, tif_data.RasterCount + 1):
                    band = tif_data.GetRasterBand(band_idx)

                    # Read partial data
                    data = band.ReadAsArray(
                        xmin_coordinate,
                        ymin_coordinate,
                        read_x,
                        read_y
                    )

                    # If data is smaller than tile size, pad it
                    if data.shape[0] < size or data.shape[1] < size:
                        padded = list(data.shape)
                        padded[0] = size
                        padded[1] = size
                        # Depending on dtype, create zeros
                        # We assume same dtype as input
                        # Actually let's just use numpy to pad
                        canvas = np.zeros((size, size), dtype=data.dtype)
                        canvas[:data.shape[0], :data.shape[1]] = data
                        data = canvas

                    out_band = out_ds.GetRasterBand(band_idx)
                    out_band.WriteArray(data)

                out_ds = None  # flush

        print(msg("Successfully Generating Test Images (.tif)", Colors.GREEN))

        if isGuide:
            print("Generating kml guide based on the output path tifs: ",
                  msg(f"{output_path}", Colors.CYAN))
            generate_guide(output_path, tif_name, output_kml)
            print(msg("Successfully Generating Tif Guide (.kml)", Colors.GREEN))

    except Exception as e:
        print("An unexpected error occurred:", e)


def create_grid(extent, grid_n, output_path):
    """
    Alternative method: create a grid_n x grid_n grid over the extent
    and warp each cell into its own GeoTIFF.
    """
    print(msg(f"FILE: {extent} (TIF)", Colors.CYAN))
    dem = gdal.Open(extent)
    if dem is None:
        raise RuntimeError(f"GDAL could not open {extent}")

    gt = dem.GetGeoTransform()

    xmin = gt[0]
    ymax = gt[3]
    res = gt[1]

    xlen = res * dem.RasterXSize
    ylen = res * dem.RasterYSize
    div = grid_n

    xsize = xlen / div
    ysize = ylen / div

    xsteps = [xmin + xsize * i for i in range(div + 1)]
    ysteps = [ymax - ysize * i for i in range(div + 1)]

    name = os.path.basename(extent).rsplit(".")[0]
    save_path_ = os.path.join(output_path, f"grid_{grid_n}x{grid_n}_{name}")
    os.makedirs(save_path_, exist_ok=True)

    for i in range(div):
        for j in range(div):
            hold_xmin = xsteps[i]
            hold_xmax = xsteps[i + 1]
            hold_ymin = ysteps[j]
            hold_ymax = ysteps[j + 1]

            output_path_tif = os.path.join(save_path_, f"{name}_{i+1}x{j+1}.tif")

            try:
                if os.path.isfile(output_path_tif):
                    print(f"Exist: {os.path.basename(output_path_tif)}")
                else:
                    print(f"Generating: {output_path_tif}")
                    gdal.Warp(
                        output_path_tif,
                        dem,
                        outputBounds=(hold_xmin, hold_ymin, hold_xmax, hold_ymax),
                        dstNodata=-9999
                    )
                    print(f"Success: {output_path_tif}")
            except Exception as e:
                print(f"Error Generating: {output_path_tif}: {str(e)}")

    print("#############")
    print(msg("Success Generating Subset (GRID)", Colors.GREEN))
