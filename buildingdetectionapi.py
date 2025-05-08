# import os
# import cv2
# import numpy as np
# import rasterio
# import geopandas as gpd
# from rasterio.features import shapes
# from shapely.geometry import shape
# from PIL import Image
# from ultralyticsplus import YOLO
# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import FileResponse
# import shutil

# app = FastAPI()

# # Load YOLO model for building segmentation
# import warnings
# warnings.simplefilter("ignore", category=FutureWarning)

# # Load YOLO model
# model = YOLO("keremberke/yolov8n-building-segmentation")  # Explicitly set weights_only=True

# model.overrides['conf'] = 0.25
# model.overrides['iou'] = 0.45
# model.overrides['max_det'] = 1000000

# def read_tiff(tiff_path):
#     with rasterio.open(tiff_path) as src:
#         image = src.read([1, 2, 3])
#         meta = src.meta
#         transform = src.transform
#     return np.dstack(image), meta, transform

# def pad_image(image, crop_size=512):
#     h, w, c = image.shape
#     pad_h = (crop_size - h % crop_size) % crop_size
#     pad_w = (crop_size - w % crop_size) % crop_size
#     padded_image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
#     return padded_image, (h, w)

# def crop_image(image, crop_size=512):
#     h, w, c = image.shape
#     tiles, positions = [], []
#     for y in range(0, h, crop_size):
#         for x in range(0, w, crop_size):
#             crop = image[y:y+crop_size, x:x+crop_size]
#             tiles.append(crop)
#             positions.append((x, y))
#     return tiles, positions, image.shape

# def run_model_on_tiles(tiles, crop_size=512):
#     predicted_tiles = []
#     for tile in tiles:
#         # Ensure tile is uint8 before processing
#         if tile.dtype == np.uint16:
#             tile = (tile / 256).astype(np.uint8)

#         pil_image = Image.fromarray(cv2.cvtColor(tile, cv2.COLOR_BGR2RGB))
#         results = model.predict(pil_image)
#         mask = np.zeros((crop_size, crop_size), dtype=np.uint8)
#         if results and results[0].masks is not None:
#             for mask_poly in results[0].masks.xy:
#                 if mask_poly is None or len(mask_poly) < 3:
#                     continue
#                 try:
#                     mask_poly = np.array(mask_poly, dtype=np.int32)
#                     cv2.fillPoly(mask, [mask_poly], 255)
#                 except Exception:
#                     continue
#         predicted_tiles.append(mask)
#     return predicted_tiles


# def merge_tiles(predicted_tiles, positions, padded_shape):
#     h, w, _ = padded_shape
#     merged_mask = np.zeros((h, w), dtype=np.uint8)
#     for tile, (x, y) in zip(predicted_tiles, positions):
#         merged_mask[y:y+tile.shape[0], x:x+tile.shape[1]] = tile
#     return merged_mask

# def remove_padding(mask, original_shape):
#     h, w = original_shape
#     return mask[:h, :w]

# def mask_to_geojson(mask, transform, geojson_path):
#     mask = (mask > 0).astype(np.uint8)
#     shapes_list = [shape(geom) for geom, value in shapes(mask, transform=transform) if value == 1]
#     gdf = gpd.GeoDataFrame(geometry=shapes_list, crs="EPSG:4326")
#     gdf.to_file(geojson_path, driver="GeoJSON")
#     return geojson_path

# @app.post("/process_tiff/")
# async def process_tiff(file: UploadFile = File(...)):
#     upload_dir = "uploads"
#     output_dir = "outputs"
#     os.makedirs(upload_dir, exist_ok=True)
#     os.makedirs(output_dir, exist_ok=True)
    
#     tiff_path = os.path.join(upload_dir, file.filename)
#     output_geojson = os.path.join(output_dir, file.filename.replace(".tif", ".geojson"))
    
#     with open(tiff_path, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)
    
#     image, meta, transform = read_tiff(tiff_path)
#     padded_image, original_shape = pad_image(image)
#     tiles, positions, padded_shape = crop_image(padded_image)
#     predicted_tiles = run_model_on_tiles(tiles)
#     merged_mask = merge_tiles(predicted_tiles, positions, padded_shape)
#     final_mask = remove_padding(merged_mask, original_shape)
#     geojson_path = mask_to_geojson(final_mask, transform, output_geojson)
    
#     return FileResponse(geojson_path, media_type="application/geo+json", filename=os.path.basename(geojson_path))
import os
import cv2
import numpy as np
import rasterio
import geopandas as gpd
from rasterio.features import shapes
from shapely.geometry import shape
from PIL import Image
from ultralyticsplus import YOLO
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import shutil
import warnings

warnings.simplefilter("ignore", category=FutureWarning)

app = FastAPI()

# Load YOLO model
model = YOLO("keremberke/yolov8n-building-segmentation")

# Improve detection quality
model.overrides['conf'] = 0.60  # Increased confidence to reduce false positives
model.overrides['iou'] = 0.45
model.overrides['max_det'] = 1000000

def read_tiff(tiff_path):
    with rasterio.open(tiff_path) as src:
        image = src.read([1, 2, 3])
        meta = src.meta
        transform = src.transform
    return np.dstack(image), meta, transform

def pad_image(image, crop_size=512):
    h, w, c = image.shape
    pad_h = (crop_size - h % crop_size) % crop_size
    pad_w = (crop_size - w % crop_size) % crop_size
    padded_image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return padded_image, (h, w)

def crop_image(image, crop_size=512):
    h, w, c = image.shape
    tiles, positions = [], []
    for y in range(0, h, crop_size):
        for x in range(0, w, crop_size):
            crop = image[y:y+crop_size, x:x+crop_size]
            tiles.append(crop)
            positions.append((x, y))
    return tiles, positions, image.shape

def run_model_on_tiles(tiles, crop_size=512):
    predicted_tiles = []
    for tile in tiles:
        if tile.dtype == np.uint16:
            tile = (tile / 256).astype(np.uint8)
        pil_image = Image.fromarray(cv2.cvtColor(tile, cv2.COLOR_BGR2RGB))
        results = model.predict(pil_image)

        mask = np.zeros((crop_size, crop_size), dtype=np.uint8)
        if results and results[0].masks is not None:
            for mask_poly in results[0].masks.xy:
                if mask_poly is None or len(mask_poly) < 3:
                    continue
                try:
                    mask_poly = np.array(mask_poly, dtype=np.int32)
                    cv2.fillPoly(mask, [mask_poly], 255)
                except Exception:
                    continue
        predicted_tiles.append(mask)
    return predicted_tiles

def merge_tiles(predicted_tiles, positions, padded_shape):
    h, w, _ = padded_shape
    merged_mask = np.zeros((h, w), dtype=np.uint8)
    for tile, (x, y) in zip(predicted_tiles, positions):
        merged_mask[y:y+tile.shape[0], x:x+tile.shape[1]] = tile
    return merged_mask

def remove_padding(mask, original_shape):
    h, w = original_shape
    return mask[:h, :w]

def filter_small_objects(mask, min_area=300):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filtered_mask = np.zeros_like(mask)
    for i in range(1, num_labels):  # Skip background
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            filtered_mask[labels == i] = 255
    return filtered_mask

def mask_to_geojson(mask, transform, geojson_path):
    mask = (mask > 0).astype(np.uint8)
    shapes_list = [shape(geom) for geom, value in shapes(mask, transform=transform) if value == 1]
    if not shapes_list:
        return None
    gdf = gpd.GeoDataFrame(geometry=shapes_list, crs="EPSG:4326")
    gdf.to_file(geojson_path, driver="GeoJSON")
    return geojson_path

@app.post("/process_tiff/")
async def process_tiff(file: UploadFile = File(...)):
    upload_dir = "uploads"
    output_dir = "outputs"
    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    tiff_path = os.path.join(upload_dir, file.filename)
    output_geojson = os.path.join(output_dir, file.filename.replace(".tif", ".geojson"))

    with open(tiff_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    image, meta, transform = read_tiff(tiff_path)
    padded_image, original_shape = pad_image(image)
    tiles, positions, padded_shape = crop_image(padded_image)
    predicted_tiles = run_model_on_tiles(tiles)
    merged_mask = merge_tiles(predicted_tiles, positions, padded_shape)
    final_mask = remove_padding(merged_mask, original_shape)

    # Remove small blobs and noise
    filtered_mask = filter_small_objects(final_mask, min_area=300)

    # Check if any buildings were detected
    if not np.any(filtered_mask):
        return {"message": "No buildings detected with high confidence."}

    geojson_path = mask_to_geojson(filtered_mask, transform, output_geojson)
    if geojson_path:
        return FileResponse(geojson_path, media_type="application/geo+json", filename=os.path.basename(geojson_path))
    else:
        return {"message": "No valid building geometries to save."}
