"""
Geospatial utility functions.
Handles coordinate transforms, tiling with georeference preservation,
and GeoJSON export of detection results.
"""

import json
import numpy as np
from pathlib import Path

try:
    import rasterio
    from rasterio.windows import Window
    from rasterio.transform import from_bounds, rowcol, xy
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("Warning: rasterio not installed. Georeferencing will be limited.")

try:
    from pyproj import Transformer
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False


def get_geotransform(tif_path):
    """
    Extract geotransform and CRS from a GeoTIFF file.

    Args:
        tif_path: Path to GeoTIFF

    Returns:
        dict with 'transform', 'crs', 'width', 'height', 'bounds'
    """
    if not HAS_RASTERIO:
        raise ImportError("rasterio required for georeferencing")

    with rasterio.open(tif_path) as src:
        return {
            "transform": src.transform,
            "crs": src.crs,
            "width": src.width,
            "height": src.height,
            "bounds": src.bounds,
        }


def pixel_to_geo(row, col, transform):
    """
    Convert pixel coordinates to geographic coordinates.

    Args:
        row: Pixel row (y)
        col: Pixel column (x)
        transform: Affine transform from rasterio

    Returns:
        (longitude, latitude) tuple
    """
    x, y = xy(transform, row, col)
    return x, y


def tile_geotiff(tif_path, tile_size=640, overlap=64, output_dir=None):
    """
    Tile a large GeoTIFF into smaller chips, preserving georeference info.

    Each tile is saved as a separate file, along with a metadata JSON
    containing the tile's geographic bounds (needed for georeferencing
    detections back to world coordinates).

    Args:
        tif_path: Path to input GeoTIFF
        tile_size: Tile dimensions in pixels
        overlap: Overlap between tiles in pixels
        output_dir: Output directory for tiles

    Returns:
        List of dicts with tile info: {path, row_off, col_off, transform, bounds}
    """
    if not HAS_RASTERIO:
        raise ImportError("rasterio is required for tiling GeoTIFFs")

    tif_path = Path(tif_path)
    if output_dir is None:
        output_dir = tif_path.parent / "tiles"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tiles_info = []
    step = tile_size - overlap

    with rasterio.open(tif_path) as src:
        height, width = src.height, src.width

        tile_idx = 0
        for row_off in range(0, height - tile_size + 1, step):
            for col_off in range(0, width - tile_size + 1, step):
                window = Window(col_off, row_off, tile_size, tile_size)
                tile_transform = src.window_transform(window)

                # Read tile data
                tile_data = src.read(window=window)

                # Skip mostly-empty tiles (nodata)
                if np.mean(tile_data == 0) > 0.5:
                    continue

                # Save tile
                tile_path = output_dir / f"tile_{tile_idx:05d}.tif"
                tile_meta = src.meta.copy()
                tile_meta.update({
                    "width": tile_size,
                    "height": tile_size,
                    "transform": tile_transform,
                })

                with rasterio.open(tile_path, "w", **tile_meta) as dst:
                    dst.write(tile_data)

                # Compute tile bounds
                tile_bounds = rasterio.transform.array_bounds(
                    tile_size, tile_size, tile_transform
                )

                tiles_info.append({
                    "path": str(tile_path),
                    "tile_idx": tile_idx,
                    "row_off": row_off,
                    "col_off": col_off,
                    "transform": list(tile_transform)[:6],
                    "bounds": list(tile_bounds),
                })

                tile_idx += 1

    # Save tile metadata
    meta_path = output_dir / "tiles_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(tiles_info, f, indent=2)

    print(f"Created {len(tiles_info)} tiles in {output_dir}/")
    return tiles_info


def detections_to_geojson(detections, tile_transforms, output_path,
                          source_crs=None, tiles_meta_path=None):
    """
    Convert pixel-space detections to a georeferenced GeoJSON file in WGS84.

    Sentinel-1 tiles are typically in a projected CRS (UTM), so this
    function reprojects coordinates to EPSG:4326 (lat/lon) for
    compatibility with web GIS tools like Kepler.gl, QGIS, etc.

    Args:
        detections: List of dicts with keys:
            - tile_idx: which tile the detection came from
            - bbox: [x1, y1, x2, y2] in pixel coords within the tile
            - confidence: detection confidence score
            - class_name: detected class label
        tile_transforms: Dict mapping tile_idx → affine transform parameters
        output_path: Path for output GeoJSON file
        source_crs: CRS of the tiles (auto-detected from first tile if None)
        tiles_meta_path: Path to tiles_metadata.json (to find a tile for CRS)

    Returns:
        Path to saved GeoJSON file
    """
    if not detections:
        print("No detections to export.")
        return output_path

    # Detect source CRS from a tile file if not provided
    if source_crs is None:
        # Try to read CRS from an actual tile
        for det in detections:
            tile_path = det.get("tile_path")
            if tile_path and Path(tile_path).exists():
                try:
                    with rasterio.open(tile_path) as src:
                        source_crs = src.crs
                    break
                except Exception:
                    pass

    # Set up reprojection if needed
    need_reproject = False
    transformer = None
    if source_crs and HAS_PYPROJ:
        crs_str = str(source_crs)
        if "4326" not in crs_str:
            # Source is not WGS84 — need to reproject
            transformer = Transformer.from_crs(source_crs, "EPSG:4326",
                                                always_xy=True)
            need_reproject = True
            print(f"  Reprojecting from {source_crs} → EPSG:4326")
    elif source_crs is None:
        print("  ⚠ Could not detect CRS — coordinates may not be lat/lon")

    features = []

    for det in detections:
        tile_idx = det["tile_idx"]
        x1, y1, x2, y2 = det["bbox"]
        conf = det["confidence"]

        # Get tile's affine transform
        if tile_idx not in tile_transforms:
            continue
        t = tile_transforms[tile_idx]
        if isinstance(t, list):
            from rasterio.transform import Affine
            transform = Affine(*t)
        else:
            transform = t

        # Convert bbox corners to map coordinates (native CRS)
        # pixel_to_geo uses rasterio.transform.xy(transform, row, col)
        map_x1, map_y1 = pixel_to_geo(y1, x1, transform)
        map_x2, map_y2 = pixel_to_geo(y2, x2, transform)

        # Reproject to WGS84 if needed
        if need_reproject and transformer:
            lon1, lat1 = transformer.transform(map_x1, map_y1)
            lon2, lat2 = transformer.transform(map_x2, map_y2)
        else:
            lon1, lat1 = map_x1, map_y1
            lon2, lat2 = map_x2, map_y2

        # Validate coordinates
        if not (-180 <= lon1 <= 180 and -90 <= lat1 <= 90):
            continue  # Skip invalid coordinates

        # Center point
        cx, cy = (lon1 + lon2) / 2, (lat1 + lat2) / 2

        # Create GeoJSON polygon for the bounding box
        coords = [
            [lon1, lat1],
            [lon2, lat1],
            [lon2, lat2],
            [lon1, lat2],
            [lon1, lat1],  # Close the ring
        ]

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [coords],
            },
            "properties": {
                "class": det.get("class_name", "ship"),
                "confidence": round(conf, 4),
                "center_lon": round(cx, 6),
                "center_lat": round(cy, 6),
                "tile_idx": tile_idx,
                "bbox_px": [int(x1), int(y1), int(x2), int(y2)],
            },
        }
        features.append(feature)

    geojson = {
        "type": "FeatureCollection",
        "features": features,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(geojson, f, indent=2)

    print(f"Saved {len(features)} detections to {output_path}")
    if features:
        lons = [f["properties"]["center_lon"] for f in features]
        lats = [f["properties"]["center_lat"] for f in features]
        print(f"  Lon range: [{min(lons):.4f}, {max(lons):.4f}]")
        print(f"  Lat range: [{min(lats):.4f}, {max(lats):.4f}]")
    return output_path
