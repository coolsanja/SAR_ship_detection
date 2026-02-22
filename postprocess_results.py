"""
Post-process detection results.

  1. Merge overlapping detections from adjacent tiles (cross-tile NMS)
  2. Georeference detections using tile metadata
  3. Export to GeoJSON 
  4. Generate visualization plots

Usage:
    python postprocess_results.py
"""

import json
import numpy as np
from pathlib import Path

from config import OUTPUT_DIR, PROCESSED_DIR, AOI_NAME, IOU_THRESHOLD
from utils.geo_utils import detections_to_geojson
from utils.viz_utils import (plot_detections_on_tile, plot_detection_grid,
                              plot_confidence_histogram)


def compute_iou(box1, box2):
    """Compute IoU between two bounding boxes [x1,y1,x2,y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / (union + 1e-10)


def cross_tile_nms(detections, tile_metadata, iou_threshold=0.3):
    """
    Non-maximum suppression across overlapping tiles.

    When tiles overlap, the same ship can be detected in multiple tiles.
    This function converts detections to a common coordinate space
    (full-scene pixel coords) and merges duplicates.

    Args:
        detections: List of detection dicts
        tile_metadata: List of tile info dicts with 'row_off', 'col_off'
        iou_threshold: IoU threshold for merging

    Returns:
        Deduplicated detections
    """
    if not detections:
        return []

    # Build tile offset lookup
    tile_offsets = {}
    for t in tile_metadata:
        tile_offsets[t["tile_idx"]] = (t["col_off"], t["row_off"])

    # Convert all bboxes to full-scene coordinates
    scene_dets = []
    for det in detections:
        tidx = det["tile_idx"]
        if tidx not in tile_offsets:
            continue

        col_off, row_off = tile_offsets[tidx]
        x1, y1, x2, y2 = det["bbox"]

        scene_dets.append({
            **det,
            "scene_bbox": [
                x1 + col_off, y1 + row_off,
                x2 + col_off, y2 + row_off
            ],
        })

    # Sort by confidence (descending)
    scene_dets.sort(key=lambda d: d["confidence"], reverse=True)

    # Greedy NMS
    keep = []
    suppressed = set()

    for i, det in enumerate(scene_dets):
        if i in suppressed:
            continue
        keep.append(det)

        for j in range(i + 1, len(scene_dets)):
            if j in suppressed:
                continue
            if compute_iou(det["scene_bbox"], scene_dets[j]["scene_bbox"]) > iou_threshold:
                suppressed.add(j)

    print(f"Cross-tile NMS: {len(detections)} → {len(keep)} detections "
          f"({len(detections) - len(keep)} duplicates removed)")

    return keep


def georeference_detections(detections, tile_metadata):
    """
    Convert detections from pixel coordinates to geographic coordinates
    and export as GeoJSON.

    Args:
        detections: List of detection dicts
        tile_metadata: Tile metadata with transform info

    Returns:
        Path to GeoJSON file
    """
    # Build transform lookup
    tile_transforms = {}
    for t in tile_metadata:
        tile_transforms[t["tile_idx"]] = t["transform"]

    # Export to GeoJSON
    geojson_path = OUTPUT_DIR / "ship_detections.geojson"
    detections_to_geojson(
        detections=detections,
        tile_transforms=tile_transforms,
        output_path=geojson_path,
    )

    return geojson_path


def generate_visualizations(detections, output_dir, tiles_dir=None):
    """
    Generate presentation-ready plots.

    Renders:
      1. Confidence histogram (if detections exist)
      2. Top-detection tiles (tiles with most detections)
      3. User-specified tiles (from VISUALIZE_TILES in config, rendered
         regardless of whether they contain detections)
    """
    output_dir = Path(output_dir)

    # Confidence histogram
    if detections:
        confidences = [d["confidence"] for d in detections]
        plot_confidence_histogram(
            confidences,
            save_path=output_dir / "confidence_distribution.png"
        )

    # Group detections by tile
    by_tile = {}
    if detections:
        for det in detections:
            tidx = det["tile_idx"]
            if tidx not in by_tile:
                by_tile[tidx] = []
            by_tile[tidx].append(det)

    # Resolve tiles directory
    if tiles_dir is None:
        tiles_dir = PROCESSED_DIR / AOI_NAME / "tiles"

    # --- Helper to render a single tile ---
    def render_tile(tidx, tile_dets):
        """Render a tile image with any detections overlaid."""
        # Try to find tile file
        tile_path = None

        # First check if detections have the path
        if tile_dets:
            candidate = tile_dets[0].get("tile_path")
            if candidate and Path(candidate).exists():
                tile_path = Path(candidate)

        # Otherwise search the tiles directory
        if tile_path is None and tiles_dir.exists():
            for pattern in [f"tile_{tidx:05d}.*", f"tile_{tidx}.*"]:
                matches = list(tiles_dir.glob(pattern))
                if matches:
                    tile_path = matches[0]
                    break

        if tile_path is None or not tile_path.exists():
            print(f"  ⚠ Tile {tidx}: file not found")
            return

        try:
            import rasterio
            with rasterio.open(tile_path) as src:
                img = src.read(1)

            n_dets = len(tile_dets)
            title = (f"Tile {tidx}: {n_dets} ship{'s' if n_dets != 1 else ''} detected"
                     if n_dets > 0
                     else f"Tile {tidx}: no detections")

            plot_detections_on_tile(
                img, tile_dets,
                save_path=output_dir / f"detections_tile_{tidx:05d}.png",
                title=title,
            )
        except Exception as e:
            print(f"  Could not plot tile {tidx}: {e}")

    # --- 1. Top-detection tiles (up to 8) ---
    if by_tile:
        top_tiles = sorted(by_tile.keys(),
                           key=lambda k: len(by_tile[k]), reverse=True)[:8]
        for tidx in top_tiles:
            render_tile(tidx, by_tile[tidx])

    # --- 2. User-specified tiles (always rendered, even if empty) ---
    try:
        from config import VISUALIZE_TILES
    except ImportError:
        VISUALIZE_TILES = []

    if VISUALIZE_TILES:
        print(f"\nRendering {len(VISUALIZE_TILES)} user-specified tiles...")
        for tidx in VISUALIZE_TILES:
            tile_dets = by_tile.get(tidx, [])
            render_tile(tidx, tile_dets)


def main():
    # Load raw detections
    raw_path = OUTPUT_DIR / "raw_detections.json"
    if not raw_path.exists():
        print(f"⚠  Raw detections not found: {raw_path}")
        print("   Run detect_ships.py first.")
        return

    with open(raw_path) as f:
        detections = json.load(f)

    print(f"Loaded {len(detections)} raw detections")

    # Load tile metadata
    tiles_meta_path = PROCESSED_DIR / AOI_NAME / "tiles" / "tiles_metadata.json"
    tile_metadata = []
    if tiles_meta_path.exists():
        with open(tiles_meta_path) as f:
            tile_metadata = json.load(f)

    # Cross-tile NMS
    if tile_metadata:
        detections = cross_tile_nms(detections, tile_metadata,
                                     iou_threshold=IOU_THRESHOLD)

    # Georeference and export GeoJSON
    if tile_metadata:
        geojson_path = georeference_detections(detections, tile_metadata)
        print(f"\n✓ GeoJSON exported: {geojson_path}")
    else:
        print("\n⚠  No tile metadata — skipping georeferencing.")
        print("   Saving detections without geographic coordinates.")
        with open(OUTPUT_DIR / "detections_pixel.json", "w") as f:
            json.dump(detections, f, indent=2)

    # Generate visualizations
    print("\nGenerating visualizations...")
    generate_visualizations(detections, OUTPUT_DIR)

    # Save final detection count summary
    summary = {
        "total_detections": len(detections),
        "mean_confidence": float(np.mean([d["confidence"] for d in detections])) if detections else 0,
        "tiles_with_detections": len(set(d["tile_idx"] for d in detections)),
    }
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)



if __name__ == "__main__":
    main()
