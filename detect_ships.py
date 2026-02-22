"""
Run ship detection on preprocessed Sentinel-1 tiles.

This script loads the trained YOLOv8 model and runs
inference on the tiled Sentinel-1 SAR chips. It collects all raw
detections before post-processing.

Usage:
    python detect_ships.py
"""

import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

from config import (MODEL_DIR, PROCESSED_DIR, OUTPUT_DIR, AOI_NAME,
                    CONF_THRESHOLD, IOU_THRESHOLD, MAX_DETECTIONS, TRAIN_IMGSZ)


def load_model(weights_path=None):
    """
    Load the YOLOv8 model.

    Priority:
      1. Fine-tuned SSDD weights (if training was done)
      2. COCO-pretrained YOLOv8n (fallback — use 'boat' class)

    Returns:
        YOLO model, bool indicating if SSDD-trained
    """
    # Try SSDD-trained weights first
    ssdd_weights = MODEL_DIR / "ssdd_yolov8n_best.pt"
    if weights_path and Path(weights_path).exists():
        print(f"Loading custom weights: {weights_path}")
        return YOLO(str(weights_path)), True
    elif ssdd_weights.exists():
        print(f"Loading SSDD-trained weights: {ssdd_weights}")
        return YOLO(str(ssdd_weights)), True
    else:
        print("No SSDD weights found. Using COCO-pretrained YOLOv8n.")
        print("(COCO class 8 = 'boat' — will filter for this class)")
        return YOLO("yolov8n.pt"), False


def run_inference_on_tiles(model, tiles_dir, is_ssdd_trained=True):
    """
    Run inference on all tiles in a directory.

    Args:
        model: Loaded YOLO model
        tiles_dir: Directory containing tile images
        is_ssdd_trained: If False, filter for COCO 'boat' class (id=8)

    Returns:
        List of detection dicts: {tile_idx, bbox, confidence, class_name}
    """
    tiles_dir = Path(tiles_dir)
    tile_files = sorted(tiles_dir.glob("tile_*.tif")) + \
                 sorted(tiles_dir.glob("tile_*.jpg")) + \
                 sorted(tiles_dir.glob("tile_*.png"))

    if not tile_files:
        print(f"⚠  No tile files found in {tiles_dir}/")
        return []

    print(f"\nRunning inference on {len(tile_files)} tiles...")

    all_detections = []
    detection_counts = []

    for tile_path in tqdm(tile_files, desc="Detecting"):
        # Extract tile index from filename
        tile_idx = int(tile_path.stem.split("_")[1])

        # Read tile — handle single-channel SAR GeoTIFFs
        import cv2
        img = cv2.imread(str(tile_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            # Try with rasterio for GeoTIFFs that OpenCV can't read
            try:
                import rasterio
                with rasterio.open(tile_path) as src:
                    img = src.read(1)
            except Exception:
                continue

        # Convert single-channel to 3-channel (YOLOv8 expects RGB)
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.shape[0] in [1, 2] and img.ndim == 3:
            # Rasterio format: (bands, H, W) → (H, W, 3)
            img = np.stack([img[0], img[0], img[0]], axis=-1)

        # Run YOLOv8 inference
        results = model.predict(
            source=img,
            imgsz=TRAIN_IMGSZ,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            max_det=MAX_DETECTIONS,
            verbose=False,
            save=False,
        )

        # Parse results
        tile_dets = 0
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                bbox = boxes.xyxy[i].cpu().numpy().tolist()

                # If using COCO model, filter for boat class (id=8)
                if not is_ssdd_trained:
                    if cls_id != 8:  # 8 = 'boat' in COCO
                        continue
                    class_name = "ship"
                else:
                    class_name = model.names.get(cls_id, "ship")

                all_detections.append({
                    "tile_idx": tile_idx,
                    "tile_path": str(tile_path),
                    "bbox": bbox,  # [x1, y1, x2, y2]
                    "confidence": conf,
                    "class_id": cls_id,
                    "class_name": class_name,
                })
                tile_dets += 1

        detection_counts.append(tile_dets)

    # Summary statistics
    total = len(all_detections)
    tiles_with_dets = sum(1 for c in detection_counts if c > 0)
    print(f"\n{'='*40}")
    print(f"Inference Summary")
    print(f"{'='*40}")
    print(f"  Total detections:   {total}")
    print(f"  Tiles with ships:   {tiles_with_dets}/{len(tile_files)}")
    if total > 0:
        confs = [d["confidence"] for d in all_detections]
        print(f"  Confidence range:   [{min(confs):.3f}, {max(confs):.3f}]")
        print(f"  Mean confidence:    {np.mean(confs):.3f}")

    return all_detections


def save_raw_detections(detections, output_path):
    """Save raw detections to JSON for post-processing."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types for JSON serialization
    serializable = []
    for d in detections:
        sd = d.copy()
        sd["bbox"] = [float(x) for x in sd["bbox"]]
        serializable.append(sd)

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"Saved {len(detections)} raw detections to {output_path}")


def main():
    # Load model
    model, is_ssdd = load_model()

    # Find tiles
    tiles_dir = PROCESSED_DIR / AOI_NAME / "tiles"
    if not tiles_dir.exists():
        print(f"⚠  Tiles not found at {tiles_dir}/")
        print("   Run preprocess_sar.py first.")
        return

    # Run inference
    detections = run_inference_on_tiles(model, tiles_dir, is_ssdd_trained=is_ssdd)

    # Save raw results
    raw_output = OUTPUT_DIR / "raw_detections.json"
    save_raw_detections(detections, raw_output)


if __name__ == "__main__":
    main()
