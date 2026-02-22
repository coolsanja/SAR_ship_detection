"""
Evaluate detection performance and generate presentation-ready outputs.

Two evaluation modes:
  A) On SSDD val set 
  B) Qualitative analysis on Sentinel-1 (visual inspection + statistics)

Generates metrics plots.

Usage:
    python evaluate.py
"""

import json
import numpy as np
from pathlib import Path
from collections import Counter

from config import (OUTPUT_DIR, YOLO_DATASET_DIR, MODEL_DIR, TRAIN_IMGSZ,
                    IOU_EVAL_THRESHOLDS)
from utils.viz_utils import plot_metrics


def evaluate_ssdd_val():
    """
    Evaluate the fine-tuned model on the SSDD validation set using
    YOLOv8's built-in val() method.

    Returns metrics dict or None if not available.
    """
    weights = MODEL_DIR / "ssdd_yolov8n_best.pt"
    dataset_yaml = YOLO_DATASET_DIR / "dataset.yaml"

    if not weights.exists():
        print("No SSDD-trained weights found. Skipping SSDD evaluation.")
        print("(Run train_model.py to train, or use pretrained weights)")
        return None

    if not dataset_yaml.exists():
        print("No SSDD dataset config found. Skipping.")
        return None

    from ultralytics import YOLO
    model = YOLO(str(weights))

    print(f"\n{'='*50}")
    print("Evaluating on validation set")
    print(f"{'='*50}")

    results = model.val(
        data=str(dataset_yaml),
        imgsz=TRAIN_IMGSZ,
        verbose=True,
    )

    metrics = {
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
        "mAP@50": float(results.box.map50),
        "mAP@50-95": float(results.box.map),
    }

    # Compute F1
    p, r = metrics["precision"], metrics["recall"]
    metrics["F1"] = 2 * p * r / (p + r + 1e-10)

    print(f"\n  Precision:    {metrics['precision']:.4f}")
    print(f"  Recall:       {metrics['recall']:.4f}")
    print(f"  F1:           {metrics['F1']:.4f}")
    print(f"  mAP@50:       {metrics['mAP@50']:.4f}")
    print(f"  mAP@50-95:    {metrics['mAP@50-95']:.4f}")

    return metrics


def analyze_sentinel1_detections():
    """
    Analyze detection results on Sentinel-1 imagery (qualitative).

    Since we don't have ground truth for the Sentinel-1 scene,
    this provides statistical analysis of the detections.
    """
    raw_path = OUTPUT_DIR / "raw_detections.json"
    if not raw_path.exists():
        print("No Sentinel-1 detections found. Skipping.")
        return None

    with open(raw_path) as f:
        detections = json.load(f)

    if not detections:
        print("No detections to analyze.")
        return None

    print(f"\n{'='*50}")
    print("Sentinel-1 Detection Analysis (Qualitative)")
    print(f"{'='*50}")

    confidences = [d["confidence"] for d in detections]
    bbox_areas = [(d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1])
                  for d in detections]

    stats = {
        "total_detections": len(detections),
        "confidence": {
            "mean": float(np.mean(confidences)),
            "std": float(np.std(confidences)),
            "min": float(np.min(confidences)),
            "max": float(np.max(confidences)),
            "median": float(np.median(confidences)),
        },
        "bbox_area_px": {
            "mean": float(np.mean(bbox_areas)),
            "std": float(np.std(bbox_areas)),
            "min": float(np.min(bbox_areas)),
            "max": float(np.max(bbox_areas)),
        },
        "high_conf_detections": int(sum(1 for c in confidences if c > 0.7)),
        "medium_conf_detections": int(sum(1 for c in confidences if 0.4 < c <= 0.7)),
        "low_conf_detections": int(sum(1 for c in confidences if c <= 0.4)),
        "tiles_with_detections": len(set(d["tile_idx"] for d in detections)),
    }

    print(f"  Total detections:     {stats['total_detections']}")
    print(f"  High conf (>0.7):     {stats['high_conf_detections']}")
    print(f"  Medium conf (0.4-0.7):{stats['medium_conf_detections']}")
    print(f"  Low conf (≤0.4):      {stats['low_conf_detections']}")
    print(f"  Mean confidence:      {stats['confidence']['mean']:.3f}")
    print(f"  Mean bbox area:       {stats['bbox_area_px']['mean']:.1f} px²")

    return stats


def main():
    # Evaluate on SSDD val set
    ssdd_metrics = evaluate_ssdd_val()

    # Analyze Sentinel-1 detections
    s1_stats = analyze_sentinel1_detections()

    # Generate visualizations
    if ssdd_metrics:
        # Load S1 confidences if available to show in the same figure
        s1_confidences = None
        raw_path = OUTPUT_DIR / "raw_detections.json"
        if raw_path.exists():
            with open(raw_path) as f:
                raw_dets = json.load(f)
            if raw_dets:
                s1_confidences = [d["confidence"] for d in raw_dets]

        plot_metrics(ssdd_metrics, save_path=OUTPUT_DIR / "ssdd_metrics.png",
                     confidences=s1_confidences)



if __name__ == "__main__":
    main()
