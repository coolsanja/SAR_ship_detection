"""
Fine-tune YOLOv8 on the SSDD dataset.

This script fine-tunes a COCO-pretrained YOLOv8 model on SAR ship imagery.


Usage:
    python train_model.py
"""

from pathlib import Path
from ultralytics import YOLO

from config import (YOLO_DATASET_DIR, YOLO_BASE_MODEL, MODEL_DIR,
                    TRAIN_EPOCHS, TRAIN_BATCH, TRAIN_IMGSZ, TRAIN_DEVICE,
                    TRAIN_PATIENCE)


def train_model():
    """Fine-tune YOLOv8 on SSDD."""

    dataset_yaml = YOLO_DATASET_DIR / "dataset.yaml"
    if not dataset_yaml.exists():
        print(f"⚠  Dataset config not found: {dataset_yaml}")
        print("   Run prepare_dataset*.py first.")
        return

    print(f"{'='*60}")
    print(f"Fine-tuning YOLOv8 on SSDD")
    print(f"{'='*60}")
    print(f"  Base model:  {YOLO_BASE_MODEL}")
    print(f"  Dataset:     {dataset_yaml}")
    print(f"  Epochs:      {TRAIN_EPOCHS}")
    print(f"  Batch size:  {TRAIN_BATCH}")
    print(f"  Image size:  {TRAIN_IMGSZ}")
    print(f"  Device:      {TRAIN_DEVICE}")

    # Load COCO-pretrained model
    model = YOLO(YOLO_BASE_MODEL)

    # Fine-tune on SSDD
    results = model.train(
        data=str(dataset_yaml),
        epochs=TRAIN_EPOCHS,
        imgsz=TRAIN_IMGSZ,
        batch=TRAIN_BATCH,
        device=TRAIN_DEVICE,
        patience=TRAIN_PATIENCE,
        # Project settings
        project=str(MODEL_DIR),
        name="ssdd_yolov8n",
        # Augmentation (conservative for SAR — don't flip vertically
        # or apply color jitter aggressively since SAR is grayscale)
        flipud=0.0,        # No vertical flip 
        fliplr=0.5,        # Horizontal flip is fine
        mosaic=1.0,        # Mosaic augmentation helps with small objects
        hsv_h=0.0,         # No hue augmentation (grayscale input)
        hsv_s=0.0,         # No saturation augmentation
        hsv_v=0.2,         # Slight brightness variation
        degrees=15.0,      # Moderate rotation
        scale=0.5,         # Scale augmentation
        # Optimization
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=3,
        # Saving
        save=True,
        save_period=10,
        verbose=True,
    )

    # Copy best weights to a known location
    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    target_path = MODEL_DIR / "ssdd_yolov8n_best.pt"

    if best_weights.exists():
        import shutil
        shutil.copy2(best_weights, target_path)
        print(f"\n✓ Best weights saved to: {target_path}")
    else:
        print(f"\n⚠  Could not find best.pt at {best_weights}")

    return results


def evaluate_on_val():
    """Quick validation of the trained model."""
    weights_path = MODEL_DIR / "ssdd_yolov8n_best.pt"

    if not weights_path.exists():
        print("No trained weights found. Run training first.")
        return

    model = YOLO(str(weights_path))
    dataset_yaml = YOLO_DATASET_DIR / "dataset.yaml"

    print(f"\nValidating on SSDD val set...")
    metrics = model.val(data=str(dataset_yaml), imgsz=TRAIN_IMGSZ)

    print(f"\n{'='*40}")
    print(f"Validation Results")
    print(f"{'='*40}")
    print(f"  mAP@50:      {metrics.box.map50:.4f}")
    print(f"  mAP@50-95:   {metrics.box.map:.4f}")
    print(f"  Precision:    {metrics.box.mp:.4f}")
    print(f"  Recall:       {metrics.box.mr:.4f}")

    return metrics


def main():
    # Train
    results = train_model()

    # Validate
    if results is not None:
        evaluate_on_val()


if __name__ == "__main__":
    main()
