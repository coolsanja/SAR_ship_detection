"""
Central configuration for the SAR Ship Detection pipeline.
Edit paths and parameters here — all scripts import from this file.
"""
from pathlib import Path

# ============================================================
# PATHS
# ============================================================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
SSDD_DIR = DATA_DIR / "ssdd"           # SSDD dataset root
S1_DIR = DATA_DIR / "sentinel1"         # Raw Sentinel-1 downloads
PROCESSED_DIR = DATA_DIR / "processed"  # Preprocessed SAR tiles
YOLO_DATASET_DIR = DATA_DIR / "yolo_ssdd"  # YOLO-formatted SSDD
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Create directories
for d in [DATA_DIR, SSDD_DIR, S1_DIR, PROCESSED_DIR, YOLO_DATASET_DIR,
          MODEL_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ============================================================
# SAR PREPROCESSING
# ============================================================
# Tile size for slicing large Sentinel-1 scenes into model-ready input
TILE_SIZE = 640          # pixels (matches YOLOv8 default input)
TILE_OVERLAP = 64        # pixels overlap between tiles
SPECKLE_FILTER = "lee"   # Options: "lee", "frost", "median", "none"
SPECKLE_KERNEL = 5       # Filter kernel size

# Preprocessing mode:
#   "ls_ssdd"  — amplitude-based normalization matching LS-SSDD training data
#   "classic"  — dB-based with percentile stretch (for SSDD or custom training)
PREPROCESS_MODE = "ls_ssdd"

# Sigma0 calibration: convert DN to backscatter coefficient
CALIBRATION_LUT = True   # Use annotation LUT if available

# ============================================================
# DATASET PREPARATION (SSDD → YOLO format)
# ============================================================
TRAIN_SPLIT = 0.8        # 80% train, 20% val
IMAGE_SIZE = 640         # Resize SSDD input to this size
CLASS_NAMES = ["ship"]   # Single-class detection

# ============================================================
# MODEL TRAINING
# ============================================================
# Base model: use COCO-pretrained YOLOv8 nano 
YOLO_BASE_MODEL = "yolov8n.pt"

# Training hyperparameters
TRAIN_EPOCHS = 50        # Increase if needed
TRAIN_BATCH = 16	 # Decrease if not enough memory
TRAIN_IMGSZ = 640
TRAIN_DEVICE = "0"       # "0" for GPU, "cpu" for CPU-only
TRAIN_PATIENCE = 10      # Early stopping patience

# ============================================================
# INFERENCE
# ============================================================
CONF_THRESHOLD = 0.25    # Minimum confidence for detections
IOU_THRESHOLD = 0.45     # NMS IoU threshold
MAX_DETECTIONS = 300     # Max detections per image

# ============================================================
# EVALUATION
# ============================================================
IOU_EVAL_THRESHOLDS = [0.5, 0.75]  # IoU thresholds for mAP

# Specific tiles to always visualize (even without detections).
# Useful for inspecting interesting areas (harbours, open sea, coastline).
VISUALIZE_TILES = [147, 232, 1131]
