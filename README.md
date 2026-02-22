# SAR_ship_detection
End-to-end pipeline for ship detection on Sentinel-1 SAR imagery using YOLOv8, demonstrating EO data preparation, model inference and output generation.

## Project Structure

```
sar_ship_detection/
├── preprocess_sar.py        # Calibrate, speckle filter, normalize, tile SAR images
├── prepare_dataset.py       # Convert SSDD dataset to YOLO format
├── train_model.py           # Fine-tune YOLOv8 on SSDD 
├── detect_ships.py          # Run inference on Sentinel-1 tiles
├── postprocess_results.py   # NMS, georeferencing, GeoJSON export
├── evaluate.py              # Compute metrics, generate plots
├── config.py                # Configuration file containing parameters for every step
├── utils/
│   ├── sar_utils.py            # SAR-specific utils (calibration, filtering)
│   ├── geo_utils.py            # Geospatial utils (coords, projections)
│   └── viz_utils.py            # Visualization utils
├── data/
│   ├── sentinel1/Baltic        # Pre-downloaded Sentinel-1 data 
│   └── ssdd/                   # Path for the training data - needs to be downloaded - step 2.
│  
├── models/                     # Model weights
└── outputs/                    # Detection results, GeoJSON, plots
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```
### 2a. Download LS-SSDD dataset
Download from: https://github.com/TianwenZhang0825/LS-SSDD-v1.0-OPEN
Place in `data/ssdd/`

### 2b. Download SSDD dataset
Download from: https://github.com/TianwenZhang0825/Official-SSDD
Place in `data/ssdd/`


### 3. Run the pipeline
```bash
python preprocess_sar.py        # Preprocess SAR image
python prepare_dataset.py       # Prepare SSDD for YOLO format
python train_model.py           # Fine-tune on SAR training data
python detect_ships.py          # Run ship detection
python postprocess_results.py   # Georeference results
python evaluate.py              # Generate metrics and plots
