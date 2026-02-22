# SAR_ship_detection
End-to-end pipeline for ship detection on Sentinel-1 SAR imagery using YOLOv8, demonstrating EO data preparation, model inference and output generation.

## Project Structure

```
sar_ship_detection/
├── preprocess_sar.py        # Calibrate, speckle filter, normalize, tile SAR images
├── prepare_dataset_ssdd.py  # Convert SSDD dataset to YOLO format
├── prepare_dataset_ls_ssdd.py  # Convert LS-SSDD dataset to YOLO format
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
│   ├── sentinel1/Baltic        # Pre-downloaded Sentinel-1 data - needs to be downloaded - step 2.
│   └── ssdd/                   # Path for the training data - needs to be downloaded - step 3.
│  
├── models/                     # Model weights
└── outputs/                    # Detection results, GeoJSON, plots
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```
### 2. Download SAR Sentinel-1 Scene
Download a scene from e.g. https://search.asf.alaska.edu/#/
More specifically: https://search.asf.alaska.edu/#/?zoom=6.210&center=20.101,55.720&polygon=POLYGON((19.1461%2058.4746,21.4187%2058.4746,21.4187%2059.1215,19.1461%2059.1215,19.1461%2058.4746))&resultsLoaded=true&granule=S1A_IW_GRDH_1SDV_20260218T162104_20260218T162129_063276_07F201_4438-GRD_HD&productTypes=GRD_HD
Place in 'data/sentinel1/Baltic/'

### 3a. Download LS-SSDD dataset
Download from: https://github.com/TianwenZhang0825/LS-SSDD-v1.0-OPEN
Place in 'data/ssdd/'

### 3b. Download SSDD dataset
Download from: https://github.com/TianwenZhang0825/Official-SSDD
Place in 'data/ssdd/'


### 4. Run the pipeline
```bash
python preprocess_sar.py        # Preprocess SAR image
python prepare_dataset.py       # Prepare SSDD for YOLO format
python train_model.py           # Fine-tune on SAR training data
python detect_ships.py          # Run ship detection
python postprocess_results.py   # Georeference results
python evaluate.py              # Generate metrics and plots
