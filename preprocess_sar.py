"""
Preprocess Sentinel-1 GRD imagery.

Pipeline:
  1. Read raw GeoTIFF band (VV polarization preferred for ship detection)
  2. Radiometric calibration → sigma0
  3. Speckle filtering (Lee filter)
  4. Normalize to uint8 for model input
  5. Tile into 640x640 input (preserving georeference)

Usage:
    python preprocess_sar.py
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm

from config import (S1_DIR, PROCESSED_DIR, AOI_NAME, TILE_SIZE, TILE_OVERLAP,
                    SPECKLE_FILTER, SPECKLE_KERNEL, PREPROCESS_MODE)
from utils.sar_utils import (calibrate_sigma0, apply_speckle_filter,
                              normalize_sar, normalize_sar_amplitude)
from utils.geo_utils import tile_geotiff
from utils.viz_utils import plot_sar_preprocessing_steps

try:
    import rasterio
except ImportError:
    raise ImportError("Install rasterio: pip install rasterio")


def find_sentinel1_tif(scene_dir):
    """
    Locate the measurement GeoTIFF within a Sentinel-1 .SAFE directory,
    or find a standalone .tif file.

    Returns:
        Path to the VV polarization GeoTIFF (preferred), or first available.
    """
    scene_dir = Path(scene_dir)

    # Check for .SAFE directory structure
    safe_dirs = list(scene_dir.glob("*.SAFE"))
    if safe_dirs:
        measurement_dir = safe_dirs[0] / "measurement"
        # Prefer VV polarization for ship detection
        vv_files = list(measurement_dir.glob("*-vv-*.tiff"))
        if vv_files:
            return vv_files[0]
        # Fallback to any available band
        all_tifs = list(measurement_dir.glob("*.tiff"))
        if all_tifs:
            return all_tifs[0]

    # Check for standalone .tif files
    tif_files = list(scene_dir.glob("*.tif")) + list(scene_dir.glob("*.tiff"))
    if tif_files:
        return tif_files[0]

    # Check for .zip files that need extraction
    zip_files = list(scene_dir.glob("*.zip"))
    if zip_files:
        print(f"Found zip file: {zip_files[0]}")
        print("Please extract it first: unzip <file.zip> -d <scene_dir>")

    return None


def preprocess_scene(tif_path, output_dir, mode="ls_ssdd"):
    """
    Full preprocessing pipeline for a single Sentinel-1 GeoTIFF.

    Args:
        tif_path: Path to input GeoTIFF
        output_dir: Directory for processed outputs
        mode: Preprocessing mode:
              "ls_ssdd"  — amplitude-based, matches LS-SSDD training data
                           (dark ocean, bright ships)
              "classic"  — dB-based with percentile normalization
                           

    Returns:
        Path to processed GeoTIFF
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Preprocessing: {tif_path.name}")
    print(f"Mode: {mode}")
    print(f"{'='*60}")

    # 1. Read raw data
    print("\n[1/4] Reading raw SAR data...")
    with rasterio.open(tif_path) as src:
        raw = src.read(1)
        profile = src.profile.copy()
        print(f"  Shape: {raw.shape}")
        print(f"  CRS: {src.crs}")
        print(f"  Resolution: {src.res}")
        print(f"  Dtype: {raw.dtype}")
        print(f"  Raw DN range: [{raw.min()}, {raw.max()}]")
        print(f"  Approx memory: {raw.nbytes / 1e9:.1f} GB (raw)")

    if mode == "ls_ssdd":
        # LS-SSDD-compatible: amplitude space, dark background
        # LS-SSDD images have: mean ~10-50, median ~9-48, max ~72-255

        # 2. Light speckle filter on amplitude (optional, keeps it simple)
        print(f"\n[2/4] Speckle filtering on amplitude...")
        raw_f32 = raw.astype(np.float32)
        del raw
        filtered = apply_speckle_filter(raw_f32, method=SPECKLE_FILTER,
                                         kernel_size=SPECKLE_KERNEL)
        del raw_f32

        # 3. Normalize amplitude to match LS-SSDD range
        # Find a clip value that gives a similar distribution:
        # median ~10-15 and p95 ~30-80 in the output
        print("\n[3/4] Normalizing (amplitude mode, matching LS-SSDD)...")
        p99 = np.percentile(filtered, 99.5)
        print(f"  Amplitude p99.5: {p99:.1f}")

        # Scale so that p99.5 maps to ~200 
        # This keeps ocean dark and ships as relative bright spots
        normalized = normalize_sar_amplitude(filtered, max_clip=p99)
        del filtered

        print(f"  Output stats: min={normalized.min()}, max={normalized.max()}, "
              f"mean={normalized.mean():.1f}, median={np.median(normalized):.1f}")

    else:
        # Classic mode: dB + percentile normalization
        # 2. Radiometric calibration
        print("\n[2/4] Radiometric calibration (DN → σ₀)...")
        _, sigma0_db = calibrate_sigma0(raw)
        del raw
        print(f"  σ₀ range: [{sigma0_db[sigma0_db > -50].min():.1f}, "
              f"{sigma0_db.max():.1f}] dB")

        # 3. Speckle filtering
        print(f"\n[3/4] Speckle filtering ({SPECKLE_FILTER}, kernel={SPECKLE_KERNEL})...")
        filtered = apply_speckle_filter(sigma0_db, method=SPECKLE_FILTER,
                                         kernel_size=SPECKLE_KERNEL)
        del sigma0_db

        # 4. Normalize
        print("\n[3b/4] Normalizing (dB percentile mode)...")
        normalized = normalize_sar(filtered)
        del filtered

    # Save processed full scene as GeoTIFF
    processed_path = output_dir / f"processed_{tif_path.stem}.tif"
    profile.update(dtype="uint8", count=1, compress="lzw")
    with rasterio.open(processed_path, "w", **profile) as dst:
        dst.write(normalized[np.newaxis, :, :])
    print(f"  Saved: {processed_path}")

    # 4. Generate preprocessing visualization (small crop only)
    print("\n[4/4] Generating preprocessing visualization...")
    crop_size = 500
    cy, cx = normalized.shape[0] // 2, normalized.shape[1] // 2
    s = slice(cy - crop_size // 2, cy + crop_size // 2)
    sc = slice(cx - crop_size // 2, cx + crop_size // 2)

    # Re-read a small crop of raw data just for the visualization
    with rasterio.open(tif_path) as src:
        window = rasterio.windows.Window(
            cx - crop_size // 2, cy - crop_size // 2, crop_size, crop_size)
        raw_crop = src.read(1, window=window)

    if mode == "ls_ssdd":
        filt_crop = apply_speckle_filter(raw_crop.astype(np.float32),
                                          method=SPECKLE_FILTER,
                                          kernel_size=SPECKLE_KERNEL)
        cal_crop = raw_crop.astype(np.float32)  # amplitude view
    else:
        _, cal_crop = calibrate_sigma0(raw_crop)
        filt_crop = apply_speckle_filter(cal_crop, method=SPECKLE_FILTER,
                                          kernel_size=SPECKLE_KERNEL)

    plot_sar_preprocessing_steps(
        raw=raw_crop,
        calibrated_db=cal_crop,
        filtered=filt_crop,
        normalized=normalized[s, sc],
        save_path=output_dir / "preprocessing_steps.png",
    )
    del raw_crop, cal_crop, filt_crop, normalized

    return processed_path


def tile_scene(processed_tif, output_dir):
    """
    Tile the preprocessed scene into model-ready input.
    """
    print(f"\n[Tiling] Creating {TILE_SIZE}x{TILE_SIZE} tiles "
          f"(overlap={TILE_OVERLAP})...")

    tiles_dir = Path(output_dir) / "tiles"
    tiles_info = tile_geotiff(
        tif_path=processed_tif,
        tile_size=TILE_SIZE,
        overlap=TILE_OVERLAP,
        output_dir=tiles_dir,
    )

    return tiles_info


def main():
    scene_dir = S1_DIR / AOI_NAME
    output_dir = PROCESSED_DIR / AOI_NAME

    # Find the input GeoTIFF
    tif_path = find_sentinel1_tif(scene_dir)

    if tif_path is None:
        print(f"\n⚠  No Sentinel-1 GeoTIFF found in {scene_dir}/")
        return

    print(f"Found: {tif_path}")

    # Preprocess
    processed_path = preprocess_scene(tif_path, output_dir, mode=PREPROCESS_MODE)

    # Tile
    tiles_info = tile_scene(processed_path, output_dir)


if __name__ == "__main__":
    main()
