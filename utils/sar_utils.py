"""
SAR-specific utility functions.
Handles radiometric calibration, speckle filtering, and land masking.
"""

import numpy as np
from scipy.ndimage import uniform_filter, median_filter


def calibrate_sigma0(dn_array, calibration_lut=None):
    """
    Convert digital numbers (DN) to radar backscatter coefficient (sigma0).

    For Sentinel-1 GRD products:
        sigma0 = (DN^2) / (calibration_LUT_value^2)

    If no calibration LUT is provided, applies a simplified conversion:
        sigma0_dB = 10 * log10(DN^2) - noise_floor

    Args:
        dn_array: 2D numpy array of raw DN values
        calibration_lut: Optional calibration lookup table from annotation XML

    Returns:
        sigma0: 2D array of calibrated backscatter values (linear scale)
        sigma0_db: 2D array in dB scale
    """
    dn_float = dn_array.astype(np.float32)

    if calibration_lut is not None:
        # Apply proper radiometric calibration using LUT
        sigma0 = (dn_float ** 2) / (calibration_lut.astype(np.float32) ** 2)
    else:
        # Simplified calibration (good enough for demonstration)
        # Avoid log of zero
        sigma0 = dn_float ** 2
        sigma0[sigma0 == 0] = 1e-10

    # Convert to dB (release linear sigma0 to save memory)
    sigma0_db = 10 * np.log10(np.clip(sigma0, 1e-10, None))
    del sigma0

    return None, sigma0_db


def lee_filter(image, kernel_size=5, block_size=2048, overlap=32):
    """
    Lee speckle filter for SAR imagery.

    The Lee filter reduces speckle noise while preserving edges by using
    local statistics (mean and variance) to adaptively weight the filtering.

    Processes the image in overlapping blocks to avoid loading everything
    into memory at once — critical for large Sentinel-1 GRD scenes.

    Args:
        image: 2D numpy array (SAR intensity image)
        kernel_size: Size of the filter window (odd number)
        block_size: Size of processing blocks in pixels
        overlap: Overlap between blocks to avoid edge artifacts

    Returns:
        Filtered image as 2D numpy array (float32)
    """
    h, w = image.shape
    filtered = np.empty((h, w), dtype=np.float32)

    # First pass: estimate global noise variance from a subsample
    step = max(1, h // 1000)
    sample = image[::step, ::step].astype(np.float32)
    sample_mean = uniform_filter(sample, size=kernel_size)
    sample_sq = uniform_filter(sample ** 2, size=kernel_size)
    sample_var = np.clip(sample_sq - sample_mean ** 2, 0, None)
    var_noise = float(np.mean(sample_var))
    del sample, sample_mean, sample_sq, sample_var

    # Second pass: filter in blocks
    for row_start in range(0, h, block_size):
        for col_start in range(0, w, block_size):
            # Compute block with overlap for seamless edges
            r0 = max(0, row_start - overlap)
            c0 = max(0, col_start - overlap)
            r1 = min(h, row_start + block_size + overlap)
            c1 = min(w, col_start + block_size + overlap)

            block = image[r0:r1, c0:c1].astype(np.float32)

            mean_local = uniform_filter(block, size=kernel_size)
            mean_sq = uniform_filter(block ** 2, size=kernel_size)
            var_local = np.clip(mean_sq - mean_local ** 2, 0, None)
            del mean_sq

            weight = var_local / (var_local + var_noise + 1e-10)
            del var_local
            result = mean_local + weight * (block - mean_local)
            del mean_local, weight, block

            # Extract the non-overlapping center
            inner_r0 = row_start - r0
            inner_c0 = col_start - c0
            inner_r1 = inner_r0 + min(block_size, h - row_start)
            inner_c1 = inner_c0 + min(block_size, w - col_start)

            filtered[row_start:row_start + (inner_r1 - inner_r0),
                     col_start:col_start + (inner_c1 - inner_c0)] = \
                result[inner_r0:inner_r1, inner_c0:inner_c1]
            del result

    return filtered


def frost_filter(image, kernel_size=5, damping=1.0):
    """
    Frost speckle filter for SAR imagery.

    Uses an exponentially damped kernel weighted by local statistics.

    Args:
        image: 2D numpy array
        kernel_size: Filter window size
        damping: Damping factor controlling filter strength

    Returns:
        Filtered image
    """
    img = image.astype(np.float64)

    # Use uniform filter for local stats as approximation
    mean_local = uniform_filter(img, size=kernel_size)
    mean_sq = uniform_filter(img ** 2, size=kernel_size)
    var_local = np.clip(mean_sq - mean_local ** 2, 0, None)

    # Coefficient of variation
    cv = np.sqrt(var_local) / (mean_local + 1e-10)

    # Simplified Frost: blend between original and mean based on CV
    alpha = damping * cv ** 2
    weight = np.exp(-alpha)
    filtered = weight * img + (1 - weight) * mean_local

    return filtered


def apply_speckle_filter(image, method="lee", kernel_size=5):
    """
    Apply speckle filter to SAR image.

    Args:
        image: 2D numpy array
        method: "lee", "frost", "median", or "none"
        kernel_size: Filter kernel size

    Returns:
        Filtered image
    """
    if method == "lee":
        return lee_filter(image, kernel_size)
    elif method == "frost":
        return frost_filter(image, kernel_size)
    elif method == "median":
        return median_filter(image, size=kernel_size)
    elif method == "none":
        return image
    else:
        raise ValueError(f"Unknown speckle filter: {method}")


def normalize_sar(image, percentile_low=2, percentile_high=98):
    """
    Normalize SAR image to 0-255 range using percentile clipping.

    Args:
        image: 2D numpy array (sigma0 or dB values)
        percentile_low: Lower percentile for clipping
        percentile_high: Upper percentile for clipping

    Returns:
        Normalized uint8 image
    """
    vmin = np.percentile(image, percentile_low)
    vmax = np.percentile(image, percentile_high)
    normalized = np.clip((image - vmin) / (vmax - vmin + 1e-10), 0, 1)
    return (normalized * 255).astype(np.uint8)


def normalize_sar_amplitude(dn_array, max_clip=5000):
    """
    Normalize raw SAR amplitude to match LS-SSDD-style imagery.


    Args:
        dn_array: 2D numpy array of raw DN values (or amplitude)
        max_clip: Clip amplitude at this value before scaling to 0-255.
                  Adjust based on your scene — higher = darker overall.

    Returns:
        Normalized uint8 image matching LS-SSDD distribution
    """
    amp = np.abs(dn_array).astype(np.float32)
    normalized = np.clip(amp / max_clip, 0, 1)
    return (normalized * 255).astype(np.uint8)


def create_land_mask_simple(sigma0_db, threshold=-15.0):
    """
    Simple land masking based on backscatter threshold.

    Land generally has higher backscatter than open ocean.


    Args:
        sigma0_db: 2D array of calibrated backscatter in dB
        threshold: dB value above which pixels are likely land

    Returns:
        Boolean mask (True = likely water/ocean)
    """
    # intentionally simple — improve with a proper coastline vector dataset
    water_mask = sigma0_db < threshold
    return water_mask
