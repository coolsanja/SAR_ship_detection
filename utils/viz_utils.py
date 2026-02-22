"""
Visualization utilities for SAR imagery and detection results.

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path


def plot_sar_preprocessing_steps(raw, calibrated_db, filtered, normalized,
                                  save_path=None):
    """
    Show SAR preprocessing pipeline as a 4-panel figure.

    Args:
        raw: Raw DN array
        calibrated_db: Calibrated sigma0 in dB
        filtered: Speckle-filtered image
        normalized: Final normalized uint8 image
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    titles = ["Raw DN", "Calibrated (σ₀ dB)", "Speckle Filtered", "Normalized"]
    images = [raw, calibrated_db, filtered, normalized]
    cmaps = ["gray", "gray", "gray", "gray"]

    for ax, img, title, cmap in zip(axes, images, titles, cmaps):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def plot_detections_on_tile(image, detections, save_path=None, title=None):
    """
    Draw bounding boxes on a SAR tile image.

    Args:
        image: 2D or 3D numpy array (the SAR tile)
        detections: List of dicts with 'bbox' [x1,y1,x2,y2] and 'confidence'
        save_path: Optional path to save
        title: Optional plot title
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(image, cmap="gray")

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        conf = det["confidence"]
        w, h = x2 - x1, y2 - y1

        # Color by confidence: green = high, yellow = medium, red = low
        if conf > 0.7:
            color = "#00ff00"
        elif conf > 0.4:
            color = "#ffff00"
        else:
            color = "#ff4444"

        rect = patches.Rectangle(
            (x1, y1), w, h,
            linewidth=2, edgecolor=color, facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, f"{conf:.2f}", color=color, fontsize=9,
                fontweight="bold", backgroundcolor="black")

    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def plot_detection_grid(images, detections_list, ncols=4, save_path=None):
    """
    Grid view of multiple tiles with their detections.
    """
    n = len(images)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for i, (img, dets) in enumerate(zip(images, detections_list)):
        ax = axes[i]
        ax.imshow(img, cmap="gray")

        for det in dets:
            x1, y1, x2, y2 = det["bbox"]
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=1.5, edgecolor="#00ff00", facecolor="none"
            )
            ax.add_patch(rect)

        ax.axis("off")
        ax.set_title(f"Tile {i} ({len(dets)} ships)", fontsize=10)

    # Hide empty axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Ship Detection Results Across Tiles", fontsize=16,
                 fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def plot_metrics(metrics_dict, save_path=None, confidences=None):
    """
    Plot evaluation metrics as a bar chart + confidence histogram.

    Args:
        metrics_dict: Dict with keys like 'precision', 'recall', 'f1', 'mAP50'
        save_path: Optional path to save
        confidences: Optional list of confidence scores for histogram
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart of key metrics
    names = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]

    bars = ax1.bar(names, values, color=colors[:len(names)], edgecolor="white",
                   linewidth=1.5)
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Score", fontsize=12)
    ax1.set_title("Detection Performance Metrics", fontsize=14,
                  fontweight="bold")

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", fontsize=11, fontweight="bold")

    ax1.grid(axis="y", alpha=0.3)

    # Confidence distribution histogram
    ax2.set_title("Confidence Score Distribution", fontsize=14,
                  fontweight="bold")
    ax2.set_xlabel("Confidence")
    ax2.set_ylabel("Count")

    if confidences is not None and len(confidences) > 0:
        ax2.hist(confidences, bins=30, color="#2196F3", edgecolor="white",
                 alpha=0.8)
        ax2.axvline(x=np.mean(confidences), color="red", linestyle="--",
                    label=f"Mean: {np.mean(confidences):.3f}")
        ax2.legend(fontsize=11)
        ax2.grid(axis="y", alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "No inference detections\navailable",
                 ha="center", va="center", transform=ax2.transAxes,
                 fontsize=14, color="gray")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def plot_confidence_histogram(confidences, save_path=None):
    """
    Histogram of detection confidence scores.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(confidences, bins=30, color="#2196F3", edgecolor="white",
            alpha=0.8)
    ax.axvline(x=np.mean(confidences), color="red", linestyle="--",
               label=f"Mean: {np.mean(confidences):.3f}")
    ax.set_xlabel("Confidence Score", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Detection Confidence Distribution", fontsize=14,
                 fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()
