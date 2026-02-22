"""
Convert SSDD (SAR Ship Detection Dataset) to YOLO format.

SSDD comes with XML annotations (Pascal VOC style). YOLOv8 expects:
  - images/ and labels/ directories
  - One .txt file per image with: class_id cx cy w h (normalized 0-1)
  - A dataset.yaml config file

Download SSDD from: https://github.com/TianwenZhang0825/Official-SSDD
Place contents in data/ssdd/

Usage:
    python prepare_dataset_ssdd.py
"""

import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image

from config import SSDD_DIR, YOLO_DATASET_DIR, TRAIN_SPLIT, IMAGE_SIZE, CLASS_NAMES


def parse_voc_xml(xml_path):
    """
    Parse a Pascal VOC format XML annotation file.

    Args:
        xml_path: Path to .xml annotation file

    Returns:
        List of dicts with 'class', 'xmin', 'ymin', 'xmax', 'ymax'
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Get image dimensions
    size = root.find("size")
    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)

    annotations = []
    for obj in root.findall("object"):
        cls_name = obj.find("name").text
        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        annotations.append({
            "class": cls_name,
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
            "img_w": img_w,
            "img_h": img_h,
        })

    return annotations


def voc_to_yolo(annotations, class_map):
    """
    Convert VOC bounding boxes to YOLO format.

    YOLO format: class_id center_x center_y width height (all normalized 0-1)

    Args:
        annotations: List of VOC annotation dicts
        class_map: Dict mapping class names to integer IDs

    Returns:
        List of YOLO format strings
    """
    yolo_lines = []
    for ann in annotations:
        if ann["class"] not in class_map:
            continue

        cls_id = class_map[ann["class"]]
        img_w, img_h = ann["img_w"], ann["img_h"]

        # Convert to YOLO normalized format
        cx = ((ann["xmin"] + ann["xmax"]) / 2) / img_w
        cy = ((ann["ymin"] + ann["ymax"]) / 2) / img_h
        w = (ann["xmax"] - ann["xmin"]) / img_w
        h = (ann["ymax"] - ann["ymin"]) / img_h

        # Clamp to [0, 1]
        cx = max(0, min(1, cx))
        cy = max(0, min(1, cy))
        w = max(0, min(1, w))
        h = max(0, min(1, h))

        yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    return yolo_lines


def create_yolo_dataset(ssdd_dir, output_dir, train_split=0.8):
    """
    Create a YOLO-formatted dataset from SSDD.

    Official-SSDD structure (from GitHub):
        Official-SSDD-OPEN/
        └── BBox_SSDD/
            └── voc_style/
                ├── JPEGImages_train/
                ├── JPEGImages_test/
                ├── Annotations_train/
                └── Annotations_test/

    Output structure:
        yolo_ssdd/
        ├── images/
        │   ├── train/
        │   └── val/
        ├── labels/
        │   ├── train/
        │   └── val/
        └── dataset.yaml
    """
    ssdd_dir = Path(ssdd_dir)
    output_dir = Path(output_dir)

    # === Try to find the correct directories ===
    # The Official-SSDD uses pre-split train/test with separate folders
    train_img_dir = None
    train_ann_dir = None
    test_img_dir = None
    test_ann_dir = None

    # Search patterns for the Official-SSDD structure
    search_roots = [
        ssdd_dir,
        ssdd_dir / "Official-SSDD-OPEN",
        ssdd_dir / "BBox_SSDD",
        ssdd_dir / "BBox_SSDD" / "voc_style",
        ssdd_dir / "Official-SSDD-OPEN" / "BBox_SSDD" / "voc_style",
    ]

    # Search one level deeper for any matching subfolder
    for child in ssdd_dir.iterdir() if ssdd_dir.exists() else []:
        if child.is_dir():
            search_roots.append(child)
            search_roots.append(child / "BBox_SSDD" / "voc_style")
            search_roots.append(child / "voc_style")

    for root in search_roots:
        if not root.exists():
            continue

        # Pattern 1: Official-SSDD split folders (JPEGImages_train / Annotations_train)
        if (root / "JPEGImages_train").exists() and (root / "Annotations_train").exists():
            train_img_dir = root / "JPEGImages_train"
            train_ann_dir = root / "Annotations_train"
            test_img_dir = root / "JPEGImages_test" if (root / "JPEGImages_test").exists() else None
            test_ann_dir = root / "Annotations_test" if (root / "Annotations_test").exists() else None
            print(f"Found Official-SSDD split structure in {root}/")
            break

        # Pattern 2: Classic VOC (JPEGImages / Annotations)
        if (root / "JPEGImages").exists() and (root / "Annotations").exists():
            train_img_dir = root / "JPEGImages"
            train_ann_dir = root / "Annotations"
            print(f"Found classic VOC structure in {root}/")
            break

        # Pattern 3: Simple (images / annotations)
        if (root / "images").exists() and (root / "annotations").exists():
            train_img_dir = root / "images"
            train_ann_dir = root / "annotations"
            print(f"Found simple structure in {root}/")
            break

    if train_img_dir is None or train_ann_dir is None:
        print(f"\n⚠  Could not find SSDD images/annotations in {ssdd_dir}/")
        print(f"   Searched in: {[str(r) for r in search_roots if r.exists()]}")
        print(f"\n   Please download Official-SSDD from:")
        print(f"   https://github.com/TianwenZhang0825/Official-SSDD")
        print(f"\n   Then place contents so that this path exists:")
        print(f"   {ssdd_dir}/BBox_SSDD/voc_style/JPEGImages_train/")
        print(f"\n   Or clone directly into data/ssdd/:")
        return

    # Class mapping (SSDD is single-class: ship)
    class_map = {name: i for i, name in enumerate(CLASS_NAMES)}
    # Also handle common label variations in SSDD
    class_map["Ship"] = 0
    class_map["ship"] = 0

    # Create output directories
    for split in ["train", "val"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    total_ships = 0

    if test_img_dir and test_ann_dir:
        # === Official-SSDD: already split into train/test ===
        splits = [
            ("train", train_img_dir, train_ann_dir),
            ("val", test_img_dir, test_ann_dir),  # Use their test as our val
        ]
    else:
        # === Classic structure: we need to split ourselves ===
        xml_files = sorted(train_ann_dir.glob("*.xml"))
        random.seed(42)
        random.shuffle(xml_files)
        split_idx = int(len(xml_files) * train_split)

        # Temporarily create split lists
        splits = [
            ("train", train_img_dir, train_ann_dir, xml_files[:split_idx]),
            ("val", train_img_dir, train_ann_dir, xml_files[split_idx:]),
        ]

    for split_info in splits:
        if len(split_info) == 3:
            split, img_dir, ann_dir = split_info
            xml_files = sorted(ann_dir.glob("*.xml"))
        else:
            split, img_dir, ann_dir, xml_files = split_info

        split_ships = 0
        for xml_file in xml_files:
            annotations = parse_voc_xml(xml_file)
            yolo_labels = voc_to_yolo(annotations, class_map)
            split_ships += len(yolo_labels)

            # Find corresponding image
            stem = xml_file.stem
            img_file = None
            for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
                candidate = img_dir / f"{stem}{ext}"
                if candidate.exists():
                    img_file = candidate
                    break

            if img_file is None:
                continue

            # Copy image
            dst_img = output_dir / "images" / split / f"{stem}.jpg"
            if img_file.suffix.lower() in [".jpg", ".jpeg"]:
                shutil.copy2(img_file, dst_img)
            else:
                # Convert to JPEG if needed
                Image.open(img_file).convert("RGB").save(dst_img)

            # Write YOLO labels
            dst_label = output_dir / "labels" / split / f"{stem}.txt"
            with open(dst_label, "w") as f:
                f.write("\n".join(yolo_labels))

        total_ships += split_ships
        print(f"  {split}: {len(xml_files)} images, {split_ships} ship annotations")

    print(f"Total ship annotations: {total_ships}")

    # Create dataset.yaml for YOLOv8
    yaml_content = f"""# SSDD - SAR Ship Detection Dataset (YOLO format)
# Auto-generated by prepare_dataset_ssdd.py

path: {output_dir.resolve()}
train: images/train
val: images/val

# Classes
nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
"""

    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"\n✓ YOLO dataset created at {output_dir}/")
    print(f"  Config: {yaml_path}")


def main():
    print("Converting SSDD to YOLO format...")
    create_yolo_dataset(
        ssdd_dir=SSDD_DIR,
        output_dir=YOLO_DATASET_DIR,
        train_split=TRAIN_SPLIT,
    )



if __name__ == "__main__":
    main()
