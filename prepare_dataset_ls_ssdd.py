"""
Convert LS-SSDD-v1.0 to YOLO format.

LS-SSDD is built from Sentinel-1 imagery — same sensor as the example scene.

LS-SSDD structure:
    LS-SSDD-v1.0-OPEN/
    ├── JPEGImages_sub/       # 9000 sub-images (800x800), named N_R_C.jpg
    ├── Annotations_sub/      # Matching XML annotations (Pascal VOC)
    ├── ImageSets/
    │   └── Main/
    │       ├── train.txt     # Training image list (images 01-10)
    │       └── test.txt      # Test image list (images 11-15)
    ├── JPEGImages/           # 15 full-scale images (24000x16000) - not used
    └── Annotations/          # Full-scale annotations - not used

Download from: https://github.com/TianwenZhang0825/LS-SSDD-v1.0-OPEN
Place in: data/ssdd/LS-SSDD-v1.0-OPEN/

Usage:
    python prepare_dataset_ls_ssdd.py
"""

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from config import DATA_DIR, YOLO_DATASET_DIR, CLASS_NAMES


LS_SSDD_DIR = DATA_DIR / "ssdd" / "LS-SSDD-v1.0-OPEN"


def parse_voc_xml(xml_path):
    """Parse Pascal VOC XML annotation file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

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
            "xmin": xmin, "ymin": ymin,
            "xmax": xmax, "ymax": ymax,
            "img_w": img_w, "img_h": img_h,
        })

    return annotations


def voc_to_yolo(annotations):
    """Convert VOC annotations to YOLO format (class 0 = ship)."""
    yolo_lines = []
    for ann in annotations:
        img_w, img_h = ann["img_w"], ann["img_h"]
        cx = ((ann["xmin"] + ann["xmax"]) / 2) / img_w
        cy = ((ann["ymin"] + ann["ymax"]) / 2) / img_h
        w = (ann["xmax"] - ann["xmin"]) / img_w
        h = (ann["ymax"] - ann["ymin"]) / img_h

        # Clamp
        cx = max(0, min(1, cx))
        cy = max(0, min(1, cy))
        w = max(0, min(1, w))
        h = max(0, min(1, h))

        yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    return yolo_lines


def find_ls_ssdd():
    """Search for LS-SSDD directory in common locations."""
    candidates = [
        LS_SSDD_DIR,
        DATA_DIR / "ssdd" / "LS-SSDD-v1.0-OPEN",
        DATA_DIR / "ls-ssdd",
        DATA_DIR / "LS-SSDD-v1.0-OPEN",
    ]

    # Also check children of data/ssdd/
    ssdd_dir = DATA_DIR / "ssdd"
    if ssdd_dir.exists():
        for child in ssdd_dir.iterdir():
            if child.is_dir():
                candidates.append(child)
                # Check one level deeper
                for grandchild in child.iterdir():
                    if grandchild.is_dir():
                        candidates.append(grandchild)

    for path in candidates:
        if (path / "JPEGImages_sub").exists() and (path / "Annotations_sub").exists():
            return path

    return None


def create_yolo_dataset_from_ls_ssdd(ls_ssdd_dir, output_dir):
    """
    Convert LS-SSDD to YOLO format.

    Uses ImageSets/Main/train.txt and test.txt for the split if available,
    otherwise uses the convention: images 01-10 = train, 11-15 = test.
    """
    ls_ssdd_dir = Path(ls_ssdd_dir)
    output_dir = Path(output_dir)

    img_dir = ls_ssdd_dir / "JPEGImages_sub"
    ann_dir = ls_ssdd_dir / "Annotations_sub"

    print(f"LS-SSDD root: {ls_ssdd_dir}")
    print(f"Images:       {img_dir}")
    print(f"Annotations:  {ann_dir}")

    # Get all sub-images
    all_images = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.jpeg"))
    all_xmls = sorted(ann_dir.glob("*.xml"))

    print(f"Found {len(all_images)} images, {len(all_xmls)} annotations")

    if len(all_images) == 0:
        print("⚠  No images found! Check the directory.")
        return

    # Determine train/test split
    train_stems = set()
    test_stems = set()

    # Try to use ImageSets split files
    train_txt = ls_ssdd_dir / "ImageSets" / "Main" / "train.txt"
    test_txt = ls_ssdd_dir / "ImageSets" / "Main" / "test.txt"

    if train_txt.exists() and test_txt.exists():
        print("Using ImageSets split files")
        with open(train_txt) as f:
            train_stems = {line.strip() for line in f if line.strip()}
        with open(test_txt) as f:
            test_stems = {line.strip() for line in f if line.strip()}
    else:
        # Default convention: images 01-10 = train, 11-15 = test
        # Sub-image names are like "1_1.jpg", "1_2.jpg", ..., "15_600.jpg"
        # The first number is the original large image index
        print("Using default split: images 01-10 = train, 11-15 = test")
        for img_path in all_images:
            stem = img_path.stem
            try:
                img_num = int(stem.split("_")[0])
                if img_num <= 10:
                    train_stems.add(stem)
                else:
                    test_stems.add(stem)
            except ValueError:
                train_stems.add(stem)  # Default to train

    print(f"Train: {len(train_stems)} | Val/Test: {len(test_stems)}")

    # Create output directories
    for split in ["train", "val"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Process
    total_ships = 0
    split_counts = {"train": 0, "val": 0}
    empty_count = 0

    for xml_path in tqdm(all_xmls, desc="Converting annotations", unit="img"):
        stem = xml_path.stem

        # Determine split
        if stem in train_stems:
            split = "train"
        elif stem in test_stems:
            split = "val"
        else:
            # If not in either set, assign by image number
            try:
                img_num = int(stem.split("_")[0])
                split = "train" if img_num <= 10 else "val"
            except ValueError:
                split = "train"

        # Parse annotations
        annotations = parse_voc_xml(xml_path)
        yolo_labels = voc_to_yolo(annotations)
        total_ships += len(yolo_labels)

        # Find matching image
        img_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            candidate = img_dir / f"{stem}{ext}"
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            continue

        # Copy image
        dst_img = output_dir / "images" / split / f"{stem}.jpg"
        shutil.copy2(img_path, dst_img)

        # Write labels (even empty ones — important for YOLO to learn backgrounds)
        dst_label = output_dir / "labels" / split / f"{stem}.txt"
        with open(dst_label, "w") as f:
            f.write("\n".join(yolo_labels))

        if len(yolo_labels) == 0:
            empty_count += 1

        split_counts[split] += 1

    print(f"\nTotal images processed: {sum(split_counts.values())}")
    print(f"  Train: {split_counts['train']}")
    print(f"  Val:   {split_counts['val']}")
    print(f"Total ship annotations: {total_ships}")
    print(f"Empty background images: {empty_count} ")

    # Create dataset.yaml
    yaml_content = f"""# LS-SSDD-v1.0 - Large-Scale SAR Ship Detection Dataset (Sentinel-1)

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
    print("=" * 60)
    print("Preparing LS-SSDD-v1.0 for YOLO training")
    print("=" * 60)

    ls_ssdd_dir = find_ls_ssdd()

    if ls_ssdd_dir is None:
        print(f"\n⚠  LS-SSDD not found!")
        print(f"   Download from: https://github.com/TianwenZhang0825/LS-SSDD-v1.0-OPEN")
        print(f"\n   Clone into data/ssdd/:")
        print(f"   cd {DATA_DIR / 'ssdd'}")
        print(f"   git clone https://github.com/TianwenZhang0825/LS-SSDD-v1.0-OPEN.git")
        return

    print(f"Found LS-SSDD at: {ls_ssdd_dir}")

    # Clean old YOLO dataset
    if YOLO_DATASET_DIR.exists():
        print(f"\nRemoving old YOLO dataset at {YOLO_DATASET_DIR}/")
        shutil.rmtree(YOLO_DATASET_DIR)

    create_yolo_dataset_from_ls_ssdd(ls_ssdd_dir, YOLO_DATASET_DIR)



if __name__ == "__main__":
    main()
