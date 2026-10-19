#!/usr/bin/env python3
"""
organize_dataset.py
Processes raw downloaded coconut leaf disease images from raw/<Class>/,
validates integrity, strips EXIF orientation, converts to RGB JPEG (max side 1024px),
detects exact duplicates via MD5 hash, and creates reproducible 75% / 12.5% / 12.5%
Train / Validation / Test splits in dataset/ ready for CoconutDoctor model training.

Classes (Exact order maintained):
["BudRot", "GreyLeafSpot", "LeafRot", "StemBleeding", "Healthy"]
"""

import os
import sys
import shutil
import hashlib
import random
from pathlib import Path
from PIL import Image, ImageOps

CLASSES = ["BudRot", "GreyLeafSpot", "LeafRot", "StemBleeding", "Healthy"]
DEFAULT_RAW_DIR = "raw"
DEFAULT_OUT_DIR = "dataset"

# Target Split Ratios (75% / 12.5% / 12.5%)
TRAIN_RATIO = 0.75
VAL_RATIO = 0.125
TEST_RATIO = 0.125
RANDOM_SEED = 42
MAX_IMAGE_SIDE = 1024


def compute_file_md5(file_path, chunk_size=8192):
    """Computes MD5 hash of an image file to detect identical duplicates."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def process_image(src_path, dst_path, max_side=MAX_IMAGE_SIDE):
    """
    Validates, transposes EXIF orientation, resizes to max_side, and saves as RGB JPEG.
    Returns True on success, False if file is corrupted/unreadable.
    """
    try:
        with Image.open(src_path) as img:
            img.verify()  # Fast structural verification

        # Re-open for actual processing (verify closes/corrupts internal pointer)
        with Image.open(src_path) as img:
            # Transpose according to EXIF orientation tag (e.g. smartphone photos)
            img = ImageOps.exif_transpose(img)

            # Convert non-RGB modes (RGBA, CMYK, Palette, Grayscale) to clean RGB
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Resize if either dimension exceeds max_side
            width, height = img.size
            if max(width, height) > max_side:
                if width > height:
                    new_w = max_side
                    new_h = int(height * (max_side / width))
                else:
                    new_h = max_side
                    new_w = int(width * (max_side / height))
                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

            # Save as optimized JPEG
            img.save(dst_path, format="JPEG", quality=90, optimize=True)
            return True
    except Exception as e:
        print(f"    [SKIP] Corrupted or invalid image: {os.path.basename(src_path)} ({e})")
        return False


def setup_raw_folders(raw_dir=DEFAULT_RAW_DIR):
    """Ensures raw/<Class> directory tree exists."""
    for cls in CLASSES:
        os.makedirs(os.path.join(raw_dir, cls), exist_ok=True)


def organize(
    raw_dir=DEFAULT_RAW_DIR,
    out_dir=DEFAULT_OUT_DIR,
    seed=RANDOM_SEED,
    clean_target=True
):
    print("=" * 75)
    print("CoconutDoctor — Raw Dataset Ingestion & Train/Val/Test Splitter")
    print("=" * 75)
    print(f"Source Directory: {os.path.abspath(raw_dir)}")
    print(f"Target Directory: {os.path.abspath(out_dir)}")
    print(f"Split Ratios:     Train {TRAIN_RATIO*100:.1f}% | Val {VAL_RATIO*100:.1f}% | Test {TEST_RATIO*100:.1f}%")
    print(f"Random Seed:      {seed} (reproducible)")
    print(f"Target Classes:   {CLASSES}\n")

    # Ensure raw directory and class folders exist
    setup_raw_folders(raw_dir)

    # Check available files in raw
    total_raw_found = 0
    raw_files_by_class = {}
    for cls in CLASSES:
        cls_dir = os.path.join(raw_dir, cls)
        files = [
            os.path.join(cls_dir, f) for f in os.listdir(cls_dir)
            if os.path.isfile(os.path.join(cls_dir, f)) and not f.startswith(".")
        ]
        raw_files_by_class[cls] = sorted(files)
        total_raw_found += len(files)

    if total_raw_found == 0:
        print("[WARNING] No raw images found in 'raw/' directory.")
        print("Please place your downloaded images into:")
        for cls in CLASSES:
            print(f"   └── {os.path.join(raw_dir, cls)}/")
        print("\nOnce images are added, re-run: python organize_dataset.py")
        print("=" * 75)
        return False

    # Clean existing dataset directory if requested to remove old dummy placeholders
    if clean_target and os.path.exists(out_dir):
        print(f"[INFO] Refreshing target directory '{out_dir}' to purge old dummy files...")
        shutil.rmtree(out_dir)

    # Prepare target folder structure
    splits = ["train", "val", "test"]
    for s in splits:
        for cls in CLASSES:
            os.makedirs(os.path.join(out_dir, s, cls), exist_ok=True)

    summary_stats = {
        cls: {"raw": len(raw_files_by_class[cls]), "duplicates": 0, "valid": 0, "train": 0, "val": 0, "test": 0}
        for cls in CLASSES
    }

    random.seed(seed)

    for cls in CLASSES:
        print(f"\nProcessing class: [{cls}] (Found {len(raw_files_by_class[cls])} raw files)...")
        file_list = raw_files_by_class[cls]

        # De-duplicate by MD5
        seen_hashes = set()
        unique_files = []
        for fp in file_list:
            file_hash = compute_file_md5(fp)
            if file_hash in seen_hashes:
                summary_stats[cls]["duplicates"] += 1
            else:
                seen_hashes.add(file_hash)
                unique_files.append(fp)

        if summary_stats[cls]["duplicates"] > 0:
            print(f"  └─ Filtered {summary_stats[cls]['duplicates']} exact duplicate images.")

        # Shuffle deterministically
        random.shuffle(unique_files)

        # Compute split indices
        total_unique = len(unique_files)
        n_train = int(total_unique * TRAIN_RATIO)
        n_val = int(total_unique * VAL_RATIO)
        # Allocate remainder to test to ensure exact total conservation
        n_test = total_unique - n_train - n_val

        # Partition files
        train_files = unique_files[:n_train]
        val_files = unique_files[n_train:n_train + n_val]
        test_files = unique_files[n_train + n_val:]

        split_assignment = [
            ("train", train_files),
            ("val", val_files),
            ("test", test_files)
        ]

        # Copy & normalize images
        valid_counter = 0
        for split_name, assigned_files in split_assignment:
            for idx, src_path in enumerate(assigned_files):
                dest_filename = f"{cls.lower()}_{split_name}_{idx+1:04d}.jpg"
                dest_path = os.path.join(out_dir, split_name, cls, dest_filename)
                if process_image(src_path, dest_path):
                    summary_stats[cls][split_name] += 1
                    valid_counter += 1

        summary_stats[cls]["valid"] = valid_counter

    # Print Ingestion & Split Summary Table
    print("\n" + "=" * 75)
    print(f"{'CLASS NAME':<16} | {'RAW':<5} | {'DUP':<4} | {'VALID':<5} | {'TRAIN':<5} | {'VAL':<5} | {'TEST':<5} | {'STATUS'}")
    print("-" * 75)

    grand_total_valid = 0
    warnings = []

    for cls in CLASSES:
        st = summary_stats[cls]
        grand_total_valid += st["valid"]
        status = "OK"
        if st["valid"] < 100:
            status = "NEEDS MORE"
            warnings.append(f"Class '{cls}' has only {st['valid']} images (target: >= 100 images).")
        print(f"{cls:<16} | {st['raw']:<5} | {st['duplicates']:<4} | {st['valid']:<5} | {st['train']:<5} | {st['val']:<5} | {st['test']:<5} | {status}")

    print("-" * 75)
    print(f"Grand Total Valid Images Across All Splits: {grand_total_valid}")
    print("=" * 75)

    if warnings:
        print("\n[ATTENTION - DATASET QUALITY WARNINGS]")
        for w in warnings:
            print(f"  ⚠️  {w}")
        print("\nTip: Refer to the dataset sources guide below to augment minority classes.")
    else:
        print("\n[SUCCESS] Dataset split generated successfully and meets quality standards!")

    print(f"\nNext command to verify generator loading:")
    print("  python prepare_data.py")
    return True


if __name__ == "__main__":
    raw_folder = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RAW_DIR
    target_folder = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUT_DIR
    organize(raw_dir=raw_folder, out_dir=target_folder)
