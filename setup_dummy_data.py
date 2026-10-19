#!/usr/bin/env python3
"""
setup_dummy_data.py
Creates the directory hierarchy for CoconutDoctor training, validation, and test sets.
Generates lightweight placeholder images and prints instructions for dropping in real datasets.
"""

import os
import sys
try:
    from PIL import Image, ImageDraw
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

CLASSES = ["BudRot", "GreyLeafSpot", "LeafRot", "StemBleeding", "Healthy"]
SPLITS = {
    "train": 10,
    "val": 4,
    "test": 4
}
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")

# Color cues for synthetic dummy images (RGB)
CLASS_COLORS = {
    "BudRot": (139, 69, 19),         # Brownish rot
    "GreyLeafSpot": (128, 128, 128),  # Grey spots
    "LeafRot": (160, 82, 45),        # Dark sienna
    "StemBleeding": (178, 34, 34),   # Rust/reddish
    "Healthy": (34, 139, 34)         # Green
}


def _create_minimal_bmp(width, height, rgb_color):
    """Generates a minimal 24-bit uncompressed BMP binary image using standard library."""
    r, g, b = rgb_color
    row_bytes = width * 3
    padding = (4 - (row_bytes % 4)) % 4
    image_size = (row_bytes + padding) * height
    file_size = 54 + image_size

    # BMP Header
    header = bytearray([
        0x42, 0x4D,               # 'BM'
        file_size & 0xFF, (file_size >> 8) & 0xFF, (file_size >> 16) & 0xFF, (file_size >> 24) & 0xFF,
        0, 0, 0, 0,               # Reserved
        54, 0, 0, 0,              # Offset to pixel data
        # DIB Header (BITMAPINFOHEADER)
        40, 0, 0, 0,              # DIB header size
        width & 0xFF, (width >> 8) & 0xFF, (width >> 16) & 0xFF, (width >> 24) & 0xFF,
        height & 0xFF, (height >> 8) & 0xFF, (height >> 16) & 0xFF, (height >> 24) & 0xFF,
        1, 0,                     # Color planes
        24, 0,                    # Bits per pixel
        0, 0, 0, 0,               # Compression (0 = none)
        image_size & 0xFF, (image_size >> 8) & 0xFF, (image_size >> 16) & 0xFF, (image_size >> 24) & 0xFF,
        0x13, 0x0B, 0, 0,         # Horizontal resolution (2835 px/m)
        0x13, 0x0B, 0, 0,         # Vertical resolution (2835 px/m)
        0, 0, 0, 0,               # Colors in palette
        0, 0, 0, 0                # Important colors
    ])

    pixel_data = bytearray()
    row = bytearray([b, g, r] * width + [0] * padding)
    for _ in range(height):
        pixel_data.extend(row)

    return bytes(header + pixel_data)


def create_dummy_dataset(base_dir=BASE_DIR, num_samples_per_split=SPLITS):
    """Creates folder structure and placeholder images for dry-run verification."""
    print("=" * 70)
    print("CoconutDoctor — Dataset Directory Setup")
    print("=" * 70)
    print(f"Base Directory: {base_dir}\n")

    created_folders = 0
    created_images = 0

    for split, count in num_samples_per_split.items():
        for class_name in CLASSES:
            folder_path = os.path.join(base_dir, split, class_name)
            os.makedirs(folder_path, exist_ok=True)
            created_folders += 1

            color = CLASS_COLORS.get(class_name, (100, 100, 100))
            for i in range(count):
                if HAS_PIL:
                    img_path = os.path.join(folder_path, f"dummy_{class_name.lower()}_{i+1:03d}.jpg")
                    if not os.path.exists(img_path):
                        img = Image.new("RGB", (224, 224), color=color)
                        draw = ImageDraw.Draw(img)
                        draw.text((10, 10), f"{split.upper()}: {class_name}", fill=(255, 255, 255))
                        draw.rectangle([20, 40, 204, 204], outline=(255, 255, 255), width=2)
                        img.save(img_path, quality=85)
                        created_images += 1
                else:
                    img_path = os.path.join(folder_path, f"dummy_{class_name.lower()}_{i+1:03d}.bmp")
                    if not os.path.exists(img_path):
                        bmp_data = _create_minimal_bmp(224, 224, color)
                        with open(img_path, "wb") as f:
                            f.write(bmp_data)
                        created_images += 1

    print(f"Created/verified {created_folders} class folders.")
    print(f"Generated {created_images} placeholder images for dry-run verification.\n")
    print("INSTRUCTIONS FOR REAL DATASET:")
    print("1. Replace or augment images in:")
    for split in SPLITS.keys():
        print(f"   - {os.path.join(base_dir, split)}/")
        for c in CLASSES:
            print(f"       └── {c}/")
    print("\n2. Recommended real dataset split ratio:")
    print("   - train/ : 70% to 80% of images per class")
    print("   - val/   : 10% to 15% of images per class")
    print("   - test/  : 10% to 15% of images per class")
    print("3. Ensure class folder names match exactly:")
    print(f"   {CLASSES}")
    print("=" * 70)


if __name__ == "__main__":
    create_dummy_dataset()
