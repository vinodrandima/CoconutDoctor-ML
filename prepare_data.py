#!/usr/bin/env python3
"""
prepare_data.py
Loads train/val/test folders using tf.keras ImageDataGenerator with data augmentation.
Enforces the exact class order required for CoconutDoctor Core ML deployment:
["BudRot", "GreyLeafSpot", "LeafRot", "StemBleeding", "Healthy"]
"""

import os
import sys
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

CLASSES = ["BudRot", "GreyLeafSpot", "LeafRot", "StemBleeding", "Healthy"]
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
DEFAULT_DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")


def check_dataset_exists(dataset_dir=DEFAULT_DATASET_DIR):
    """Verifies that dataset directories and subfolders exist."""
    splits = ["train", "val", "test"]
    missing = []
    for split in splits:
        split_path = os.path.join(dataset_dir, split)
        if not os.path.isdir(split_path):
            missing.append(split_path)
            continue
        for class_name in CLASSES:
            class_path = os.path.join(split_path, class_name)
            if not os.path.isdir(class_path):
                missing.append(class_path)
    return len(missing) == 0, missing


def get_data_generators(dataset_dir=DEFAULT_DATASET_DIR, batch_size=BATCH_SIZE, img_size=IMAGE_SIZE):
    """
    Constructs and returns train, val, and test data generators.
    Uses MobileNetV2 preprocess_input to scale RGB inputs [0, 255] to [-1, 1].
    Crucially passes `classes=CLASSES` to lock down label indices to match Core ML.
    """
    exists, missing = check_dataset_exists(dataset_dir)
    if not exists:
        print("[WARNING] Dataset directory is incomplete or missing.")
        print(f"Missing paths: {missing[:5]} (total {len(missing)} missing)")
        print("\nAttempting to generate placeholder directory structure with dummy images...")
        try:
            from setup_dummy_data import create_dummy_dataset
            create_dummy_dataset(base_dir=dataset_dir)
        except Exception as e:
            print(f"[ERROR] Could not auto-generate placeholder dataset: {e}")
            print("Please run: python setup_dummy_data.py or add your real images to dataset/")
            sys.exit(1)

    train_dir = os.path.join(dataset_dir, "train")
    val_dir = os.path.join(dataset_dir, "val")
    test_dir = os.path.join(dataset_dir, "test")

    # Data Augmentation for training
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=False,
        brightness_range=[0.85, 1.15],
        fill_mode="nearest"
    )

    # Validation and testing generators: only preprocessing, NO augmentation
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    print(f"[INFO] Initializing generators from: {dataset_dir}")
    print(f"[INFO] Enforcing class list and index order: {CLASSES}")

    # Flow from directory with explicit class order!
    train_generator = train_datagen.flow_from_directory(
        directory=train_dir,
        target_size=img_size,
        color_mode="rgb",
        classes=CLASSES,  # CRITICAL: Fixes class index order [0..4]
        class_mode="categorical",
        batch_size=batch_size,
        shuffle=True,
        seed=42
    )

    val_generator = val_datagen.flow_from_directory(
        directory=val_dir,
        target_size=img_size,
        color_mode="rgb",
        classes=CLASSES,  # CRITICAL: Same class index order
        class_mode="categorical",
        batch_size=batch_size,
        shuffle=False,
        seed=42
    )

    test_generator = test_datagen.flow_from_directory(
        directory=test_dir,
        target_size=img_size,
        color_mode="rgb",
        classes=CLASSES,  # CRITICAL: Same class index order
        class_mode="categorical",
        batch_size=batch_size,
        shuffle=False,  # DO NOT SHUFFLE test set for consistent metric evaluation
        seed=42
    )

    return train_generator, val_generator, test_generator


if __name__ == "__main__":
    print("Testing data pipeline initialization...")
    train_gen, val_gen, test_gen = get_data_generators()

    print("\nVerified Class Indices in Generator:")
    for cls_name, idx in train_gen.class_indices.items():
        print(f"  Index {idx} => {cls_name}")

    # Assert exact order matching
    expected_indices = {cls: idx for idx, cls in enumerate(CLASSES)}
    assert train_gen.class_indices == expected_indices, "FATAL: class index mismatch!"
    print("\n[SUCCESS] Class indices successfully locked to exact required order!")
    print(f"Train samples: {train_gen.samples}")
    print(f"Val samples:   {val_gen.samples}")
    print(f"Test samples:  {test_gen.samples}")
