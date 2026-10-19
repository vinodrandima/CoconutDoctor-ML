#!/usr/bin/env python3
"""
train_model.py
Two-phase transfer learning with MobileNetV2 for 5-class coconut leaf disease classification:
Phase 1: Feature Extraction (Frozen MobileNetV2 base + trained classification top)
Phase 2: Fine-Tuning (Unfreeze top layers of MobileNetV2 with a reduced learning rate)

Callbacks:
- EarlyStopping (restores best weights)
- ModelCheckpoint (saves best model based on validation accuracy)
- ReduceLROnPlateau (smooth learning rate decay upon plateau)
"""

import os
import argparse
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from prepare_data import get_data_generators, CLASSES, IMAGE_SIZE, BATCH_SIZE

MODEL_CHECKPOINT_KERAS = "best_coconut_doctor_model.keras"
MODEL_CHECKPOINT_H5 = "best_coconut_doctor_model.h5"


def build_model(num_classes=len(CLASSES), input_shape=(224, 224, 3)):
    """
    Builds transfer learning model based on MobileNetV2 pretrained on ImageNet.
    Pretrained base is initially frozen for Phase 1.
    """
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )
    base_model.trainable = False

    inputs = layers.Input(shape=input_shape, name="input_image")
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="head_batchnorm")(x)
    x = layers.Dense(128, activation="relu", name="head_dense")(x)
    x = layers.Dropout(0.3, name="head_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="CoconutDoctor_MobileNetV2")
    return model, base_model


def train(
    dataset_dir="dataset",
    batch_size=BATCH_SIZE,
    phase1_epochs=15,
    phase2_epochs=20,
    fine_tune_at=100,
    auto_balance=False,
    target_accuracy=0.90
):
    print("=" * 70)
    print("CoconutDoctor — Model Training Pipeline (Two-Phase Transfer Learning)")
    print("=" * 70)
    print(f"Target Validation Accuracy: >= {target_accuracy * 100:.1f}%\n")

    # Load data generators
    train_gen, val_gen, _ = get_data_generators(
        dataset_dir=dataset_dir,
        batch_size=batch_size,
        img_size=IMAGE_SIZE
    )

    # Build model
    model, base_model = build_model(
        num_classes=len(CLASSES),
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    )
    model.summary()

    # Compute class weights if requested
    class_weights_dict = None
    if auto_balance:
        from sklearn.utils.class_weight import compute_class_weight
        import numpy as np
        unique_classes = np.unique(train_gen.classes)
        weights = compute_class_weight(
            class_weight="balanced",
            classes=unique_classes,
            y=train_gen.classes
        )
        class_weights_dict = dict(zip(unique_classes, weights))
        print("\n[INFO] Auto-balance enabled. Computed class weights:")
        for cls_idx, weight in class_weights_dict.items():
            print(f"  Class {cls_idx} ({CLASSES[cls_idx]}): weight = {weight:.3f}")

    callbacks = [
        EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=6,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=MODEL_CHECKPOINT_KERAS,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.2,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]

    # ---------------------------------------------------------
    # PHASE 1: Train Classification Head (Feature Extraction)
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 1: Training Classification Head (MobileNetV2 base frozen)")
    print("=" * 70)
    base_model.trainable = False

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    history_p1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=phase1_epochs,
        callbacks=callbacks,
        class_weight=class_weights_dict
    )

    p1_best_val_acc = max(history_p1.history.get("val_accuracy", [0.0]))
    print(f"\n[INFO] Phase 1 Completed. Best val_accuracy: {p1_best_val_acc * 100:.2f}%")

    # ---------------------------------------------------------
    # PHASE 2: Fine-Tuning Top Layers of MobileNetV2
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"PHASE 2: Fine-Tuning Top Layers of MobileNetV2 (from layer {fine_tune_at} onward)")
    print("=" * 70)

    # Unfreeze top layers of base model while keeping BatchNorm layers in inference mode
    base_model.trainable = True

    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False  # Freeze BN statistics to avoid distribution shock

    trainable_count = sum(len(layer.weights) for layer in model.layers if layer.trainable)
    print(f"[INFO] Base model unfrozen from layer {fine_tune_at} onwards.")
    print(f"[INFO] Trainable weight tensors in model: {trainable_count}")

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-5),  # Lower LR for fine-tuning
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    history_p2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=phase2_epochs,
        callbacks=callbacks,
        class_weight=class_weights_dict
    )

    # Also save .h5 format for maximum toolchain compatibility
    try:
        model.save(MODEL_CHECKPOINT_H5)
        print(f"[INFO] Also saved fallback format: {MODEL_CHECKPOINT_H5}")
    except Exception as e:
        print(f"[NOTE] .h5 save skipped or failed: {e}")

    # Evaluate best saved model
    print("\n" + "=" * 70)
    print("TRAINING EVALUATION & TARGET VERIFICATION")
    print("=" * 70)
    best_model = tf.keras.models.load_model(MODEL_CHECKPOINT_KERAS)
    val_loss, val_acc = best_model.evaluate(val_gen, verbose=0)
    print(f"Validation Loss:     {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc * 100:.2f}%")

    if val_acc >= target_accuracy:
        print(f"\n[SUCCESS] Target validation accuracy ({target_accuracy*100:.1f}%) ACHIEVED! ({val_acc*100:.2f}%)")
    else:
        print(f"\n[NOTE] Validation accuracy is {val_acc*100:.2f}% (Target: {target_accuracy*100:.1f}%).")
        print("Tip: Train on real coconut leaf disease images with 100+ images per class to reach >90%.")

    return best_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CoconutDoctor MobileNetV2 Model")
    parser.add_argument("--dataset-dir", type=str, default="dataset", help="Path to dataset directory")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--epochs-phase1", type=int, default=15, help="Epochs for feature extraction")
    parser.add_argument("--epochs-phase2", type=int, default=20, help="Epochs for fine-tuning")
    parser.add_argument("--fine-tune-at", type=int, default=100, help="Layer index from which to unfreeze MobileNetV2 in Phase 2")
    parser.add_argument("--auto-balance", action="store_true", help="Automatically compute balanced class weights")
    parser.add_argument("--target-accuracy", type=float, default=0.90, help="Target validation accuracy")
    args = parser.parse_args()

    train(
        dataset_dir=args.dataset_dir,
        batch_size=args.batch_size,
        phase1_epochs=args.epochs_phase1,
        phase2_epochs=args.epochs_phase2,
        fine_tune_at=args.fine_tune_at,
        auto_balance=args.auto_balance,
        target_accuracy=args.target_accuracy
    )
