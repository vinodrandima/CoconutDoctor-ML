#!/usr/bin/env python3
"""
evaluate.py
Evaluates the trained CoconutDoctor MobileNetV2 model on the held-out test dataset.
Generates all formal coursework report artifacts:
1. classification_report.txt — per-class precision, recall, F1-score, and support
2. confusion_matrix.png     — annotated high-res seaborn/matplotlib heatmap
3. metrics.json             — machine-readable evaluation metrics & summary stats
"""

import os
import json
import argparse
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from prepare_data import get_data_generators, CLASSES, IMAGE_SIZE, BATCH_SIZE


def evaluate_model(
    model_path="best_coconut_doctor_model.keras",
    dataset_dir="dataset",
    output_cm_path="confusion_matrix.png",
    output_report_path="classification_report.txt",
    output_metrics_path="metrics.json",
    batch_size=BATCH_SIZE
):
    print("=" * 70)
    print("CoconutDoctor — Model Evaluation & Academic Artifact Generator")
    print("=" * 70)

    # Check model path
    if not os.path.exists(model_path):
        fallback_h5 = "best_coconut_doctor_model.h5"
        if os.path.exists(fallback_h5):
            print(f"[INFO] '{model_path}' not found, falling back to '{fallback_h5}'")
            model_path = fallback_h5
        else:
            raise FileNotFoundError(
                f"Model checkpoint '{model_path}' not found. Please run train_model.py first."
            )

    print(f"Loading model checkpoint from: {model_path}")
    model = tf.keras.models.load_model(model_path)

    # Load test data generator (shuffle must be False for consistent ground truth)
    print(f"Loading test dataset from: {dataset_dir}/test")
    _, _, test_gen = get_data_generators(
        dataset_dir=dataset_dir,
        batch_size=batch_size,
        img_size=IMAGE_SIZE
    )

    if test_gen.samples == 0:
        raise ValueError(f"No test images found in {dataset_dir}/test")

    print(f"Running inference on {test_gen.samples} test samples...")
    test_gen.reset()
    predictions = model.predict(test_gen, verbose=1)

    y_pred = np.argmax(predictions, axis=1)
    y_true = test_gen.classes

    # Compute Classification Metrics (as dictionary and formatted string)
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=CLASSES,
        digits=4,
        zero_division=0,
        output_dict=True
    )

    report_str = classification_report(
        y_true,
        y_pred,
        target_names=CLASSES,
        digits=4,
        zero_division=0
    )

    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)
    print(report_str)

    # 1. Save classification_report.txt
    with open(output_report_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("CoconutDoctor — Test Set Classification Report\n")
        f.write("=" * 70 + "\n")
        f.write(report_str)
        f.write("\n" + "=" * 70 + "\n")
    print(f"[SUCCESS] Saved text report to: {os.path.abspath(output_report_path)}")

    # 2. Compute & Plot Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))
    print("\nCONFUSION MATRIX (Counts):")
    print(cm)

    plt.figure(figsize=(9, 7.5))
    sns.set_theme(style="white")
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        cbar=True,
        linewidths=1.0,
        linecolor="#dddddd",
        annot_kws={"size": 13, "weight": "bold"}
    )
    plt.title("CoconutDoctor — Test Confusion Matrix", fontsize=15, pad=18, fontweight="bold")
    plt.ylabel("True Disease Label (Ground Truth)", fontsize=12, labelpad=12, fontweight="semibold")
    plt.xlabel("Predicted Disease Label", fontsize=12, labelpad=12, fontweight="semibold")
    plt.xticks(rotation=30, ha="right", fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    plt.tight_layout()

    plt.savefig(output_cm_path, dpi=300)
    plt.close()
    print(f"[SUCCESS] Saved confusion matrix plot to: {os.path.abspath(output_cm_path)}")

    # 3. Save metrics.json
    overall_acc = float(np.mean(y_pred == y_true))
    metrics_data = {
        "model_file": model_path,
        "total_test_samples": int(test_gen.samples),
        "test_accuracy": round(overall_acc, 6),
        "macro_avg": {
            "precision": round(report_dict["macro avg"]["precision"], 4),
            "recall": round(report_dict["macro avg"]["recall"], 4),
            "f1_score": round(report_dict["macro avg"]["f1-score"], 4)
        },
        "weighted_avg": {
            "precision": round(report_dict["weighted avg"]["precision"], 4),
            "recall": round(report_dict["weighted avg"]["recall"], 4),
            "f1_score": round(report_dict["weighted avg"]["f1-score"], 4)
        },
        "per_class_metrics": {
            cls: {
                "precision": round(report_dict[cls]["precision"], 4),
                "recall": round(report_dict[cls]["recall"], 4),
                "f1_score": round(report_dict[cls]["f1-score"], 4),
                "support": int(report_dict[cls]["support"])
            }
            for cls in CLASSES
        },
        "confusion_matrix": cm.tolist()
    }

    with open(output_metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=4)
    print(f"[SUCCESS] Saved structured metrics to: {os.path.abspath(output_metrics_path)}")

    print("\n" + "=" * 70)
    print(f"[SUMMARY] Overall Test Accuracy: {overall_acc * 100:.2f}%")
    print(f"[SUMMARY] Macro F1-Score:      {report_dict['macro avg']['f1-score'] * 100:.2f}%")
    print("=" * 70)

    return overall_acc, report_dict, cm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CoconutDoctor Model and Generate Report Artifacts")
    parser.add_argument("--model-path", type=str, default="best_coconut_doctor_model.keras", help="Path to trained model")
    parser.add_argument("--dataset-dir", type=str, default="dataset", help="Path to dataset directory")
    parser.add_argument("--output-cm", type=str, default="confusion_matrix.png", help="Output path for confusion matrix")
    parser.add_argument("--output-report", type=str, default="classification_report.txt", help="Output path for classification report text")
    parser.add_argument("--output-metrics", type=str, default="metrics.json", help="Output path for JSON metrics")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    args = parser.parse_args()

    evaluate_model(
        model_path=args.model_path,
        dataset_dir=args.dataset_dir,
        output_cm_path=args.output_cm,
        output_report_path=args.output_report,
        output_metrics_path=args.output_metrics,
        batch_size=args.batch_size
    )
