#!/usr/bin/env python3
"""
predict.py
Tests single coconut leaf image prediction using CoconutDoctor.mlpackage (Core ML).

Usage:
  python predict.py path/to/leaf_image.jpg
  python predict.py   (runs on a sample test image if no path is given)
"""

import os
import sys
import glob
from PIL import Image
import coremltools as ct

MODEL_PATH = "CoconutDoctor.mlpackage"
CLASSES = ["BudRot", "GreyLeafSpot", "LeafRot", "StemBleeding", "Healthy"]

# Sinhala translation & short recommendations for CoconutDoctor
DISEASE_INFO = {
    "BudRot": {
        "name_si": "කරටි කුණුවීම (Bud Rot)",
        "symptom": "Crown rotting, central spear leaf wilting",
        "action": "Apply 1% Bordeaux mixture / Copper oxychloride paste to the crown."
    },
    "GreyLeafSpot": {
        "name_si": "අළු පැහැ පත්‍ර පුල්ලි රෝගය (Grey Leaf Spot)",
        "symptom": "Oval grey spots with dark brown margins",
        "action": "Cut and burn heavily affected lower fronds; spray copper fungicide."
    },
    "LeafRot": {
        "name_si": "පත්‍ර කුණුවීම (Leaf Rot)",
        "symptom": "Blackening and rotting of leaflet tips",
        "action": "Remove infected parts; apply fungicide and improve tree nutrition (Potash)."
    },
    "StemBleeding": {
        "name_si": "කඳෙන් දිය ගැලීම (Stem Bleeding)",
        "symptom": "Rust-brown liquid oozing from stem cracks",
        "action": "Chisel affected bark, apply coal tar or Bordeaux paste; improve drainage."
    },
    "Healthy": {
        "name_si": "නිරෝගී ශාකය (Healthy Palm)",
        "symptom": "Normal green foliage without lesions",
        "action": "Maintain regular fertilizer application (APM/CRI recommended)."
    }
}


def predict_single_image(image_path, model_path=MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Core ML model '{model_path}' not found. Please run convert_coreml.py first.")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image '{image_path}' not found.")

    print("=" * 65)
    print("🌴 CoconutDoctor — On-Device Single Image Diagnosis")
    print("=" * 65)
    print(f"Loading Core ML Model: {model_path}...")
    model = ct.models.MLModel(model_path)

    print(f"Loading Image:         {image_path}")
    pil_img = Image.open(image_path).convert("RGB").resize((224, 224))

    # Run Core ML prediction
    # (scale and bias are automatically handled by ImageType inside the model)
    prediction = model.predict({"input_image": pil_img})

    top_class = prediction["classLabel"]
    probs = prediction["classLabel_probs"]
    top_confidence = probs.get(top_class, 0.0)

    # Sort all probabilities descending
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)

    info = DISEASE_INFO.get(top_class, {})

    print("\n" + "-" * 65)
    print("🎯 DIAGNOSIS RESULTS (ප්‍රතිඵලය):")
    print("-" * 65)
    print(f"  • Detected Condition: {top_class}")
    print(f"  • සිංහල නම:           {info.get('name_si', 'N/A')}")
    print(f"  • Confidence:         {top_confidence * 100:.2f}%")
    print("-" * 65)

    print("\n📊 PROBABILITY BREAKDOWN ACROSS ALL 5 CLASSES:")
    for cls_name, prob in sorted_probs:
        bar_len = int(prob * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        pointer = " ⬅ (TOP)" if cls_name == top_class else ""
        print(f"  {cls_name:<14} : [{bar}] {prob * 100:6.2f}%{pointer}")

    print("\n💡 FIELD ADVISORY (ගොවියා සඳහා උපදෙස්):")
    print(f"  • Symptoms: {info.get('symptom', '-')}")
    print(f"  • Action:   {info.get('action', '-')}")
    print("=" * 65)

    return top_class, top_confidence, probs


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_img = sys.argv[1]
    else:
        # Default: Pick one real image from dataset/test
        samples = glob.glob("dataset/test/*/*.jpg")
        if samples:
            target_img = samples[0]
            print(f"[NOTE] No image path provided. Testing with a sample image: {target_img}\n")
        else:
            print("Usage: python predict.py <path_to_image>")
            sys.exit(1)

    predict_single_image(target_img)
