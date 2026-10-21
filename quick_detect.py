#!/usr/bin/env python3
"""
quick_detect.py
Instant, one-line disease diagnosis for CoconutDoctor.

Usage:
  python quick_detect.py <image_path>
  python quick_detect.py <folder_path>
"""

import sys
import os
import glob
from PIL import Image
import coremltools as ct

MODEL_PATH = "CoconutDoctor.mlpackage"

TRANSLATIONS = {
    "BudRot": "කරටි කුණුවීම (Bud Rot)",
    "GreyLeafSpot": "අළු පැහැ පත්‍ර පුල්ලි රෝගය (Grey Leaf Spot)",
    "LeafRot": "පත්‍ර කුණුවීම (Leaf Rot)",
    "StemBleeding": "කඳෙන් දිය ගැලීම (Stem Bleeding)",
    "Healthy": "නිරෝගී ශාකයකි (Healthy - ලෙඩක් නැත)"
}

# Cache model so repeated predictions are instant
_MODEL = None


def get_model():
    global _MODEL
    if _MODEL is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model '{MODEL_PATH}' not found. Run convert_coreml.py first.")
        _MODEL = ct.models.MLModel(MODEL_PATH)
    return _MODEL


def detect_image(img_path):
    model = get_model()
    pil_img = Image.open(img_path).convert("RGB").resize((224, 224))
    pred = model.predict({"input_image": pil_img})
    top_label = pred["classLabel"]
    conf = pred["classLabel_probs"].get(top_label, 0.0) * 100
    si_name = TRANSLATIONS.get(top_label, top_label)
    return top_label, si_name, conf


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nභාවිතා කරන ආකාරය (Usage):")
        print("  python quick_detect.py <image_path>")
        print("  උදා: python quick_detect.py my_leaf.jpg\n")
        # Run on a quick sample
        samples = glob.glob("dataset/test/*/*.jpg")
        if samples:
            sample = samples[0]
            print(f"[Sample Test]: {sample}")
            lbl, si, conf = detect_image(sample)
            print(f"👉 ලෙඩේ: {si} ({lbl}) | විශ්වාසනීයත්වය: {conf:.1f}%\n")
        sys.exit(0)

    target = sys.argv[1]

    # If user passed a single image
    if os.path.isfile(target):
        lbl, si, conf = detect_image(target)
        print(f"\n=======================================================")
        print(f"🌿 Image : {os.path.basename(target)}")
        print(f"🎯 ලෙඩේ  : {si}")
        print(f"📊 Accuracy / Confidence: {conf:.1f}%")
        print(f"=======================================================\n")

    # If user passed a directory of images
    elif os.path.isdir(target):
        files = [f for f in os.listdir(target) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.avif', '.webp'))]
        print(f"\nScanning {len(files)} images in '{target}':\n")
        for f in files:
            fp = os.path.join(target, f)
            lbl, si, conf = detect_image(fp)
            print(f"• {f:<25} ➔ {si:<35} ({conf:5.1f}%)")
        print()
    else:
        print(f"Error: Path '{target}' not found.")
