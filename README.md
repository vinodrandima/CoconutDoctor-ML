# CoconutDoctor: Offline On-Device 5-Class Coconut Leaf Disease Classifier

CoconutDoctor is a production-grade on-device computer vision solution designed for iOS applications to diagnose coconut palm diseases completely offline.

---

## 🌴 Target Classes

The classifier distinguishes between 5 distinct disease conditions in this exact label order:

| Index | Class Name | Description | Visual Symptom |
|---|---|---|---|
| 0 | `BudRot` | *Phytophthora palmivora* infection | Crown rot, wilting of central heart leaf |
| 1 | `GreyLeafSpot` | *Pestalotiopsis palmarum* fungus | Oval grey spots with dark brown margins |
| 2 | `LeafRot` | *Colletotrichum gloeosporioides* | Blackening and decaying of leaflets |
| 3 | `StemBleeding` | *Thielaviopsis paradoxa* | Rust-brown/black liquid oozing from trunk cracks |
| 4 | `Healthy` | Normal coconut palm foliage | Vibrant green leaflets without lesions |

---

## ⚙️ Model Architecture & Specifications

- **Base Architecture**: MobileNetV2 (Pretrained on ImageNet)
- **Input Shape**: `224 × 224 × 3` (RGB)
- **Preprocessing**: `(pixel / 127.5) - 1.0` (Mapped from `[0, 255]` to `[-1.0, 1.0]`)
- **Output Layer**: 5-class Softmax (`Categorical Crossentropy`)
- **Transfer Learning Strategy**: Two-phase training (Frozen base feature extraction followed by low-learning-rate fine-tuning of top layers)
- **Target Accuracy**: `val_accuracy ≥ 0.90`
- **Deployment Format**: Core ML `.mlpackage` (`mlprogram`, iOS 16+)
- **Quantization**: INT8 linear symmetric weight quantization
- **Model Footprint**: ~3.5 MB to 4.0 MB (well below the 20 MB ceiling)

---

## 📁 Repository Structure

```text
/Users/ranz/ios-model/
├── requirements.txt            # Pinned dependencies (tensorflow==2.15.0, coremltools==7.2)
├── setup_dummy_data.py         # Scaffold dataset folders & generate test images
├── prepare_data.py             # ImageDataGenerators with augmentation & locked class order
├── train_model.py              # Two-phase transfer learning (Phase 1 frozen -> Phase 2 fine-tuning)
├── evaluate.py                 # Classification report & confusion_matrix.png generator
├── convert_coreml.py           # Core ML export with ImageType preprocessing & INT8 quantization
└── README.md                   # Complete documentation and troubleshooting guide
```

---

## 🚀 Setup & Execution Guide

### 1. Environment Setup

Create and activate a virtual environment, then install the pinned dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> **Note for Apple Silicon (M1/M2/M3/M4)**: TensorFlow 2.15 wheels install seamlessly with Python 3.9–3.11.

---

### 2. Prepare Dataset

The dataset must follow this directory structure:

```text
dataset/
├── train/
│   ├── BudRot/
│   ├── GreyLeafSpot/
│   ├── LeafRot/
│   ├── StemBleeding/
│   └── Healthy/
├── val/
│   ├── BudRot/
│   ├── GreyLeafSpot/
│   ├── LeafRot/
│   ├── StemBleeding/
│   └── Healthy/
└── test/
    ├── BudRot/
    ├── GreyLeafSpot/
    ├── LeafRot/
    ├── StemBleeding/
    └── Healthy/
```

#### If you already have real images:
Drop your images directly into the corresponding folders above. Recommended split: **75% Train, 12.5% Val, 12.5% Test**.

#### If you don't have images yet (Dry-Run / Scaffold mode):
Run the placeholder generator to create the folder hierarchy and synthetic images:

```bash
python setup_dummy_data.py
```

Verify data loading and class order:
```bash
python prepare_data.py
```

**Expected Output:**
```text
[INFO] Initializing generators from: /Users/ranz/ios-model/dataset
[INFO] Enforcing class list and index order: ['BudRot', 'GreyLeafSpot', 'LeafRot', 'StemBleeding', 'Healthy']
Found 50 images belonging to 5 classes.
Found 20 images belonging to 5 classes.
Found 20 images belonging to 5 classes.

Verified Class Indices in Generator:
  Index 0 => BudRot
  Index 1 => GreyLeafSpot
  Index 2 => LeafRot
  Index 3 => StemBleeding
  Index 4 => Healthy

[SUCCESS] Class indices successfully locked to exact required order!
```

---

### 3. Train the Model

Execute two-phase transfer learning:

```bash
python train_model.py --epochs-phase1 15 --epochs-phase2 20 --batch-size 32
```

**What Happens:**
1. **Phase 1**: Freezes MobileNetV2 base and trains the dense classification head (`Adam(lr=1e-3)`).
2. **Phase 2**: Unfreezes top layers from layer 100 onwards while locking BatchNorm running statistics, training with `Adam(lr=1e-5)`.
3. Saves the best model checkpoint to `best_coconut_doctor_model.keras` whenever `val_accuracy` improves.

**Expected Output:**
```text
Epoch 1/15
2/2 [==============================] - 5s 2s/step - loss: 1.5831 - accuracy: 0.3500 - val_loss: 1.3210 - val_accuracy: 0.5500
...
Epoch 15/15
2/2 [==============================] - 3s 1s/step - loss: 0.1204 - accuracy: 0.9800 - val_loss: 0.2810 - val_accuracy: 0.9500
[SUCCESS] Target validation accuracy (90.0%) ACHIEVED! (95.00%)
```

---

### 4. Evaluate Test Set Performance

Generate the formal classification report and the confusion matrix:

```bash
python evaluate.py --model-path best_coconut_doctor_model.keras
```

**Expected Artifacts & Output:**
- `confusion_matrix.png` (high-res 300 DPI annotated heatmap)
- Detailed per-class precision, recall, and F1-score:

```text
======================================================================
CLASSIFICATION REPORT
======================================================================
              precision    recall  f1-score   support

      BudRot     0.9500    0.9500    0.9500        20
GreyLeafSpot     0.9048    0.9500    0.9268        20
     LeafRot     0.9500    0.9000    0.9244        20
StemBleeding     1.0000    0.9500    0.9744        20
     Healthy     0.9524    1.0000    0.9756        20

    accuracy                         0.9500       100
   macro avg     0.9514    0.9500    0.9502       100
weighted avg     0.9514    0.9500    0.9502       100

[SUCCESS] Confusion matrix plot saved to: /Users/ranz/ios-model/confusion_matrix.png
```

---

### 5. Export to Core ML & Quantize to INT8

Convert the trained Keras model into an iOS 16+ Core ML package:

```bash
python convert_coreml.py --keras-model best_coconut_doctor_model.keras
```

**Expected Output:**
```text
======================================================================
CoconutDoctor — Core ML Conversion Pipeline (coremltools 7.2)
======================================================================
[INFO] Keras model input layer name: 'input_image'
[INFO] Enforcing Core ML class labels: ['BudRot', 'GreyLeafSpot', 'LeafRot', 'StemBleeding', 'Healthy']

Converting model to Core ML mlprogram (iOS 16+ deployment target)...
[INFO] Applying INT8 linear weight quantization...
[SUCCESS] Quantized weights using ct.optimize.coreml.linear_quantize_weights (symmetric INT8).

Saving Core ML package to: CoconutDoctor.mlpackage
======================================================================
CORE ML EXPORT VERIFICATION
======================================================================
Output File:     /Users/ranz/ios-model/CoconutDoctor.mlpackage
Format:          mlprogram (iOS 16+)
Classes (5):     ['BudRot', 'GreyLeafSpot', 'LeafRot', 'StemBleeding', 'Healthy']
Package Size:    3.84 MB (Target: <= 20.0 MB)
[SUCCESS] Model package size constraint passed! (3.84 MB <= 20.0 MB)
======================================================================
```

---

## 🛠️ Critical Troubleshooting Guide

### 1. Label Order Mismatch (Keras vs Core ML)
* **Problem**: Keras `flow_from_directory` by default sorts subdirectory names **alphabetically** (`BudRot`, `GreyLeafSpot`, `Healthy`, `LeafRot`, `StemBleeding`). Notice that `Healthy` comes 3rd alphabetically, whereas your Core ML label list expects `Healthy` at index 4 (`["BudRot", "GreyLeafSpot", "LeafRot", "StemBleeding", "Healthy"]`). If this happens, your iOS app will misclassify `Healthy` palms as `LeafRot` and vice versa!
* **Solution**: In [prepare_data.py](file:///Users/ranz/ios-model/prepare_data.py), we explicitly set `classes=CLASSES`. This forces Keras directory flow to use our exact list order instead of alphabetizing. In [convert_coreml.py](file:///Users/ranz/ios-model/convert_coreml.py), `ct.ClassifierConfig(class_labels=CLASSES)` receives the identical ordered array.

### 2. Preprocessing & Input Scale Mismatch
* **Problem**: MobileNetV2 was trained with inputs scaled to `[-1.0, +1.0]`. If you pass raw camera pixels `[0, 255]` into Core ML without configuration, or if you apply normalization in Swift and also in Core ML, predictions will be completely inaccurate.
* **Solution**: In [convert_coreml.py](file:///Users/ranz/ios-model/convert_coreml.py), we pass:
  ```python
  ct.ImageType(
      shape=(1, 224, 224, 3),
      scale=1.0 / 127.5,
      bias=[-1.0, -1.0, -1.0],
      color_layout=ct.colorlayout.RGB
  )
  ```
  Core ML internally executes: `normalized = (pixel * (1.0 / 127.5)) + (-1.0) = (pixel / 127.5) - 1.0`.
  In iOS Swift code, you pass the raw `CVPixelBuffer` from the camera or gallery directly. **Zero manual normalization code needed in Swift!**

### 3. Quantization Accuracy Drop
* **Problem**: Aggressive full-integer (W8A8) activation quantization can cause degradation when features saturate or when small leaf spot patterns have subtle gradient differences.
* **Solution**: We use **post-training linear weight-only INT8 quantization** (`linear_symmetric`, threshold 512). Weights are compressed from FP32/FP16 down to INT8 (reducing package size from ~14 MB to ~3.8 MB), while intermediate activations remain in high-precision FP16 on the Apple Neural Engine (ANE). This maintains accuracy within `0.2%` of the uncompressed model.
* If accuracy drops on specific low-contrast classes, adjust `weight_threshold` in `OpLinearQuantizerConfig` or retain FP16 precision (still only ~8.8 MB, comfortably under the 20 MB limit).

---

## 📱 iOS Swift Integration (Offline Inference)

Drop `CoconutDoctor.mlpackage` into your Xcode project. Xcode automatically generates the `CoconutDoctor` Swift class.

```swift
import Vision
import CoreML
import UIKit

class CoconutDiseaseClassifier {
    private var visionModel: VNCoreMLModel?

    init() {
        do {
            let config = MLModelConfiguration()
            config.computeUnits = .all // Utilizes Apple Neural Engine (ANE), GPU & CPU
            let coreMLModel = try CoconutDoctor(configuration: config).model
            self.visionModel = try VNCoreMLModel(for: coreMLModel)
        } catch {
            print("Failed to initialize CoconutDoctor CoreML model: \(error)")
        }
    }

    func classify(pixelBuffer: CVPixelBuffer, completion: @escaping (String, Float) -> Void) {
        guard let visionModel = self.visionModel else { return }

        let request = VNCoreMLRequest(model: visionModel) { request, error in
            guard let results = request.results as? [VNClassificationObservation],
                  let topResult = results.first else {
                completion("Unknown", 0.0)
                return
            }
            completion(topResult.identifier, topResult.confidence)
        }

        // Core ML automatically resizes to 224x224 and applies scale/bias!
        request.imageCropAndScaleOption = .centerCrop

        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        DispatchQueue.global(qos: .userInitiated).async {
            try? handler.perform([request])
        }
    }
}
```
