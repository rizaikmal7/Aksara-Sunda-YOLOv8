# Aksara Sunda (Sundanese Script) Character Recognition using YOLOv8

This repository implements a high-performance computer vision pipeline utilizing the **YOLOv8 Nano (YOLOv8n)** architecture to automatically detect, localize, and classify **30 distinct characters** of the traditional Sundanese script (*Aksara Sunda*).

---

## 📊 Dataset Specifications
- **Classes:** 30 distinct script characters (`'a'`, `'ae'`, `'ba'`, `'ca'`, `'da'`, `'e'`, `'eu'`, `'fa'`, `'ga'`, `'ha'`, `'i'`, `'ja'`, `'ka'`, `'la'`, `'ma'`, `'na'`, `'nga'`, `'nya'`, `'o'`, `'pa'`, `'qa'`, `'ra'`, `'sa'`, `'ta'`, `'u'`, `'va'`, `'wa'`, `'xa'`, `'ya'`, `'za'`).
- **Format:** YOLOv8 PyTorch bounding box coordinates annotations.
- **Source Workspace:** Publicly sourced and managed via Roboflow (`workspace: aksarasunda`, `project: aksara-sunda-eayhq`, version 3).

---

## 🏗️ Model Architecture & Training Workflow
- **Base Framework:** `ultralytics` (YOLOv8n - Nano variant optimized for rapid edge deployment and device compatibility).
- **Input Image Size:** 640x640 pixels (BCHW format).
- **Hardware Acceleration:** Trained using Automatic Mixed Precision (AMP) checks on a cloud-hosted Tesla T4 GPU environment.
- **Optimization Strategy:** Evaluated over 20 structured training epochs utilizing the `AdamW` optimizer.

---

## 📈 Evaluation & Performance Metrics
The trained model achieved highly precise performance metrics during validation across the entire character set:

| Metric | Evaluation Value |
| :--- | :---: |
| **Mean F1-Score** | **97.79%** |
| **Mean Precision** | **97.37%** |
| **Mean Recall** | **98.32%** |
| **mAP @ 0.5** | **98.99%** |
| **mAP @ 0.5:0.95 (IoU)** | **91.65%** |

### Execution Speed Per Image:
- **Preprocessing:** 0.3ms
- **Inference Speed:** 3.9ms
- **Post-processing:** 2.4ms

### Core Engineering Highlights
1. **Character Matrix Assessment:** Built custom evaluation modules leveraging `scikit-learn` to isolate individual character boundary confusion tables (`ConfusionMatrixDisplay`).
2. **Production Export:** Successfully slimmed and exported the final optimized weights into **ONNX (`best.onnx`)** format with an operational input shape of `(1, 3, 640, 640)` to support runtime engines.
3. **Automated Batch Processing:** Fully configured background automated zip compilation batches to quickly compress cross-validation outputs.

---

## 🚀 Environment & Dependencies
To run inference or modify the script metrics locally, ensure you have the following core frameworks installed:
- `ultralytics` (YOLOv8)
- `numpy` & `pandas`
- `matplotlib` & `seaborn`
- `scikit-learn`
- `onnx` / `onnxslim` / `onnxruntime-gpu`

---

## 📂 Repository Structure
```text
├── Aksara Sunda YOLOv8n.ipynb               # Core YOLOv8n model optimization pipeline
├── .gitignore                               # Prevents tracking caches, zip logs, and runs folder
└── README.md                                # Project documentation and performance overview
