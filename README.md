# Road Rage Reasoning

Binary detection of three hazardous behavior categories is performed using visual-only features extracted from a vision–language model (VLM), together with a lightweight classification layer.

---

## 🧠 Method Overview

We adopt a **two-stage pipeline**:

1. **VLM Feature Extraction**

   * Video frames are uniformly sampled from each video.
   * A pretrained **InternVL image encoder** is used as a frozen visual backbone.
   * Only visual (single-modality) features are extracted, without language input or finetuning.

2. **Lightweight Classification**

   * Extracted frame-level features are aggregated along the temporal dimension.
   * A lightweight **Temporal Conv + MLP** classification head is trained on top of the frozen VLM features.
   * Binary classification is performed independently for each hazardous behavior category.


---

## 🎯 Task Definition

The task is formulated as **binary classification** for three hazardous behavior categories:

* Category 1 (Dangerous Behavior): normal vs. dangerous
* Category 2 (Aggressive Behavior): normal vs. Aggressive
* Category 3 (Obstructive Behavior): normal vs. Obstructive

Each category is detected independently using the same feature extraction pipeline.

---

## ⚙️ Environment Setup

### Requirements

* Python = 3.10
* PyTorch = 2.4
* CUDA = 12.4 

### Installation

```bash
pip install -r requirements.txt
```

---

### Model Weights

Pretrained InternVL weights are not included due to license restrictions.
Please download the official checkpoint from:
https://huggingface.co/OpenGVLab/InternVL/blob/main/internvl_c_13b_224px.pth

---
## 📁 Data Preparation

Each video should be converted into a folder of RGB frames:

```text
dataset/
 ├── 0_110/
 │    ├── 0001.jpg
 │    ├── 0002.jpg
 │    └── ...
 └── 2_110/
```

During preprocessing, frames are **uniformly sampled to a fixed number (e.g., 20 frames)** per video.

---

## 🚀 Quick Start

We provide a **pre-extracted VLM feature file**, allowing you to **skip the VLM feature extraction stage** and directly run the evaluation pipeline.

### One-Click Execution

Simply run:

```bash
python main.py
```

### Test Results

| Category    | Accuracy | F1-score |
| ----------- | -------- | -------- |
| Dangerous   | 100.00%  | 1.0000   |
| Aggressive  | 82.35%   | 0.6667   |
| Obstructive | 88.24%   | 0.8333   |
| Normal      | 100.00%  | 1.0000   |

---

## 🧩Functional Modules

### 1️⃣ Feature Extraction

Extract visual features using the pretrained InternVL image encoder:

```bash
python scripts/extract_features.py \
    --video_dir path/to/video_frames \
    --save_dir features/
```

The extracted features are stored as `.pt` files and reused for classifier training and inference.

---

### 2️⃣ Classification

Train the lightweight temporal classification head on frozen VLM features:

```bash
python scripts/classifier.py
```

Only the lightweight classifier parameters are updated during training.

---

### 3️⃣ Inference

Run inference on a single video:

```bash
python scripts/run_inference.py --video_dir path/to/video_frames
```

The output includes predicted probabilities and binary labels for each hazardous behavior category.

---

## 📜 License

This project is released under the **MIT License**.

---



