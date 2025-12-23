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

The task is formulated as **independent binary classification** for three hazardous driving behavior categories.  
Each category is annotated using **0/1 binary labels**, where `1` indicates the presence of a hazardous behavior.

| Category ID | Hazardous Behavior  | Label = 0 (Normal) | Label = 1 (Hazardous) |
|------------ |---------------------|--------------------|-----------------------|
| 1           | Dangerous Behavior  | Normal driving     | Dangerous driving     |
| 2           | Aggressive Behavior | Normal driving     | Aggressive driving    |
| 3           | Obstructive Behavior| Normal driving     | Obstructive driving   |

Each hazardous behavior category is detected **independently**, while sharing the same visual feature extraction pipeline.

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
video2img_2fps/
 ├── 0_110/
 │    ├── 0001.png
 │    ├── 0002.png
 │    └── ...
 └── 2_110/
```

During preprocessing, frames are **uniformly sampled to a fixed number (e.g., 20 frames)** per video.

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

Train the lightweight temporal classification head on the extracted VLM features to perform binary detection of three road-rage hazardous behavior categories.

```bash
python scripts/classifier.py
```

Test Results(Example)

| Category    | Accuracy | F1-score |
| ----------- | -------- | -------- |
| Dangerous   | 94.12%   | 0.9697   |
| Aggressive  | 88.24%   | 0.8000   |
| Obstructive | 82.35%   | 0.7692   |
| Normal      | 100.00%  | 1.0000   |

#### 🚀 Quick Start

Due to the large size of VLM-extracted features, they are not included in this repository.
If you would like to skip the VLM feature extraction stage and directly run the classification and evaluation pipeline, feel free to contact us to obtain the extracted VLM features.

Once the features are prepared,Simply run:

```bash
python main.py
```
---

This will directly load the provided features and output the classification results for the three hazardous behavior categories.

### 3️⃣ Inference

Run inference on a single video:

```bash
python scripts/run_inference.py --video_dir path/to/video_frames
```

The output includes predicted probabilities and binary labels for each hazardous behavior category.

**Note**: We provide pretrained weights for the lightweight MLP classifier, so inference can be performed directly without retraining. The VLM is used only as a fixed feature extractor during inference.

---

## 📬 Data & Resource Availability

For dataset access or additional experimental resources, please contact **roadragereasoning@163.com**.

---

## 📜 License

This project is released under the **MIT License**.

---




