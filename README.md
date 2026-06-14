# FakeCatcher: Automated Deepfake Video Detection Framework

A hybrid deep learning system that detects deepfake videos by combining **EfficientNet-Lite0** for spatial feature extraction with an **LSTM** network for temporal modelling. Trained and evaluated on the [Celeb-DF v2](https://www.kaggle.com/datasets/reubensuju/celeb-df-v2) dataset.

---

## Results

| Metric | Score |
|--------|-------|
| Test Accuracy | **96.31%** |
| ROC-AUC | **0.9850** |
| PR-AUC | **0.9920** |

**Per-class performance:**

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Real  | 95.2%     | 92.4%  | 93.7%    |
| Fake  | 96.8%     | 98.0%  | 97.4%    |

---

## Architecture

```
Input Video
    │
    ▼
Frame Extraction (30 evenly-spaced frames)
    │
    ▼
Face Detection & Cropping (Haar Cascade)
    │
    ▼
Preprocessing (resize to 224×224, normalize, augment)
    │
    ▼
EfficientNet-Lite0  ──►  1280-d feature vector (per frame)
    │
    ▼
Dropout (p=0.4)
    │
    ▼
LSTM (2 layers, hidden=256)
    │
    ▼
FC Layers (256 → 128 → 2)
    │
    ▼
Output: REAL / FAKE + confidence score
```

---

## Tech Stack

- **ML:** PyTorch, EfficientNet-Lite0 (`timm`), LSTM
- **Preprocessing:** OpenCV (Haar Cascade), NumPy
- **Training:** Google Colab (T4 GPU) / Kaggle (T4 x2)
- **Dataset:** Celeb-DF v2

---

## Dataset

[Celeb-DF v2](https://www.kaggle.com/datasets/reubensuju/celeb-df-v2) — a large-scale benchmark dataset of high-quality real and synthetic videos.

- 890 real videos + 5,639 fake videos
- Split: 70% train / 20% val / 10% test

> Download the dataset from Kaggle before running the notebooks.

---

## Notebooks

| Notebook | Environment | Description |
|----------|-------------|-------------|
| `FakeCatcher_Kaggle_Pipeline.ipynb` | Kaggle | Full executed pipeline — training logs, metrics, and visualisations included |
| `FakeCatcher_Colab_Pipeline.ipynb` | Google Colab | Full pipeline for Colab + Google Drive setup |

Both cover the complete end-to-end pipeline:
1. Dataset preparation & face extraction
2. Data splitting (70 / 20 / 10)
3. Model training with early stopping & mixed precision (AMP)
4. Evaluation — confusion matrix, ROC curve, PR curve, per-class metrics
5. Inference on new videos

---

## How to Run

### Kaggle
1. Open `FakeCatcher_Kaggle_Pipeline.ipynb` on Kaggle
2. Add `reubensuju/celeb-df-v2` via *Add Data*
3. Enable T4 GPU via *Settings → Accelerator*
4. Run all cells

### Google Colab
1. Open `FakeCatcher_Colab_Pipeline.ipynb` in Google Colab
2. Place `Celeb-DF-v2.zip` in your Drive at `MyDrive/Celeb-DF-v2.zip`
3. Enable T4 GPU via *Runtime → Change runtime type*
4. Run all cells

---

## Authors

- **Shriya Chaubey** (22052334) — Data preprocessing, model training
- **Vaibhav Sharma** (22053123) — System design, evaluation, inference pipeline

*B.Tech CSE, KIIT Deemed to be University, 2026*  
*Guide: Dr. Partha Sarthi Paul*
