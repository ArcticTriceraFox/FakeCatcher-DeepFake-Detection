# FakeCatcher: Automated Deepfake Video Detection Framework

> B.Tech Final Year Project — KIIT Deemed to be University, 2026  
> Guide: Dr. Partha Sarthi Paul

A hybrid spatio-temporal deep learning system for automated deepfake video detection. The model combines **EfficientNet-Lite0** (spatial feature extraction) with an **LSTM** network (temporal modelling) to identify both visual artefacts and frame-to-frame inconsistencies in deepfake videos. Trained and evaluated on the [Celeb-DF v2](https://www.kaggle.com/datasets/reubensuju/celeb-df-v2) benchmark dataset.

---

## Motivation

The rapid advancement of Generative Adversarial Networks (GANs) has made it increasingly difficult to distinguish synthetic videos from authentic ones using the human eye alone. Deepfakes pose serious risks in the form of misinformation, identity fraud, and digital manipulation — particularly in journalism, politics, and security domains.

Traditional frame-level detection methods often fail on modern deepfakes due to their high visual realism. However, subtle **temporal inconsistencies** persist across consecutive frames — unnatural blinking, irregular facial movements, and frame-to-frame lighting variations. This project addresses detection by analysing **both spatial and temporal features** jointly.

---

## Results

| Metric | Score |
|--------|-------|
| Test Accuracy | **96.31%** |
| ROC-AUC | **0.9850** |
| PR-AUC | **0.9920** |
| Best Validation Accuracy | **96.9%** |

**Per-class performance:**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Real  | 95.2%     | 92.4%  | 93.7%    | 170     |
| Fake  | 96.8%     | 98.0%  | 97.4%    | 399     |

**Confusion Matrix (Test Set — 569 videos):**

|                | Predicted Real | Predicted Fake |
|----------------|---------------|----------------|
| **True Real**  | 157           | 13             |
| **True Fake**  | 8             | 391            |

Training converged at epoch 10 (best validation loss), with early stopping triggered at epoch 15.

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

**Why this architecture?**
- **EfficientNet-Lite0** — chosen for its balance between accuracy and computational efficiency; extracts high-level spatial features and subtle visual artefacts per frame
- **LSTM** — captures temporal dependencies across the frame sequence, detecting motion-related inconsistencies that single-frame methods miss
- **Hybrid approach** — outperforms pure spatial or pure temporal methods by jointly modelling both dimensions

---

## Tech Stack

- **ML:** PyTorch, EfficientNet-Lite0 (`timm`), LSTM
- **Preprocessing:** OpenCV (Haar Cascade), NumPy
- **Training:** Google Colab (T4 GPU) / Kaggle (T4 x2)
- **Optimiser:** AdamW + CosineAnnealing LR scheduler
- **Training tricks:** Mixed precision (AMP), early stopping (patience=5), data augmentation

---

## Dataset

[Celeb-DF v2](https://www.kaggle.com/datasets/reubensuju/celeb-df-v2) — a large-scale benchmark of high-quality real and synthetic celebrity videos, known for its realism and diversity.

- **890** real videos (Celeb-real + YouTube-real merged)
- **5,639** fake videos (Celeb-synthesis)
- Class distribution: ~70.3% fake / ~29.7% real
- Split: **70% train / 20% val / 10% test**

> Download the dataset from Kaggle before running the notebooks.

---

## Data Augmentation

Applied during training to improve generalisation:
- Random horizontal flip
- Random rotation (±10°)
- Colour jitter (brightness, contrast, saturation)
- Random grayscale (p=0.05)

---

## Notebooks

| Notebook | Environment | Description |
|----------|-------------|-------------|
| `FakeCatcher_Kaggle_Pipeline.ipynb` | Kaggle | Full executed pipeline — training logs, metrics, and visualisations included |
| `FakeCatcher_Colab_Pipeline.ipynb` | Google Colab | Full pipeline for Colab + Google Drive setup |

Both cover the complete end-to-end pipeline:
1. Dataset preparation & face extraction
2. Data splitting (70 / 20 / 10)
3. Model training with early stopping & AMP
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

## Future Scope

- **Advanced face detection** — Replace Haar Cascade with MTCNN or RetinaFace for better accuracy under challenging conditions
- **Transformer-based models** — Incorporate attention mechanisms to capture longer-range temporal dependencies
- **Frequency domain analysis** — Detect artefacts invisible in the spatial domain using FFT-based features
- **Real-time inference** — Optimise for deployment in surveillance systems and social media platforms
- **Cross-dataset generalisation** — Train across multiple datasets (FaceForensics++, DFDC) for broader robustness
- **Explainability** — Integrate Grad-CAM to provide visual explanations for model predictions

---

## References

1. Y. Li et al., *Celeb-DF: A Large-Scale Challenging Dataset for DeepFake Forensics*, CVPR 2020
2. M. Tan and Q. V. Le, *EfficientNet: Rethinking Model Scaling for CNNs*, arXiv:1905.11946, 2019
3. S. Hochreiter and J. Schmidhuber, *Long Short-Term Memory*, Neural Computation, 1997
4. A. Paszke et al., *PyTorch: An Imperative Style, High-Performance Deep Learning Library*, arXiv:1912.01703, 2019
5. G. Bradski, *The OpenCV Library*, Dr. Dobb's Journal, 2000

---

## Authors

- **Shriya Chaubey** (22052334) — Data preprocessing, model training
- **Vaibhav Sharma** (22053123) — System design, evaluation, inference pipeline

*B.Tech CSE, KIIT Deemed to be University, 2026*  
*Guide: Dr. Partha Sarthi Paul*
