# S4 Neural Network — Audio Classification

Implementation of the **S4 (Structured State Space Sequence Model)** from scratch in PyTorch for spoken command recognition using the Google Speech Commands dataset.

---

## 📋 Overview

This project trains an S4 model to classify 10 spoken words:
`yes, no, up, down, left, right, on, off, stop, go`

S4 is a modern sequence model that outperforms RNN and Transformer architectures on long sequences thanks to its efficient FFT-based convolution.

---

## 🏗️ Model Architecture
Raw audio input (B, 16000)
↓
Linear Encoder → (B, 16000, d_model=64)
↓
S4Layer + FFN + LayerNorm + Residual  × N_LAYERS
↓
Global Average Pooling → (B, d_model)
↓
Classifier → Output logits (B, 10)

---

## 📦 Dataset

- **Google Speech Commands v0.01**
- ~65,000 audio files of 1 second each
- Mono WAV, 16kHz
- 10 classes used out of 30 available

---

## ⚙️ Requirements

```bash
pip install torch torchaudio datasets soundfile librosa matplotlib seaborn scikit-learn
apt-get install libsndfile1 ffmpeg
```

---

## 🚀 Usage

Open the notebook in Google Colab:

1. Go to **Runtime → Change runtime type → GPU (T4)**
2. Run all cells from top to bottom
3. Training takes ~20–30 minutes on a T4 GPU

---

## 📐 Why S4?

| Model       | Complexity     | Memory   | Long sequences        |
|-------------|----------------|----------|-----------------------|
| RNN/LSTM    | O(L)           | O(1)     | ❌ Gradient vanishing |
| Transformer | O(L²)          | O(L²)    | ❌ Too expensive      |
| **S4**      | **O(L log L)** | **O(L)** | **✅ Excellent**      |

S4 reformulates the state space model as a discrete convolution computed via FFT, making it highly efficient for long audio sequences (16,000 time steps per second).

---

## 📁 Notebook Structure

1. Install dependencies
2. Theory: State Space Models & S4
3. S4Layer implementation (core of the model)
4. Full S4Model architecture
5. Dataset loading and preprocessing
6. Training
7. Evaluation & results visualization
8. Inference on a custom audio file
9. Comparison: S4 vs RNN vs Transformer

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Training Accuracy | ~85–90% |
| Validation Accuracy | ~80–85% |
| Test Accuracy | ~80–85% |

---

## 📜 License

MIT License
