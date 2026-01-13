# Conditional Flow VAE (cFVAE) for Network Performance Evaluation

This repository contains the implementation and experimental evaluation of the **Conditional Flow Variational Autoencoder (cFVAE)**, a conditional generative model for uncertainty-aware network performance prediction.
The model combines **conditional VAEs** with **normalizing flows** to learn the full conditional distribution of per-path delay given the network state.

---

## 📁 Repository Structure

```
.
├── scripts/
│   ├── Training and evaluation scripts
│   └── Utility scripts for data processing and inference
│
├── notebooks/
│   └── Jupyter notebooks for reproducing experiments,
│       generating plots, and printing quantitative results
│
├── data/
│   ├── Datasets
│   ├── Model outputs
│   └── Ground-truth labels
│
├── models/
│   └── Pretrained cFVAE models used in the experiments
│
└── README.md
```

---

## 🚀 Getting Started

### Requirements

The code is written in Python and relies on standard deep learning and scientific computing libraries (e.g., PyTorch, NumPy, Matplotlib).
Exact requirements will be documented soon.

---

## 🧠 Training a Model from Scratch

**TBD**

Instructions for training the cFVAE model from scratch—including data preparation, configuration, and execution—will be added in a future update.

---

## 📊 Experiments & Results

All experiments, figures, and printed results reported in the paper can be reproduced using the notebooks in the `notebooks/` directory.
Pretrained models corresponding to these experiments are provided in `models/`.

