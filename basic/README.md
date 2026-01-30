# 🟢 Basic Exercises — Digit Recognition (Pre‑Neural Network Approach)

## 📘 Overview
This folder contains two related notebooks from my university coursework, where I approached the **digit recognition problem** before diving into neural networks.

The task was to start with a **dummy classifier** (constant prediction, ~10% accuracy) and then achieve an increase in accuracy using any available methods. At the time, I had not yet studied neural networks, so my solution became a **behavioral check approach** — rule‑based classification based on pixel distribution rather than learned weights.

---

## 📂 Contents

| File | Description | Accuracy |
|------|-------------|----------|
| **digit_recognition_constant.ipynb** | Trivial baseline classifier that always predicts the same digit. Demonstrates dataset loading and evaluation pipeline. | ~10% |
| **digit_recognition_quadrant.ipynb** | Rule‑based classifier dividing digit images into quadrants and comparing pixel counts. Behavioral check approach to improve accuracy. | ~29% |

---

## 🎯 Learning Goal
- Understand dataset loading and evaluation pipeline.  
- Explore **non‑neural approaches** to classification.  
- Show how even simple rule‑based heuristics can improve accuracy over a constant baseline.  
- Provide a stepping stone toward later **neural network implementations**.

---

## ⚙️ Requirements
- Python 3.x  
- Jupyter Notebook or Google Colaboratory  
- Libraries:  
  - `numpy`  
  - `pandas`  
  - `matplotlib`  
  - `scikit-learn`  

---

## 📌 Notes
- These notebooks are **archived coursework exercises**.  
- They are **not actively maintained**, but preserved for reference and learning.  
> ⚠️ **Data Disclaimer:** No input data files are included in this folder. The digit dataset is publicly available via `scikit-learn` (`load_digits()`).

---

## 🌍 Author
- 👤 Semyon Kim  
- 📍 Uzbekistan  
- 🔗 [GitHub](https://github.com/SemyonKim)
