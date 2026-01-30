# 🟡 Medium-Level Neural Network Exercises — Keras Lab Work

This directory contains **medium-level exercises** in neural networks using **Keras**.  
The lab work progresses from simple models to deeper architectures, exploring normalization, regularization, convolutional layers, and optimizer comparisons.  
Additionally, this folder includes an exercise on **Intersection over Union (IoU)** for rectangles, a key metric in object detection tasks.

---

## 📘 Overview

**Objective:**  
- Understand the interface of Keras layers used in building neural networks.  
- Learn how to train and optimize neural networks effectively.  
- Explore the impact of depth, normalization, regularization, and optimizers on performance.  
- Apply evaluation metrics such as **IoU (Intersection over Union)** for bounding boxes in computer vision.  

**Dataset:**  
- Most tasks use the **MNIST handwritten digits dataset**.  
- The IoU exercise uses synthetic rectangle data to demonstrate bounding box overlap evaluation.  

**Structure of the lab (Tasks 1–7):**
1. **Single-layer network (baseline):** ~92–93% accuracy.  
2. **Two-layer network:** ~97.6% accuracy.  
3. **Three-layer network:** ~97.9% accuracy.  
4. **Normalization layers (BatchNorm, Dropout, L2):** ~98.1% accuracy, faster convergence.  
5. **Deep fully connected network:** ~98.3% accuracy.  
6. **Network with one convolutional layer:** ~98.34% accuracy, faster convergence.  
7. **Optimizer comparison:** Adadelta achieved best accuracy (~98.3%) and fastest convergence.  

---

## 📂 Files in This Directory

- **`Introduction_to_Keras_Lab.ipynb`** — Complete lab notebook with Tasks 1–7 (MNIST classification).  
- **`IoU_Rectangle.ipynb`** — Exercise on calculating Intersection over Union (IoU) for rectangles, demonstrating bounding box overlap evaluation in object detection.  

---

## 🔗 Note

This directory is part of the [Neural-Network-Exercises](https://github.com/SemyonKim/Neural-Network-Exercises) project:


---

## ✅ Key Takeaways
- Deeper networks generally improve accuracy, but gains diminish and training becomes more complex.  
- Normalization and dropout are essential for stable training and preventing overfitting.  
- Convolutional layers provide a clear advantage for image classification tasks.  
- Optimizer choice significantly affects both accuracy and convergence speed.  
- IoU is a critical metric for evaluating bounding box predictions in object detection tasks.  
