# 🧠 Neural Network Exercises (University Coursework)

![Repo Size](https://img.shields.io/github/repo-size/SemyonKim/Neural-Network-Exercises?style=for-the-badge&logo=github)
![Status](https://img.shields.io/badge/Status-lightgrey?style=for-the-badge&logo=python)
![Academic Coursework](https://img.shields.io/badge/Academic%20Coursework-blue?style=for-the-badge)
![Archived](https://img.shields.io/badge/Archived-red?style=for-the-badge)
![Language: Python](https://img.shields.io/badge/Language-Python-blue?style=for-the-badge&logo=python)
![Platform: Jupyter/Colab](https://img.shields.io/badge/Platform-Jupyter%20%2F%20Colab-orange?style=for-the-badge&logo=googlecolab)




## 📖 Overview
This repository is an **archived collection** of selected works from my university coursework in neural networks, implemented in **Python notebooks**.  
Exercises range from **basic concepts** (perceptron, activation functions) to **medium-level implementations** (multi-layer networks, backpropagation, convolutional examples), and extend into **advanced cooperative learning experiments** inspired by [NeurIPS 2018](https://arxiv.org/pdf/1806.04606) research.  

The archive includes: 
- Introductory and intermediate coursework exercises.
- A complete Python & Machine Learning course archive (SHARE program).
- A synthetic license plate generation and recognition project.
- **Knowledge Distillation with ONE (On-the-Fly Native Ensemble)** — my own experiments based on Xu et al. (NeurIPS 2018) [PDF](https://arxiv.org/pdf/1806.04606) and [github](https://github.com/Lan1991Xu/ONE_NeurIPS2018), refactored and extended into multi-branch cooperative neural networks.
- A **Digital Marketing block** combining classic and neural network approaches for clustering, classification, and regression tasks.

---

## 📂 Structure
- **basic/** → introductory exercises (perceptron, activation functions, gradient descent).  
- **medium/** → intermediate topics (multi-layer networks, backpropagation, convolutional examples).  
- **visualization/** → training curves, accuracy plots.  
- **utils/** → helper functions for reuse.
- **PythonMLCourse/** → complete archive of the [SHARE](https://sharemsu.ru/) program’s Python & Machine Learning course.
    - **Part1/** → 8 seminars + 2 competitions covering fundamentals: Python basics, data preprocessing, linear/logistic regression, decision trees, and early ML challenges.
    - **Part2/** → 8 seminars (including 1 competition) focused on applied ML: feature engineering, ensemble methods, PyTorch models, image classification with augmentation, and advanced regression tasks.
    - **[PythonMLCourse.ipynb](PythonMLCourse/PythonMLCourse.ipynb)** → consolidated notebook combining all seminars and competitions into one file for quick review.
- **OptimisationNN/** → Implementation of the project on synthetic license plate generation and recognition.
    - **[generator.py](OptimisationNN/Plate_Generator/generator.py)** → synthetic license plate generator.
    - **[augmenter.py](OptimisationNN/Plate_Generator/augmenter.py)** → image distortion functions (rotation, smudging, noise).
    - **[train.py](OptimisationNN/CNN_model/train.py)** → CNN training pipeline with custom NormLayer.
    - **[test.py](OptimisationNN/CNN_model/test.py)** → evaluation and prediction script.
- **ONE_KnowledgeDistillation/** → experiments inspired by [Xu et al. (NeurIPS 2018).](https://arxiv.org/pdf/1806.04606)  
    - **TwoStageCooperativeNN/Step1/** → README and code for Step1 experiments (5 subgroups: baseline, 3‑branch, 5‑branch, voting, transfer 3→5).
    - **TwoStageCooperativeNN/Step2/** → README and code for Step2 experiments (4 subgroups: merge baseline+branch models, merge three one‑branch models, expand single branch into 3, single branch inside cooperative model).
        - **utils/** → helper functions (logging, accuracy, branch replacement, etc.).
        - **CooperativeNeuralNetworks.ipynb** → Colab notebook where Part1 and Part2 experiments are executed.
- **DigitalMarketing/** → completed Digital Marketing course notebooks:
  - **Sales_Stores/** → store performance and sales prediction.
  - **TimeSeries_PanelData/** → time series and panel data modeling.
  - **Analysis_Combined/** → unified notebook combining clustering, classification, and regression (classic + neural).

---

## 📂 Contents

### 🟢 Basic Level
- **[digit_recognition_constant.ipynb](basic/digit_recognition_constant.ipynb)**  
  Trivial baseline classifier that always predicts the same digit.  
  *Problem solved:* Demonstrates dataset loading and evaluation pipeline, even though accuracy is ~10%.

- **[digit_recognition_quadrant.ipynb](basic/digit_recognition_quadrant.ipynb)**  
  Rule-based classifier dividing digit images into quadrants and comparing pixel counts.  
  *Problem solved:* Shows feature engineering and manual classification logic, achieving ~29% accuracy.

### 🟡 Medium Level
- **[iou_and_polygon_area.ipynb](medium/iou_and_polygon_area.ipynb)**  
  Demonstrates geometric computations: Intersection over Union (IoU) between rectangles and polygon area calculation using Shapely.
*Problem solved:* Introduces bounding box overlap metrics (IoU) widely used in computer vision, and shows how to compute polygon areas programmatically.

### 📊 Visualization
- *(to be added: training curves, accuracy plots)*

### 🛠️ Utilities
- *(to be added: helper functions for reuse)*

### 📘 PythonMLCourse ([SHARE](https://sharemsu.ru/) Program)
- **Part1/** → Fundamentals + competitions (linear/logistic regression, decision trees, preprocessing, Kaggle-style challenges).  
- **Part2/** → Applied ML & advanced topics (feature engineering, ensembles, PyTorch CNNs, image classification, regression).  
- **[PythonMLCourse.ipynb](PythonMLCourse/PythonMLCourse.ipynb)** → consolidated notebook merging all seminars and competitions.
 
### 🎓 Optimisation NN Project
- **[generator.py](OptimisationNN/Plate_Generator/generator.py)** → synthetic license plate generator.  
- **[augmenter.py](OptimisationNN/Plate_Generator/augmenter.py)**   → image distortion functions (rotation, smudging, noise).  
- **[train.py](OptimisationNN/CNN_model/train.py)** → CNN training pipeline with custom NormLayer.  
- **[test.py](OptimisationNN/CNN_model/test.py)**  → evaluation and prediction script.

### 🔬 Knowledge Distillation (ONE)
- Experiments inspired by [Xu et al. (NeurIPS 2018).](https://arxiv.org/pdf/1806.04606)  
- Based on the original [Lan1991Xu/ONE_NeurIPS2018](https://github.com/Lan1991Xu/ONE_NeurIPS2018) repo, with my own modifications and parameter explorations.  
- **Step1:** 5 subgroups (baseline, 3‑branch, 5‑branch, voting, transfer 3→5).  
- **Step2:** 4 subgroups (merge baseline+branch models, merge three one‑branch models, expand single branch into 3, single branch inside cooperative model).  
> ⚠️ **Note:** No pretrained weights included — viewers must train their own checkpoints.

### 📈 Digital Marketing Block 
- **Sales_Stores/** → sales and store performance analysis (classification, regression).
- **TimeSeries_PanelData/** → combined solutions for time series and panel data (ARIMA, LSTM, RNN, PooledOLS).
- **Analysis_Combined/** → unified notebook integrating clustering, classification, and regression with both classical and neural methods.
- *Highlights:* Balanced integration of classical and neural approaches, reproducible workflows, professional documentation. 

---

## ⚙️ Requirements
- Python 3.x
- Jupyter Notebook or Google Colaboratory
- Libraries:
    - Core: `numpy`, `pandas`, `matplotlib`, `seaborn`
    - ML/NN: `scikit-learn`, `statsmodels`, `tensorflow`, `torch`, `torchvision`
    - SOM libraries: `SimpSOM`, `MiniSom`

> ⚠️ **Note**: These notebooks were created during university coursework. Dependencies may vary between files, and not all notebooks have been tested recently.

---

## 🚀 Usage
- Each .ipynb notebook demonstrates a specific algorithm or neural network concept.
- You are welcome to open the notebooks in **Jupyter** or **Google Colab** to explore the code and visualizations.
- Since this is an archive, some notebooks may require adjustments to run correctly in modern environments.

---

## 📌 Notes
- This project is an **academic exercise archive**.
- It is **not actively maintained**, but preserved for reference and learning.

---

## 🌍 Author
- 👤 Semyon Kim
- 📍 Uzbekistan
- 🗣️ Languages: Russian (native), English (intermediate), Korean (elementary)
- 🔗 [GitHub](https://github.com/SemyonKim)

---

## 📜 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
