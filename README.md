# 🧠 Neural Network Exercises (University Coursework)

![Repo Size](https://img.shields.io/github/repo-size/SemyonKim/Neural-Network-Exercises?style=for-the-badge&logo=github)
![Status](https://img.shields.io/badge/Status-lightgrey?style=for-the-badge&logo=python)
![Academic Coursework](https://img.shields.io/badge/Academic%20Coursework-blue?style=for-the-badge)
![Archived](https://img.shields.io/badge/Archived-red?style=for-the-badge)
![Language: Python](https://img.shields.io/badge/Language-Python-blue?style=for-the-badge&logo=python)
![Platform: Jupyter/Colab](https://img.shields.io/badge/Platform-Jupyter%20%2F%20Colab-orange?style=for-the-badge&logo=googlecolab)




## 📖 Overview
This repository is an **archived collection** of selected works from my university coursework in neural networks, implemented in **Python notebooks**. I uploaded works as it is without refactoring the solution code. Exercises range from **basic concepts** (perceptron, activation functions) to **medium-level implementations** (multi-layer networks, backpropagation, convolutional examples).  

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
- *(to be added: multi-layer networks, backpropagation, convolutional examples)*

### 📊 Visualization
- *(to be added: training curves, accuracy plots)*

### 🛠️ Utilities
- *(to be added: helper functions for reuse)*

### 📘 PythonMLCourse ([SHARE](https://sharemsu.ru/) Program)

#### Combined Notebook
- **[PythonMLCourse.ipynb](PythonMLCourse/PythonMLCourse.ipynb)**
  Consolidated notebook merging all seminars and competitions for quick review.

#### Part1 — Fundamentals + Competitions
- **Seminar01**
    - Problem: Load and explore tabular dataset.
    - Solution: Applied pandas for preprocessing and simple statistics.

- **Seminar02**
    - Problem: Implement Python functions for data manipulation.
    - Solution: Wrote reusable functions and tested on sample inputs.

- **Seminar03**
    - Problem: Linear regression on numeric dataset.
    - Solution: Used scikit-learn LinearRegression and plotted predictions.

- **Seminar04**
    - Problem: Binary classification task.
    - Solution: Trained logistic regression and evaluated accuracy.

- **Seminar05**
    - Problem: Handle missing values and categorical features.
    - Solution: Applied imputation and one-hot encoding.

- **Seminar06**
    - Problem: Decision tree modeling.
    - Solution: Built DecisionTreeClassifier and visualized splits.

- **Competition01**
    - Problem: Predict categorical labels in a Kaggle-style challenge.
    - Solution: Combined preprocessing + logistic regression baseline.

- **Competition02**
    - Problem: Regression competition with larger dataset.
    - Solution: Used ensemble methods and tuned hyperparameters.

- **Seminar07–08**
    - Problem: More advanced ML tasks (feature engineering, evaluation).
    - Solution: Applied RandomForest and gradient boosting models.

#### Part2 — Applied ML & Advanced Topics

- **Seminar01–06**
    - Problem: Progressively harder ML exercises (classification, regression, visualization).
    - Solution: Implemented models with scikit-learn and PyTorch, added plots for evaluation.

- **Seminar07 (Competition)**
    - Problem: Image classification competition with augmentation.
    - Solution: Built CNN in PyTorch, added augmentation, tuned epochs.

- **Seminar08**
    - Problem: Regression with feature engineering.
    - Solution: Applied ensemble methods and optimized performance.
 
### 🎓 Optimisation NN Project
- **[generator.py](OptimisationNN/Plate_Generator/generator.py)**  
  Generates synthetic license plate images using PIL and OpenCV.  
  Problem solved: Creates realistic training data for end-to-end recognition.
- **[augmenter.py](OptimisationNN/Plate_Generator/augmenter.py)**  
  Implements image transformations: rotation, smudging, Gaussian blur, and noise injection.  
  Problem solved: Simulates real-world distortions to improve model robustness.
- **[train.py](OptimisationNN/CNN_model/train.py)**  
  Defines and trains a CNN with custom normalization layers (NormLayer).  
  Problem solved: Learns to recognize 8-character license plates from synthetic data.
- **[test.py](OptimisationNN/CNN_model/test.py)**  
  Loads the trained model, runs predictions, and compares outputs with true labels.  
  Problem solved: Evaluates recognition accuracy and decodes predicted license plates.

---

## ⚙️ Requirements
- Python 3.x  
- Jupyter Notebook or Google Colaboratory  
- Libraries:
   - Core: `numpy`, `matplotlib`
   - ML/NN: `scikit-learn`, `tensorflow`

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
- Future updates may include documentation of my **Bachelor/Master thesis experiments**.

---

## 🌍 Author
- 👤 Semyon Kim
- 📍 Uzbekistan
- 🗣️ Languages: Russian (native), English (intermediate), Korean (elementary)
- 🔗 [GitHub](https://github.com/SemyonKim)

---

## 📜 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
