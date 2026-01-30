# NumericalMethods

## 📘 Overview
This directory contains the notebook **`NumericalMethods.ipynb`**, which documents a series of numerical methods exercises. Each homework task demonstrates a classical algorithm or optimization technique, implemented in Python with clear visualization and interpretive notes.

This notebook is the result of an **academic course on Optimisations and Numerical Methods** from my university study. It serves as both a **learning archive** and a **reference resource**, showcasing how numerical methods can be applied to problems in optimization, regression, dimensionality reduction, and image processing.

---

## 📂 Contents

### Notebook: `NumericalMethods.ipynb`
The notebook is organized into sequential homework tasks:

| HW # | Topic | Method | Key Notes |
|------|-------|--------|-----------|
| HW1 | PCA on Wine Dataset | Principal Component Analysis | Dimensionality reduction, ROC analysis |
| HW2 | LDA on Breast Cancer Dataset | Linear Discriminant Analysis | Supervised projection, ROC comparison |
| HW3 | ICA on Synthetic Signals | Independent Component Analysis | Blind source separation |
| HW4 | Robust Circle Fitting | RANSAC | Outlier‑resistant model fitting |
| HW5 | ODE Solver | Euler Method | Accuracy vs step size |
| HW6 | Polynomial Regression | Least Squares (7th degree) | Overfitting vs generalization |
| HW7 | Dimensionality Reduction | t‑SNE, Isomap, PCA | Comparative embeddings |
| HW8 | Integration | Monte Carlo | Uniform vs importance sampling |
| HW9 | Optimization | Gradient Descent on Rosenbrock | Step size decay, convergence |
| HW10 | Image Processing | Hybrid Images (FFT filters) | Low‑pass + high‑pass combination |
| HW11 | Orbit Fitting | Polynomial Regression | Robustness to perturbations |
| HW12 | Image Reconstruction | Regression from low‑light inputs | MSE evaluation |
| HW13 | Optimization | Nelder–Mead & Gradient Descent | Numerical vs closed‑form gradients |
| HW14 | Optimization | Coordinate Descent & Conjugate Gradient | Fletcher–Reeves vs PRP |

---

## ⚙️ Requirements
- Python 3.8+  
- Libraries: `numpy`, `pandas`, `matplotlib`, `scipy`, `scikit-learn`, `seaborn`, `imageio`, `numdifftools`  

---

## 📊 Usage
Open the notebook in Jupyter or Google Colab:
  - jupyter notebook [NumericalMethods.ipynb](NumericalMethods/NumericalMethods.ipynb)

Note: Each section is self‑contained, with:
- **Markdown storytelling** explaining objectives and interpretive notes.  
- **Code cells** implementing algorithms.  
- **Plots and outputs** visualizing results.  

---

## 📌 Data Disclaimer
This project **does not contain any input data files**.  
All datasets used (e.g., Wine, Breast Cancer, Digits) are publicly available through **`scikit-learn`** or other open repositories.  
For image‑based tasks (hybrid images, low‑light reconstruction), you can substitute with any publicly available datasets or your own images.

---

## 🎯 Learning Goals
- Understand classical numerical methods.  
- Compare linear vs nonlinear dimensionality reduction.  
- Explore robust fitting and optimization techniques.  
- Apply regression and matrix factorization to image tasks.  
- Build a cohesive archive of numerical exercises with professional documentation.  

---

## 📘 Comparative Overview of Methods

### Dimensionality Reduction
| Method | Goal | Nature | Strengths | Limitations |
|--------|------|--------|-----------|-------------|
| PCA | Maximize variance | Linear, unsupervised | Simple, interpretable, fast | Sensitive to outliers, linear only |
| LDA | Maximize class separability | Linear, supervised | Good for classification | Requires labels, assumes linear boundaries |
| ICA | Maximize independence | Nonlinear, unsupervised | Recovers hidden sources | Sensitive to noise, convergence issues |
| t-SNE | Preserve local neighborhoods | Nonlinear, unsupervised | Excellent visualization | Distorts global distances |
| Isomap | Preserve global geometry | Nonlinear, unsupervised | Captures manifold structure | Sensitive to neighbor choice |

### Regression & Fitting
| Method | Goal | Nature | Strengths | Limitations |
|--------|------|--------|-----------|-------------|
| Polynomial Regression | Approximate nonlinear trends | Linear in parameters | Flexible, captures curvature | Overfitting at high degree |
| RANSAC | Robust model fitting | Iterative sampling | Resistant to outliers | May miss global optimum |
| Orbit Fitting | Approximate trajectories | Polynomial regression | Models elliptical paths | Sensitive to noise |

### Integration & ODEs
| Method | Goal | Nature | Strengths | Limitations |
|--------|------|--------|-----------|-------------|
| Euler Method | Solve ODEs | First-order | Simple, intuitive | Requires small step size |
| Monte Carlo | Approximate integrals | Probabilistic | Works in high dimensions | Slow convergence |

### Optimization
| Method | Goal | Nature | Strengths | Limitations |
|--------|------|--------|-----------|-------------|
| Gradient Descent | Minimize functions | Iterative, gradient-based | Simple, widely used | Sensitive to step size |
| Nelder–Mead | Minimize functions | Derivative-free | Robust to noise | Slow in high dimensions |
| Coordinate Descent | Minimize functions | Variable-wise updates | Simple, intuitive | Inefficient for correlated variables |
| Conjugate Gradient | Minimize quadratic functions | Iterative, matrix-based | Fast convergence | Requires SPD matrices |
| Fletcher–Reeves / PRP | Refined conjugate gradient | Gradient-based | Faster convergence | Requires line search |

### Image Processing
| Method | Goal | Nature | Strengths | Limitations |
|--------|------|--------|-----------|-------------|
| Hybrid Images | Combine low/high frequencies | Frequency-domain filtering | Exploits human perception | Sensitive to cutoff choice |
| Low-Light Reconstruction | Recover full image | Regression-based | Restores missing info | Sensitive to noise |
| SVD Compression | Reduce dimensionality | Matrix factorization | Efficient compression | Loss of fine detail |

---

## 📘 License
This project is for **educational and research purposes**.  
Datasets referenced are publicly available; please check their respective licenses before reuse.
