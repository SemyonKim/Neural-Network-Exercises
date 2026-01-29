# Analysis_Combined

## Overview
The `Analysis_Combined.ipynb` notebook consolidates multiple analytical workflows for **Digital Marketing data** into a single, structured environment. It demonstrates how classical machine learning methods and neural network approaches can be applied to marketing datasets, providing a unified view of clustering, classification, and regression tasks. The notebook is designed as both a technical exercise and a professional reference, highlighting reproducible workflows, clear documentation, and interpretive insights.

---

## Objectives
- Integrate **Classic methods** (Clustering, Classification, Regression) and **Neural Network approaches** (SOM/miniSOM, MLPClassifier, MLPRegressor) into one cohesive notebook.
- Provide a **benchmark archive** of machine learning exercises applied to digital marketing data.
- Demonstrate **best practices** in preprocessing, feature engineering, model training, evaluation, and visualization.
- Offer **interpretive notes** and structured markdown storytelling for clarity and usability.

---

## Structure
The notebook is organized into two major blocks:

### 1. Classic Methods
- **Clustering**: K‑Means, hierarchical clustering, manual grouping, and 3D visualization.
- **Classification**: Correlation analysis, Random Forest, Decision Tree, Logistic Regression, and model comparison.
- **Regression**: Linear Regression, model comparison (KNN, SVR), Poisson Regression, Negative Binomial Regression.

### 2. Neural Network Approaches
- **Clustering with SOM/miniSOM**: Self‑Organizing Maps for unsupervised grouping and visualization.
- **Classification with MLPClassifier**: Multi‑layer perceptron for categorizing institutions into performance levels.
- **Regression with MLPRegressor**: Neural network regression for predicting continuous scores.

---

## Key Features
- **Unified workflow**: Combines classical and neural approaches in one notebook.
- **Reproducibility**: Each block includes preprocessing, training, evaluation, and visualization steps.
- **Professional documentation**: Markdown storytelling, inline comments, and interpretive notes.
- **Balanced summaries**: Each block begins with a concise description of its purpose and scope.
- **Evaluation metrics**: Accuracy, confusion matrices, R² scores, residual plots, and cluster visualizations.

---

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/SemyonKim/Neural-Network-Exercises.git
   ```
2. Navigate to the notebook directory:
    ```bash
   cd Neural-Network-Exercises/DigitalMarketing/Analysis_Combined
   ```
3. Open the notebook:
   ```bash
   jupyter notebook Analysis_Combined.ipynb
   ```
4. Run cells sequentially to reproduce results.

--- 

## Dependencies
- Python 3.x
- Core libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`
- Scikit‑learn: `sklearn.preprocessing`, `sklearn.model_selection`, `sklearn.linear_model`, `sklearn.neural_network`
- Statsmodels: `statsmodels.api`, `statsmodels.formula.api`
- SOM libraries: `SimpSOM`, `MiniSom`

Install missing packages with:
  ```bash
   pip install -r requirements.txt
   ```

---

## Interpretive Notes
- **Classic block**: Provides a foundation with interpretable models and statistical regression.
- **Neural block**: Extends analysis with nonlinear, flexible models capable of capturing complex patterns.
- **Combined perspective**: Enables comparison of traditional and modern approaches on the same dataset, highlighting strengths and limitations.

---

## Contribution
This notebook is part of the **Neural Network Exercises** archive curated by [SemyonKim](https://github.com/SemyonKim). Contributions, refinements, and extensions are welcome via pull requests.

---

## License
This project is released under the MIT License. See [LICENSE](https://github.com/SemyonKim/Neural-Network-Exercises/blob/main/LICENSE) for details.

