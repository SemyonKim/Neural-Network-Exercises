# 📘 Project Overview

The notebook demonstrates:
- **Data Loading & Preparation**  
Handling multiple datasets (features, sales, stores) and preparing them for analysis.

- **Exploratory Data Analysis (EDA)**
  - Store type distribution (pie chart)
  - Weekly sales comparison: holiday vs. non‑holiday
  - Time series plots of temperature, fuel price, and weekly sales across stores
  - Pairplots and categorical plots with Seaborn

- **Machine Learning Models**
  - K‑Nearest Neighbors (KNN): Classifies store types based on features
  - Linear Regression: Predicts weekly sales from store attributes
  - Error diagnostics: confusion matrix, misclassification counts, train/test MSE comparison

- **Key Findings**
  - Store type strongly influences sales distribution
  - Holidays impact weekly sales averages
  - Simple ML models can capture store‑type classification with high accuracy

---

# 📂 Repository Structure

```
Neural-Network-Exercises/  
└── DigitalMarketing/  
    └── Sales_Stores/  
        ├── DM_Marketing_Analysis.ipynb  
        └── README.md   ← (this file)
```

---

# ⚙️ Requirements
This notebook is intended as a **practice archive**.  
Datasets are **not included**; file paths are placeholders (`data/Features.csv`, `data/Sales.csv`, `data/Stores.csv`).  

Dependencies:
* Python 3.x
* numpy
* pandas
* matplotlib
* seaborn
* scikit‑learn  

Install with:
```
pip install numpy pandas matplotlib seaborn scikit-learn
```

---

# 🚀 Usage
Clone the repository and navigate to the project folder:
```
git clone https://github.com/SemyonKim/Neural-Network-Exercises.git
cd Neural-Network-Exercises/DigitalMarketing/Sales_Stores
```

Open the notebook:
```
jupyter notebook DM_Marketing_Analysis.ipynb
```

---

# 📝 Notes
* This notebook is **archived for educational purposes** and does not include original datasets.
* All file paths are placeholders; replace them with actual dataset locations if you wish to rerun the analysis.
* Markdown explanations and inline comments have been added for clarity.
* The notebook demonstrates **workflow structure and analysis methodology**, not production‑ready modeling.

---

## License
This project is released under the MIT License. See [LICENSE](https://github.com/SemyonKim/Neural-Network-Exercises/blob/main/LICENSE) for details.
