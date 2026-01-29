# Time Series & Panel Data Analysis (Academic Exercise)
This project demonstrates forecasting and econometric modeling using both **time series methods** and **panel data approaches**. It combines classical statistical techniques with machine learning models to provide a comprehensive practice archive.

---

## 📂 Project Structure
```
DigitalMarketing/
└── TimeSeries_PanelData/
    ├── TimeSeries_PanelData.ipynb   # Main notebook
    ├── data/                        # Input datasets (Alcohol Sales, Stores, Features, Sales)
    └── README.md                    # Documentation
```

---

## 📑 Contents of the Notebook
- **Introduction**
  - Overview of time series forecasting and panel data econometrics.
  - Motivation for combining statistical and neural network approaches.
- **Time Series Analysis**
  - **ARIMA Model:**
    - Stationarity checks (ADF test, differencing).
    - Autocorrelation diagnostics (ACF/PACF).
    - Walk‑forward validation and RMSE evaluation.
    - Grid search for optimal parameters.
  - **LSTM Model:**
    - Data transformation (differencing, supervised framing, scaling).
    - Stateful LSTM training.
    - Forecasting with walk‑forward validation.
    - RMSE evaluation and visualization.
- **Panel Data Models**
  - **Pooled OLS, Fixed Effects, Random Effects:**
    - Estimation of store‑level weekly sales.
    - Diagnostic tests (F‑test, LM test, Hausman test).
    - Robust covariance estimation.
  - **Airline Example Dataset:**
    - Demonstration of pooled, between, first differences, fixed, and random effects estimators.
    - Robust covariance and alternative Hausman test.
  - **RNN for Panel Data:**
    - Normalization and supervised framing.
    - LSTM applied to store‑level panel data.
    - Per‑store evaluation with RMSE metrics and plots.
  - **Store‑Level Split Strategy:**
    - Random assignment of stores into training, validation, and testing groups.
    - Ensures robust evaluation across distinct entities.
- **Conclusion**
  - ARIMA captures linear time series dynamics.
  - LSTM demonstrates neural network forecasting on sequential data.
  - Panel data models illustrate econometric approaches to multi‑entity datasets.
  - Airline example provides estimator comparison.
  - RNN extends forecasting to panel data with per‑store evaluation.
  - Store‑level splits ensure robust training/validation/testing.
 
  ---

## ⚙️ Requirements
- Python 3.x
- Jupyter/Colab environment
- Libraries:
    - `numpy`, `pandas`, `matplotlib`, `seaborn`
    - `statsmodels`, `pmdarima`
    - `scikit-learn`
    - `keras` / `tensorflow`
    - `linearmodels`
    - `scipy`

  ---

## 🚀 Usage
1. Place datasets in the `data/` folder:
  - `Alcohol_Sales.csv`
  - `stores.csv`
  - `features.csv`
  - `sales.csv`
2. Open `TimeSeries_PanelData.ipynb` in Jupyter or Google Colab.
3. Run cells sequentially to reproduce analysis and visualizations.

---

## 🎯 Purpose
This notebook serves as a **practice archive** for combining econometric and machine learning approaches to time series and panel data. It is part of the Neural-Network-Exercises (github.com in Bing) repository under the **DigitalMarketing** section.

---

## License
This project is released under the MIT License. See [LICENSE](https://github.com/SemyonKim/Neural-Network-Exercises/blob/main/LICENSE) for details.

