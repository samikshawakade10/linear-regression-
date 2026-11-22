# House Price Prediction using Linear Regression

A complete Machine Learning academic project demonstrating end-to-end regression modeling, data preprocessing pipelines, exploratory data analysis (EDA), interactive desktop GUI, and web-based prediction dashboards.

---

## 1. Project Introduction

House Price Prediction is a fundamental supervised machine learning regression problem. Real estate prices are influenced by multiple quantitative and qualitative factors such as square footage, number of rooms, construction vintage, land area, and neighborhood rating. This project builds a reliable, reproducible, and transparent Machine Learning pipeline that maps these physical and geographical features to an estimated market property value.

---

## 2. Objective

- Build a production-ready Machine Learning pipeline strictly utilizing **Linear Regression**.
- Enforce strict statistical integrity by preventing data leakage via Scikit-Learn `Pipeline` and `ColumnTransformer`.
- Conduct exploratory data analysis (EDA) and persist publication-quality visualization figures.
- Evaluate model performance rigorously using four academic regression metrics: **MAE, MSE, RMSE, and R² Score**.
- Provide multiple user interfaces:
  1. **Native Desktop GUI Application (`gui.py`)** with tabs for prediction, evaluation, graphs, and dataset browsing.
  2. **Modern Web Dashboard (`web_app.py`)** with interactive sliders and real-time linear equation decomposition.
  3. **Interactive Python Terminal Menu (`main.py`)** for command-line valuation.

---

## 3. Algorithm Explanation: Linear Regression

Linear Regression models the relationship between a continuous dependent target variable ($y$) and one or more independent predictor features ($X_1, X_2, \dots, X_n$).

### Mathematical Formulation:
$$\hat{y} = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_n X_n + \epsilon$$

Where:
- $\hat{y}$: Predicted House Price.
- $\beta_0$: Intercept (expected baseline price when all explanatory features equal zero).
- $\beta_i$: Regression coefficient (weight) for feature $X_i$, denoting the marginal change in house price per unit change in $X_i$, holding other variables constant.
- $\epsilon$: Residual error term.

### Optimization Criterion (Ordinary Least Squares - OLS):
The parameters $\beta$ are computed by minimizing the Residual Sum of Squares (RSS):
$$\text{RSS} = \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$

### Key Assumptions of Classical Linear Regression:
1. **Linearity**: The relationship between independent and dependent variables is linear.
2. **Homoscedasticity**: The variance of residual errors is constant across all predicted values.
3. **Normality of Residuals**: The prediction errors are normally distributed.
4. **No Multicollinearity**: Features should not be linearly dependent on each other.

---

## 4. Technologies & Libraries Used

| Technology / Library | Version | Purpose |
|----------------------|---------|---------|
| **Python** | 3.13+ | Core programming language |
| **Pandas** | ≥ 2.0.0 | Tabular dataset loading, inspection, and manipulation |
| **NumPy** | ≥ 1.24.0 | Numerical operations, matrix computations, and RMSE calculation |
| **Scikit-Learn** | ≥ 1.3.0 | `LinearRegression`, `Pipeline`, `ColumnTransformer`, `SimpleImputer`, `OneHotEncoder`, metrics |
| **Tkinter & Pillow** | Standard / ≥ 9.0 | Native Windows desktop GUI (`gui.py`) |
| **HTML5 / Tailwind CSS** | CDN | Responsive Web Dashboard (`web_app.py`) |
| **Matplotlib** | ≥ 3.7.0 | Figure rendering, regression subplots, and residual charts |
| **Seaborn** | ≥ 0.12.0 | Statistical visualization (correlation heatmap, distributions) |
| **Joblib** | ≥ 1.3.0 | Pipeline persistence (`house_price_model.pkl`) |

---

## 5. Dataset Description

The project uses the actual house price dataset (`house_price_regression_dataset.csv`) containing 1,000 real-world housing records with 7 independent features and 1 continuous target:

| Column Name | Data Type | Description | Observed Range |
|-------------|-----------|-------------|----------------|
| `Square_Footage` | Integer | Total interior living space in square feet | 503 – 4,999 sq ft |
| `Num_Bedrooms` | Integer | Count of full bedrooms | 1 – 5 bedrooms |
| `Num_Bathrooms` | Integer | Count of bathrooms | 1 – 3 bathrooms |
| `Year_Built` | Integer | Construction calendar year | 1950 – 2022 |
| `Lot_Size` | Float | Property land area in acres | 0.51 – 4.99 acres |
| `Garage_Size` | Integer | Garage capacity in passenger vehicles | 0 – 2 cars |
| `Neighborhood_Quality` | Integer | Neighborhood tier rating (scale 1 to 10) | 1 – 10 |
| **`House_Price`** *(Target)* | Float | Market price of the house in US Dollars ($) | $111,626 – $1,108,237 |

- **Missing Values**: 0 (Clean dataset; preprocessing pipeline includes `SimpleImputer` for future-proof robustness).
- **Duplicate Rows**: 0.

---

## 6. Project Directory Structure

```
D:\project\
│
├── dataset/
│   └── house_price_regression_dataset.csv   # Safe copy of dataset
│
├── visualizations/
│   ├── correlation_heatmap.png              # Feature correlation matrix
│   ├── house_price_distribution.png         # Distribution histogram & KDE
│   ├── feature_vs_house_price.png           # Feature relationship subplots
│   ├── actual_vs_predicted.png              # Scatter plot of y_test vs y_pred
│   └── residual_plot.png                    # Errors vs predicted price
│
├── model/
│   ├── house_price_model.pkl                # Serialized preprocessing + OLS pipeline
│   └── metrics.json                         # Saved evaluation scores & coefficients
│
├── main.py                                  # Multi-interface launcher & terminal menu
├── gui.py                                   # Native Windows Desktop GUI (Tkinter)
├── web_app.py                               # Modern Web Browser Dashboard (HTTP/REST)
├── preprocessing.py                         # Data loading, inspection, and pipeline transformers
├── train.py                                 # Model training, metrics calculation, and plot generation
├── predict.py                               # Terminal prediction with input validation
├── requirements.txt                         # Python dependencies
└── README.md                                # Academic project documentation
```

---

## 7. How to Run the Interfaces

### 1. Launch Native Desktop GUI (Tkinter Window)
```powershell
python gui.py
```
*Opens a native desktop application with:*
- **Price Predictor Tab**: Sliders, inputs, instant valuation, and feature dollar impact.
- **Model Evaluation Tab**: MAE, MSE, RMSE, R² cards and OLS weights table.
- **Visualizations Tab**: High-res chart viewer with captions.
- **Dataset Tab**: Interactive data table of housing records.

---

### 2. Launch Web Browser Dashboard
```powershell
python web_app.py
```
*Starts local web server at `http://127.0.0.1:5000` with:*
- Real-time prediction sliders
- Interactive linear equation decomposition
- Full visual gallery with tabs

---

### 3. Launch Terminal Interface
```powershell
python main.py
```
```
==========================================
        HOUSE PRICE PREDICTION
        Linear Regression Model
==========================================

  1. Train Model
  2. Evaluate Model
  3. Predict House Price (Terminal)
  4. Launch Desktop GUI (Tkinter Window)
  5. Launch Web Dashboard (Browser)
  6. Exit

Enter your choice (1-6):
```

---

## 8. Model Evaluation Metrics

Evaluated on the held-out 20% test dataset ($N = 200$ samples, `random_state=42`):

| Evaluation Metric | Formula | Model Value | Academic Interpretation |
|-------------------|---------|-------------|-------------------------|
| **Mean Absolute Error (MAE)** | $\frac{1}{n} \sum \|y_i - \hat{y}_i\|$ | **$8,174.58** | On average, predictions deviate by approximately \$8,174 from true prices. |
| **Mean Squared Error (MSE)** | $\frac{1}{n} \sum (y_i - \hat{y}_i)^2$ | **101,434,798.51** | Squared error measure; heavily penalizes rare large outliers. |
| **Root Mean Squared Error (RMSE)** | $\sqrt{\text{MSE}}$ | **$10,071.48** | Measures error standard deviation in original currency units (\$); closely aligns with MAE. |
| **R² Score ($R^2$)** | $1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$ | **0.9984 (99.84%)** | 99.84% of total variance in house prices is explained by the Linear Regression model. |

### Learned Model Equation:
$$\text{House\_Price} = -2,010,567.42 + 199.51(\text{Square\_Footage}) + 10,225.20(\text{Bedrooms}) + 8,208.43(\text{Bathrooms}) + 993.54(\text{Year\_Built}) + 14,885.38(\text{Lot\_Size}) + 5,146.15(\text{Garage\_Size}) + 115.07(\text{Neighborhood\_Quality})$$


 ### Faculty
(https://github.com/complexiota)