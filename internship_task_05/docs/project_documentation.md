# Project Documentation: Regression Modeling From Scratch to Production

## Internship Task 05: Medical Insurance Cost Prediction

### 1. Introduction
This document provides comprehensive technical documentation for the "Regression Modeling: From Scratch to Production" project, focusing on the Medical Insurance Cost Prediction task. The project aims to develop a robust understanding of regression algorithms, their implementation from first principles, and their deployment in a practical application. It covers data preprocessing, model development (from scratch and using scikit-learn), evaluation, diagnostics, and a user-friendly Streamlit interface.

### 2. System Architecture
The project is structured into three main components:
-   **Data**: Contains the raw dataset (`insurance.csv`).
-   **Notebooks**: Houses the development and analysis work, including custom regression implementations and the Google Colab notebook for benchmarking and diagnostics.
-   **Application**: Contains the Streamlit web application for interactive visualization and real-world predictions.

```
internship_task_05/
├── data/
│   └── insurance.csv
├── notebooks/
│   ├── linear_regression_scratch.py  # From-scratch implementations of Linear, Ridge, Lasso Regression
│   └── regression_from_scratch_to_production.ipynb # Google Colab notebook for analysis and benchmarking
├── app/
│   └── app.py                      # Streamlit web application
├── docs/
│   └── project_documentation.md    # This documentation file
└── README.md                       # Project README file
```

### 3. Data Source and Preprocessing

#### 3.1. Dataset
The **Medical Cost Personal Dataset** (`insurance.csv`) is sourced from Kaggle [1]. It comprises 1,338 entries with 7 features and one target variable (`charges`).

**Features:**
-   `age`: Age of the primary beneficiary (numerical).
-   `sex`: Gender of the insurance contractor (categorical: male, female).
-   `bmi`: Body Mass Index (numerical).
-   `children`: Number of children covered by health insurance (numerical).
-   `smoker`: Smoking status (categorical: yes, no).
-   `region`: Residential area in the US (categorical: northeast, southeast, southwest, northwest).

**Target Variable:**
-   `charges`: Individual medical costs billed by health insurance (numerical).

#### 3.2. Preprocessing Steps
Data preprocessing is handled using `sklearn.compose.ColumnTransformer` to apply different transformations to numerical and categorical features:
-   **Numerical Features (`age`, `bmi`, `children`)**: Scaled using `StandardScaler` to standardize their range, which is crucial for gradient-descent-based algorithms.
-   **Categorical Features (`sex`, `smoker`, `region`)**: Encoded using `OneHotEncoder` to convert them into a numerical format suitable for regression models. `handle_unknown=\'ignore\'` is used to manage unseen categories during prediction gracefully.

### 4. Regression Model Implementations

#### 4.1. From-Scratch Implementations (`linear_regression_scratch.py`)
This module contains Python classes for implementing Linear, Ridge, and Lasso Regression from fundamental mathematical principles using NumPy. These implementations utilize Gradient Descent for optimizing model parameters.

-   **`LinearRegressionFromScratch`**: Implements ordinary least squares regression. The `fit` method iteratively updates weights and bias using the gradient of the Mean Squared Error loss function.
-   **`RidgeRegressionFromScratch`**: Extends `LinearRegressionFromScratch` by adding an L2 regularization term to the loss function. This term penalizes large coefficients, helping to prevent overfitting. The gradient update includes an additional term proportional to the weights.
-   **`LassoRegressionFromScratch`**: Extends `LinearRegressionFromScratch` by adding an L1 regularization term to the loss function. This term encourages sparsity in the coefficients, effectively performing feature selection. The gradient update incorporates the sign of the weights (subgradient).

#### 4.2. Scikit-learn Implementations
For benchmarking purposes, standard implementations from the `sklearn.linear_model` module are used:
-   `sklearn.linear_model.LinearRegression`
-   `sklearn.linear_model.Ridge`
-   `sklearn.linear_model.Lasso`

These models are trained on the same preprocessed data as the from-scratch versions to ensure a fair comparison.

### 5. Model Evaluation and Diagnostics

#### 5.1. Evaluation Metrics
Model performance is assessed using the following metrics:
-   **Mean Squared Error (MSE)**: Average of the squared differences between predicted and actual values.
-   **Root Mean Squared Error (RMSE)**: The square root of MSE, providing an error measure in the same units as the target variable.
-   **Mean Absolute Error (MAE)**: Average of the absolute differences between predicted and actual values.
-   **R-squared (R2)**: Represents the proportion of the variance in the dependent variable that is predictable from the independent variables.

#### 5.2. Residual Analysis
Residual analysis is performed to check the assumptions of linear regression and identify potential issues:
-   **Residuals vs. Predicted Values Plot**: Helps to detect non-linearity, heteroscedasticity (non-constant variance of errors), and outliers. Ideally, residuals should be randomly scattered around zero.
-   **Distribution of Residuals (Histogram/KDE)**: Checks for normality of residuals. A bell-shaped distribution centered at zero is desirable.
-   **Q-Q Plot of Residuals**: Compares the distribution of residuals to a normal distribution. Points should ideally fall along the 45-degree line.

#### 5.3. Feature Importance
For the scikit-learn Linear Regression model, feature importance is inferred from the magnitude of the coefficients. A bar plot visualizes the absolute coefficients, indicating the relative impact of each feature on the predicted insurance charges.

### 6. Streamlit Application (`app/app.py`)
The Streamlit application provides an interactive interface for exploring the dataset and making predictions.

#### 6.1. Project Overview
An introductory page explaining the project's purpose and features.

#### 6.2. Interactive Visualization
-   **Distribution Plots**: Histograms and box plots to visualize the distribution of `charges` and its relationship with categorical features like `smoker`.
-   **Dynamic Scatter Plot**: A scatter plot of `age` vs. `charges`, with interactive sliders for `age` and `bmi` ranges, allowing users to filter the data and observe trends.

#### 6.3. Real-world Prediction Tool
-   **User Input Form**: Allows users to input values for `age`, `sex`, `bmi`, `children`, `smoker`, and `region`.
-   **Cost Estimation**: Uses the trained scikit-learn `LinearRegression` model to predict the insurance cost based on the provided inputs.
-   **Comparison with Average**: Displays the estimated cost and compares it to the average cost in the dataset, highlighting whether the prediction is above or below the average.

### 7. Development Environment and Dependencies

#### 7.1. Python Libraries
-   `numpy`
-   `pandas`
-   `matplotlib`
-   `seaborn`
-   `scikit-learn`
-   `streamlit`
-   `nbformat`
-   `statsmodels`

#### 7.2. Setup Instructions
To set up the environment and run the project, refer to the `README.md` file in the project root directory.

### 8. Future Enhancements
-   **Advanced Models**: Explore more complex regression models (e.g., Gradient Boosting, Random Forests) for potentially improved accuracy.
-   **Hyperparameter Tuning**: Implement advanced hyperparameter tuning techniques for optimized model performance.
-   **Model Interpretability**: Integrate tools like SHAP or LIME for deeper insights into model predictions.
-   **Deployment**: Containerize the Streamlit application using Docker for easier deployment.
-   **Database Integration**: Connect the Streamlit app to a database for storing user inputs and predictions.

### 9. References
[1] Medical Cost Personal Dataset. Kaggle. Available at: [https://www.kaggle.com/datasets/mirichoi0218/insurance](https://www.kaggle.com/datasets/mirichoi0218/insurance)
